# C:\Users\18449\Desktop\researchguard_workspace\researchguard\agent\replanner.py
from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from researchguard.agent.state import ResearchAgentState, utc_timestamp
from researchguard.agent.subgoal_evidence import refined_subgoal_query
from researchguard.skills import SkillResult
from researchguard.tools import ToolResult


PLAN_REVISION_SCHEMA_VERSION = "researchguard.plan_revision.v2"


@dataclass(frozen=True)
class PlanRevision:
    previous_plan: tuple[Mapping[str, Any], ...]
    observation: Mapping[str, Any]
    new_plan: tuple[Mapping[str, Any], ...]
    reason: str
    subgoal_changes: tuple[Mapping[str, Any], ...] = ()
    revision_id: str = ""
    created_at: str = ""
    version: str = "1.0.0"
    schema_version: str = PLAN_REVISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("PlanRevision.reason must not be empty.")
        object.__setattr__(
            self,
            "revision_id",
            self.revision_id or f"revision-{uuid.uuid4().hex}",
        )
        object.__setattr__(self, "created_at", self.created_at or utc_timestamp())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "revision_id": self.revision_id,
            "created_at": self.created_at,
            "previous_plan": copy.deepcopy(list(self.previous_plan)),
            "observation": copy.deepcopy(dict(self.observation)),
            "new_plan": copy.deepcopy(list(self.new_plan)),
            "previous_skill": self.observation.get(
                "skill_name",
                self.observation.get("tool_name"),
            ),
            "new_skill_sequence": [
                str(step.get("skill"))
                for step in self.new_plan
                if step.get("skill")
            ],
            "subgoal_changes": copy.deepcopy(list(self.subgoal_changes)),
            "reason": self.reason,
        }


class BoundedReplanner:
    max_runtime_subgoals = 4

    def __init__(
        self,
        *,
        max_revisions: int = 2,
        expanded_top_k: int = 20,
        expanded_candidate_k: int = 160,
    ):
        if not 0 <= max_revisions <= 2:
            raise ValueError("max_revisions must be between 0 and 2.")
        self.max_revisions = max_revisions
        self.expanded_top_k = max(1, int(expanded_top_k))
        self.expanded_candidate_k = max(1, int(expanded_candidate_k))

    def revise(
        self,
        state: ResearchAgentState,
        *,
        tool_name: str,
        result: ToolResult,
        available_tools: tuple[str, ...],
    ) -> PlanRevision | None:
        if len(state.plan_revisions) >= self.max_revisions:
            return None
        observation = self._observation(tool_name, result)
        reasons = {
            str(item.get("reason", "")) for item in state.plan_revisions
        }
        has_expanded_retrieval = any(
            reason.startswith("expanded_retrieval") for reason in reasons
        )

        if tool_name == "retrieve_evidence":
            retrieval_failed = result.status == "failed"
            evidence_count = int(observation.get("evidence_count", 0))
            coverage_low = bool(observation.get("coverage_low", False))
            if retrieval_failed or evidence_count == 0 or coverage_low:
                if not has_expanded_retrieval:
                    return self._expanded_retrieval(
                        state,
                        observation,
                        reason="expanded_retrieval_after_miss",
                    )
                if (
                    "scholarly_discovery_after_retrieval_miss" not in reasons
                    and "search_scholarly_sources" in available_tools
                ):
                    return self._scholarly_fallback(
                        state,
                        observation,
                        reason="scholarly_discovery_after_retrieval_miss",
                    )
            return None

        if tool_name == "assess_evidence":
            support_level = str(observation.get("support_level", "")).casefold()
            if (
                result.status == "rejected"
                and support_level == "partial"
                and not has_expanded_retrieval
            ):
                return self._expanded_retrieval(
                    state,
                    observation,
                    reason="expanded_retrieval_after_partial_evidence",
                )
            return None

        return None

    def revise_skill(
        self,
        state: ResearchAgentState,
        *,
        skill_name: str,
        result: SkillResult,
        available_skills: tuple[str, ...],
        step: Mapping[str, Any] | None = None,
    ) -> PlanRevision | None:
        if len(state.plan_revisions) >= self.max_revisions:
            return None
        observation = self._skill_observation(
            state,
            skill_name,
            result,
            step=step,
        )
        subgoal_id = str(observation.get("subgoal_id") or "").strip() or None
        reasons = {
            str(item.get("reason", "")) for item in state.plan_revisions
        }
        has_expanded_retrieval = any(
            str(item.get("reason", "")).startswith("expanded_retrieval")
            and (
                not subgoal_id
                or item.get("observation", {}).get("subgoal_id") == subgoal_id
            )
            for item in state.plan_revisions
        )

        if skill_name == "retrieve_evidence":
            evidence_count = int(observation.get("evidence_count", 0))
            coverage_low = bool(observation.get("coverage_low", False))
            if result.status == "failed" or evidence_count == 0 or coverage_low:
                if not has_expanded_retrieval:
                    if subgoal_id:
                        return self._expanded_subgoal_skill_retrieval(
                            state,
                            observation,
                            reason="expanded_retrieval_for_missing_subgoal",
                        )
                    return self._expanded_skill_retrieval(
                        state,
                        observation,
                        reason="expanded_retrieval_after_miss",
                    )
                if (
                    not self._has_scholarly_fallback(
                        state,
                        subgoal_id=subgoal_id,
                    )
                    and "search_papers" in available_skills
                ):
                    return self._skill_scholarly_fallback(
                        state,
                        observation,
                        reason="scholarly_discovery_after_retrieval_miss",
                    )
            support_level = str(
                observation.get("support_level", "")
            ).casefold()
            if (
                subgoal_id
                and result.status == "success"
                and support_level in {"partial", "unsupported"}
                and not self._has_subgoal_gate_revision(
                    state,
                    subgoal_id=subgoal_id,
                )
            ):
                return self._refined_subgoal_skill_retrieval(
                    state,
                    observation,
                    reason="refine_subgoal_query_after_evidence_gap",
                )
            return None

        if skill_name in {"generate_report", "assess_evidence"}:
            support_level = str(
                observation.get("support_level", "")
            ).casefold()
            if state.subgoal_results:
                if (
                    result.status == "rejected"
                    and support_level == "partial"
                    and all(
                        item.get("status") == "satisfied"
                        for item in state.subgoal_results.values()
                    )
                    and len(state.subgoals) < self.max_runtime_subgoals
                    and not any(
                        item.get("reason") == "insert_subgoal_for_aggregate_gate_gap"
                        for item in state.plan_revisions
                    )
                ):
                    return self._insert_aggregate_gap_subgoal(
                        state,
                        observation,
                        reason="insert_subgoal_for_aggregate_gate_gap",
                    )
                return None
            if (
                result.status == "rejected"
                and support_level == "partial"
                and not has_expanded_retrieval
            ):
                return self._expanded_skill_retrieval(
                    state,
                    observation,
                    reason="expanded_retrieval_after_partial_evidence",
                )
        return None

    def _refined_subgoal_skill_retrieval(
        self,
        state: ResearchAgentState,
        observation: Mapping[str, Any],
        *,
        reason: str,
    ) -> PlanRevision:
        subgoal_id = str(observation.get("subgoal_id") or "").strip()
        subgoal = state.subgoal_results[subgoal_id]
        current = copy.deepcopy(state.plan[state.current_step])
        parameters = copy.deepcopy(dict(current.get("parameters", {}) or {}))
        new_query = refined_subgoal_query(subgoal)
        parameters.update(
            {
                "query": new_query,
                "rewrite": True,
                "multi_query": True,
                "top_k": self.expanded_top_k,
                "candidate_k": self.expanded_candidate_k,
                "read_cache": False,
            }
        )
        current.update(
            {
                "purpose": "Resolve the observed subgoal evidence gap with a refined query.",
                "parameters": parameters,
                "recovery_terminal": False,
                "max_retry": 1,
            }
        )
        return self._revision(
            state,
            observation,
            [current],
            reason,
            preserve_remaining=True,
            subgoal_changes=[
                {
                    "operation": "revise_query",
                    "subgoal_id": subgoal_id,
                    "previous_query": observation.get("subgoal_query"),
                    "new_query": new_query,
                    "success_criterion": subgoal.get("success_criterion"),
                }
            ],
        )

    def _insert_aggregate_gap_subgoal(
        self,
        state: ResearchAgentState,
        observation: Mapping[str, Any],
        *,
        reason: str,
    ) -> PlanRevision:
        suffix = len(state.plan_revisions) + 1
        subgoal_id = f"sg-gap-{suffix}"
        gap_reason = " ".join(
            str(observation.get("gate_reason") or "aggregate evidence remained partial")
            .split()
        )
        if len(gap_reason) > 480:
            gap_reason = gap_reason[:480].rsplit(" ", 1)[0]
        question = (
            f"{state.query} Focus only on the unresolved evidence requirement: "
            f"{gap_reason}"
        )
        subgoal = {
            "subgoal_id": subgoal_id,
            "question": question,
            "objective": "Resolve the evidence gap found by the final aggregate Gate.",
            "success_criterion": (
                "Canonical evidence directly resolves this Gate observation: "
                f"{gap_reason}"
            ),
        }
        recovery_steps = [
            {
                "skill": "retrieve_evidence",
                "tool": "",
                "purpose": "Collect canonical evidence for the aggregate Gate gap.",
                "parameters": {
                    "query": question,
                    "rewrite": True,
                    "multi_query": True,
                    "top_k": self.expanded_top_k,
                    "candidate_k": self.expanded_candidate_k,
                    "read_cache": False,
                },
                "subgoal_id": subgoal_id,
                "recovery_terminal": False,
                "max_retry": 1,
            },
            {
                "skill": "generate_report",
                "tool": "",
                "purpose": "Re-run the protected aggregate Gate and finish if supported.",
                "parameters": {},
                "subgoal_id": None,
                "recovery_terminal": False,
                "max_retry": 0,
            },
        ]
        return self._revision(
            state,
            observation,
            recovery_steps,
            reason,
            subgoal_changes=[{"operation": "add", "subgoal": subgoal}],
        )

    def _expanded_subgoal_skill_retrieval(
        self,
        state: ResearchAgentState,
        observation: Mapping[str, Any],
        *,
        reason: str,
    ) -> PlanRevision:
        current = copy.deepcopy(state.plan[state.current_step])
        parameters = copy.deepcopy(dict(current.get("parameters", {}) or {}))
        parameters.update(
            {
                "rewrite": True,
                "multi_query": True,
                "top_k": self.expanded_top_k,
                "candidate_k": self.expanded_candidate_k,
                "read_cache": False,
            }
        )
        current.update(
            {
                "purpose": (
                    "Target the missing subgoal with expanded canonical retrieval."
                ),
                "parameters": parameters,
                "recovery_terminal": False,
                "max_retry": 1,
            }
        )
        return self._revision(
            state,
            observation,
            [current],
            reason,
            preserve_remaining=True,
        )

    def _expanded_skill_retrieval(
        self,
        state: ResearchAgentState,
        observation: Mapping[str, Any],
        *,
        reason: str,
    ) -> PlanRevision:
        recovery_steps: list[dict[str, Any]] = [
            {
                "skill": "retrieve_evidence",
                "tool": "",
                "purpose": "Expand query rewriting and canonical retrieval.",
                "parameters": {
                    "rewrite": True,
                    "multi_query": True,
                    "top_k": self.expanded_top_k,
                    "candidate_k": self.expanded_candidate_k,
                    "read_cache": False,
                },
                "recovery_terminal": False,
                "max_retry": 1,
            }
        ]
        if state.task_type in {"comparison", "paper_comparison"}:
            recovery_steps.append(
                {
                    "skill": "compare_evidence",
                    "tool": "",
                    "purpose": "Rebuild comparison groups from expanded evidence.",
                    "parameters": {},
                    "recovery_terminal": False,
                    "max_retry": 0,
                }
            )
        elif state.task_type == "claim_audit":
            recovery_steps.append(
                {
                    "skill": "analyze_claims",
                    "tool": "",
                    "purpose": "Re-extract claims from expanded evidence.",
                    "parameters": {},
                    "recovery_terminal": False,
                    "max_retry": 0,
                }
            )
        terminal_skill = (
            "audit_claims"
            if state.task_type in {"claim_audit", "audit"}
            else "generate_report"
        )
        recovery_steps.append(
            {
                "skill": terminal_skill,
                "tool": "",
                "purpose": "Finish through the protected evidence boundary.",
                "parameters": {},
                "recovery_terminal": False,
                "max_retry": 0,
            }
        )
        return self._revision(state, observation, recovery_steps, reason)

    def _skill_scholarly_fallback(
        self,
        state: ResearchAgentState,
        observation: Mapping[str, Any],
        *,
        reason: str,
    ) -> PlanRevision:
        subgoal_id = str(observation.get("subgoal_id") or "").strip() or None
        query = str(observation.get("subgoal_question") or state.query)
        return self._revision(
            state,
            observation,
            [
                {
                    "skill": "search_papers",
                    "tool": "",
                    "purpose": (
                        "Discover metadata-only candidates after corpus retrieval "
                        "failed; do not treat them as evidence."
                    ),
                    "parameters": {"limit": 5, "query": query},
                    "subgoal_id": subgoal_id,
                    "recovery_terminal": True,
                    "max_retry": 1,
                }
            ],
            reason,
        )

    def _expanded_retrieval(
        self,
        state: ResearchAgentState,
        observation: Mapping[str, Any],
        *,
        reason: str,
    ) -> PlanRevision:
        recovery_steps = [
            {
                "tool": "retrieve_evidence",
                "purpose": "Rewrite the query and expand retrieval after weak evidence.",
                "optional": False,
                "parameters": {
                    "rewrite": True,
                    "multi_query": True,
                    "top_k": self.expanded_top_k,
                    "candidate_k": self.expanded_candidate_k,
                    "read_cache": False,
                },
                "recovery_terminal": False,
                "max_retry": 1,
            },
            {
                "tool": "assess_evidence",
                "purpose": "Reassess the expanded canonical evidence.",
                "optional": False,
                "parameters": {},
                "recovery_terminal": False,
                "max_retry": 1,
            },
            {
                "tool": "generate_grounded_answer",
                "purpose": "Generate only when the revised evidence gate is strong.",
                "optional": False,
                "parameters": {},
                "recovery_terminal": False,
                "max_retry": 0,
            },
            {
                "tool": "audit_answer",
                "purpose": "Audit the revised answer against the reused evidence bundle.",
                "optional": False,
                "parameters": {},
                "recovery_terminal": False,
                "max_retry": 1,
            },
        ]
        return self._revision(state, observation, recovery_steps, reason)

    def _scholarly_fallback(
        self,
        state: ResearchAgentState,
        observation: Mapping[str, Any],
        *,
        reason: str,
    ) -> PlanRevision:
        recovery_steps = [
            {
                "tool": "search_scholarly_sources",
                "purpose": (
                    "Discover metadata-only candidates after corpus retrieval failed; "
                    "do not treat them as answer evidence."
                ),
                "optional": False,
                "parameters": {"limit": 5},
                "recovery_terminal": True,
                "max_retry": 1,
            }
        ]
        return self._revision(state, observation, recovery_steps, reason)

    @staticmethod
    def _revision(
        state: ResearchAgentState,
        observation: Mapping[str, Any],
        recovery_steps: list[dict[str, Any]],
        reason: str,
        *,
        preserve_remaining: bool = False,
        subgoal_changes: list[dict[str, Any]] | None = None,
    ) -> PlanRevision:
        previous = copy.deepcopy(state.plan)
        completed = copy.deepcopy(state.plan[: state.current_step + 1])
        remaining = (
            copy.deepcopy(state.plan[state.current_step + 1 :])
            if preserve_remaining
            else []
        )
        combined = completed + recovery_steps + remaining
        for index, step in enumerate(combined, start=1):
            step["step_id"] = index
        return PlanRevision(
            previous_plan=tuple(previous),
            observation=copy.deepcopy(dict(observation)),
            new_plan=tuple(combined),
            reason=reason,
            subgoal_changes=tuple(copy.deepcopy(subgoal_changes or [])),
        )

    @staticmethod
    def _observation(tool_name: str, result: ToolResult) -> dict[str, Any]:
        evidence = result.data.get("evidence", ())
        evidence_count = len(evidence) if isinstance(evidence, list) else 0
        retrieval = result.data.get("retrieval", {})
        coverage = (
            retrieval.get("coverage")
            if isinstance(retrieval, Mapping)
            else None
        )
        assessment = result.data.get("assessment", {})
        support_level = (
            assessment.get("support_level")
            if isinstance(assessment, Mapping)
            else None
        )
        error = result.error.to_dict() if result.error else None
        return {
            "tool_name": tool_name,
            "status": result.status,
            "reason": result.reason,
            "evidence_count": evidence_count,
            "coverage": coverage,
            "coverage_low": (
                isinstance(coverage, (int, float)) and float(coverage) < 0.25
            ),
            "support_level": support_level,
            "error_type": result.error.category if result.error else None,
            "error_code": result.error.code if result.error else None,
            "retryable": result.error.retryable if result.error else False,
            "error": error,
            "trace_id": result.trace_id,
        }

    @staticmethod
    def _skill_observation(
        state: ResearchAgentState,
        skill_name: str,
        result: SkillResult,
        *,
        step: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        step_value = dict(step or {})
        subgoal_id = str(step_value.get("subgoal_id") or "").strip() or None
        parameters = dict(step_value.get("parameters", {}) or {})
        coverage = result.output.get("coverage")
        subgoal = state.subgoal_results.get(subgoal_id or "", {})
        subgoal_gate = subgoal.get("gate_decision")
        gate_status = (
            str(subgoal_gate.get("status", ""))
            if isinstance(subgoal_gate, Mapping)
            else ""
        )
        result_gate = result.output.get("gate_decision")
        gate_reason = (
            str(result_gate.get("reason", ""))
            if isinstance(result_gate, Mapping)
            else ""
        )
        return {
            "skill_name": skill_name,
            "status": result.status,
            "reason": result.reason,
            "evidence_count": (
                int(result.output.get("evidence_count", len(result.evidence)))
            ),
            "coverage": coverage,
            "coverage_low": (
                isinstance(coverage, (int, float)) and float(coverage) < 0.25
            ),
            "support_level": gate_status or result.output.get("support_level"),
            "gate_reason": (
                str(subgoal_gate.get("reason", ""))
                if isinstance(subgoal_gate, Mapping)
                else gate_reason
            ),
            "error_type": result.output.get("error_type"),
            "error_code": result.output.get("error_code"),
            "retryable": bool(result.output.get("retryable", False)),
            "trace_id": result.trace_id,
            "subgoal_id": subgoal_id,
            "subgoal_question": subgoal.get("question"),
            "subgoal_query": parameters.get("query"),
            "subgoal_status": subgoal.get("status"),
            "success_criterion": subgoal.get("success_criterion"),
            "completion_criterion_met": subgoal.get(
                "completion_criterion_met"
            ),
            "missing_subgoal_ids": [
                key
                for key, value in state.subgoal_results.items()
                if value.get("status") != "satisfied"
            ],
        }

    @staticmethod
    def _has_subgoal_gate_revision(
        state: ResearchAgentState,
        *,
        subgoal_id: str,
    ) -> bool:
        return any(
            item.get("reason") == "refine_subgoal_query_after_evidence_gap"
            and item.get("observation", {}).get("subgoal_id") == subgoal_id
            for item in state.plan_revisions
        )

    @staticmethod
    def _has_scholarly_fallback(
        state: ResearchAgentState,
        *,
        subgoal_id: str | None,
    ) -> bool:
        return any(
            item.get("reason") == "scholarly_discovery_after_retrieval_miss"
            and (
                not subgoal_id
                or item.get("observation", {}).get("subgoal_id") == subgoal_id
            )
            for item in state.plan_revisions
        )
