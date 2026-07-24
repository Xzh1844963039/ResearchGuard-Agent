# C:\Users\18449\Desktop\researchguard_workspace\researchguard\agent\replanner.py
from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from researchguard.agent.state import ResearchAgentState, utc_timestamp
from researchguard.tools import ToolResult


PLAN_REVISION_SCHEMA_VERSION = "researchguard.plan_revision.v1"


@dataclass(frozen=True)
class PlanRevision:
    previous_plan: tuple[Mapping[str, Any], ...]
    observation: Mapping[str, Any]
    new_plan: tuple[Mapping[str, Any], ...]
    reason: str
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
            "reason": self.reason,
        }


class BoundedReplanner:
    def __init__(
        self,
        *,
        max_revisions: int = 2,
        expanded_top_k: int = 20,
        expanded_candidate_k: int = 160,
    ):
        if max_revisions < 0:
            raise ValueError("max_revisions must not be negative.")
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
            },
            {
                "tool": "assess_evidence",
                "purpose": "Reassess the expanded canonical evidence.",
                "optional": False,
                "parameters": {},
                "recovery_terminal": False,
            },
            {
                "tool": "generate_grounded_answer",
                "purpose": "Generate only when the revised evidence gate is strong.",
                "optional": False,
                "parameters": {},
                "recovery_terminal": False,
            },
            {
                "tool": "audit_answer",
                "purpose": "Audit the revised answer against the reused evidence bundle.",
                "optional": False,
                "parameters": {},
                "recovery_terminal": False,
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
            }
        ]
        return self._revision(state, observation, recovery_steps, reason)

    @staticmethod
    def _revision(
        state: ResearchAgentState,
        observation: Mapping[str, Any],
        recovery_steps: list[dict[str, Any]],
        reason: str,
    ) -> PlanRevision:
        previous = copy.deepcopy(state.plan)
        completed = copy.deepcopy(state.plan[: state.current_step + 1])
        combined = completed + recovery_steps
        for index, step in enumerate(combined, start=1):
            step["step_id"] = index
        return PlanRevision(
            previous_plan=tuple(previous),
            observation=copy.deepcopy(dict(observation)),
            new_plan=tuple(combined),
            reason=reason,
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
            "error": result.error.to_dict() if result.error else None,
            "trace_id": result.trace_id,
        }
