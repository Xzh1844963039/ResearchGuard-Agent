# C:\Users\18449\Desktop\researchguard_workspace\researchguard\agent\controller.py
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from researchguard.agent.hybrid_planner import (
    DEFAULT_PLANNER_CONFIG,
    DeterministicPlanner,
    HybridPlanner,
    PlannerOutcome,
    load_hybrid_planner_settings,
)
from researchguard.agent.planner import PlannerError
from researchguard.agent.planner_schema import (
    PlanBudget,
    StructuredPlan,
    StructuredPlanStep,
)
from researchguard.agent.planner_validator import PlanValidationResult
from researchguard.agent.policy import AgentPolicy
from researchguard.agent.replanner import BoundedReplanner
from researchguard.agent.state import ResearchAgentState, utc_timestamp
from researchguard.memory import DEFAULT_MEMORY_ROOT, ResearchMemory
from researchguard.skills import SkillRegistry, build_default_skill_registry
from researchguard.tools import (
    EvidenceBundle,
    EvidenceRecord,
    GateDecision,
    ScholarPaperRecord,
    ToolError,
    ToolRegistry,
    ToolResult,
    build_default_registry,
)
from researchguard.workflows import (
    WorkflowLimits,
    WorkflowRegistry,
    build_default_workflow_registry,
)


class BoundedResearchAgentController:
    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        workflow_registry: WorkflowRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        planner: Any | None = None,
        replanner: Any | None = None,
        policy: AgentPolicy | None = None,
        memory: ResearchMemory | None = None,
        memory_enabled: bool = True,
        memory_root: str | Path = DEFAULT_MEMORY_ROOT,
        config_path: str | Path = "configs/pipeline_v1.yaml",
        planner_config_path: str | Path = DEFAULT_PLANNER_CONFIG,
    ):
        custom_registry = registry is not None
        self.policy = policy or AgentPolicy()
        self.registry = registry or build_default_registry(config_path)
        self.skill_registry = skill_registry or build_default_skill_registry(
            self.registry
        )
        self.memory = memory if memory is not None else (
            ResearchMemory(memory_root) if memory_enabled else None
        )
        if workflow_registry is not None:
            self.workflow_registry = workflow_registry
        elif registry is None or {
            "retrieve_evidence",
            "assess_evidence",
            "generate_grounded_answer",
            "audit_answer",
            "search_scholarly_sources",
        }.issubset(self.registry.names):
            self.workflow_registry = build_default_workflow_registry(
                self.registry,
                limits=WorkflowLimits(
                    max_steps=self.policy.max_steps,
                    max_tool_calls=self.policy.max_tool_calls,
                    max_retry=self.policy.max_retry,
                    timeout=self.policy.timeout,
                ),
            )
        else:
            self.workflow_registry = WorkflowRegistry()
        deterministic = DeterministicPlanner(
            registry=self.registry,
            skills=self.skill_registry,
            policy=self.policy,
            workflow_names=self.workflow_registry.names,
        )
        if planner is not None:
            self.planner = planner
        elif custom_registry:
            self.planner = deterministic
        else:
            self.planner = HybridPlanner(
                registry=self.registry,
                skills=self.skill_registry,
                policy=self.policy,
                deterministic=deterministic,
                settings=load_hybrid_planner_settings(planner_config_path),
                workflow_names=self.workflow_registry.names,
            )
        self.replanner = replanner or BoundedReplanner(
            max_revisions=self.policy.max_plan_revisions,
        )

    def run(
        self,
        query: str,
        *,
        task_type: str | None = None,
        answer_artifact: Mapping[str, Any] | None = None,
        evidence: Iterable[EvidenceRecord | Mapping[str, Any]] | None = None,
        workflow_input: Mapping[str, Any] | None = None,
    ) -> ResearchAgentState:
        serialized_evidence = self._serialize_evidence(evidence or ())
        state = ResearchAgentState(
            query=query,
            answer=copy.deepcopy(dict(answer_artifact)) if answer_artifact is not None else None,
            evidence=serialized_evidence,
            workflow_input=copy.deepcopy(dict(workflow_input or {})),
            memory_status={
                "enabled": self.memory is not None,
                "started": False,
                "persisted": False,
                "errors": [],
            },
        )
        if serialized_evidence:
            state.evidence_bundle = EvidenceBundle.create(
                query=state.query,
                evidence=serialized_evidence,
                provenance={"source": "controller_supplied_evidence"},
            ).to_dict()
        self._load_memory_context(state)
        self._start_memory(state)
        return self._finalize_memory(
            self._plan_and_execute(
                state,
                task_type=task_type,
            )
        )

    def _plan_and_execute(
        self,
        state: ResearchAgentState,
        *,
        task_type: str | None,
    ) -> ResearchAgentState:
        try:
            outcome = self._generate_plan(state, task_type=task_type)
        except (PlannerError, ValueError, TypeError) as exc:
            state.set_status("failed", f"planner_error: {exc}")
            return state

        plan = outcome.executable_plan
        state.task_type = plan.task_type
        state.plan = [step.to_dict() for step in plan.steps]
        state.planner_plan = outcome.structured_plan.copy_dict()
        state.planner_metadata = outcome.metadata()
        state.workflow_name = plan.workflow
        state.set_status("planned")
        plan_error = self.policy.validate_plan(state)
        if plan_error:
            state.set_status("failed", plan_error)
            return state
        if state.workflow_name:
            return self.execute_workflow(state)
        return self.execute(state)

    def _generate_plan(
        self,
        state: ResearchAgentState,
        *,
        task_type: str | None,
    ) -> PlannerOutcome:
        kwargs = {
            "task_type": task_type,
            "has_evidence": bool(state.evidence),
            "has_answer": state.answer is not None,
            "memory_context": state.memory_context,
        }
        generate = getattr(self.planner, "generate_plan", None)
        if callable(generate):
            outcome = generate(state.query, **kwargs)
            if not isinstance(outcome, PlannerOutcome):
                raise TypeError("Planner Interface must return PlannerOutcome.")
            return outcome

        create = getattr(self.planner, "create_plan", None)
        if not callable(create):
            raise TypeError(
                "Planner must implement generate_plan(query) or create_plan(query)."
            )
        plan = create(state.query, **kwargs)
        tool_to_skill = {
            "retrieve_evidence": "retrieve_evidence",
            "search_scholarly_sources": "search_scholarly_sources",
            "assess_evidence": "assess_evidence",
            "generate_grounded_answer": "generate_report",
            "audit_answer": "audit_claims",
        }
        structured_steps: list[StructuredPlanStep] = []
        for step in plan.steps:
            skill = tool_to_skill.get(step.tool)
            if skill is None:
                raise PlannerError(
                    f"Legacy planner references an unsupported tool: {step.tool}"
                )
            spec = self.skill_registry.get(skill)
            structured_steps.append(
                StructuredPlanStep(
                    skill=skill,
                    purpose=step.purpose,
                    expected_observation=spec.output_type,
                    max_retry=(
                        self.policy.max_retry
                        if step.max_retry is None
                        else step.max_retry
                    ),
                )
            )
        if not structured_steps and plan.workflow is not None:
            structured_steps.append(
                StructuredPlanStep(
                    skill="retrieve_evidence",
                    purpose=f"Execute the registered {plan.workflow} workflow.",
                    expected_observation="WorkflowResult",
                    max_retry=self.policy.max_retry,
                )
            )
        structured = StructuredPlan(
            task_type=plan.task_type,
            goal=state.query,
            steps=tuple(structured_steps),
            budget=PlanBudget(
                max_steps=self.policy.max_steps,
                max_tool_calls=self.policy.max_tool_calls,
                max_retries=self.policy.max_retry,
                max_plan_revisions=self.policy.max_plan_revisions,
            ),
            reasoning_summary=(
                "Adapt a legacy create_plan implementation to the Planner Interface."
            ),
            planner_version="legacy_create_plan_adapter_v1",
        )
        return PlannerOutcome(
            structured_plan=structured,
            executable_plan=plan,
            mode="legacy_planner_adapter",
            fallback_used=False,
            fallback_reason=None,
            validation=PlanValidationResult(
                valid=True,
                reason=None,
                errors=(),
                validator_version="legacy_adapter_compatibility",
            ),
            planner_model=type(self.planner).__name__,
            prompt_version="legacy_create_plan_adapter_v1",
            latency_ms=0.0,
            api_call_count=0,
        )

    def _load_memory_context(self, state: ResearchAgentState) -> None:
        if self.memory is None or not hasattr(self.memory, "search_context"):
            return
        try:
            state.memory_context = self.memory.search_context(state.query)
        except Exception as exc:
            state.memory_context = {}
            state.memory_status["errors"].append(
                f"search_context: {type(exc).__name__}: {exc}"
            )
        state.touch()

    def resume(self, state: ResearchAgentState) -> ResearchAgentState:
        if state.status in {"completed", "rejected", "failed"}:
            return state
        if state.workflow_name:
            return self._finalize_memory(self.execute_workflow(state))
        if not state.plan:
            state.set_status("failed", "missing_plan")
            return self._finalize_memory(state)
        plan_error = self.policy.validate_plan(state)
        if plan_error:
            state.set_status("failed", plan_error)
            return self._finalize_memory(state)
        return self._finalize_memory(self.execute(state))

    def _start_memory(self, state: ResearchAgentState) -> None:
        if self.memory is None:
            return
        try:
            self.memory.start_run(state)
            state.memory_status["started"] = True
        except Exception as exc:
            state.memory_status["errors"].append(
                f"start_run: {type(exc).__name__}: {exc}"
            )
        state.touch()

    def _finalize_memory(self, state: ResearchAgentState) -> ResearchAgentState:
        if self.memory is None:
            return state
        try:
            result = self.memory.complete_run(state)
            previous_errors = list(state.memory_status.get("errors", []))
            result_errors = list(result.get("errors", []))
            state.memory_status.update(
                {key: value for key, value in result.items() if key != "errors"}
            )
            state.memory_status["errors"] = previous_errors + result_errors
        except Exception as exc:
            state.memory_status["persisted"] = False
            state.memory_status["errors"].append(
                f"complete_run: {type(exc).__name__}: {exc}"
            )
        state.touch()
        return state

    def execute_workflow(self, state: ResearchAgentState) -> ResearchAgentState:
        state.set_status("running")
        workflow_name = state.workflow_name or ""
        if workflow_name not in self.workflow_registry.names:
            state.set_status("failed", f"unregistered_workflow: {workflow_name or 'missing'}")
            return state
        try:
            result = self.workflow_registry.run(workflow_name, state)
        except Exception as exc:
            state.set_status(
                "failed",
                f"workflow_error: {type(exc).__name__}: {exc}",
            )
            return state
        state.workflow_result = result.to_dict()
        state.current_step = len(state.workflow_steps)
        if result.status == "success":
            state.set_status("completed")
        elif result.status == "rejected":
            state.set_status("rejected", result.reason or "workflow_rejected")
        else:
            state.set_status("failed", result.reason or "workflow_failed")
        return state

    def execute(self, state: ResearchAgentState) -> ResearchAgentState:
        started = time.perf_counter()
        state.set_status("running")

        while state.current_step < len(state.plan):
            stop_reason = self.policy.stop_reason(
                state,
                elapsed_seconds=time.perf_counter() - started,
            )
            if stop_reason:
                state.set_status("failed", stop_reason)
                return state

            step = state.plan[state.current_step]
            tool_name = str(step.get("tool", ""))
            if tool_name not in self.registry.names:
                state.set_status("failed", f"unregistered_tool: {tool_name or 'missing'}")
                return state

            try:
                tool_input = self._build_tool_input(tool_name, state, step)
            except (ValueError, TypeError) as exc:
                state.set_status("failed", f"invalid_tool_input: {exc}")
                return state

            try:
                result = self.registry.invoke(tool_name, **tool_input)
            except Exception as exc:
                result = ToolResult.create(
                    status="failed",
                    message="Registered tool raised an unhandled exception.",
                    reason="unhandled_tool_exception",
                    tool_name=tool_name,
                    tool_version=self.registry.version,
                    latency_ms=0.0,
                    error=ToolError(
                        code="unhandled_tool_exception",
                        category="execution_failure",
                        message=str(exc),
                        retryable=False,
                        details={"exception_type": type(exc).__name__},
                    ),
                )
            self._record_tool_result(state, tool_name, tool_input, result)
            try:
                self._apply_result(state, tool_name, result)
            except (TypeError, ValueError) as exc:
                state.observations[-1]["output_validation_error"] = {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
                state.set_status("failed", "invalid_tool_output")
                return state

            if time.perf_counter() - started >= self.policy.timeout:
                state.set_status("failed", "timeout_exceeded")
                return state

            revision = self.replanner.revise(
                state,
                tool_name=tool_name,
                result=result,
                available_tools=self.registry.names,
            )
            if revision is not None:
                state.plan_revisions.append(revision.to_dict())
                state.plan = copy.deepcopy(list(revision.new_plan))
                state.current_step += 1
                state.retry_counts.clear()
                state.touch()
                plan_error = self.policy.validate_plan(state)
                if plan_error:
                    state.set_status("failed", plan_error)
                    return state
                continue

            if result.status == "failed":
                retry_key = str(state.current_step)
                retry_count = state.retry_counts.get(retry_key, 0)
                retryable = bool(result.error and result.error.retryable)
                step_retry_limit = step.get("max_retry")
                retry_limit = (
                    self.policy.max_retry
                    if step_retry_limit is None
                    else min(
                        self.policy.max_retry,
                        max(0, int(step_retry_limit)),
                    )
                )
                if retryable and retry_count < retry_limit:
                    state.retry_counts[retry_key] = retry_count + 1
                    state.touch()
                    continue
                state.set_status("failed", "tool_error")
                return state

            if result.status == "rejected":
                reason = self._rejection_reason(tool_name, result)
                state.set_status("rejected", reason)
                return state

            if bool(step.get("recovery_terminal", False)):
                state.current_step += 1
                state.set_status(
                    "rejected",
                    "no_corpus_evidence_scholarly_candidates_only",
                )
                return state

            state.retry_counts.pop(str(state.current_step), None)
            state.current_step += 1
            state.touch()

        state.set_status("completed")
        return state

    def _build_tool_input(
        self,
        tool_name: str,
        state: ResearchAgentState,
        step: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        parameters = dict((step or {}).get("parameters", {}) or {})
        if tool_name == "retrieve_evidence":
            allowed = {
                key: parameters[key]
                for key in (
                    "top_k",
                    "candidate_k",
                    "read_cache",
                    "rewrite",
                    "multi_query",
                )
                if key in parameters
            }
            return {"query": state.query, **allowed}
        if tool_name == "search_scholarly_sources":
            allowed = {
                key: parameters[key]
                for key in ("limit", "sources")
                if key in parameters
            }
            return {"query": state.query, **allowed}
        if tool_name == "assess_evidence":
            if not state.evidence_bundle:
                raise ValueError("assess_evidence requires an EvidenceBundle.")
            return {"evidence_bundle": state.evidence_bundle}
        if tool_name == "generate_grounded_answer":
            if not state.evidence_bundle or not state.gate_decision:
                raise ValueError(
                    "generate_grounded_answer requires EvidenceBundle and GateDecision."
                )
            return {
                "evidence_bundle": state.evidence_bundle,
                "gate_decision": state.gate_decision,
            }
        if tool_name == "audit_answer":
            if state.answer is None:
                raise ValueError("audit_answer requires a provenance-bearing answer artifact.")
            if not state.evidence_bundle:
                raise ValueError("audit_answer requires the canonical EvidenceBundle.")
            return {
                "answer": state.answer,
                "evidence_bundle": state.evidence_bundle,
            }
        raise ValueError(f"No input adapter exists for tool: {tool_name}")

    def _apply_result(
        self,
        state: ResearchAgentState,
        tool_name: str,
        result: ToolResult,
    ) -> None:
        data = result.data
        if tool_name == "retrieve_evidence":
            evidence = data.get("evidence", [])
            if isinstance(evidence, list):
                state.evidence = self._serialize_evidence(evidence)
            bundle = data.get("evidence_bundle")
            if isinstance(bundle, Mapping):
                normalized_bundle = EvidenceBundle.from_mapping(bundle)
                state.evidence_bundle = normalized_bundle.to_dict()
                state.evidence = [
                    record.to_dict()
                    for record in normalized_bundle.evidence_records
                ]
            elif state.evidence:
                state.evidence_bundle = EvidenceBundle.create(
                    query=state.query,
                    evidence=state.evidence,
                    retrieval_metadata=(
                        data.get("retrieval")
                        if isinstance(data.get("retrieval"), Mapping)
                        else {}
                    ),
                    provenance={"source": "controller_legacy_retrieval_result"},
                ).to_dict()
            else:
                state.evidence_bundle = None
            state.gate_decision = None
            state.answer = None
            state.audit_result = None
        elif tool_name == "search_scholarly_sources":
            candidate_papers = data.get("candidate_papers", [])
            if isinstance(candidate_papers, list):
                state.candidate_papers = self._serialize_candidate_papers(candidate_papers)
            else:
                raise TypeError("candidate_papers must be a list.")
        elif tool_name == "assess_evidence":
            gate = data.get("gate_decision")
            if isinstance(gate, Mapping):
                normalized_gate = GateDecision.from_mapping(gate)
                if (
                    state.evidence_bundle
                    and normalized_gate.evidence_bundle_id
                    != state.evidence_bundle.get("bundle_id")
                ):
                    raise ValueError(
                        "Evidence assessment returned a GateDecision for another bundle."
                    )
                state.gate_decision = normalized_gate.to_dict()
            else:
                assessment = data.get("assessment")
                if isinstance(assessment, Mapping) and state.evidence_bundle:
                    state.gate_decision = GateDecision.from_assessment(
                        evidence_bundle_id=str(
                            state.evidence_bundle.get("bundle_id", "")
                        ),
                        assessment=assessment,
                    ).to_dict()
        elif tool_name == "generate_grounded_answer":
            answer = data.get("answer", data.get("answer_artifact"))
            if isinstance(answer, Mapping):
                state.answer = copy.deepcopy(dict(answer))
            else:
                raise TypeError("Answer Tool returned an invalid answer artifact.")
        elif tool_name == "audit_answer":
            audit = data.get("audit")
            if isinstance(audit, Mapping):
                state.audit_result = copy.deepcopy(dict(audit))
        state.touch()

    def _record_tool_result(
        self,
        state: ResearchAgentState,
        tool_name: str,
        tool_input: Mapping[str, Any],
        result: ToolResult,
    ) -> None:
        input_summary = self._input_summary(tool_input)
        bundle_payload = result.data.get("evidence_bundle")
        output_bundle_id = (
            bundle_payload.get("bundle_id")
            if isinstance(bundle_payload, Mapping)
            else None
        )
        state.tool_history.append(
            {
                "tool_name": tool_name,
                "input_summary": input_summary,
                "output_status": result.status,
                "latency_ms": result.latency_ms,
                "timestamp": result.timestamp,
                "trace_id": result.trace_id,
                "api_call_count": self._api_call_count(result.data),
                "evidence_bundle_id": (
                    result.data.get("evidence_bundle_id")
                    or output_bundle_id
                    or input_summary.get("evidence_bundle_id")
                ),
            }
        )
        state.observations.append(
            {
                "tool_name": tool_name,
                "status": result.status,
                "message": result.message,
                "reason": result.reason,
                "trace_id": result.trace_id,
                "data": copy.deepcopy(dict(result.data)),
                "error": result.error.to_dict() if result.error else None,
            }
        )
        state.touch()

    @staticmethod
    def _input_summary(tool_input: Mapping[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        query = tool_input.get("query")
        if query is not None:
            text = str(query)
            summary["query_preview"] = text[:120]
            summary["query_length"] = len(text)
        for key in (
            "top_k",
            "candidate_k",
            "rewrite",
            "multi_query",
            "read_cache",
            "filters",
            "limit",
            "sources",
        ):
            if key in tool_input:
                summary[key] = copy.deepcopy(tool_input[key])
        evidence = tool_input.get("evidence")
        if isinstance(evidence, list):
            summary["evidence_count"] = len(evidence)
            summary["evidence_chunk_ids"] = [
                str(item.get("chunk_id"))
                for item in evidence
                if isinstance(item, Mapping) and item.get("chunk_id")
            ]
        evidence_bundle = tool_input.get("evidence_bundle")
        if isinstance(evidence_bundle, Mapping):
            records = evidence_bundle.get("evidence_records", [])
            summary["evidence_bundle_id"] = evidence_bundle.get("bundle_id")
            summary["evidence_count"] = len(records) if isinstance(records, list) else 0
            summary["evidence_chunk_ids"] = [
                str(item.get("chunk_id"))
                for item in records
                if isinstance(item, Mapping) and item.get("chunk_id")
            ]
        gate = tool_input.get("gate_decision")
        if isinstance(gate, Mapping):
            summary["gate_status"] = gate.get("status")
            summary["gate_bundle_id"] = gate.get("evidence_bundle_id")
        answer = tool_input.get("answer")
        if isinstance(answer, Mapping):
            summary["answer_length"] = len(str(answer.get("answer", "")))
            citations = answer.get("citations", [])
            summary["citation_count"] = len(citations) if isinstance(citations, list) else 0
        return summary

    @staticmethod
    def _api_call_count(value: Any) -> int:
        if isinstance(value, Mapping):
            direct = value.get("api_call_count")
            if isinstance(direct, int) and not isinstance(direct, bool):
                return max(0, direct)
            return sum(
                BoundedResearchAgentController._api_call_count(item)
                for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return sum(
                BoundedResearchAgentController._api_call_count(item)
                for item in value
            )
        return 0

    @staticmethod
    def _serialize_evidence(
        evidence: Iterable[EvidenceRecord | Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in evidence:
            record = item if isinstance(item, EvidenceRecord) else EvidenceRecord.from_mapping(item)
            if record.chunk_id in seen:
                continue
            seen.add(record.chunk_id)
            serialized.append(record.to_dict())
        return serialized

    @staticmethod
    def _serialize_candidate_papers(
        candidate_papers: Iterable[ScholarPaperRecord | Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in candidate_papers:
            record = (
                item
                if isinstance(item, ScholarPaperRecord)
                else ScholarPaperRecord.from_dict(item)
            )
            if record.paper_id in seen:
                continue
            seen.add(record.paper_id)
            serialized.append(record.to_dict())
        return serialized

    @staticmethod
    def _rejection_reason(tool_name: str, result: ToolResult) -> str:
        if tool_name == "assess_evidence":
            return "insufficient_evidence"
        if tool_name == "generate_grounded_answer":
            return result.reason or "answer_not_grounded"
        if tool_name == "audit_answer":
            return "answer_not_grounded"
        return result.reason or "tool_rejected"
