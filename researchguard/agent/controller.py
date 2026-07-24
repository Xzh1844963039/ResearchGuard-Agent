# C:\Users\18449\Desktop\researchguard_workspace\researchguard\agent\controller.py
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from researchguard.agent.planner import BoundedPlanner, PlannerError
from researchguard.agent.policy import AgentPolicy
from researchguard.agent.state import ResearchAgentState, utc_timestamp
from researchguard.memory import DEFAULT_MEMORY_ROOT, ResearchMemory
from researchguard.tools import (
    EvidenceRecord,
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
        planner: Any | None = None,
        policy: AgentPolicy | None = None,
        memory: ResearchMemory | None = None,
        memory_enabled: bool = True,
        memory_root: str | Path = DEFAULT_MEMORY_ROOT,
        config_path: str | Path = "configs/pipeline_v1.yaml",
    ):
        self.policy = policy or AgentPolicy()
        self.registry = registry or build_default_registry(config_path)
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
        self.planner = planner or BoundedPlanner(
            self.registry,
            max_steps=self.policy.max_steps,
            workflow_names=self.workflow_registry.names,
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
        state = ResearchAgentState(
            query=query,
            answer=copy.deepcopy(dict(answer_artifact)) if answer_artifact is not None else None,
            evidence=self._serialize_evidence(evidence or ()),
            workflow_input=copy.deepcopy(dict(workflow_input or {})),
            memory_status={
                "enabled": self.memory is not None,
                "started": False,
                "persisted": False,
                "errors": [],
            },
        )
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
            plan = self.planner.create_plan(
                state.query,
                task_type=task_type,
                has_evidence=bool(state.evidence),
                has_answer=state.answer is not None,
                memory_context=state.memory_context,
            )
        except (PlannerError, ValueError, TypeError) as exc:
            state.set_status("failed", f"planner_error: {exc}")
            return state

        state.task_type = plan.task_type
        state.plan = [step.to_dict() for step in plan.steps]
        state.workflow_name = plan.workflow
        state.set_status("planned")
        plan_error = self.policy.validate_plan(state)
        if plan_error:
            state.set_status("failed", plan_error)
            return state
        if state.workflow_name:
            return self.execute_workflow(state)
        return self.execute(state)

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
                tool_input = self._build_tool_input(tool_name, state)
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

            if result.status == "failed":
                retry_key = str(state.current_step)
                retry_count = state.retry_counts.get(retry_key, 0)
                retryable = bool(result.error and result.error.retryable)
                if retryable and self.policy.can_retry(retry_count):
                    state.retry_counts[retry_key] = retry_count + 1
                    state.touch()
                    continue
                state.set_status("failed", "tool_error")
                return state

            if result.status == "rejected":
                reason = self._rejection_reason(tool_name, result)
                state.set_status("rejected", reason)
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
    ) -> dict[str, Any]:
        if tool_name == "retrieve_evidence":
            return {"query": state.query}
        if tool_name == "search_scholarly_sources":
            return {"query": state.query}
        if tool_name == "assess_evidence":
            if not state.evidence:
                raise ValueError("assess_evidence requires retrieved evidence.")
            return {"query": state.query, "evidence": state.evidence}
        if tool_name == "generate_grounded_answer":
            return {"query": state.query}
        if tool_name == "audit_answer":
            if state.answer is None:
                raise ValueError("audit_answer requires a provenance-bearing answer artifact.")
            if not state.evidence:
                raise ValueError("audit_answer requires canonical evidence.")
            return {"answer": state.answer, "evidence": state.evidence}
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
        elif tool_name == "search_scholarly_sources":
            candidate_papers = data.get("candidate_papers", [])
            if isinstance(candidate_papers, list):
                state.candidate_papers = self._serialize_candidate_papers(candidate_papers)
            else:
                raise TypeError("candidate_papers must be a list.")
        elif tool_name == "generate_grounded_answer":
            pipeline_result = data.get("pipeline_result", {})
            if isinstance(pipeline_result, Mapping):
                answer_stage = pipeline_result.get("answer_generation", {})
                audit_stage = pipeline_result.get("citation_audit", {})
                retrieval_stage = pipeline_result.get("retrieval", {})
                answer_output = (
                    answer_stage.get("output") if isinstance(answer_stage, Mapping) else None
                )
                audit_output = audit_stage.get("output") if isinstance(audit_stage, Mapping) else None
                retrieval_output = (
                    retrieval_stage.get("output") if isinstance(retrieval_stage, Mapping) else None
                )
                if isinstance(answer_output, Mapping):
                    state.answer = copy.deepcopy(dict(answer_output))
                if isinstance(audit_output, Mapping):
                    state.audit_result = copy.deepcopy(dict(audit_output))
                if isinstance(retrieval_output, Mapping):
                    hits = retrieval_output.get("hits", [])
                    if isinstance(hits, list) and hits:
                        state.evidence = self._serialize_evidence(hits)
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
        state.tool_history.append(
            {
                "tool_name": tool_name,
                "input_summary": self._input_summary(tool_input),
                "output_status": result.status,
                "latency_ms": result.latency_ms,
                "timestamp": result.timestamp,
                "trace_id": result.trace_id,
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
        evidence = tool_input.get("evidence")
        if isinstance(evidence, list):
            summary["evidence_count"] = len(evidence)
            summary["evidence_chunk_ids"] = [
                str(item.get("chunk_id"))
                for item in evidence
                if isinstance(item, Mapping) and item.get("chunk_id")
            ]
        answer = tool_input.get("answer")
        if isinstance(answer, Mapping):
            summary["answer_length"] = len(str(answer.get("answer", "")))
            citations = answer.get("citations", [])
            summary["citation_count"] = len(citations) if isinstance(citations, list) else 0
        return summary

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
            pipeline_result = result.data.get("pipeline_result", {})
            final_status = (
                pipeline_result.get("final_status")
                if isinstance(pipeline_result, Mapping)
                else None
            )
            return "insufficient_evidence" if final_status == "rejected" else "answer_not_grounded"
        if tool_name == "audit_answer":
            return "answer_not_grounded"
        return result.reason or "tool_rejected"
