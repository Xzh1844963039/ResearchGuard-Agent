# C:\Users\18449\Desktop\researchguard_workspace\researchguard\workflows\base.py
from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from researchguard.tools import EvidenceRecord, ToolError, ToolRegistry, ToolResult


WORKFLOW_RESULT_SCHEMA_VERSION = "researchguard.workflow_result.v1"
WORKFLOW_SPEC_SCHEMA_VERSION = "researchguard.workflow_spec.v1"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WorkflowLimits:
    max_steps: int = 6
    max_tool_calls: int = 10
    max_retry: int = 2
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.max_tool_calls < 1:
            raise ValueError("Workflow step and tool-call limits must be positive.")
        if self.max_retry < 0 or self.timeout <= 0:
            raise ValueError("Workflow retry and timeout limits are invalid.")


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_name: str
    description: str
    required_tools: tuple[str, ...]
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    version: str
    schema_version: str = WORKFLOW_SPEC_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workflow_name": self.workflow_name,
            "description": self.description,
            "required_tools": list(self.required_tools),
            "input_schema": copy.deepcopy(dict(self.input_schema)),
            "output_schema": copy.deepcopy(dict(self.output_schema)),
            "version": self.version,
        }


@dataclass(frozen=True)
class WorkflowResult:
    workflow_name: str
    workflow_version: str
    status: str
    message: str
    reason: str | None
    output: Mapping[str, Any]
    trace: tuple[Mapping[str, Any], ...]
    started_at: str
    completed_at: str
    latency_ms: float
    schema_version: str = WORKFLOW_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"success", "rejected", "failed"}:
            raise ValueError(f"Unsupported workflow status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workflow_name": self.workflow_name,
            "workflow_version": self.workflow_version,
            "status": self.status,
            "message": self.message,
            "reason": self.reason,
            "output": copy.deepcopy(dict(self.output)),
            "trace": copy.deepcopy(list(self.trace)),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "latency_ms": self.latency_ms,
        }


class WorkflowExecutionError(RuntimeError):
    pass


class WorkflowLimitError(WorkflowExecutionError):
    pass


class ResearchWorkflow(ABC):
    workflow_name: str
    version: str
    description: str
    required_tools: tuple[str, ...]
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        limits: WorkflowLimits | None = None,
    ):
        self.registry = registry
        self.limits = limits or WorkflowLimits()
        missing = [name for name in self.required_tools if name not in registry.names]
        if missing:
            raise ValueError(
                f"Workflow {self.workflow_name} requires unregistered tools: {', '.join(missing)}"
            )

    @property
    def spec(self) -> WorkflowSpec:
        return WorkflowSpec(
            workflow_name=self.workflow_name,
            description=self.description,
            required_tools=self.required_tools,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            version=self.version,
        )

    @abstractmethod
    def run(self, state: Any) -> WorkflowResult:
        raise NotImplementedError

    def _invoke(
        self,
        state: Any,
        trace: list[dict[str, Any]],
        workflow_started: float,
        tool_name: str,
        **tool_input: Any,
    ) -> ToolResult:
        retry_count = 0
        while True:
            self._check_limits(state, trace, workflow_started)
            call_started = time.perf_counter()
            try:
                result = self.registry.invoke(tool_name, **tool_input)
            except Exception as exc:
                result = ToolResult.create(
                    status="failed",
                    message="Registered workflow tool raised an unhandled exception.",
                    reason="unhandled_tool_exception",
                    tool_name=tool_name,
                    tool_version=self.registry.version,
                    latency_ms=(time.perf_counter() - call_started) * 1000.0,
                    error=ToolError(
                        code="unhandled_tool_exception",
                        category="execution_failure",
                        message=str(exc),
                        retryable=False,
                        details={"exception_type": type(exc).__name__},
                    ),
                )
            self._record_tool_call(state, trace, tool_name, tool_input, result)
            if time.perf_counter() - workflow_started >= self.limits.timeout:
                raise WorkflowLimitError("workflow_timeout_exceeded")
            if not (
                result.status == "failed"
                and result.error is not None
                and result.error.retryable
                and retry_count < self.limits.max_retry
            ):
                return result
            retry_count += 1

    def _check_limits(
        self,
        state: Any,
        trace: list[dict[str, Any]],
        workflow_started: float,
    ) -> None:
        if len(trace) >= self.limits.max_steps:
            raise WorkflowLimitError("workflow_max_steps_exceeded")
        if len(state.tool_history) >= self.limits.max_tool_calls:
            raise WorkflowLimitError("workflow_max_tool_calls_exceeded")
        if time.perf_counter() - workflow_started >= self.limits.timeout:
            raise WorkflowLimitError("workflow_timeout_exceeded")

    def _record_tool_call(
        self,
        state: Any,
        trace: list[dict[str, Any]],
        tool_name: str,
        tool_input: Mapping[str, Any],
        result: ToolResult,
    ) -> None:
        input_summary = self._input_summary(tool_input)
        history = {
            "workflow_name": self.workflow_name,
            "tool_name": tool_name,
            "input_summary": input_summary,
            "output_status": result.status,
            "latency_ms": result.latency_ms,
            "timestamp": result.timestamp,
            "trace_id": result.trace_id,
        }
        state.tool_history.append(history)
        observation = {
            "workflow_name": self.workflow_name,
            "tool_name": tool_name,
            "status": result.status,
            "message": result.message,
            "reason": result.reason,
            "trace_id": result.trace_id,
            "data": copy.deepcopy(dict(result.data)),
            "error": result.error.to_dict() if result.error else None,
        }
        state.observations.append(observation)
        step = {
            "step": len(trace) + 1,
            "tool_name": tool_name,
            "status": result.status,
            "latency_ms": result.latency_ms,
            "timestamp": result.timestamp,
            "trace_id": result.trace_id,
        }
        trace.append(step)
        state.workflow_steps.append(copy.deepcopy(step))
        state.touch()

    @staticmethod
    def _input_summary(tool_input: Mapping[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        if "query" in tool_input:
            query = str(tool_input["query"])
            summary["query_preview"] = query[:120]
            summary["query_length"] = len(query)
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
        if "sources" in tool_input:
            sources = tool_input["sources"]
            summary["sources"] = [sources] if isinstance(sources, str) else list(sources)
        return summary

    def _finish(
        self,
        *,
        status: str,
        message: str,
        reason: str | None,
        output: Mapping[str, Any],
        trace: list[dict[str, Any]],
        started_at: str,
        started: float,
    ) -> WorkflowResult:
        return WorkflowResult(
            workflow_name=self.workflow_name,
            workflow_version=self.version,
            status=status,
            message=message,
            reason=reason,
            output=output,
            trace=tuple(trace),
            started_at=started_at,
            completed_at=utc_timestamp(),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _evidence_from_result(result: ToolResult) -> list[dict[str, Any]]:
        value = result.data.get("evidence", [])
        if not isinstance(value, list):
            raise WorkflowExecutionError("Tool returned an invalid evidence list.")
        records: list[dict[str, Any]] = []
        for item in value:
            record = item if isinstance(item, EvidenceRecord) else EvidenceRecord.from_mapping(item)
            records.append(record.to_dict())
        return records

    @staticmethod
    def _guarded_artifacts(
        result: ToolResult,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
        pipeline_result = result.data.get("pipeline_result")
        if not isinstance(pipeline_result, Mapping):
            raise WorkflowExecutionError("Guarded Answer Tool did not return a pipeline result.")
        answer_stage = pipeline_result.get("answer_generation")
        retrieval_stage = pipeline_result.get("retrieval")
        audit_stage = pipeline_result.get("citation_audit")
        answer = answer_stage.get("output") if isinstance(answer_stage, Mapping) else None
        retrieval = retrieval_stage.get("output") if isinstance(retrieval_stage, Mapping) else None
        audit = audit_stage.get("output") if isinstance(audit_stage, Mapping) else None
        if not isinstance(answer, Mapping) or not isinstance(retrieval, Mapping):
            raise WorkflowExecutionError("Guarded Answer Tool returned incomplete artifacts.")
        hits = retrieval.get("hits", [])
        if not isinstance(hits, list):
            raise WorkflowExecutionError("Guarded Answer Tool returned invalid retrieval hits.")
        evidence = [EvidenceRecord.from_mapping(hit).to_dict() for hit in hits]
        return dict(answer), evidence, dict(audit) if isinstance(audit, Mapping) else None
