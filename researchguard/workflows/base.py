# C:\Users\18449\Desktop\researchguard_workspace\researchguard\workflows\base.py
from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from researchguard.tools import (
    EvidenceBundle,
    EvidenceRecord,
    GateDecision,
    ToolError,
    ToolRegistry,
    ToolResult,
)


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
        bundle_payload = result.data.get("evidence_bundle")
        output_bundle_id = (
            bundle_payload.get("bundle_id")
            if isinstance(bundle_payload, Mapping)
            else None
        )
        history = {
            "workflow_name": self.workflow_name,
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
        for key in (
            "top_k",
            "candidate_k",
            "rewrite",
            "multi_query",
            "read_cache",
            "filters",
            "limit",
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
        gate = tool_input.get("gate_decision")
        if isinstance(gate, Mapping):
            summary["gate_status"] = gate.get("status")
        answer = tool_input.get("answer")
        if isinstance(answer, Mapping):
            summary["answer_length"] = len(str(answer.get("answer", "")))
            citations = answer.get("citations", [])
            summary["citation_count"] = len(citations) if isinstance(citations, list) else 0
        if "sources" in tool_input:
            sources = tool_input["sources"]
            summary["sources"] = [sources] if isinstance(sources, str) else list(sources)
        return summary

    @staticmethod
    def _api_call_count(value: Any) -> int:
        if isinstance(value, Mapping):
            direct = value.get("api_call_count")
            if isinstance(direct, int) and not isinstance(direct, bool):
                return max(0, direct)
            return sum(ResearchWorkflow._api_call_count(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(ResearchWorkflow._api_call_count(item) for item in value)
        return 0

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

    @classmethod
    def _bundle_from_result(
        cls,
        result: ToolResult,
        *,
        query: str,
    ) -> EvidenceBundle:
        value = result.data.get("evidence_bundle")
        if isinstance(value, Mapping):
            return EvidenceBundle.from_mapping(value)
        evidence = cls._evidence_from_result(result)
        if not evidence:
            raise WorkflowExecutionError("Retrieval returned no canonical evidence.")
        return EvidenceBundle.create(
            query=query,
            evidence=evidence,
            provenance={"source": "workflow_legacy_retrieval_result"},
        )

    @staticmethod
    def _gate_from_result(
        result: ToolResult,
        *,
        bundle: EvidenceBundle,
    ) -> GateDecision:
        value = result.data.get("gate_decision")
        if isinstance(value, Mapping):
            gate = GateDecision.from_mapping(value)
        else:
            assessment = result.data.get("assessment")
            if not isinstance(assessment, Mapping):
                raise WorkflowExecutionError(
                    "Evidence Tool returned no GateDecision or assessment."
                )
            gate = GateDecision.from_assessment(
                evidence_bundle_id=bundle.bundle_id,
                assessment=assessment,
            )
        if gate.evidence_bundle_id != bundle.bundle_id:
            raise WorkflowExecutionError(
                "Evidence Tool returned a GateDecision for another bundle."
            )
        return gate

    @staticmethod
    def _answer_from_result(
        result: ToolResult,
        *,
        bundle: EvidenceBundle,
    ) -> dict[str, Any]:
        answer = result.data.get("answer", result.data.get("answer_artifact"))
        if not isinstance(answer, Mapping):
            raise WorkflowExecutionError(
                "Answer Tool returned no complete answer artifact."
            )
        bundle_id = result.data.get("evidence_bundle_id")
        if bundle_id != bundle.bundle_id:
            raise WorkflowExecutionError(
                "Answer Tool returned an artifact from another evidence bundle."
            )
        return copy.deepcopy(dict(answer))
