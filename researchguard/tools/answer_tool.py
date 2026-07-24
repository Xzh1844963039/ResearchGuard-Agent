# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tools\answer_tool.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from researchguard.pipeline import DEFAULT_CONFIG_PATH, PipelineSettings, load_pipeline_settings
from researchguard.retrieval.answer_generator import load_answer_generation_settings
from researchguard.retrieval.answer_pipeline import AnswerGenerationPipeline
from researchguard.tools.contracts import (
    EvidenceBundle,
    GateDecision,
    ToolError,
    ToolResult,
    ToolSpec,
)


class GuardedAnswerTool:
    name = "generate_grounded_answer"
    version = "2.0.0"
    description = (
        "Generate an answer from one pre-assessed EvidenceBundle without retrieval or re-judging."
    )

    def __init__(
        self,
        *,
        answer_pipeline: Any | None = None,
        pipeline: Any | None = None,
        settings: PipelineSettings | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        if answer_pipeline is not None and pipeline is not None:
            raise ValueError("Provide answer_pipeline or pipeline, not both.")
        self._pipeline = answer_pipeline if answer_pipeline is not None else pipeline
        self._settings = settings
        self._config_path = config_path

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema={
                "evidence_bundle": "EvidenceBundle mapping produced by Retrieval Tool",
                "gate_decision": "strong GateDecision for the same evidence_bundle_id",
                "read_cache": "boolean",
            },
        )

    def _pipeline_settings(self) -> PipelineSettings:
        if self._settings is None:
            _, self._settings = load_pipeline_settings(self._config_path)
        return self._settings

    def _answer_pipeline(self) -> Any:
        if self._pipeline is None:
            settings = self._pipeline_settings()
            _, answer_settings = load_answer_generation_settings(
                settings.answer_config_path
            )
            self._pipeline = AnswerGenerationPipeline(answer_settings)
        return self._pipeline

    def invoke(self, **kwargs: Any) -> ToolResult:
        return self.generate_grounded_answer(**kwargs)

    def generate_grounded_answer(
        self,
        evidence_bundle: EvidenceBundle | Mapping[str, Any] | None = None,
        gate_decision: GateDecision | Mapping[str, Any] | None = None,
        *,
        read_cache: bool = True,
    ) -> ToolResult:
        started = time.perf_counter()
        try:
            bundle = self._normalize_bundle(evidence_bundle)
            gate = self._normalize_gate(gate_decision)
            self._validate_gate(bundle, gate)
            if gate.status != "strong" or not gate.answerable:
                return ToolResult.create(
                    status="rejected",
                    message="Answer generation blocked by the evidence gate.",
                    reason=f"evidence_{gate.status}",
                    tool_name=self.name,
                    tool_version=self.version,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    data={
                        "evidence_bundle_id": bundle.bundle_id,
                        "gate_decision": gate.to_dict(),
                    },
                )

            result = self._answer_pipeline().generate(
                bundle.query,
                [
                    record.to_retrieval_mapping()
                    for record in bundle.evidence_records
                ],
                gate.to_sufficiency_result(),
                read_cache=read_cache,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            data = {
                "answer": result.to_dict(),
                "answer_artifact": result.to_dict(),
                "evidence_bundle_id": bundle.bundle_id,
                "evidence_chunk_ids": list(bundle.chunk_ids),
                "supporting_chunk_ids": list(gate.supporting_chunk_ids),
                "generation": {
                    "api_call_count": result.api_call_count,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cache_hit": result.cache_hit,
                },
            }
            if result.fallback_used:
                return ToolResult.create(
                    status="failed",
                    message="Answer generation failed closed.",
                    reason=result.fallback_reason or "answer_generation_fallback",
                    tool_name=self.name,
                    tool_version=self.version,
                    latency_ms=latency_ms,
                    data=data,
                    error=ToolError(
                        code="answer_generation_fallback",
                        category="api_failure",
                        message=result.fallback_reason or "Answer generation used fallback.",
                        retryable=True,
                    ),
                )
            if result.refused:
                return ToolResult.create(
                    status="rejected",
                    message="Answer generator refused to release an answer.",
                    reason=result.refusal_reason or "answer_refused",
                    tool_name=self.name,
                    tool_version=self.version,
                    latency_ms=latency_ms,
                    data=data,
                )
            return ToolResult.create(
                status="success",
                message="Grounded answer generated from the supplied evidence bundle.",
                tool_name=self.name,
                tool_version=self.version,
                latency_ms=latency_ms,
                data=data,
            )
        except (ValueError, TypeError) as exc:
            return self._failure(
                started, exc, "invalid_input", "invalid_answer_input", False
            )
        except TimeoutError as exc:
            return self._failure(
                started, exc, "timeout", "answer_generation_timeout", True
            )
        except Exception as exc:
            return self._failure(
                started, exc, "execution_failure", "answer_generation_failed", True
            )

    @staticmethod
    def _normalize_bundle(
        value: EvidenceBundle | Mapping[str, Any] | None,
    ) -> EvidenceBundle:
        if isinstance(value, EvidenceBundle):
            return value
        if isinstance(value, Mapping):
            return EvidenceBundle.from_mapping(value)
        raise TypeError("evidence_bundle must be an EvidenceBundle or mapping.")

    @staticmethod
    def _normalize_gate(
        value: GateDecision | Mapping[str, Any] | None,
    ) -> GateDecision:
        if isinstance(value, GateDecision):
            return value
        if isinstance(value, Mapping):
            return GateDecision.from_mapping(value)
        raise TypeError("gate_decision must be a GateDecision or mapping.")

    @staticmethod
    def _validate_gate(bundle: EvidenceBundle, gate: GateDecision) -> None:
        if gate.evidence_bundle_id != bundle.bundle_id:
            raise ValueError("GateDecision was produced for a different EvidenceBundle.")
        missing = set(gate.supporting_chunk_ids).difference(bundle.chunk_ids)
        if missing:
            raise ValueError(
                "GateDecision references chunks outside the EvidenceBundle: "
                + ", ".join(sorted(missing))
            )

    def _failure(
        self,
        started: float,
        exc: Exception,
        category: str,
        code: str,
        retryable: bool,
    ) -> ToolResult:
        return ToolResult.create(
            status="failed",
            message="Grounded answer generation failed.",
            reason=code,
            tool_name=self.name,
            tool_version=self.version,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=ToolError(
                code=code,
                category=category,
                message=str(exc),
                retryable=retryable,
                details={"exception_type": type(exc).__name__},
            ),
        )
