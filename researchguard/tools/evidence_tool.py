# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tools\evidence_tool.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from researchguard.pipeline import DEFAULT_CONFIG_PATH, PipelineSettings, load_pipeline_settings
from researchguard.retrieval.evidence_judge import load_evidence_judge_settings
from researchguard.retrieval.evidence_pipeline import EvidenceSufficiencyPipeline
from researchguard.tools.contracts import (
    EvidenceBundle,
    EvidenceRecord,
    GateDecision,
    ToolError,
    ToolResult,
    ToolSpec,
)


def normalize_evidence(
    evidence: Iterable[EvidenceRecord | Mapping[str, Any]],
) -> tuple[EvidenceRecord, ...]:
    records: list[EvidenceRecord] = []
    for item in evidence:
        if isinstance(item, EvidenceRecord):
            records.append(item)
        elif isinstance(item, Mapping):
            records.append(EvidenceRecord.from_mapping(item))
        else:
            raise TypeError("Evidence items must be EvidenceRecord instances or mappings.")
    if not records:
        raise ValueError("At least one evidence record is required.")
    chunk_ids = [record.chunk_id for record in records]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Evidence contains duplicate chunk_id values.")
    return tuple(records)


class EvidenceTool:
    name = "assess_evidence"
    version = "1.0.0"
    description = "Assess whether canonical evidence is strong, partial, or unsupported."

    def __init__(
        self,
        *,
        pipeline: Any | None = None,
        settings: PipelineSettings | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        self._pipeline = pipeline
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
                "read_cache": "boolean",
            },
        )

    def _pipeline_settings(self) -> PipelineSettings:
        if self._settings is None:
            _, self._settings = load_pipeline_settings(self._config_path)
        return self._settings

    def _evidence_pipeline(self) -> Any:
        if self._pipeline is None:
            settings = self._pipeline_settings()
            _, evidence_settings = load_evidence_judge_settings(settings.evidence_config_path)
            self._pipeline = EvidenceSufficiencyPipeline(evidence_settings)
        return self._pipeline

    def invoke(self, **kwargs: Any) -> ToolResult:
        return self.assess_evidence(**kwargs)

    def assess_evidence(
        self,
        evidence_bundle: EvidenceBundle | Mapping[str, Any] | None = None,
        *,
        query: str | None = None,
        evidence: Iterable[EvidenceRecord | Mapping[str, Any]] | None = None,
        read_cache: bool = True,
    ) -> ToolResult:
        started = time.perf_counter()
        try:
            if evidence_bundle is None:
                normalized_query = str(query or "").strip()
                if not normalized_query:
                    raise ValueError("Query must not be empty.")
                records = normalize_evidence(evidence or ())
                bundle = EvidenceBundle.create(
                    query=normalized_query,
                    evidence=records,
                    provenance={"source": "legacy_evidence_tool_input"},
                )
            elif isinstance(evidence_bundle, EvidenceBundle):
                bundle = evidence_bundle
                records = bundle.evidence_records
            elif isinstance(evidence_bundle, Mapping):
                bundle = EvidenceBundle.from_mapping(evidence_bundle)
                records = bundle.evidence_records
            else:
                raise TypeError("evidence_bundle must be an EvidenceBundle or mapping.")
            result = self._evidence_pipeline().assess(
                bundle.query,
                [record.to_retrieval_mapping() for record in records],
                read_cache=read_cache,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            assessment = result.to_dict()
            gate_decision = GateDecision.from_assessment(
                evidence_bundle_id=bundle.bundle_id,
                assessment=assessment,
            )
            data = {
                "assessment": assessment,
                "gate_decision": gate_decision.to_dict(),
                "evidence_bundle_id": bundle.bundle_id,
                "evidence_chunk_ids": [record.chunk_id for record in records],
            }
            if result.fallback_used:
                error = ToolError(
                    code="evidence_assessment_fallback",
                    category="api_failure",
                    message=result.fallback_reason or "Evidence assessment used a fail-closed fallback.",
                    retryable=True,
                )
                return ToolResult.create(
                    status="failed",
                    message="Evidence assessment failed closed.",
                    reason=result.fallback_reason or "evidence_assessment_fallback",
                    tool_name=self.name,
                    tool_version=self.version,
                    latency_ms=latency_ms,
                    data=data,
                    error=error,
                )
            status = "success" if result.support_level == "strong" and result.answerable else "rejected"
            return ToolResult.create(
                status=status,
                message=f"Evidence support level: {result.support_level}.",
                reason=None if status == "success" else result.reason,
                tool_name=self.name,
                tool_version=self.version,
                latency_ms=latency_ms,
                data=data,
            )
        except (ValueError, TypeError) as exc:
            return self._failure(started, exc, "invalid_input", "invalid_evidence_input", False)
        except TimeoutError as exc:
            return self._failure(started, exc, "timeout", "evidence_assessment_timeout", True)
        except Exception as exc:
            return self._failure(started, exc, "api_failure", "evidence_assessment_failed", True)

    def _failure(
        self,
        started: float,
        exc: Exception,
        category: str,
        code: str,
        retryable: bool,
    ) -> ToolResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ToolResult.create(
            status="failed",
            message="Evidence assessment failed.",
            reason=code,
            tool_name=self.name,
            tool_version=self.version,
            latency_ms=latency_ms,
            error=ToolError(
                code=code,
                category=category,
                message=str(exc),
                retryable=retryable,
                details={"exception_type": type(exc).__name__},
            ),
        )
