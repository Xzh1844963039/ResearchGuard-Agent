# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tools\audit_tool.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from researchguard.pipeline import DEFAULT_CONFIG_PATH, PipelineSettings, load_pipeline_settings
from researchguard.retrieval.answer_generator import AnswerCitation, AnswerGenerationResult
from researchguard.retrieval.citation_audit import CitationAuditPipeline
from researchguard.retrieval.claim_extractor import load_citation_audit_settings
from researchguard.tools.contracts import EvidenceRecord, ToolError, ToolResult, ToolSpec
from researchguard.tools.evidence_tool import normalize_evidence


def _answer_result_from_mapping(value: Mapping[str, Any]) -> AnswerGenerationResult:
    required = ("answer", "citations", "evidence_chunk_ids")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"Answer artifact is missing required fields: {', '.join(missing)}.")
    citations = tuple(
        AnswerCitation(
            chunk_id=str(citation.get("chunk_id", "")),
            doc_id=str(citation.get("doc_id", "")),
            section=str(citation.get("section", "")),
            page=int(citation["page"]) if citation.get("page") is not None else None,
        )
        for citation in value.get("citations", [])
    )
    if not citations and not bool(value.get("refused", False)):
        raise ValueError("A non-refused answer artifact must contain citations.")
    return AnswerGenerationResult(
        answer=str(value.get("answer", "")),
        citations=citations,
        confidence=float(value.get("confidence", 0.0)),
        refused=bool(value.get("refused", False)),
        refusal_reason=value.get("refusal_reason"),
        evidence_chunk_ids=tuple(str(item) for item in value.get("evidence_chunk_ids", [])),
        model=str(value.get("model", "unknown")),
        prompt_version=str(value.get("prompt_version", "unknown")),
        config_version=str(value.get("config_version", "unknown")),
        timestamp=str(value.get("timestamp", "")),
        cache_hit=bool(value.get("cache_hit", False)),
        fallback_used=bool(value.get("fallback_used", False)),
        fallback_reason=value.get("fallback_reason"),
        api_call_count=int(value.get("api_call_count", 0)),
        input_tokens=int(value.get("input_tokens", 0)),
        output_tokens=int(value.get("output_tokens", 0)),
        latency_ms=float(value.get("latency_ms", 0.0)),
    )


class CitationAuditTool:
    name = "audit_answer"
    version = "1.0.0"
    description = "Audit a provenance-bearing generated answer against its canonical evidence."

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
                "answer": "AnswerGenerationResult or its complete serialized mapping; raw strings are rejected",
                "evidence": "non-empty list of EvidenceRecord-compatible mappings",
                "read_cache": "boolean",
            },
        )

    def _pipeline_settings(self) -> PipelineSettings:
        if self._settings is None:
            _, self._settings = load_pipeline_settings(self._config_path)
        return self._settings

    def _audit_pipeline(self) -> Any:
        if self._pipeline is None:
            settings = self._pipeline_settings()
            _, audit_settings = load_citation_audit_settings(settings.citation_audit_config_path)
            self._pipeline = CitationAuditPipeline(audit_settings)
        return self._pipeline

    def invoke(self, **kwargs: Any) -> ToolResult:
        return self.audit_answer(**kwargs)

    def audit_answer(
        self,
        answer: AnswerGenerationResult | Mapping[str, Any],
        evidence: Iterable[EvidenceRecord | Mapping[str, Any]],
        *,
        read_cache: bool = True,
    ) -> ToolResult:
        started = time.perf_counter()
        try:
            if isinstance(answer, AnswerGenerationResult):
                answer_result = answer
            elif isinstance(answer, Mapping):
                answer_result = _answer_result_from_mapping(answer)
            else:
                raise TypeError(
                    "answer must be an AnswerGenerationResult or complete serialized answer artifact."
                )
            records = normalize_evidence(evidence)
            result = self._audit_pipeline().audit(
                answer_result,
                [record.to_retrieval_mapping() for record in records],
                read_cache=read_cache,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            data = {
                "audit": result.to_dict(),
                "evidence_chunk_ids": [record.chunk_id for record in records],
            }
            if result.fallback_used or not result.audit_completed:
                error = ToolError(
                    code="citation_audit_fallback",
                    category="api_failure",
                    message=result.fallback_reason or result.audit_reason or "Citation audit failed closed.",
                    retryable=True,
                )
                return ToolResult.create(
                    status="failed",
                    message="Citation audit failed closed.",
                    reason=result.fallback_reason or result.audit_reason,
                    tool_name=self.name,
                    tool_version=self.version,
                    latency_ms=latency_ms,
                    data=data,
                    error=error,
                )
            status = "success" if result.overall_grounded else "rejected"
            return ToolResult.create(
                status=status,
                message="Citation audit completed.",
                reason=None if status == "success" else "answer_not_fully_grounded",
                tool_name=self.name,
                tool_version=self.version,
                latency_ms=latency_ms,
                data=data,
            )
        except (ValueError, TypeError) as exc:
            return self._failure(started, exc, "invalid_input", "invalid_audit_input", False)
        except TimeoutError as exc:
            return self._failure(started, exc, "timeout", "citation_audit_timeout", True)
        except Exception as exc:
            return self._failure(started, exc, "api_failure", "citation_audit_failed", True)

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
            message="Citation audit failed.",
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
