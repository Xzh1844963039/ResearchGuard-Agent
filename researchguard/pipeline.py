# C:\Users\18449\Desktop\researchguard_workspace\researchguard\pipeline.py
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from researchguard.indexing.corpus_loader import load_yaml
from researchguard.retrieval import (
    AnswerGenerationPipeline,
    CitationAuditPipeline,
    EvidenceSufficiencyPipeline,
    RetrievalEngine,
)
from researchguard.retrieval.answer_generator import load_answer_generation_settings
from researchguard.retrieval.claim_extractor import load_citation_audit_settings
from researchguard.retrieval.evidence_judge import load_evidence_judge_settings, normalize_question


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline_v1.yaml"
STAGE_NAMES = (
    "rewrite",
    "retrieval",
    "reranking",
    "evidence_check",
    "answer_generation",
    "citation_audit",
)


class PipelineConfigurationError(ValueError):
    pass


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.isoformat()


@dataclass(frozen=True)
class PipelineSettings:
    schema_version: str
    config_version: str
    read_cache: bool
    include_retrieval_text: bool
    rewrite_enabled: bool
    multi_query_enabled: bool
    retrieval_enabled: bool
    retrieval_config_path: Path
    retrieval_mode: str
    retrieval_top_k: int
    retrieval_candidate_k: int
    reranker_enabled: bool
    reranker_candidate_k: int
    evidence_check_enabled: bool
    evidence_config_path: Path
    answer_generation_enabled: bool
    answer_config_path: Path
    citation_audit_enabled: bool
    citation_audit_config_path: Path


def load_pipeline_settings(path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[dict[str, Any], PipelineSettings]:
    config_path = _project_path(path)
    config = load_yaml(config_path)
    pipeline = config.get("pipeline", {}) or {}
    modules = pipeline.get("modules", {}) or {}

    def module(name: str) -> dict[str, Any]:
        value = modules.get(name, {}) or {}
        if not isinstance(value, dict):
            raise PipelineConfigurationError(f"pipeline.modules.{name} must be a mapping.")
        return value

    rewrite = module("rewrite")
    retrieval = module("retrieval")
    reranker = module("reranker")
    evidence = module("evidence_check")
    answer = module("answer_generation")
    audit = module("citation_audit")
    settings = PipelineSettings(
        schema_version=str(pipeline.get("schema_version", "researchguard_pipeline_v1")),
        config_version=str(pipeline.get("config_version", "pipeline_v1.0")),
        read_cache=bool(pipeline.get("read_cache", True)),
        include_retrieval_text=bool(pipeline.get("include_retrieval_text", True)),
        rewrite_enabled=bool(rewrite.get("enabled", True)),
        multi_query_enabled=bool(rewrite.get("multi_query", True)),
        retrieval_enabled=bool(retrieval.get("enabled", True)),
        retrieval_config_path=_project_path(retrieval.get("config_path", "configs/retrieval_v1.yaml")),
        retrieval_mode=str(retrieval.get("mode", "hybrid")),
        retrieval_top_k=max(1, int(retrieval.get("top_k", 10))),
        retrieval_candidate_k=max(1, int(retrieval.get("candidate_k", 80))),
        reranker_enabled=bool(reranker.get("enabled", True)),
        reranker_candidate_k=max(1, int(reranker.get("candidate_k", 20))),
        evidence_check_enabled=bool(evidence.get("enabled", True)),
        evidence_config_path=_project_path(evidence.get("config_path", "configs/evidence_sufficiency_v1.yaml")),
        answer_generation_enabled=bool(answer.get("enabled", True)),
        answer_config_path=_project_path(answer.get("config_path", "configs/answer_generation_v1.yaml")),
        citation_audit_enabled=bool(audit.get("enabled", True)),
        citation_audit_config_path=_project_path(audit.get("config_path", "configs/citation_audit_v1.yaml")),
    )
    _validate_dependencies(settings)
    return config, settings


def _validate_dependencies(settings: PipelineSettings) -> None:
    if not settings.schema_version or not settings.config_version:
        raise PipelineConfigurationError("Pipeline schema_version and config_version must not be empty.")
    if not settings.retrieval_enabled and any(
        (
            settings.rewrite_enabled,
            settings.reranker_enabled,
            settings.evidence_check_enabled,
            settings.answer_generation_enabled,
            settings.citation_audit_enabled,
        )
    ):
        raise PipelineConfigurationError("Retrieval must be enabled when any downstream pipeline module is enabled.")
    if settings.answer_generation_enabled and not settings.evidence_check_enabled:
        raise PipelineConfigurationError("Answer generation requires evidence_check to be enabled.")
    if settings.citation_audit_enabled and not settings.answer_generation_enabled:
        raise PipelineConfigurationError("Citation audit requires answer_generation to be enabled.")
    if settings.multi_query_enabled and not settings.rewrite_enabled:
        raise PipelineConfigurationError("multi_query requires rewrite to be enabled.")
    if (settings.rewrite_enabled or settings.reranker_enabled) and settings.retrieval_mode != "hybrid":
        raise PipelineConfigurationError("Rewrite and reranker stages require hybrid retrieval in v1.")


def _stage(
    *,
    status: str,
    model: str | None,
    config_version: str,
    output: Any = None,
    latency_ms: float = 0.0,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    end = end_time or _utc_now()
    start = start_time or end - timedelta(milliseconds=max(0.0, latency_ms))
    return {
        "status": status,
        "start_time": _timestamp(start),
        "end_time": _timestamp(end),
        "latency_ms": float(latency_ms),
        "model": model,
        "config_version": config_version,
        "reason": reason,
        "output": output,
    }


class ResearchGuardPipeline:
    def __init__(
        self,
        settings: PipelineSettings,
        *,
        retrieval_engine: Any | None = None,
        evidence_pipeline: Any | None = None,
        answer_pipeline: Any | None = None,
        citation_audit_pipeline: Any | None = None,
    ):
        self.settings = settings
        self._retrieval_engine = retrieval_engine
        self._evidence_pipeline = evidence_pipeline
        self._answer_pipeline = answer_pipeline
        self._citation_audit_pipeline = citation_audit_pipeline

    @classmethod
    def from_config(
        cls,
        path: str | Path = DEFAULT_CONFIG_PATH,
        **dependencies: Any,
    ) -> "ResearchGuardPipeline":
        _, settings = load_pipeline_settings(path)
        return cls(settings, **dependencies)

    def _engine(self) -> Any:
        if self._retrieval_engine is None:
            self._retrieval_engine = RetrievalEngine.from_config(self.settings.retrieval_config_path)
        return self._retrieval_engine

    def _evidence(self) -> Any:
        if self._evidence_pipeline is None:
            _, settings = load_evidence_judge_settings(self.settings.evidence_config_path)
            self._evidence_pipeline = EvidenceSufficiencyPipeline(settings)
        return self._evidence_pipeline

    def _answer(self) -> Any:
        if self._answer_pipeline is None:
            _, settings = load_answer_generation_settings(self.settings.answer_config_path)
            self._answer_pipeline = AnswerGenerationPipeline(settings)
        return self._answer_pipeline

    def _audit(self) -> Any:
        if self._citation_audit_pipeline is None:
            _, settings = load_citation_audit_settings(self.settings.citation_audit_config_path)
            self._citation_audit_pipeline = CitationAuditPipeline(settings)
        return self._citation_audit_pipeline

    def run(self, query: str) -> dict[str, Any]:
        normalized_query = normalize_question(query)
        if not normalized_query:
            raise ValueError("Query must not be empty.")

        pipeline_started_at = _utc_now()
        pipeline_started = time.perf_counter()
        stages = self._initial_stages(pipeline_started_at)
        result: dict[str, Any] = {
            "query": normalized_query,
            **stages,
            "final_status": "running",
            "pipeline": {
                "schema_version": self.settings.schema_version,
                "config_version": self.settings.config_version,
                "start_time": _timestamp(pipeline_started_at),
                "end_time": None,
                "latency_ms": 0.0,
            },
        }

        if not self.settings.retrieval_enabled:
            return self._finish(result, "disabled", pipeline_started)

        retrieval_started_at = _utc_now()
        try:
            engine = self._engine()
            response = engine.retrieve(
                normalized_query,
                mode=self.settings.retrieval_mode,
                top_k=self.settings.retrieval_top_k,
                candidate_k=self.settings.retrieval_candidate_k,
                rerank=self.settings.reranker_enabled,
                rerank_candidate_k=self.settings.reranker_candidate_k,
                rerank_read_cache=self.settings.read_cache,
                rewrite=self.settings.rewrite_enabled,
                multi_query=self.settings.multi_query_enabled,
                rewrite_read_cache=self.settings.read_cache,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            now = _utc_now()
            result["retrieval"] = _stage(
                status="failed",
                model=self._retrieval_model(self._retrieval_engine),
                config_version="retrieval_v1",
                start_time=retrieval_started_at,
                end_time=now,
                latency_ms=(now - retrieval_started_at).total_seconds() * 1000.0,
                reason=reason,
            )
            for name in ("rewrite", "reranking"):
                if result[name]["status"] == "pending":
                    result[name] = {**result[name], "status": "failed", "reason": reason}
            return self._finish(result, "failed", pipeline_started)

        retrieval_finished_at = _utc_now()
        self._record_retrieval_stages(result, response, retrieval_started_at, retrieval_finished_at)
        if not self.settings.evidence_check_enabled:
            return self._finish(result, "retrieved", pipeline_started)

        evidence_started_at = _utc_now()
        try:
            evidence_result = self._evidence().assess(
                normalized_query,
                response.hits,
                read_cache=self.settings.read_cache,
            )
        except Exception as exc:
            result["evidence_check"] = self._failed_component_stage(
                self._evidence_pipeline, evidence_started_at, exc, "evidence_sufficiency_v1"
            )
            return self._finish(result, "failed", pipeline_started)
        result["evidence_check"] = self._result_stage(
            evidence_result,
            evidence_started_at,
            status="fallback" if evidence_result.fallback_used else "completed",
        )

        if not evidence_result.answerable or evidence_result.support_level != "strong":
            reason = f"evidence_{evidence_result.support_level}"
            result["answer_generation"] = self._skipped_stage(
                result["answer_generation"], reason=reason
            )
            result["citation_audit"] = self._skipped_stage(result["citation_audit"], reason=reason)
            return self._finish(result, "rejected", pipeline_started)
        if not self.settings.answer_generation_enabled:
            return self._finish(result, "evidence_sufficient", pipeline_started)

        answer_started_at = _utc_now()
        try:
            answer_result = self._answer().generate(
                normalized_query,
                response.hits,
                evidence_result,
                read_cache=self.settings.read_cache,
            )
        except Exception as exc:
            result["answer_generation"] = self._failed_component_stage(
                self._answer_pipeline, answer_started_at, exc, "answer_generation_v1"
            )
            return self._finish(result, "failed", pipeline_started)
        answer_status = "fallback" if answer_result.fallback_used else "completed"
        result["answer_generation"] = self._result_stage(answer_result, answer_started_at, status=answer_status)
        if answer_result.refused:
            result["citation_audit"] = self._skipped_stage(
                result["citation_audit"], reason="answer_refused"
            )
            final = "failed" if answer_result.fallback_used else "rejected"
            return self._finish(result, final, pipeline_started)
        if not self.settings.citation_audit_enabled:
            return self._finish(result, "answered", pipeline_started)

        audit_started_at = _utc_now()
        try:
            audit_result = self._audit().audit(
                answer_result,
                response.hits,
                read_cache=self.settings.read_cache,
            )
        except Exception as exc:
            result["citation_audit"] = self._failed_component_stage(
                self._citation_audit_pipeline, audit_started_at, exc, "citation_audit_v1"
            )
            return self._finish(result, "failed", pipeline_started)
        audit_status = "fallback" if audit_result.fallback_used else "completed"
        result["citation_audit"] = self._result_stage(audit_result, audit_started_at, status=audit_status)
        if not audit_result.audit_completed:
            final_status = "failed"
        elif audit_result.overall_grounded:
            final_status = "grounded"
        else:
            final_status = "needs_review"
        return self._finish(result, final_status, pipeline_started)

    def _initial_stages(self, now: datetime) -> dict[str, dict[str, Any]]:
        engine = self._retrieval_engine
        rewrite_model = getattr(getattr(engine, "query_rewrite_settings", None), "model", None)
        reranker_model = getattr(getattr(engine, "reranker_settings", None), "model_name", None)
        values = {
            "rewrite": (self.settings.rewrite_enabled, rewrite_model, "query_rewrite_v1"),
            "retrieval": (self.settings.retrieval_enabled, self._retrieval_model(engine), "retrieval_v1"),
            "reranking": (self.settings.reranker_enabled, reranker_model, "reranker_v1"),
            "evidence_check": (self.settings.evidence_check_enabled, None, "evidence_sufficiency_v1"),
            "answer_generation": (self.settings.answer_generation_enabled, None, "answer_generation_v1"),
            "citation_audit": (self.settings.citation_audit_enabled, None, "citation_audit_v1"),
        }
        return {
            name: _stage(
                status="pending" if enabled else "disabled",
                model=model,
                config_version=version,
                start_time=now,
                end_time=now,
                reason=None if enabled else "disabled_by_config",
            )
            for name, (enabled, model, version) in values.items()
        }

    def _record_retrieval_stages(
        self,
        result: dict[str, Any],
        response: Any,
        started_at: datetime,
        finished_at: datetime,
    ) -> None:
        engine = self._retrieval_engine
        trace = response.trace
        result["retrieval"] = _stage(
            status="completed",
            model=self._retrieval_model(engine),
            config_version="retrieval_v1",
            output=response.to_dict(include_text=self.settings.include_retrieval_text),
            latency_ms=response.retrieval_latency_ms,
            start_time=started_at,
            end_time=finished_at,
        )
        if self.settings.rewrite_enabled:
            rewrite_output = trace.get("query_rewrite")
            result["rewrite"] = _stage(
                status="fallback" if rewrite_output and rewrite_output.get("fallback_used") else "completed",
                model=getattr(engine.query_rewrite_settings, "model", None),
                config_version=getattr(engine.query_rewrite_settings, "prompt_version", "query_rewrite_v1"),
                output=rewrite_output,
                latency_ms=response.rewrite_latency_ms,
                end_time=finished_at,
            )
        if self.settings.reranker_enabled:
            result["reranking"] = _stage(
                status="completed",
                model=getattr(engine.reranker_settings, "model_name", None),
                config_version=getattr(engine.reranker_settings, "config_version", "reranker_v1"),
                output=trace.get("reranker"),
                latency_ms=response.rerank_latency_ms,
                end_time=finished_at,
            )

    @staticmethod
    def _retrieval_model(engine: Any | None) -> str | None:
        provider = getattr(engine, "embedding_provider", None)
        config = getattr(provider, "config", None)
        return getattr(config, "model", None)

    @staticmethod
    def _result_stage(value: Any, started_at: datetime, *, status: str) -> dict[str, Any]:
        latency = float(getattr(value, "latency_ms", 0.0))
        return _stage(
            status=status,
            model=getattr(value, "model", None),
            config_version=str(getattr(value, "config_version", "unknown")),
            output=value.to_dict(),
            latency_ms=latency,
            start_time=started_at,
            end_time=_utc_now(),
            reason=getattr(value, "fallback_reason", None) if status == "fallback" else None,
        )

    @staticmethod
    def _failed_component_stage(component: Any, started_at: datetime, exc: Exception, version: str) -> dict[str, Any]:
        settings = getattr(component, "settings", None)
        now = _utc_now()
        return _stage(
            status="failed",
            model=getattr(settings, "model", None),
            config_version=str(getattr(settings, "config_version", version)),
            latency_ms=(now - started_at).total_seconds() * 1000.0,
            start_time=started_at,
            end_time=now,
            reason=f"{type(exc).__name__}: {exc}",
        )

    @staticmethod
    def _skipped_stage(stage: dict[str, Any], *, reason: str) -> dict[str, Any]:
        now = _utc_now()
        return {
            **stage,
            "status": "skipped",
            "start_time": _timestamp(now),
            "end_time": _timestamp(now),
            "latency_ms": 0.0,
            "reason": reason,
        }

    @staticmethod
    def _finish(result: dict[str, Any], final_status: str, started: float) -> dict[str, Any]:
        finished_at = _utc_now()
        for name in STAGE_NAMES:
            if result[name]["status"] == "pending":
                result[name] = ResearchGuardPipeline._skipped_stage(
                    result[name], reason=f"pipeline_finished:{final_status}"
                )
        result["final_status"] = final_status
        result["pipeline"]["end_time"] = _timestamp(finished_at)
        result["pipeline"]["latency_ms"] = (time.perf_counter() - started) * 1000.0
        return result


def run_pipeline(
    query: str,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    return ResearchGuardPipeline.from_config(config_path).run(query)
