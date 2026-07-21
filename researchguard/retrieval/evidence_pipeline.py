# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\evidence_pipeline.py
from __future__ import annotations

import math
import time
from typing import Any, Iterable, Mapping

from researchguard.indexing.corpus_loader import stable_json_hash
from researchguard.retrieval.evidence_cache import EvidenceSufficiencyCache
from researchguard.retrieval.evidence_judge import (
    SUPPORT_LEVELS,
    BackendEvidenceJudgment,
    EvidenceJudgeBackend,
    EvidenceJudgeBackendError,
    EvidenceJudgeSettings,
    EvidencePassage,
    EvidenceSufficiencyResult,
    OpenAIEvidenceJudgeBackend,
    build_evidence_model_input,
    normalize_question,
    prepare_evidence_passages,
    utc_timestamp,
)
from researchguard.retrieval.models import RetrievalHit
from researchguard.retrieval.query_rewriter import extract_preserved_entities, missing_entities


class EvidenceSufficiencyPipeline:
    def __init__(
        self,
        settings: EvidenceJudgeSettings,
        *,
        backend: EvidenceJudgeBackend | None = None,
        cache: EvidenceSufficiencyCache | None = None,
    ):
        self.settings = settings
        self.backend = backend or OpenAIEvidenceJudgeBackend(settings)
        self.cache = cache or EvidenceSufficiencyCache(
            settings.cache_directory,
            enabled=settings.cache_enabled,
        )

    def assess(
        self,
        query: str,
        hits: Iterable[RetrievalHit | Mapping[str, Any]],
        *,
        read_cache: bool = True,
    ) -> EvidenceSufficiencyResult:
        started = time.perf_counter()
        question = normalize_question(query)
        if not question:
            return self._fallback_result("empty_question", started=started)
        passages = prepare_evidence_passages(
            hits,
            max_chunks=self.settings.max_evidence_chunks,
            max_chars_per_chunk=self.settings.max_chars_per_chunk,
        )
        if not passages:
            return self._fallback_result("empty_evidence", started=started)

        model_input = build_evidence_model_input(question, passages)
        input_hash = stable_json_hash(model_input)
        chunk_ids = [passage.chunk_id for passage in passages]
        key = self.cache.make_key(
            query=question,
            chunk_ids=chunk_ids,
            input_hash=input_hash,
            settings=self.settings,
        )
        cached = self.cache.get(key, input_hash=input_hash) if read_cache else None
        if cached is not None:
            result = self._result_from_cache(cached, question, passages)
            if result is not None:
                return EvidenceSufficiencyResult(
                    **{
                        **result.to_dict(),
                        "supporting_chunk_ids": result.supporting_chunk_ids,
                        "cache_hit": True,
                        "api_call_count": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "latency_ms": (time.perf_counter() - started) * 1000.0,
                    }
                )

        try:
            backend_result = self.backend.judge(question, passages)
            result = self._validated_result(question, backend_result, passages, started=started)
        except EvidenceJudgeBackendError as exc:
            result = self._fallback_result(
                f"backend_failure:{type(exc.__cause__ or exc).__name__}",
                started=started,
                api_call_count=exc.api_call_count,
            )
        except Exception as exc:
            result = self._fallback_result(
                f"judge_failure:{type(exc).__name__}",
                started=started,
            )
        self.cache.put(
            key,
            input_hash=input_hash,
            output=result.to_dict(),
            timestamp=result.timestamp,
        )
        return result

    def _validated_result(
        self,
        question: str,
        backend_result: BackendEvidenceJudgment,
        passages: tuple[EvidencePassage, ...],
        *,
        started: float,
    ) -> EvidenceSufficiencyResult:
        level = backend_result.support_level.strip().casefold()
        reason = " ".join(backend_result.reason.split()).strip()
        confidence = float(backend_result.confidence)
        supporting_ids = tuple(dict.fromkeys(item.strip() for item in backend_result.supporting_chunk_ids if item.strip()))
        supported_requirements = tuple(
            dict.fromkeys(item.strip() for item in backend_result.supported_requirements if item.strip())
        )
        missing_requirements = tuple(
            dict.fromkeys(item.strip() for item in backend_result.missing_requirements if item.strip())
        )
        available_ids = {passage.chunk_id for passage in passages}

        if (
            level not in SUPPORT_LEVELS
            or not reason
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or (not supported_requirements and not missing_requirements)
        ):
            return self._fallback_from_backend("schema_failure", backend_result, started=started)
        if any(chunk_id not in available_ids for chunk_id in supporting_ids):
            return self._fallback_from_backend("missing_chunk_reference", backend_result, started=started)
        if supported_requirements and missing_requirements:
            level = "partial"
        elif supported_requirements:
            level = "strong"
        else:
            level = "unsupported"
        answerable = level == "strong"
        if level in {"strong", "partial"} and not supporting_ids:
            return self._fallback_from_backend("schema_failure", backend_result, started=started)
        if level == "unsupported" and supporting_ids:
            return self._fallback_from_backend("schema_failure", backend_result, started=started)

        combined_evidence = "\n".join(passage.text for passage in passages)
        absent_entities = missing_entities(combined_evidence, extract_preserved_entities(question))
        if level == "strong" and absent_entities:
            missing_labels = self._remove_subsumed_entities(absent_entities)
            is_partial = self._looks_compound(question) and bool(supporting_ids)
            return EvidenceSufficiencyResult(
                answerable=False,
                support_level="partial" if is_partial else "unsupported",
                confidence=min(confidence, 0.9),
                reason=(
                    "The supplied evidence does not mention required query entities: "
                    f"{', '.join(missing_labels)}. Omission cannot establish a negative answer."
                ),
                supporting_chunk_ids=supporting_ids if is_partial else (),
                model=self.settings.model,
                prompt_version=self.settings.prompt_version,
                config_version=self.settings.config_version,
                timestamp=utc_timestamp(),
                cache_hit=False,
                fallback_used=False,
                fallback_reason=None,
                api_call_count=backend_result.api_call_count,
                input_tokens=backend_result.input_tokens,
                output_tokens=backend_result.output_tokens,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        return EvidenceSufficiencyResult(
            answerable=answerable,
            support_level=level,
            confidence=confidence,
            reason=reason,
            supporting_chunk_ids=supporting_ids,
            model=self.settings.model,
            prompt_version=self.settings.prompt_version,
            config_version=self.settings.config_version,
            timestamp=utc_timestamp(),
            cache_hit=False,
            fallback_used=False,
            fallback_reason=None,
            api_call_count=backend_result.api_call_count,
            input_tokens=backend_result.input_tokens,
            output_tokens=backend_result.output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _remove_subsumed_entities(entities: list[str]) -> list[str]:
        normalized = [(entity, entity.casefold()) for entity in entities]
        return [
            entity
            for entity, key in normalized
            if not any(key != other and key in other for _, other in normalized)
        ]

    @staticmethod
    def _looks_compound(question: str) -> bool:
        normalized = f" {question.casefold()} "
        return any(marker in normalized for marker in (" and ", " compare ", " versus ", " vs. ", " vs "))

    def _fallback_from_backend(
        self,
        reason: str,
        backend_result: BackendEvidenceJudgment,
        *,
        started: float,
    ) -> EvidenceSufficiencyResult:
        return self._fallback_result(
            reason,
            started=started,
            api_call_count=backend_result.api_call_count,
            input_tokens=backend_result.input_tokens,
            output_tokens=backend_result.output_tokens,
        )

    def _fallback_result(
        self,
        reason: str,
        *,
        started: float,
        api_call_count: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> EvidenceSufficiencyResult:
        return EvidenceSufficiencyResult(
            answerable=False,
            support_level="unsupported",
            confidence=0.0,
            reason="Evidence sufficiency could not be established safely; answer generation must remain disabled.",
            supporting_chunk_ids=(),
            model=self.settings.model,
            prompt_version=self.settings.prompt_version,
            config_version=self.settings.config_version,
            timestamp=utc_timestamp(),
            cache_hit=False,
            fallback_used=True,
            fallback_reason=reason,
            api_call_count=api_call_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _result_from_cache(
        self,
        payload: dict[str, Any],
        question: str,
        passages: tuple[EvidencePassage, ...],
    ) -> EvidenceSufficiencyResult | None:
        try:
            result = EvidenceSufficiencyResult(
                answerable=bool(payload["answerable"]),
                support_level=str(payload["support_level"]),
                confidence=float(payload["confidence"]),
                reason=str(payload["reason"]),
                supporting_chunk_ids=tuple(str(item) for item in payload.get("supporting_chunk_ids", [])),
                model=str(payload["model"]),
                prompt_version=str(payload["prompt_version"]),
                config_version=str(payload["config_version"]),
                timestamp=str(payload["timestamp"]),
                cache_hit=True,
                fallback_used=bool(payload.get("fallback_used", False)),
                fallback_reason=payload.get("fallback_reason"),
                api_call_count=0,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
            )
        except (KeyError, TypeError, ValueError):
            return None
        if result.model != self.settings.model:
            return None
        if result.prompt_version != self.settings.prompt_version or result.config_version != self.settings.config_version:
            return None
        available_ids = {passage.chunk_id for passage in passages}
        if not self._valid_result_shape(result, available_ids):
            return None
        if result.support_level == "strong":
            evidence_text = "\n".join(passage.text for passage in passages)
            if missing_entities(evidence_text, extract_preserved_entities(question)):
                return None
        return result

    @staticmethod
    def _valid_result_shape(result: EvidenceSufficiencyResult, available_ids: set[str]) -> bool:
        if (
            result.support_level not in SUPPORT_LEVELS
            or not result.reason.strip()
            or not math.isfinite(result.confidence)
            or not 0 <= result.confidence <= 1
            or len(result.supporting_chunk_ids) != len(set(result.supporting_chunk_ids))
            or any(chunk_id not in available_ids for chunk_id in result.supporting_chunk_ids)
        ):
            return False
        if result.support_level == "strong":
            return result.answerable and bool(result.supporting_chunk_ids)
        if result.support_level == "partial":
            return not result.answerable and bool(result.supporting_chunk_ids)
        return not result.answerable and not result.supporting_chunk_ids
