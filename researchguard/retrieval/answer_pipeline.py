# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\answer_pipeline.py
from __future__ import annotations

import math
import time
from typing import Any, Iterable, Mapping

from researchguard.indexing.corpus_loader import stable_json_hash
from researchguard.retrieval.answer_cache import AnswerGenerationCache
from researchguard.retrieval.answer_generator import (
    REFUSAL_ANSWER,
    AnswerCitation,
    AnswerGenerationBackendError,
    AnswerGenerationResult,
    AnswerGenerationSettings,
    AnswerGeneratorBackend,
    AnswerPassage,
    BackendGeneratedAnswer,
    OpenAIAnswerGeneratorBackend,
    build_answer_model_input,
    prepare_answer_passages,
    utc_timestamp,
)
from researchguard.retrieval.evidence_judge import EvidenceSufficiencyResult, normalize_question
from researchguard.retrieval.models import RetrievalHit


class AnswerGenerationPipeline:
    def __init__(
        self,
        settings: AnswerGenerationSettings,
        *,
        backend: AnswerGeneratorBackend | None = None,
        cache: AnswerGenerationCache | None = None,
    ):
        self.settings = settings
        self.backend = backend or OpenAIAnswerGeneratorBackend(settings)
        self.cache = cache or AnswerGenerationCache(settings.cache_directory, enabled=settings.cache_enabled)

    def generate(
        self,
        query: str,
        hits: Iterable[RetrievalHit | Mapping[str, Any]],
        sufficiency: EvidenceSufficiencyResult,
        *,
        read_cache: bool = True,
    ) -> AnswerGenerationResult:
        started = time.perf_counter()
        question = normalize_question(query)
        if not sufficiency.answerable or sufficiency.support_level != "strong":
            return self._refusal_result("evidence_not_answerable", started=started)
        if not question:
            return self._fallback_result("empty_question", started=started)

        supporting_ids = tuple(dict.fromkeys(sufficiency.supporting_chunk_ids))
        passages = prepare_answer_passages(
            hits,
            supporting_ids,
            max_chunks=self.settings.max_evidence_chunks,
            max_chars_per_chunk=self.settings.max_chars_per_chunk,
        )
        if not supporting_ids or tuple(passage.chunk_id for passage in passages) != supporting_ids:
            return self._fallback_result(
                "missing_supporting_evidence",
                started=started,
                evidence_chunk_ids=tuple(passage.chunk_id for passage in passages),
            )

        model_input = build_answer_model_input(question, passages)
        input_hash = stable_json_hash(model_input)
        evidence_ids = [passage.chunk_id for passage in passages]
        key = self.cache.make_key(
            query=question,
            evidence_chunk_ids=evidence_ids,
            input_hash=input_hash,
            settings=self.settings,
        )
        cached = self.cache.get(key, input_hash=input_hash) if read_cache else None
        if cached is not None:
            result = self._result_from_cache(cached, passages)
            if result is not None:
                return AnswerGenerationResult(
                    **{
                        **result.to_dict(),
                        "citations": result.citations,
                        "evidence_chunk_ids": result.evidence_chunk_ids,
                        "cache_hit": True,
                        "api_call_count": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "latency_ms": (time.perf_counter() - started) * 1000.0,
                    }
                )

        try:
            backend_result = self.backend.generate(question, passages)
            result = self._validated_result(backend_result, passages, started=started)
        except AnswerGenerationBackendError as exc:
            result = self._fallback_result(
                f"backend_failure:{type(exc.__cause__ or exc).__name__}",
                started=started,
                evidence_chunk_ids=tuple(evidence_ids),
                api_call_count=exc.api_call_count,
            )
        except Exception as exc:
            result = self._fallback_result(
                f"generation_failure:{type(exc).__name__}",
                started=started,
                evidence_chunk_ids=tuple(evidence_ids),
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
        backend_result: BackendGeneratedAnswer,
        passages: tuple[AnswerPassage, ...],
        *,
        started: float,
    ) -> AnswerGenerationResult:
        answer = " ".join(backend_result.answer.split()).strip()
        confidence = float(backend_result.confidence)
        passage_by_id = {passage.chunk_id: passage for passage in passages}
        citation_ids = [citation.chunk_id.strip() for citation in backend_result.citations]
        if (
            not answer
            or answer == REFUSAL_ANSWER
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or not citation_ids
            or len(citation_ids) != len(set(citation_ids))
        ):
            return self._fallback_from_backend("schema_failure", backend_result, passages, started=started)
        for citation in backend_result.citations:
            passage = passage_by_id.get(citation.chunk_id.strip())
            if passage is None:
                return self._fallback_from_backend("citation_outside_evidence", backend_result, passages, started=started)
            if (
                citation.doc_id.strip() != passage.doc_id
                or citation.section.strip() != passage.section
                or citation.page != passage.page
            ):
                return self._fallback_from_backend("citation_metadata_mismatch", backend_result, passages, started=started)

        canonical_citations = tuple(
            AnswerCitation(
                chunk_id=passage_by_id[chunk_id].chunk_id,
                doc_id=passage_by_id[chunk_id].doc_id,
                section=passage_by_id[chunk_id].section,
                page=passage_by_id[chunk_id].page,
            )
            for chunk_id in citation_ids
        )
        return AnswerGenerationResult(
            answer=answer,
            citations=canonical_citations,
            confidence=confidence,
            refused=False,
            refusal_reason=None,
            evidence_chunk_ids=tuple(passage_by_id),
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

    def _fallback_from_backend(
        self,
        reason: str,
        backend_result: BackendGeneratedAnswer,
        passages: tuple[AnswerPassage, ...],
        *,
        started: float,
    ) -> AnswerGenerationResult:
        return self._fallback_result(
            reason,
            started=started,
            evidence_chunk_ids=tuple(passage.chunk_id for passage in passages),
            api_call_count=backend_result.api_call_count,
            input_tokens=backend_result.input_tokens,
            output_tokens=backend_result.output_tokens,
        )

    def _refusal_result(self, reason: str, *, started: float) -> AnswerGenerationResult:
        return AnswerGenerationResult(
            answer=REFUSAL_ANSWER,
            citations=(),
            confidence=0.0,
            refused=True,
            refusal_reason=reason,
            evidence_chunk_ids=(),
            model=self.settings.model,
            prompt_version=self.settings.prompt_version,
            config_version=self.settings.config_version,
            timestamp=utc_timestamp(),
            cache_hit=False,
            fallback_used=False,
            fallback_reason=None,
            api_call_count=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _fallback_result(
        self,
        reason: str,
        *,
        started: float,
        evidence_chunk_ids: tuple[str, ...] = (),
        api_call_count: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> AnswerGenerationResult:
        result = self._refusal_result("generation_failed_closed", started=started)
        return AnswerGenerationResult(
            **{
                **result.to_dict(),
                "citations": (),
                "evidence_chunk_ids": evidence_chunk_ids,
                "fallback_used": True,
                "fallback_reason": reason,
                "api_call_count": api_call_count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        )

    def _result_from_cache(
        self,
        payload: dict[str, Any],
        passages: tuple[AnswerPassage, ...],
    ) -> AnswerGenerationResult | None:
        try:
            result = AnswerGenerationResult(
                answer=str(payload["answer"]),
                citations=tuple(
                    AnswerCitation(
                        chunk_id=str(item["chunk_id"]),
                        doc_id=str(item["doc_id"]),
                        section=str(item["section"]),
                        page=item.get("page"),
                    )
                    for item in payload.get("citations", [])
                ),
                confidence=float(payload["confidence"]),
                refused=bool(payload["refused"]),
                refusal_reason=payload.get("refusal_reason"),
                evidence_chunk_ids=tuple(str(item) for item in payload.get("evidence_chunk_ids", [])),
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
        if result.evidence_chunk_ids != tuple(passage.chunk_id for passage in passages):
            return None
        if result.refused:
            return result if self._valid_refusal(result) else None
        return result if self._valid_generated(result, passages) else None

    @staticmethod
    def _valid_refusal(result: AnswerGenerationResult) -> bool:
        return result.answer == REFUSAL_ANSWER and not result.citations and result.confidence == 0

    @staticmethod
    def _valid_generated(result: AnswerGenerationResult, passages: tuple[AnswerPassage, ...]) -> bool:
        passage_by_id = {passage.chunk_id: passage for passage in passages}
        if (
            not result.answer.strip()
            or result.answer == REFUSAL_ANSWER
            or not result.citations
            or not math.isfinite(result.confidence)
            or not 0 <= result.confidence <= 1
        ):
            return False
        for citation in result.citations:
            passage = passage_by_id.get(citation.chunk_id)
            if passage is None or citation.to_dict() != AnswerCitation(
                chunk_id=passage.chunk_id,
                doc_id=passage.doc_id,
                section=passage.section,
                page=passage.page,
            ).to_dict():
                return False
        return len(result.citations) == len({citation.chunk_id for citation in result.citations})
