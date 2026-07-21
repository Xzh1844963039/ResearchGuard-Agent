# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\citation_audit.py
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from researchguard.indexing.corpus_loader import stable_json_hash
from researchguard.retrieval.answer_generator import (
    AnswerCitation,
    AnswerGenerationResult,
    AnswerPassage,
    utc_timestamp,
)
from researchguard.retrieval.citation_cache import CitationAuditCache
from researchguard.retrieval.claim_extractor import (
    BackendClaimExtraction,
    CitationAuditSettings,
    ClaimExtractionBackendError,
    ClaimExtractorBackend,
    ExtractedClaim,
    OpenAIClaimExtractorBackend,
)
from researchguard.retrieval.claim_verifier import (
    SUPPORT_LEVELS,
    BackendClaimVerification,
    ClaimVerificationBackendError,
    ClaimVerifierBackend,
    OpenAIClaimVerifierBackend,
)
from researchguard.retrieval.models import RetrievalHit
from researchguard.retrieval.query_rewriter import extract_preserved_entities, missing_entities


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class AuditedClaim:
    claim_id: str
    text: str
    support_level: str
    confidence: float
    citations: tuple[AnswerCitation, ...]
    reason: str
    candidate_chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.claim_id,
            "text": self.text,
            "support_level": self.support_level,
            "confidence": self.confidence,
            "citations": [citation.to_dict() for citation in self.citations],
            "reason": self.reason,
            "candidate_chunk_ids": list(self.candidate_chunk_ids),
        }


@dataclass(frozen=True)
class CitationAuditResult:
    answer: str
    claims: tuple[AuditedClaim, ...]
    overall_grounded: bool
    unsupported_claim_count: int
    partial_claim_count: int
    grounding_score: float
    audit_completed: bool
    audit_reason: str | None
    evidence_chunk_ids: tuple[str, ...]
    model: str
    extraction_prompt_version: str
    verification_prompt_version: str
    config_version: str
    timestamp: str
    cache_hit: bool
    fallback_used: bool
    fallback_reason: str | None
    api_call_count: int
    extraction_api_calls: int
    verification_api_calls: int
    input_tokens: int
    output_tokens: int
    extraction_latency_ms: float
    verification_latency_ms: float
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "claims": [claim.to_dict() for claim in self.claims],
            "overall_grounded": self.overall_grounded,
            "unsupported_claim_count": self.unsupported_claim_count,
            "partial_claim_count": self.partial_claim_count,
            "grounding_score": self.grounding_score,
            "audit_completed": self.audit_completed,
            "audit_reason": self.audit_reason,
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
            "model": self.model,
            "extraction_prompt_version": self.extraction_prompt_version,
            "verification_prompt_version": self.verification_prompt_version,
            "config_version": self.config_version,
            "timestamp": self.timestamp,
            "cache_hit": self.cache_hit,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "api_call_count": self.api_call_count,
            "extraction_api_calls": self.extraction_api_calls,
            "verification_api_calls": self.verification_api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "extraction_latency_ms": self.extraction_latency_ms,
            "verification_latency_ms": self.verification_latency_ms,
            "latency_ms": self.latency_ms,
        }


def _hit_mapping(hit: RetrievalHit | Mapping[str, Any]) -> Mapping[str, Any]:
    return hit.to_dict(include_text=True) if isinstance(hit, RetrievalHit) else hit


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(text) if len(token) > 1}


def build_audit_input(
    answer_result: AnswerGenerationResult,
    passages: tuple[AnswerPassage, ...],
) -> dict[str, Any]:
    return {
        "answer": answer_result.answer,
        "answer_citations": [citation.to_dict() for citation in answer_result.citations],
        "generation_evidence_chunk_ids": list(answer_result.evidence_chunk_ids),
        "evidence_passages": [passage.to_dict() for passage in passages],
    }


class CitationAuditPipeline:
    def __init__(
        self,
        settings: CitationAuditSettings,
        *,
        extractor: ClaimExtractorBackend | None = None,
        verifier: ClaimVerifierBackend | None = None,
        cache: CitationAuditCache | None = None,
    ):
        self.settings = settings
        self.extractor = extractor or OpenAIClaimExtractorBackend(settings)
        self.verifier = verifier or OpenAIClaimVerifierBackend(settings)
        self.cache = cache or CitationAuditCache(settings.cache_directory, enabled=settings.cache_enabled)

    def audit(
        self,
        answer_result: AnswerGenerationResult,
        hits: Iterable[RetrievalHit | Mapping[str, Any]],
        *,
        read_cache: bool = True,
    ) -> CitationAuditResult:
        started = time.perf_counter()
        if answer_result.refused:
            return self._not_audited_result(answer_result.answer, "answer_refused", started=started)
        if answer_result.fallback_used or not answer_result.answer.strip():
            return self._fallback_result(answer_result.answer, "invalid_answer_result", started=started)

        passages = self._prepare_generation_passages(answer_result, hits)
        expected_ids = tuple(answer_result.evidence_chunk_ids)
        if not expected_ids or tuple(passage.chunk_id for passage in passages) != expected_ids:
            return self._fallback_result(
                answer_result.answer,
                "missing_generation_evidence",
                started=started,
                evidence_chunk_ids=tuple(passage.chunk_id for passage in passages),
            )
        passage_by_id = {passage.chunk_id: passage for passage in passages}
        if not self._valid_answer_citations(answer_result.citations, passage_by_id):
            return self._fallback_result(
                answer_result.answer,
                "invalid_answer_citations",
                started=started,
                evidence_chunk_ids=expected_ids,
            )

        model_input = build_audit_input(answer_result, passages)
        input_hash = stable_json_hash(model_input)
        answer_hash = stable_json_hash({"answer": answer_result.answer})
        key = self.cache.make_key(
            answer_hash=answer_hash,
            evidence_chunk_ids=list(expected_ids),
            input_hash=input_hash,
            settings=self.settings,
        )
        cached = self.cache.get(key, input_hash=input_hash) if read_cache else None
        if cached is not None:
            result = self._result_from_cache(cached, answer_result, passages)
            if result is not None:
                return CitationAuditResult(
                    **{
                        **result.to_dict(),
                        "claims": result.claims,
                        "evidence_chunk_ids": result.evidence_chunk_ids,
                        "cache_hit": True,
                        "api_call_count": 0,
                        "extraction_api_calls": 0,
                        "verification_api_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "extraction_latency_ms": 0.0,
                        "verification_latency_ms": 0.0,
                        "latency_ms": (time.perf_counter() - started) * 1000.0,
                    }
                )

        extraction_started = time.perf_counter()
        try:
            extraction = self.extractor.extract(answer_result.answer)
        except ClaimExtractionBackendError as exc:
            result = self._fallback_result(
                answer_result.answer,
                f"claim_extraction_failure:{type(exc.__cause__ or exc).__name__}",
                started=started,
                evidence_chunk_ids=expected_ids,
                api_call_count=exc.api_call_count,
                extraction_api_calls=exc.api_call_count,
                extraction_latency_ms=(time.perf_counter() - extraction_started) * 1000.0,
            )
            self._cache_result(key, input_hash, result)
            return result
        except Exception as exc:
            result = self._fallback_result(
                answer_result.answer,
                f"claim_extraction_failure:{type(exc).__name__}",
                started=started,
                evidence_chunk_ids=expected_ids,
                extraction_latency_ms=(time.perf_counter() - extraction_started) * 1000.0,
            )
            self._cache_result(key, input_hash, result)
            return result
        extraction_latency = (time.perf_counter() - extraction_started) * 1000.0
        claims = self._validate_extraction(extraction, answer_result.answer)
        if claims is None:
            result = self._fallback_result(
                answer_result.answer,
                "claim_extraction_schema_failure",
                started=started,
                evidence_chunk_ids=expected_ids,
                api_call_count=extraction.api_call_count,
                extraction_api_calls=extraction.api_call_count,
                input_tokens=extraction.input_tokens,
                output_tokens=extraction.output_tokens,
                extraction_latency_ms=extraction_latency,
            )
            self._cache_result(key, input_hash, result)
            return result

        citation_ids = tuple(citation.chunk_id for citation in answer_result.citations)
        audited_claims: list[AuditedClaim] = []
        verification_calls = 0
        verification_input_tokens = 0
        verification_output_tokens = 0
        verification_started = time.perf_counter()
        for claim in claims:
            candidates = self._candidate_passages(claim, passages, citation_ids)
            try:
                verification = self.verifier.verify(claim, candidates)
            except ClaimVerificationBackendError as exc:
                verification_calls += exc.api_call_count
                result = self._fallback_result(
                    answer_result.answer,
                    f"claim_verification_failure:{type(exc.__cause__ or exc).__name__}",
                    started=started,
                    evidence_chunk_ids=expected_ids,
                    api_call_count=extraction.api_call_count + verification_calls,
                    extraction_api_calls=extraction.api_call_count,
                    verification_api_calls=verification_calls,
                    input_tokens=extraction.input_tokens + verification_input_tokens,
                    output_tokens=extraction.output_tokens + verification_output_tokens,
                    extraction_latency_ms=extraction_latency,
                    verification_latency_ms=(time.perf_counter() - verification_started) * 1000.0,
                )
                self._cache_result(key, input_hash, result)
                return result
            except Exception as exc:
                result = self._fallback_result(
                    answer_result.answer,
                    f"claim_verification_failure:{type(exc).__name__}",
                    started=started,
                    evidence_chunk_ids=expected_ids,
                    api_call_count=extraction.api_call_count + verification_calls,
                    extraction_api_calls=extraction.api_call_count,
                    verification_api_calls=verification_calls,
                    input_tokens=extraction.input_tokens + verification_input_tokens,
                    output_tokens=extraction.output_tokens + verification_output_tokens,
                    extraction_latency_ms=extraction_latency,
                    verification_latency_ms=(time.perf_counter() - verification_started) * 1000.0,
                )
                self._cache_result(key, input_hash, result)
                return result
            verification_calls += verification.api_call_count
            verification_input_tokens += verification.input_tokens
            verification_output_tokens += verification.output_tokens
            audited = self._validate_verification(claim, verification, candidates)
            if audited is None:
                result = self._fallback_result(
                    answer_result.answer,
                    "claim_verification_schema_failure",
                    started=started,
                    evidence_chunk_ids=expected_ids,
                    api_call_count=extraction.api_call_count + verification_calls,
                    extraction_api_calls=extraction.api_call_count,
                    verification_api_calls=verification_calls,
                    input_tokens=extraction.input_tokens + verification_input_tokens,
                    output_tokens=extraction.output_tokens + verification_output_tokens,
                    extraction_latency_ms=extraction_latency,
                    verification_latency_ms=(time.perf_counter() - verification_started) * 1000.0,
                )
                self._cache_result(key, input_hash, result)
                return result
            audited_claims.append(audited)

        verification_latency = (time.perf_counter() - verification_started) * 1000.0
        supported_count = sum(claim.support_level == "supported" for claim in audited_claims)
        unsupported_count = sum(claim.support_level == "unsupported" for claim in audited_claims)
        partial_count = sum(claim.support_level == "partial" for claim in audited_claims)
        result = CitationAuditResult(
            answer=answer_result.answer,
            claims=tuple(audited_claims),
            overall_grounded=supported_count == len(audited_claims),
            unsupported_claim_count=unsupported_count,
            partial_claim_count=partial_count,
            grounding_score=supported_count / len(audited_claims),
            audit_completed=True,
            audit_reason=None,
            evidence_chunk_ids=expected_ids,
            model=self.settings.model,
            extraction_prompt_version=self.settings.extraction_prompt_version,
            verification_prompt_version=self.settings.verification_prompt_version,
            config_version=self.settings.config_version,
            timestamp=utc_timestamp(),
            cache_hit=False,
            fallback_used=False,
            fallback_reason=None,
            api_call_count=extraction.api_call_count + verification_calls,
            extraction_api_calls=extraction.api_call_count,
            verification_api_calls=verification_calls,
            input_tokens=extraction.input_tokens + verification_input_tokens,
            output_tokens=extraction.output_tokens + verification_output_tokens,
            extraction_latency_ms=extraction_latency,
            verification_latency_ms=verification_latency,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        self._cache_result(key, input_hash, result)
        return result

    def _prepare_generation_passages(
        self,
        answer_result: AnswerGenerationResult,
        hits: Iterable[RetrievalHit | Mapping[str, Any]],
    ) -> tuple[AnswerPassage, ...]:
        expected = tuple(answer_result.evidence_chunk_ids)
        expected_set = set(expected)
        found: dict[str, AnswerPassage] = {}
        for hit in hits:
            row = _hit_mapping(hit)
            chunk_id = str(row.get("chunk_id", "")).strip()
            text = str(row.get("text", "")).strip()
            if chunk_id not in expected_set or chunk_id in found or not text:
                continue
            found[chunk_id] = AnswerPassage(
                chunk_id=chunk_id,
                doc_id=str(row.get("doc_id", "")).strip(),
                section=str(row.get("section", "")).strip(),
                page=row.get("page_start"),
                text=text[: self.settings.max_chars_per_chunk],
            )
        return tuple(found[chunk_id] for chunk_id in expected if chunk_id in found)

    @staticmethod
    def _valid_answer_citations(
        citations: tuple[AnswerCitation, ...],
        passage_by_id: dict[str, AnswerPassage],
    ) -> bool:
        if not citations or len(citations) != len({citation.chunk_id for citation in citations}):
            return False
        for citation in citations:
            passage = passage_by_id.get(citation.chunk_id)
            if passage is None:
                return False
            if (
                citation.doc_id != passage.doc_id
                or citation.section != passage.section
                or citation.page != passage.page
            ):
                return False
        return True

    def _validate_extraction(
        self,
        extraction: BackendClaimExtraction,
        answer: str,
    ) -> tuple[ExtractedClaim, ...] | None:
        claims = tuple(
            ExtractedClaim(claim.claim_id.strip(), " ".join(claim.text.split()).strip())
            for claim in extraction.claims
        )
        if not claims or len(claims) > self.settings.max_claims:
            return None
        expected_ids = tuple(f"c{index}" for index in range(1, len(claims) + 1))
        if tuple(claim.claim_id for claim in claims) != expected_ids:
            return None
        normalized_texts = [claim.text.casefold() for claim in claims]
        if any(not text for text in normalized_texts) or len(normalized_texts) != len(set(normalized_texts)):
            return None
        combined_claims = " ".join(claim.text for claim in claims)
        if missing_entities(combined_claims, extract_preserved_entities(answer)):
            return None
        return claims

    def _candidate_passages(
        self,
        claim: ExtractedClaim,
        passages: tuple[AnswerPassage, ...],
        citation_ids: tuple[str, ...],
    ) -> tuple[AnswerPassage, ...]:
        passage_by_id = {passage.chunk_id: passage for passage in passages}
        ordered_ids = [chunk_id for chunk_id in citation_ids if chunk_id in passage_by_id]
        selected = set(ordered_ids)
        claim_tokens = _tokens(claim.text)
        remaining = [passage for passage in passages if passage.chunk_id not in selected]
        remaining.sort(
            key=lambda passage: (
                -len(claim_tokens & _tokens(passage.text)),
                passages.index(passage),
            )
        )
        ordered_ids.extend(passage.chunk_id for passage in remaining)
        ordered_ids = ordered_ids[: self.settings.max_evidence_chunks_per_claim]
        return tuple(passage_by_id[chunk_id] for chunk_id in ordered_ids)

    @staticmethod
    def _validate_verification(
        claim: ExtractedClaim,
        verification: BackendClaimVerification,
        candidates: tuple[AnswerPassage, ...],
    ) -> AuditedClaim | None:
        level = verification.support_level.strip().casefold()
        reason = " ".join(verification.reason.split()).strip()
        confidence = float(verification.confidence)
        supporting_ids = tuple(
            dict.fromkeys(item.strip() for item in verification.supporting_chunk_ids if item.strip())
        )
        candidate_by_id = {passage.chunk_id: passage for passage in candidates}
        if (
            verification.claim_id.strip() != claim.claim_id
            or level not in SUPPORT_LEVELS
            or not reason
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or any(chunk_id not in candidate_by_id for chunk_id in supporting_ids)
        ):
            return None
        if level in {"supported", "partial"} and not supporting_ids:
            return None
        if level == "unsupported" and supporting_ids:
            return None
        citations = tuple(
            AnswerCitation(
                chunk_id=candidate_by_id[chunk_id].chunk_id,
                doc_id=candidate_by_id[chunk_id].doc_id,
                section=candidate_by_id[chunk_id].section,
                page=candidate_by_id[chunk_id].page,
            )
            for chunk_id in supporting_ids
        )
        return AuditedClaim(
            claim_id=claim.claim_id,
            text=claim.text,
            support_level=level,
            confidence=confidence,
            citations=citations,
            reason=reason,
            candidate_chunk_ids=tuple(passage.chunk_id for passage in candidates),
        )

    def _cache_result(self, key: str, input_hash: str, result: CitationAuditResult) -> None:
        self.cache.put(
            key,
            input_hash=input_hash,
            output=result.to_dict(),
            timestamp=result.timestamp,
        )

    def _not_audited_result(
        self,
        answer: str,
        reason: str,
        *,
        started: float,
    ) -> CitationAuditResult:
        return CitationAuditResult(
            answer=answer,
            claims=(),
            overall_grounded=False,
            unsupported_claim_count=0,
            partial_claim_count=0,
            grounding_score=0.0,
            audit_completed=False,
            audit_reason=reason,
            evidence_chunk_ids=(),
            model=self.settings.model,
            extraction_prompt_version=self.settings.extraction_prompt_version,
            verification_prompt_version=self.settings.verification_prompt_version,
            config_version=self.settings.config_version,
            timestamp=utc_timestamp(),
            cache_hit=False,
            fallback_used=False,
            fallback_reason=None,
            api_call_count=0,
            extraction_api_calls=0,
            verification_api_calls=0,
            input_tokens=0,
            output_tokens=0,
            extraction_latency_ms=0.0,
            verification_latency_ms=0.0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _fallback_result(
        self,
        answer: str,
        reason: str,
        *,
        started: float,
        evidence_chunk_ids: tuple[str, ...] = (),
        api_call_count: int = 0,
        extraction_api_calls: int = 0,
        verification_api_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        extraction_latency_ms: float = 0.0,
        verification_latency_ms: float = 0.0,
    ) -> CitationAuditResult:
        return CitationAuditResult(
            answer=answer,
            claims=(),
            overall_grounded=False,
            unsupported_claim_count=0,
            partial_claim_count=0,
            grounding_score=0.0,
            audit_completed=False,
            audit_reason="audit_failed_closed",
            evidence_chunk_ids=evidence_chunk_ids,
            model=self.settings.model,
            extraction_prompt_version=self.settings.extraction_prompt_version,
            verification_prompt_version=self.settings.verification_prompt_version,
            config_version=self.settings.config_version,
            timestamp=utc_timestamp(),
            cache_hit=False,
            fallback_used=True,
            fallback_reason=reason,
            api_call_count=api_call_count,
            extraction_api_calls=extraction_api_calls,
            verification_api_calls=verification_api_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            extraction_latency_ms=extraction_latency_ms,
            verification_latency_ms=verification_latency_ms,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _result_from_cache(
        self,
        payload: dict[str, Any],
        answer_result: AnswerGenerationResult,
        passages: tuple[AnswerPassage, ...],
    ) -> CitationAuditResult | None:
        try:
            claims = tuple(
                AuditedClaim(
                    claim_id=str(item["id"]),
                    text=str(item["text"]),
                    support_level=str(item["support_level"]),
                    confidence=float(item["confidence"]),
                    citations=tuple(
                        AnswerCitation(
                            chunk_id=str(citation["chunk_id"]),
                            doc_id=str(citation["doc_id"]),
                            section=str(citation["section"]),
                            page=citation.get("page"),
                        )
                        for citation in item.get("citations", [])
                    ),
                    reason=str(item["reason"]),
                    candidate_chunk_ids=tuple(str(value) for value in item.get("candidate_chunk_ids", [])),
                )
                for item in payload.get("claims", [])
            )
            result = CitationAuditResult(
                answer=str(payload["answer"]),
                claims=claims,
                overall_grounded=bool(payload["overall_grounded"]),
                unsupported_claim_count=int(payload["unsupported_claim_count"]),
                partial_claim_count=int(payload.get("partial_claim_count", 0)),
                grounding_score=float(payload["grounding_score"]),
                audit_completed=bool(payload["audit_completed"]),
                audit_reason=payload.get("audit_reason"),
                evidence_chunk_ids=tuple(str(item) for item in payload.get("evidence_chunk_ids", [])),
                model=str(payload["model"]),
                extraction_prompt_version=str(payload["extraction_prompt_version"]),
                verification_prompt_version=str(payload["verification_prompt_version"]),
                config_version=str(payload["config_version"]),
                timestamp=str(payload["timestamp"]),
                cache_hit=True,
                fallback_used=bool(payload.get("fallback_used", False)),
                fallback_reason=payload.get("fallback_reason"),
                api_call_count=0,
                extraction_api_calls=0,
                verification_api_calls=0,
                input_tokens=0,
                output_tokens=0,
                extraction_latency_ms=0.0,
                verification_latency_ms=0.0,
                latency_ms=0.0,
            )
        except (KeyError, TypeError, ValueError):
            return None
        if (
            result.answer != answer_result.answer
            or result.model != self.settings.model
            or result.extraction_prompt_version != self.settings.extraction_prompt_version
            or result.verification_prompt_version != self.settings.verification_prompt_version
            or result.config_version != self.settings.config_version
            or result.evidence_chunk_ids != tuple(passage.chunk_id for passage in passages)
        ):
            return None
        if result.fallback_used:
            return result if not result.audit_completed and not result.overall_grounded and not result.claims else None
        return result if self._valid_completed_result(result, passages) else None

    @staticmethod
    def _valid_completed_result(
        result: CitationAuditResult,
        passages: tuple[AnswerPassage, ...],
    ) -> bool:
        if not result.audit_completed or not result.claims or not 0 <= result.grounding_score <= 1:
            return False
        passage_by_id = {passage.chunk_id: passage for passage in passages}
        if tuple(claim.claim_id for claim in result.claims) != tuple(
            f"c{index}" for index in range(1, len(result.claims) + 1)
        ):
            return False
        for claim in result.claims:
            if claim.support_level not in SUPPORT_LEVELS or not claim.text.strip() or not claim.reason.strip():
                return False
            if claim.support_level in {"supported", "partial"} and not claim.citations:
                return False
            if claim.support_level == "unsupported" and claim.citations:
                return False
            for citation in claim.citations:
                passage = passage_by_id.get(citation.chunk_id)
                if passage is None or citation.to_dict() != AnswerCitation(
                    passage.chunk_id, passage.doc_id, passage.section, passage.page
                ).to_dict():
                    return False
        supported = sum(claim.support_level == "supported" for claim in result.claims)
        unsupported = sum(claim.support_level == "unsupported" for claim in result.claims)
        partial = sum(claim.support_level == "partial" for claim in result.claims)
        return (
            result.overall_grounded == (supported == len(result.claims))
            and result.unsupported_claim_count == unsupported
            and result.partial_claim_count == partial
            and math.isclose(result.grounding_score, supported / len(result.claims))
        )
