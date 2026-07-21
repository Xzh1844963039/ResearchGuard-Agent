# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_citation_audit_v1.py
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(r"C:\Users\18449\Desktop\researchguard_workspace")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.indexing.corpus_loader import read_jsonl, stable_json_hash, write_json, write_jsonl  # noqa: E402
from researchguard.retrieval import (  # noqa: E402
    AnswerGenerationPipeline,
    CitationAuditPipeline,
    EvidenceSufficiencyPipeline,
    RetrievalEngine,
)
from researchguard.retrieval.answer_generator import (  # noqa: E402
    AnswerCitation,
    AnswerGenerationResult,
    AnswerPassage,
    load_answer_generation_settings,
    utc_timestamp,
)
from researchguard.retrieval.citation_audit import CitationAuditResult, build_audit_input  # noqa: E402
from researchguard.retrieval.citation_cache import CitationAuditCache  # noqa: E402
from researchguard.retrieval.claim_extractor import (  # noqa: E402
    BackendClaimExtraction,
    ClaimExtractorBackend,
    CitationAuditSettings,
    ExtractedClaim,
    build_claim_extraction_input,
    load_citation_audit_settings,
)
from researchguard.retrieval.claim_verifier import (  # noqa: E402
    BackendClaimVerification,
    ClaimVerifierBackend,
    build_claim_verification_input,
)
from researchguard.retrieval.evidence_judge import load_evidence_judge_settings  # noqa: E402


DEFAULT_CONFIG = Path(r"C:\Users\18449\Desktop\researchguard_workspace\configs\citation_audit_v1.yaml")
HARD_CHECK_KEYS = (
    "claim_parse_failure",
    "verification_schema_failure",
    "citation_missing",
    "unsupported_claim_leakage",
    "benchmark_leakage",
    "cache_inconsistency",
    "retrieval_regression",
    "answer_generation_regression",
)
FORBIDDEN_INPUT_KEYS = {
    "category",
    "expected_claims",
    "expected_support_level",
    "supporting_chunk_ids",
    "relevant_chunk_ids",
    "benchmark_answer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Claim Verification / Citation Audit v1.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to citation_audit_v1.yaml.")
    return parser.parse_args()


class FixedExtractor(ClaimExtractorBackend):
    def __init__(self, claims: tuple[ExtractedClaim, ...]):
        self.claims = claims
        self.calls = 0
        self.answers: list[str] = []

    def extract(self, answer: str) -> BackendClaimExtraction:
        self.calls += 1
        self.answers.append(answer)
        return BackendClaimExtraction(self.claims, 1, 20, 10)


class FixedVerifier(ClaimVerifierBackend):
    def __init__(self, results: dict[str, BackendClaimVerification]):
        self.results = results
        self.calls = 0
        self.candidate_ids: dict[str, list[str]] = {}

    def verify(
        self,
        claim: ExtractedClaim,
        passages: tuple[AnswerPassage, ...],
    ) -> BackendClaimVerification:
        self.calls += 1
        self.candidate_ids[claim.claim_id] = [passage.chunk_id for passage in passages]
        return self.results[claim.claim_id]


def verification(
    claim_id: str,
    level: str,
    supporting_ids: tuple[str, ...],
    *,
    confidence: float = 0.9,
) -> BackendClaimVerification:
    return BackendClaimVerification(
        claim_id=claim_id,
        support_level=level,
        confidence=confidence,
        supporting_chunk_ids=supporting_ids,
        reason=f"Synthetic {level} judgment.",
        api_call_count=1,
        input_tokens=30,
        output_tokens=12,
    )


def synthetic_hits() -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": "paper_demo_chunk_00001",
            "doc_id": "paper_demo",
            "section": "method",
            "page_start": 2,
            "page_end": 2,
            "text": "CRAG uses a lightweight evaluator to assess retrieval quality.",
        },
        {
            "chunk_id": "paper_demo_chunk_00002",
            "doc_id": "paper_demo",
            "section": "results",
            "page_start": 3,
            "page_end": 3,
            "text": "The evaluator improves document selection in the experiment.",
        },
    ]


def synthetic_answer(*, refused: bool = False) -> AnswerGenerationResult:
    answer = (
        "Insufficient evidence in the current corpus."
        if refused
        else "CRAG uses a lightweight evaluator. It improves document selection."
    )
    return AnswerGenerationResult(
        answer=answer,
        citations=()
        if refused
        else (AnswerCitation("paper_demo_chunk_00002", "paper_demo", "results", 3),),
        confidence=0.0 if refused else 0.9,
        refused=refused,
        refusal_reason="evidence_not_answerable" if refused else None,
        evidence_chunk_ids=()
        if refused
        else ("paper_demo_chunk_00001", "paper_demo_chunk_00002"),
        model="synthetic-answer",
        prompt_version="synthetic",
        config_version="synthetic",
        timestamp=utc_timestamp(),
        cache_hit=False,
        fallback_used=False,
        fallback_reason=None,
        api_call_count=0,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
    )


def stable_audit(result: CitationAuditResult) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "claims": [claim.to_dict() for claim in result.claims],
        "overall_grounded": result.overall_grounded,
        "unsupported_claim_count": result.unsupported_claim_count,
        "partial_claim_count": result.partial_claim_count,
        "grounding_score": result.grounding_score,
        "audit_completed": result.audit_completed,
        "audit_reason": result.audit_reason,
        "evidence_chunk_ids": list(result.evidence_chunk_ids),
        "fallback_used": result.fallback_used,
        "fallback_reason": result.fallback_reason,
        "model": result.model,
        "extraction_prompt_version": result.extraction_prompt_version,
        "verification_prompt_version": result.verification_prompt_version,
        "config_version": result.config_version,
    }


def run_synthetic_tests(settings: CitationAuditSettings) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    hits = synthetic_hits()
    answer = synthetic_answer()
    claims = (
        ExtractedClaim("c1", "CRAG uses a lightweight evaluator."),
        ExtractedClaim("c2", "It improves document selection."),
    )

    def record(name: str, passed: bool, details: dict[str, Any] | None = None) -> None:
        tests.append({"name": name, "passed": bool(passed), "details": details or {}})

    extraction_input = build_claim_extraction_input(answer.answer)
    verification_input = build_claim_verification_input(
        claims[0],
        (AnswerPassage("paper_demo_chunk_00001", "paper_demo", "method", 2, hits[0]["text"]),),
    )
    serialized_inputs = json.dumps([extraction_input, verification_input], ensure_ascii=False)
    record(
        "model_inputs_exclude_benchmark_labels",
        not any(key in serialized_inputs for key in FORBIDDEN_INPUT_KEYS),
    )

    no_call_extractor = FixedExtractor(claims)
    no_call_verifier = FixedVerifier({
        "c1": verification("c1", "supported", ("paper_demo_chunk_00001",)),
        "c2": verification("c2", "supported", ("paper_demo_chunk_00002",)),
    })
    refused = CitationAuditPipeline(
        replace(settings, cache_enabled=False),
        extractor=no_call_extractor,
        verifier=no_call_verifier,
    ).audit(synthetic_answer(refused=True), hits)
    record(
        "refused_answer_skips_audit_model_calls",
        no_call_extractor.calls == 0
        and no_call_verifier.calls == 0
        and not refused.audit_completed
        and refused.audit_reason == "answer_refused",
    )

    verifier = FixedVerifier({
        "c1": verification("c1", "supported", ("paper_demo_chunk_00001",)),
        "c2": verification("c2", "unsupported", ()),
    })
    result = CitationAuditPipeline(
        replace(settings, cache_enabled=False),
        extractor=FixedExtractor(claims),
        verifier=verifier,
    ).audit(answer, hits)
    record(
        "atomic_claims_keep_order_and_independent_verdicts",
        [claim.claim_id for claim in result.claims] == ["c1", "c2"]
        and [claim.support_level for claim in result.claims] == ["supported", "unsupported"],
    )
    record(
        "answer_citations_are_prioritized_before_generation_evidence",
        verifier.candidate_ids["c1"] == ["paper_demo_chunk_00002", "paper_demo_chunk_00001"],
        {"candidate_ids": verifier.candidate_ids["c1"]},
    )
    record(
        "unsupported_claim_has_no_citation_and_blocks_grounded_answer",
        not result.overall_grounded
        and result.unsupported_claim_count == 1
        and not result.claims[1].citations,
    )
    record(
        "claim_citations_are_canonicalized",
        result.claims[0].citations[0].to_dict()
        == {
            "chunk_id": "paper_demo_chunk_00001",
            "doc_id": "paper_demo",
            "section": "method",
            "page": 2,
        },
    )

    partial = CitationAuditPipeline(
        replace(settings, cache_enabled=False),
        extractor=FixedExtractor((claims[0],)),
        verifier=FixedVerifier({
            "c1": verification("c1", "partial", ("paper_demo_chunk_00001",)),
        }),
    ).audit(answer, hits)
    record(
        "partial_claim_keeps_supporting_citation_and_blocks_grounded_answer",
        partial.claims[0].support_level == "partial"
        and bool(partial.claims[0].citations)
        and partial.partial_claim_count == 1
        and not partial.overall_grounded,
    )

    invalid_extraction = CitationAuditPipeline(
        replace(settings, cache_enabled=False),
        extractor=FixedExtractor((ExtractedClaim("wrong", "CRAG uses a lightweight evaluator."),)),
        verifier=no_call_verifier,
    ).audit(answer, hits)
    record(
        "invalid_claim_schema_fails_closed",
        invalid_extraction.fallback_used
        and invalid_extraction.fallback_reason == "claim_extraction_schema_failure",
    )

    invalid_verification = CitationAuditPipeline(
        replace(settings, cache_enabled=False),
        extractor=FixedExtractor((claims[0],)),
        verifier=FixedVerifier({
            "c1": verification("wrong", "supported", ("paper_demo_chunk_00001",)),
        }),
    ).audit(answer, hits)
    record(
        "invalid_verification_schema_fails_closed",
        invalid_verification.fallback_used
        and invalid_verification.fallback_reason == "claim_verification_schema_failure",
    )

    outside_verification = CitationAuditPipeline(
        replace(settings, cache_enabled=False),
        extractor=FixedExtractor((claims[0],)),
        verifier=FixedVerifier({
            "c1": verification("c1", "supported", ("paper_demo_chunk_99999",)),
        }),
    ).audit(answer, hits)
    record(
        "citation_outside_candidate_evidence_fails_closed",
        outside_verification.fallback_used
        and outside_verification.fallback_reason == "claim_verification_schema_failure",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_settings = replace(settings, cache_enabled=True, cache_directory=Path(temp_dir))
        cache_extractor = FixedExtractor(claims)
        cache_verifier = FixedVerifier({
            "c1": verification("c1", "supported", ("paper_demo_chunk_00001",)),
            "c2": verification("c2", "supported", ("paper_demo_chunk_00002",)),
        })
        pipeline = CitationAuditPipeline(
            cache_settings,
            extractor=cache_extractor,
            verifier=cache_verifier,
        )
        first = pipeline.audit(answer, hits, read_cache=False)
        second = pipeline.audit(answer, hits, read_cache=True)
        record(
            "cache_round_trip_is_deterministic",
            cache_extractor.calls == 1
            and cache_verifier.calls == 2
            and second.cache_hit
            and stable_audit(first) == stable_audit(second),
        )
        passages = (
            AnswerPassage("paper_demo_chunk_00001", "paper_demo", "method", 2, hits[0]["text"]),
            AnswerPassage("paper_demo_chunk_00002", "paper_demo", "results", 3, hits[1]["text"]),
        )
        input_hash = stable_json_hash(build_audit_input(answer, passages))
        answer_hash = stable_json_hash({"answer": answer.answer})
        cache = CitationAuditCache(Path(temp_dir), enabled=True)
        keys = {
            cache.make_key(
                answer_hash=answer_hash,
                evidence_chunk_ids=list(answer.evidence_chunk_ids),
                input_hash=input_hash,
                settings=cache_settings,
            ),
            cache.make_key(
                answer_hash="changed",
                evidence_chunk_ids=list(answer.evidence_chunk_ids),
                input_hash=input_hash,
                settings=cache_settings,
            ),
            cache.make_key(
                answer_hash=answer_hash,
                evidence_chunk_ids=["different"],
                input_hash=input_hash,
                settings=cache_settings,
            ),
            cache.make_key(
                answer_hash=answer_hash,
                evidence_chunk_ids=list(answer.evidence_chunk_ids),
                input_hash=input_hash,
                settings=replace(cache_settings, extraction_prompt_version="changed"),
            ),
            cache.make_key(
                answer_hash=answer_hash,
                evidence_chunk_ids=list(answer.evidence_chunk_ids),
                input_hash=input_hash,
                settings=replace(cache_settings, verification_prompt_version="changed"),
            ),
            cache.make_key(
                answer_hash=answer_hash,
                evidence_chunk_ids=list(answer.evidence_chunk_ids),
                input_hash=input_hash,
                settings=replace(cache_settings, model="changed"),
            ),
        }
        record("cache_key_covers_required_identity", len(keys) == 6)
    return tests


def load_benchmark(path: Path, corpus: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    expected_fields = {
        "case_id",
        "category",
        "query",
        "answer",
        "citation_chunk_ids",
        "evidence_chunk_ids",
        "expected_claims",
    }
    errors: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    cases: list[dict[str, Any]] = []
    for row_no, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id", "")).strip()
        if set(row) != expected_fields:
            errors.append({"row": row_no, "type": "unexpected_fields", "fields": sorted(row)})
        if not case_id or case_id in seen_ids:
            errors.append({"row": row_no, "type": "empty_or_duplicate_case_id"})
        seen_ids.add(case_id)
        evidence_ids = [str(item) for item in row.get("evidence_chunk_ids", [])]
        citation_ids = [str(item) for item in row.get("citation_chunk_ids", [])]
        if not row.get("query") or not row.get("answer") or not evidence_ids or not citation_ids:
            errors.append({"row": row_no, "type": "empty_required_value"})
        if any(chunk_id not in corpus for chunk_id in evidence_ids):
            errors.append({"row": row_no, "type": "missing_corpus_evidence"})
        if any(chunk_id not in evidence_ids for chunk_id in citation_ids):
            errors.append({"row": row_no, "type": "citation_outside_evidence"})
        claims = row.get("expected_claims", [])
        if not claims:
            errors.append({"row": row_no, "type": "empty_expected_claims"})
        for claim in claims:
            if set(claim) != {"text", "support_level", "supporting_chunk_ids"}:
                errors.append({"row": row_no, "type": "invalid_expected_claim_schema"})
                continue
            if claim.get("support_level") not in {"supported", "partial", "unsupported"}:
                errors.append({"row": row_no, "type": "invalid_expected_support_level"})
            expected_ids = [str(item) for item in claim.get("supporting_chunk_ids", [])]
            if any(chunk_id not in evidence_ids for chunk_id in expected_ids):
                errors.append({"row": row_no, "type": "expected_citation_outside_evidence"})
        cases.append(dict(row))
    return cases, errors


def answer_result_from_case(case: dict[str, Any], corpus: dict[str, dict[str, Any]]) -> AnswerGenerationResult:
    citations = tuple(
        AnswerCitation(
            chunk_id=chunk_id,
            doc_id=str(corpus[chunk_id]["doc_id"]),
            section=str(corpus[chunk_id]["section"]),
            page=corpus[chunk_id].get("page_start"),
        )
        for chunk_id in case["citation_chunk_ids"]
    )
    return AnswerGenerationResult(
        answer=str(case["answer"]),
        citations=citations,
        confidence=1.0,
        refused=False,
        refusal_reason=None,
        evidence_chunk_ids=tuple(str(item) for item in case["evidence_chunk_ids"]),
        model="benchmark_fixture",
        prompt_version="benchmark_fixture",
        config_version="benchmark_fixture",
        timestamp=utc_timestamp(),
        cache_hit=False,
        fallback_used=False,
        fallback_reason=None,
        api_call_count=0,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
    )


def hits_from_case(case: dict[str, Any], corpus: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(corpus[str(chunk_id)]) for chunk_id in case["evidence_chunk_ids"]]


def claim_tokens(text: str) -> set[str]:
    cleaned = "".join(character.casefold() if character.isalnum() else " " for character in text)
    return {token for token in cleaned.split() if token}


def claim_similarity(left: str, right: str) -> float:
    left_tokens = claim_tokens(left)
    right_tokens = claim_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return 2 * len(left_tokens & right_tokens) / (len(left_tokens) + len(right_tokens))


def align_claims(
    predicted: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    pairs: list[tuple[float, int, int]] = []
    for predicted_index, predicted_claim in enumerate(predicted):
        for expected_index, expected_claim in enumerate(expected):
            pairs.append(
                (
                    claim_similarity(predicted_claim["text"], expected_claim["text"]),
                    predicted_index,
                    expected_index,
                )
            )
    matches: list[dict[str, Any]] = []
    used_predicted: set[int] = set()
    used_expected: set[int] = set()
    for score, predicted_index, expected_index in sorted(pairs, reverse=True):
        if score < 0.55 or predicted_index in used_predicted or expected_index in used_expected:
            continue
        used_predicted.add(predicted_index)
        used_expected.add(expected_index)
        matches.append(
            {
                "predicted_index": predicted_index,
                "expected_index": expected_index,
                "similarity": score,
            }
        )
    return (
        sorted(matches, key=lambda item: item["predicted_index"]),
        sorted(set(range(len(predicted))) - used_predicted),
        sorted(set(range(len(expected))) - used_expected),
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def run_integration_regression(
    validation: dict[str, Any],
    audit_settings: CitationAuditSettings,
    hard_checks: Counter[str],
) -> dict[str, Any]:
    retrieval_config = PROJECT_ROOT / validation.get("retrieval_config_path", "configs/retrieval_v1.yaml")
    evidence_config = PROJECT_ROOT / validation.get(
        "evidence_config_path", "configs/evidence_sufficiency_v1.yaml"
    )
    answer_config = PROJECT_ROOT / validation.get("answer_config_path", "configs/answer_generation_v1.yaml")
    query = str(validation.get("integration_query", "What is the difference between RAG-Sequence and RAG-Token?"))
    engine = RetrievalEngine.from_config(retrieval_config)
    response = engine.retrieve(
        query,
        mode="hybrid",
        top_k=int(validation.get("top_k", 10)),
        candidate_k=int(validation.get("retrieval_candidate_k", 80)),
        rerank=True,
        rerank_candidate_k=int(validation.get("rerank_candidate_k", 20)),
        rewrite=True,
        multi_query=True,
    )
    ids_before = [hit.chunk_id for hit in response.hits]
    _, evidence_settings = load_evidence_judge_settings(evidence_config)
    evidence_result = EvidenceSufficiencyPipeline(evidence_settings).assess(query, response.hits, read_cache=True)
    _, answer_settings = load_answer_generation_settings(answer_config)
    answer_result = AnswerGenerationPipeline(answer_settings).generate(
        query, response.hits, evidence_result, read_cache=True
    )
    answer_before = {
        "answer": answer_result.answer,
        "citations": [citation.to_dict() for citation in answer_result.citations],
        "confidence": answer_result.confidence,
        "refused": answer_result.refused,
        "evidence_chunk_ids": list(answer_result.evidence_chunk_ids),
    }
    audit_result = CitationAuditPipeline(audit_settings).audit(answer_result, response.hits, read_cache=False)
    ids_after = [hit.chunk_id for hit in response.hits]
    answer_after = {
        "answer": answer_result.answer,
        "citations": [citation.to_dict() for citation in answer_result.citations],
        "confidence": answer_result.confidence,
        "refused": answer_result.refused,
        "evidence_chunk_ids": list(answer_result.evidence_chunk_ids),
    }
    if ids_before != ids_after:
        hard_checks["retrieval_regression"] += 1
    if answer_before != answer_after or answer_result.refused or not answer_result.citations:
        hard_checks["answer_generation_regression"] += 1
    if not audit_result.audit_completed:
        if audit_result.fallback_reason and "extraction" in audit_result.fallback_reason:
            hard_checks["claim_parse_failure"] += 1
        else:
            hard_checks["verification_schema_failure"] += 1
    return {
        "query": query,
        "retrieved_chunk_ids": ids_before,
        "evidence_support_level": evidence_result.support_level,
        "answer_cache_hit": answer_result.cache_hit,
        "answer": answer_result.answer,
        "answer_citations": [citation.to_dict() for citation in answer_result.citations],
        "audit": audit_result.to_dict(),
    }


def render_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    metrics = summary["metrics"]
    latency = summary["latency_ms"]
    lines = [
        "# Claim Verification / Citation Audit v1 Validation Report",
        "",
        f"Conclusion: `{summary['conclusion']}`",
        f"Benchmark: `{summary['benchmark_count']}` answers and `{summary['expected_claim_count']}` labeled claims.",
        "",
        "## Scope",
        "",
        "The audit extracts atomic claims and verifies them against answer-time citations first, then other chunks that were visible to Answer Generation. It never calls the Retriever and does not alter the answer or ranking.",
        "",
        "## Hard Checks",
        "",
        "| Check | Count |",
        "| --- | ---: |",
    ]
    for key in HARD_CHECK_KEYS:
        lines.append(f"| `{key}` | {summary['hard_checks'][key]} |")
    lines.extend(
        [
            "",
            "## Claim Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Claim Support Accuracy | {metrics['claim_support_accuracy']:.4f} |",
            f"| Supported Claim Precision | {metrics['supported_claim_precision']:.4f} |",
            f"| Unsupported Claim Detection Recall | {metrics['unsupported_claim_detection_recall']:.4f} |",
            f"| Unsupported Claim Rate | {metrics['unsupported_claim_rate']:.4f} |",
            f"| Citation Precision | {metrics['citation_precision']:.4f} |",
            f"| Citation Recall | {metrics['citation_recall']:.4f} |",
            f"| Answer Grounded Rate | {metrics['answer_grounded_rate']:.4f} |",
            "",
            "## Latency And Cache",
            "",
            f"- Extraction average / P95: `{latency['extraction_average']:.2f}` / `{latency['extraction_p95']:.2f} ms`.",
            f"- Verification average / P95 per answer: `{latency['verification_average']:.2f}` / `{latency['verification_p95']:.2f} ms`.",
            f"- Audit average / P95: `{latency['audit_average']:.2f}` / `{latency['audit_p95']:.2f} ms`.",
            f"- API calls: `{summary['api']['calls']}`; warm cache hit rate: `{summary['cache']['warm_hit_rate']:.4f}`.",
            "",
            "## Representative Cases",
            "",
        ]
    )
    for category in ("fully_supported", "partially_supported_answer", "unsupported_claim", "partial_claim"):
        row = next((item for item in rows if item["category"] == category), None)
        if row is None:
            continue
        lines.extend(
            [
                f"### {category}",
                "",
                f"- Answer: {row['answer']}",
                f"- Predicted levels: `{', '.join(claim['support_level'] for claim in row['claims'])}`.",
                f"- Overall grounded: `{row['overall_grounded']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "This is an LLM-based claim and citation audit over the evidence visible at answer time. It is not an Agent workflow, does not prove source truth, does not search the corpus again, and is not a complete scientific reasoning system.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config, settings = load_citation_audit_settings(args.config)
    validation = config.get("validation", {}) or {}
    benchmark_path = PROJECT_ROOT / validation.get("benchmark_path", "data/eval/citation_audit_v1_queries.jsonl")
    output_dir = PROJECT_ROOT / validation.get(
        "output_directory", "outputs/citation_audit_validation_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_rows = read_jsonl(PROJECT_ROOT / "data/indexes/index_v1/corpus_manifest.jsonl")
    corpus = {str(row["chunk_id"]): row for row in corpus_rows}

    hard_checks: Counter[str] = Counter({key: 0 for key in HARD_CHECK_KEYS})
    synthetic_tests = run_synthetic_tests(settings)
    for test in synthetic_tests:
        if test["passed"]:
            continue
        if "benchmark" in test["name"]:
            hard_checks["benchmark_leakage"] += 1
        elif "cache" in test["name"]:
            hard_checks["cache_inconsistency"] += 1
        elif "claim" in test["name"] or "atomic" in test["name"]:
            hard_checks["claim_parse_failure"] += 1
        elif "citation" in test["name"]:
            hard_checks["citation_missing"] += 1
        else:
            hard_checks["verification_schema_failure"] += 1

    cases, benchmark_errors = load_benchmark(benchmark_path, corpus)
    hard_checks["claim_parse_failure"] += len(benchmark_errors)
    pipeline = CitationAuditPipeline(settings)
    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    extraction_latencies: list[float] = []
    verification_latencies: list[float] = []
    audit_latencies: list[float] = []
    api_calls = input_tokens = output_tokens = 0
    warm_hits = 0

    support_correct = support_total = 0
    predicted_supported = supported_true_positive = 0
    expected_unsupported = detected_unsupported = 0
    predicted_unsupported = predicted_claim_total = 0
    citation_true_positive = citation_predicted = citation_expected = 0

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['case_id']}: extract -> verify -> audit", flush=True)
        answer_result = answer_result_from_case(case, corpus)
        hits = hits_from_case(case, corpus)
        result = pipeline.audit(answer_result, hits, read_cache=False)
        if not result.audit_completed:
            failure_rows.append({"case_id": case["case_id"], "result": result.to_dict()})
            if result.fallback_reason and "claim_extraction" in result.fallback_reason:
                hard_checks["claim_parse_failure"] += 1
            else:
                hard_checks["verification_schema_failure"] += 1
        warm_result = pipeline.audit(answer_result, hits, read_cache=True)
        if not warm_result.cache_hit or stable_audit(result) != stable_audit(warm_result):
            hard_checks["cache_inconsistency"] += 1
        else:
            warm_hits += 1

        predicted_claims = [claim.to_dict() for claim in result.claims]
        expected_claims = list(case["expected_claims"])
        matches, unmatched_predicted, unmatched_expected = align_claims(predicted_claims, expected_claims)
        if unmatched_predicted or unmatched_expected:
            hard_checks["claim_parse_failure"] += len(unmatched_predicted) + len(unmatched_expected)

        match_rows: list[dict[str, Any]] = []
        for match in matches:
            predicted = predicted_claims[match["predicted_index"]]
            expected = expected_claims[match["expected_index"]]
            predicted_level = predicted["support_level"]
            expected_level = expected["support_level"]
            support_total += 1
            support_correct += int(predicted_level == expected_level)
            predicted_supported += int(predicted_level == "supported")
            supported_true_positive += int(
                predicted_level == "supported" and expected_level == "supported"
            )
            expected_unsupported += int(expected_level == "unsupported")
            detected_unsupported += int(
                expected_level == "unsupported" and predicted_level == "unsupported"
            )
            predicted_unsupported += int(predicted_level == "unsupported")
            predicted_claim_total += 1
            predicted_ids = {item["chunk_id"] for item in predicted["citations"]}
            expected_ids = {str(item) for item in expected["supporting_chunk_ids"]}
            citation_true_positive += len(predicted_ids & expected_ids)
            citation_predicted += len(predicted_ids)
            citation_expected += len(expected_ids)
            if predicted_level in {"supported", "partial"} and not predicted_ids:
                hard_checks["citation_missing"] += 1
            if predicted_level == "unsupported" and predicted_ids:
                hard_checks["unsupported_claim_leakage"] += 1
            match_rows.append(
                {
                    "predicted": predicted,
                    "expected": expected,
                    "similarity": match["similarity"],
                    "support_correct": predicted_level == expected_level,
                }
            )
        if result.overall_grounded and any(claim["support_level"] != "supported" for claim in predicted_claims):
            hard_checks["unsupported_claim_leakage"] += 1

        extraction_latencies.append(result.extraction_latency_ms)
        verification_latencies.append(result.verification_latency_ms)
        audit_latencies.append(result.latency_ms)
        api_calls += result.api_call_count
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        result_rows.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "query": case["query"],
                "answer": case["answer"],
                "claims": predicted_claims,
                "overall_grounded": result.overall_grounded,
                "unsupported_claim_count": result.unsupported_claim_count,
                "partial_claim_count": result.partial_claim_count,
                "grounding_score": result.grounding_score,
                "matches": match_rows,
                "unmatched_predicted_claim_indexes": unmatched_predicted,
                "unmatched_expected_claim_indexes": unmatched_expected,
                "cache_hit": result.cache_hit,
                "warm_cache_hit": warm_result.cache_hit,
                "fallback_used": result.fallback_used,
                "fallback_reason": result.fallback_reason,
                "latency_ms": result.latency_ms,
            }
        )

    integration = run_integration_regression(validation, settings, hard_checks)
    metrics = {
        "claim_support_accuracy": support_correct / max(support_total, 1),
        "supported_claim_precision": supported_true_positive / max(predicted_supported, 1),
        "unsupported_claim_detection_recall": detected_unsupported / max(expected_unsupported, 1),
        "unsupported_claim_rate": predicted_unsupported / max(predicted_claim_total, 1),
        "citation_precision": citation_true_positive / max(citation_predicted, 1),
        "citation_recall": citation_true_positive / max(citation_expected, 1),
        "answer_grounded_rate": sum(row["overall_grounded"] for row in result_rows) / max(len(result_rows), 1),
    }
    hard_failed = any(hard_checks[key] for key in HARD_CHECK_KEYS)
    quality_passed = (
        metrics["claim_support_accuracy"] >= 0.85
        and metrics["supported_claim_precision"] >= 0.85
        and metrics["unsupported_claim_detection_recall"] >= 0.80
        and metrics["citation_precision"] >= 0.80
        and metrics["citation_recall"] >= 0.75
    )
    conclusion = "PASS" if not hard_failed and quality_passed else "FAIL"
    latency = {
        "extraction_average": statistics.mean(extraction_latencies) if extraction_latencies else 0.0,
        "extraction_p95": percentile(extraction_latencies, 0.95),
        "verification_average": statistics.mean(verification_latencies) if verification_latencies else 0.0,
        "verification_p95": percentile(verification_latencies, 0.95),
        "audit_average": statistics.mean(audit_latencies) if audit_latencies else 0.0,
        "audit_p95": percentile(audit_latencies, 0.95),
    }
    expected_claim_count = sum(len(case["expected_claims"]) for case in cases)
    summary = {
        "schema_version": "citation_audit_validation_v1",
        "conclusion": conclusion,
        "benchmark_path": str(benchmark_path),
        "benchmark_count": len(cases),
        "expected_claim_count": expected_claim_count,
        "predicted_claim_count": sum(len(row["claims"]) for row in result_rows),
        "category_distribution": dict(sorted(Counter(case["category"] for case in cases).items())),
        "model": settings.model,
        "temperature": settings.temperature,
        "extraction_prompt_version": settings.extraction_prompt_version,
        "verification_prompt_version": settings.verification_prompt_version,
        "config_version": settings.config_version,
        "hard_checks": {key: int(hard_checks[key]) for key in HARD_CHECK_KEYS},
        "metrics": metrics,
        "latency_ms": latency,
        "api": {
            "calls": api_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "cache": {
            "warm_hits": warm_hits,
            "warm_checks": len(cases),
            "warm_hit_rate": warm_hits / max(len(cases), 1),
        },
        "synthetic_tests": synthetic_tests,
        "benchmark_errors": benchmark_errors,
        "failure_count": len(failure_rows),
        "integration_regression": integration,
        "scope_boundary": {
            "retriever_called_by_audit": False,
            "answer_modified": False,
            "agent_workflow": False,
            "langgraph": False,
            "multi_agent": False,
        },
    }
    write_json(output_dir / "citation_audit_validation_summary.json", summary)
    write_jsonl(output_dir / "citation_audit_results.jsonl", result_rows)
    write_jsonl(output_dir / "failure_cases.jsonl", failure_rows)
    write_json(output_dir / "synthetic_tests.json", synthetic_tests)
    write_json(output_dir / "integration_regression.json", integration)
    write_json(output_dir / "latency_report.json", latency)
    write_json(output_dir / "api_cache_report.json", {"api": summary["api"], "cache": summary["cache"]})
    (output_dir / "citation_audit_validation_report.md").write_text(
        render_report(summary, result_rows), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if conclusion == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
