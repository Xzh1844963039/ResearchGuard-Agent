# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_evidence_sufficiency_v1.py
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
from researchguard.retrieval import RetrievalEngine  # noqa: E402
from researchguard.retrieval.evidence_cache import EvidenceSufficiencyCache  # noqa: E402
from researchguard.retrieval.evidence_judge import (  # noqa: E402
    BackendEvidenceJudgment,
    EvidenceJudgeBackend,
    EvidenceJudgeBackendError,
    EvidencePassage,
    EvidenceSufficiencyResult,
    OpenAIEvidenceJudgeBackend,
    build_evidence_model_input,
    load_evidence_judge_settings,
)
from researchguard.retrieval.evidence_pipeline import EvidenceSufficiencyPipeline  # noqa: E402


DEFAULT_CONFIG = Path(r"C:\Users\18449\Desktop\researchguard_workspace\configs\evidence_sufficiency_v1.yaml")
HARD_CHECK_KEYS = (
    "json_parse_failure",
    "empty_result",
    "benchmark_leakage",
    "cache_inconsistency",
    "missing_chunk_reference",
    "retrieval_regression",
    "schema_failure",
)
FORBIDDEN_MODEL_INPUT_KEYS = {
    "answerable",
    "support_level",
    "expected_label",
    "query_id",
    "relevant_chunk_ids",
    "benchmark_answer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Evidence Sufficiency v1 on frozen ResearchGuard retrieval.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to evidence_sufficiency_v1.yaml.")
    return parser.parse_args()


class FixedBackend(EvidenceJudgeBackend):
    def __init__(self, result: BackendEvidenceJudgment):
        self.result = result
        self.calls = 0

    def judge(self, question: str, passages: tuple[EvidencePassage, ...]) -> BackendEvidenceJudgment:
        self.calls += 1
        return self.result


class FailingBackend(EvidenceJudgeBackend):
    def judge(self, question: str, passages: tuple[EvidencePassage, ...]) -> BackendEvidenceJudgment:
        raise EvidenceJudgeBackendError("synthetic API failure", api_call_count=1)


class InvalidJsonResponses:
    def create(self, **kwargs: Any) -> Any:
        return type("Response", (), {"output_text": "{not-json", "usage": None})()


class InvalidJsonClient:
    def __init__(self) -> None:
        self.responses = InvalidJsonResponses()


def backend_result(
    *,
    answerable: bool,
    support_level: str,
    supporting_chunk_ids: tuple[str, ...],
    reason: str = "Synthetic direct-support judgment.",
) -> BackendEvidenceJudgment:
    if support_level == "strong":
        supported_requirements = ("the requested method property",)
        missing_requirements: tuple[str, ...] = ()
    elif support_level == "partial":
        supported_requirements = ("one material requirement",)
        missing_requirements = ("another material requirement",)
    else:
        supported_requirements = ()
        missing_requirements = ("the requested claim",)
    return BackendEvidenceJudgment(
        answerable=answerable,
        support_level=support_level,
        confidence=0.9,
        reason=reason,
        supporting_chunk_ids=supporting_chunk_ids,
        api_call_count=1,
        input_tokens=10,
        output_tokens=10,
        supported_requirements=supported_requirements,
        missing_requirements=missing_requirements,
    )


def stable_judgment(result: EvidenceSufficiencyResult) -> dict[str, Any]:
    return {
        "answerable": result.answerable,
        "support_level": result.support_level,
        "confidence": result.confidence,
        "reason": result.reason,
        "supporting_chunk_ids": list(result.supporting_chunk_ids),
        "fallback_used": result.fallback_used,
        "fallback_reason": result.fallback_reason,
        "model": result.model,
        "prompt_version": result.prompt_version,
        "config_version": result.config_version,
    }


def synthetic_passages() -> tuple[EvidencePassage, ...]:
    return (
        EvidencePassage(
            chunk_id="paper_demo_chunk_00001",
            doc_id="paper_demo",
            section="method",
            page_start=2,
            page_end=2,
            text="The method uses retrieved passages to provide non-parametric memory.",
        ),
        EvidencePassage(
            chunk_id="paper_demo_chunk_00002",
            doc_id="paper_demo",
            section="results",
            page_start=3,
            page_end=3,
            text="The experiment reports improved factual accuracy.",
        ),
    )


def run_synthetic_tests(settings: Any) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    passages = synthetic_passages()
    hits = [
        {
            "chunk_id": passage.chunk_id,
            "doc_id": passage.doc_id,
            "section": passage.section,
            "page_start": passage.page_start,
            "page_end": passage.page_end,
            "text": passage.text,
        }
        for passage in passages
    ]
    test_settings = replace(settings, cache_enabled=False, max_retries=0)

    def record(name: str, passed: bool, details: dict[str, Any] | None = None) -> None:
        tests.append({"name": name, "passed": bool(passed), "details": details or {}})

    model_input = build_evidence_model_input("How does the method use retrieval?", passages)
    input_keys = set(model_input)
    nested_keys = set().union(*(set(item) for item in model_input["evidence_passages"]))
    serialized = json.dumps(model_input, ensure_ascii=False)
    record(
        "model_input_excludes_benchmark_labels",
        not (FORBIDDEN_MODEL_INPUT_KEYS & input_keys)
        and not (FORBIDDEN_MODEL_INPUT_KEYS & nested_keys)
        and "relevant_chunk_ids" not in serialized,
        {"input_keys": sorted(input_keys), "passage_keys": sorted(nested_keys)},
    )

    cases = [
        ("strong_schema", backend_result(answerable=True, support_level="strong", supporting_chunk_ids=(passages[0].chunk_id,))),
        ("partial_schema", backend_result(answerable=False, support_level="partial", supporting_chunk_ids=(passages[0].chunk_id,))),
        ("unsupported_schema", backend_result(answerable=False, support_level="unsupported", supporting_chunk_ids=())),
    ]
    for name, fixed in cases:
        result = EvidenceSufficiencyPipeline(test_settings, backend=FixedBackend(fixed)).assess(
            "How does the method use retrieval?",
            hits,
        )
        record(name, not result.fallback_used and result.support_level == fixed.support_level, result.to_dict())

    invalid_reference = EvidenceSufficiencyPipeline(
        test_settings,
        backend=FixedBackend(
            backend_result(
                answerable=True,
                support_level="strong",
                supporting_chunk_ids=("invented_chunk_99999",),
            )
        ),
    ).assess("How does the method use retrieval?", hits)
    record(
        "invented_chunk_id_falls_back",
        invalid_reference.fallback_used
        and invalid_reference.fallback_reason == "missing_chunk_reference"
        and not invalid_reference.answerable,
        invalid_reference.to_dict(),
    )

    omitted_entity = EvidenceSufficiencyPipeline(
        test_settings,
        backend=FixedBackend(
            backend_result(
                answerable=True,
                support_level="strong",
                supporting_chunk_ids=(passages[0].chunk_id,),
            )
        ),
    ).assess("Does CRAG use GPT-5?", hits)
    record(
        "omitted_entity_cannot_prove_negative",
        not omitted_entity.fallback_used
        and not omitted_entity.answerable
        and omitted_entity.support_level == "unsupported",
        omitted_entity.to_dict(),
    )

    inconsistent = EvidenceSufficiencyPipeline(
        test_settings,
        backend=FixedBackend(
            backend_result(
                answerable=True,
                support_level="partial",
                supporting_chunk_ids=(passages[0].chunk_id,),
            )
        ),
    ).assess("How does the method use retrieval?", hits)
    record(
        "coverage_canonicalizes_inconsistent_label",
        not inconsistent.fallback_used
        and not inconsistent.answerable
        and inconsistent.support_level == "partial",
        inconsistent.to_dict(),
    )

    failed = EvidenceSufficiencyPipeline(test_settings, backend=FailingBackend()).assess(
        "How does the method use retrieval?",
        hits,
    )
    record(
        "api_failure_is_conservative",
        failed.fallback_used and not failed.answerable and failed.support_level == "unsupported",
        failed.to_dict(),
    )

    invalid_json_backend = OpenAIEvidenceJudgeBackend(test_settings)
    invalid_json_backend._client = InvalidJsonClient()  # type: ignore[assignment]
    invalid_json = EvidenceSufficiencyPipeline(test_settings, backend=invalid_json_backend).assess(
        "How does the method use retrieval?",
        hits,
    )
    record(
        "invalid_json_is_conservative",
        invalid_json.fallback_used
        and invalid_json.fallback_reason == "backend_failure:JSONDecodeError"
        and not invalid_json.answerable,
        invalid_json.to_dict(),
    )

    with tempfile.TemporaryDirectory() as tmp:
        cache_settings = replace(settings, cache_enabled=True, cache_directory=Path(tmp), max_retries=0)
        fixed_backend = FixedBackend(
            backend_result(
                answerable=True,
                support_level="strong",
                supporting_chunk_ids=(passages[0].chunk_id,),
            )
        )
        pipeline = EvidenceSufficiencyPipeline(cache_settings, backend=fixed_backend)
        first = pipeline.assess("How does the method use retrieval?", hits, read_cache=False)
        second = pipeline.assess("How does the method use retrieval?", hits, read_cache=True)
        record(
            "deterministic_cache_roundtrip",
            fixed_backend.calls == 1 and second.cache_hit and stable_judgment(first) == stable_judgment(second),
            {"backend_calls": fixed_backend.calls, "second_cache_hit": second.cache_hit},
        )

        model_payload = build_evidence_model_input("How does the method use retrieval?", passages)
        input_hash = stable_json_hash(model_payload)
        cache = EvidenceSufficiencyCache(Path(tmp), enabled=True)
        base_key = cache.make_key(
            query="How does the method use retrieval?",
            chunk_ids=[passage.chunk_id for passage in passages],
            input_hash=input_hash,
            settings=cache_settings,
        )
        changed_query_key = cache.make_key(
            query="Different question",
            chunk_ids=[passage.chunk_id for passage in passages],
            input_hash=input_hash,
            settings=cache_settings,
        )
        changed_chunks_key = cache.make_key(
            query="How does the method use retrieval?",
            chunk_ids=[passages[0].chunk_id],
            input_hash=input_hash,
            settings=cache_settings,
        )
        changed_prompt_key = cache.make_key(
            query="How does the method use retrieval?",
            chunk_ids=[passage.chunk_id for passage in passages],
            input_hash=input_hash,
            settings=replace(cache_settings, prompt_version="changed"),
        )
        changed_config_key = cache.make_key(
            query="How does the method use retrieval?",
            chunk_ids=[passage.chunk_id for passage in passages],
            input_hash=input_hash,
            settings=replace(cache_settings, config_version="changed"),
        )
        record(
            "cache_key_covers_required_identity",
            len({base_key, changed_query_key, changed_chunks_key, changed_prompt_key, changed_config_key}) == 5,
        )
    return tests


def load_benchmark(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    cases: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_no, row in enumerate(rows, start=1):
        query = " ".join(str(row.get("query", "")).split()).strip()
        level = str(row.get("support_level", "")).strip().casefold()
        answerable = row.get("answerable")
        if set(row) != {"query", "answerable", "support_level"}:
            errors.append({"row": row_no, "type": "unexpected_fields", "fields": sorted(row)})
        if not query or query.casefold() in seen:
            errors.append({"row": row_no, "type": "empty_or_duplicate_query"})
        seen.add(query.casefold())
        if type(answerable) is not bool or level not in {"strong", "partial", "unsupported"}:
            errors.append({"row": row_no, "type": "invalid_label"})
        if (level == "strong") != (answerable is True):
            errors.append({"row": row_no, "type": "inconsistent_answerability"})
        cases.append(
            {
                "case_id": f"esv1_q{row_no:03d}",
                "query": query,
                "answerable": bool(answerable),
                "support_level": level,
            }
        )
    return cases, errors


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


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for row in rows if row["expected_answerable"] and row["predicted_answerable"])
    fp = sum(1 for row in rows if not row["expected_answerable"] and row["predicted_answerable"])
    tn = sum(1 for row in rows if not row["expected_answerable"] and not row["predicted_answerable"])
    fn = sum(1 for row in rows if row["expected_answerable"] and not row["predicted_answerable"])
    total = max(len(rows), 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    unsupported = [row for row in rows if row["expected_support_level"] == "unsupported"]
    all_unanswerable = [row for row in rows if not row["expected_answerable"]]
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "no_answer_false_positive_rate": (
            sum(1 for row in unsupported if row["predicted_answerable"]) / len(unsupported)
            if unsupported
            else 0.0
        ),
        "all_unanswerable_false_positive_rate": (
            sum(1 for row in all_unanswerable if row["predicted_answerable"]) / len(all_unanswerable)
            if all_unanswerable
            else 0.0
        ),
    }


def support_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    levels = ("strong", "partial", "unsupported")
    per_level: dict[str, Any] = {}
    for level in levels:
        expected = [row for row in rows if row["expected_support_level"] == level]
        predicted = [row for row in rows if row["predicted_support_level"] == level]
        correct = [row for row in expected if row["predicted_support_level"] == level]
        precision = len(correct) / len(predicted) if predicted else 0.0
        recall = len(correct) / len(expected) if expected else 0.0
        per_level[level] = {
            "count": len(expected),
            "precision": precision,
            "recall": recall,
        }
    return {
        "accuracy": (
            sum(1 for row in rows if row["expected_support_level"] == row["predicted_support_level"])
            / max(len(rows), 1)
        ),
        "per_level": per_level,
    }


def validate_result(
    result: EvidenceSufficiencyResult,
    available_ids: set[str],
    hard_checks: Counter[str],
) -> None:
    if not result.reason.strip():
        hard_checks["empty_result"] += 1
    if (
        result.support_level not in {"strong", "partial", "unsupported"}
        or not math.isfinite(result.confidence)
        or not 0 <= result.confidence <= 1
    ):
        hard_checks["schema_failure"] += 1
    if any(chunk_id not in available_ids for chunk_id in result.supporting_chunk_ids):
        hard_checks["missing_chunk_reference"] += 1
    if result.support_level == "strong" and (not result.answerable or not result.supporting_chunk_ids):
        hard_checks["schema_failure"] += 1
    if result.support_level == "partial" and (result.answerable or not result.supporting_chunk_ids):
        hard_checks["schema_failure"] += 1
    if result.support_level == "unsupported" and (result.answerable or result.supporting_chunk_ids):
        hard_checks["schema_failure"] += 1
    if result.fallback_used:
        reason = str(result.fallback_reason or "")
        if "JSONDecodeError" in reason:
            hard_checks["json_parse_failure"] += 1
        elif reason == "missing_chunk_reference":
            hard_checks["missing_chunk_reference"] += 1
        else:
            hard_checks["schema_failure"] += 1


def render_report(summary: dict[str, Any], examples: dict[str, Any]) -> str:
    answerability = summary["answerability_metrics"]
    support = summary["support_level_metrics"]
    latency = summary["latency_ms"]
    lines = [
        "# Evidence Sufficiency v1 Validation Report",
        "",
        f"Conclusion: `{summary['conclusion']}`",
        f"Benchmark: {summary['benchmark_count']} queries ({summary['class_distribution']})",
        "",
        "## Scope",
        "",
        "Evidence Sufficiency v1 judges whether the final retrieved passages are enough to answer. It does not generate an answer, extract claims, audit citations, alter retrieval ranking, or inspect benchmark labels during inference.",
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
            "## Answerability",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Accuracy | {answerability['accuracy']:.4f} |",
            f"| Precision | {answerability['precision']:.4f} |",
            f"| Recall | {answerability['recall']:.4f} |",
            f"| F1 | {answerability['f1']:.4f} |",
            f"| No-answer false positive rate | {answerability['no_answer_false_positive_rate']:.4f} |",
            f"| All-unanswerable false positive rate | {answerability['all_unanswerable_false_positive_rate']:.4f} |",
            "",
            "## Support Level",
            "",
            f"Support-level accuracy: `{support['accuracy']:.4f}`",
            "",
            "| Level | Count | Precision | Recall |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for level, values in support["per_level"].items():
        lines.append(f"| {level} | {values['count']} | {values['precision']:.4f} | {values['recall']:.4f} |")
    lines.extend(
        [
            "",
            "## Latency And Usage",
            "",
            f"- Judge latency: average `{latency['judge_average']:.2f} ms`, P95 `{latency['judge_p95']:.2f} ms`.",
            f"- End-to-end latency: average `{latency['total_average']:.2f} ms`, P95 `{latency['total_p95']:.2f} ms`.",
            f"- Judge API calls: `{summary['api']['calls']}`; input tokens: `{summary['api']['input_tokens']}`; output tokens: `{summary['api']['output_tokens']}`.",
            f"- Warm cache hit rate: `{summary['cache']['warm_hit_rate']:.4f}`.",
            "",
            "## Representative Cases",
            "",
        ]
    )
    for level in ("strong", "partial", "unsupported"):
        sample = examples.get(level)
        if not sample:
            continue
        lines.extend(
            [
                f"### {level}",
                "",
                f"- Query: {sample['query']}",
                f"- Expected / predicted: `{sample['expected_support_level']}` / `{sample['predicted_support_level']}`",
                f"- Answerable: `{sample['predicted_answerable']}`",
                f"- Reason: {sample['reason']}",
                f"- Supporting chunks: `{', '.join(sample['supporting_chunk_ids'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "The judge estimates support only from the supplied Top-k passages. It cannot prove that the whole corpus lacks an answer, and it does not establish factual truth, generate an answer, extract claims, or audit citations. A retrieval miss can therefore become an answerability false negative.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config, settings = load_evidence_judge_settings(args.config)
    validation = config.get("validation", {}) or {}
    benchmark_path = PROJECT_ROOT / validation.get(
        "benchmark_path", "data/eval/evidence_sufficiency_v1_queries.jsonl"
    )
    retrieval_config_path = PROJECT_ROOT / validation.get("retrieval_config_path", "configs/retrieval_v1.yaml")
    output_dir = PROJECT_ROOT / validation.get(
        "output_directory", "outputs/evidence_sufficiency_validation_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_k = int(validation.get("retrieval_candidate_k", 80))
    rerank_candidate_k = int(validation.get("rerank_candidate_k", 20))
    evidence_top_k = int(validation.get("evidence_top_k", 10))
    rewrite_enabled = bool(validation.get("rewrite_enabled", True))
    multi_query_enabled = bool(validation.get("multi_query_enabled", True))

    hard_checks: Counter[str] = Counter({key: 0 for key in HARD_CHECK_KEYS})
    synthetic_tests = run_synthetic_tests(settings)
    for test in synthetic_tests:
        if test["passed"]:
            continue
        if test["name"] == "model_input_excludes_benchmark_labels":
            hard_checks["benchmark_leakage"] += 1
        elif "cache" in test["name"]:
            hard_checks["cache_inconsistency"] += 1
        else:
            hard_checks["schema_failure"] += 1

    cases, benchmark_errors = load_benchmark(benchmark_path)
    hard_checks["schema_failure"] += len(benchmark_errors)
    engine = RetrievalEngine.from_config(retrieval_config_path)
    pipeline = EvidenceSufficiencyPipeline(settings)

    result_rows: list[dict[str, Any]] = []
    misclassified: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    retrieval_latencies: list[float] = []
    judge_latencies: list[float] = []
    total_latencies: list[float] = []
    api_calls = 0
    input_tokens = 0
    output_tokens = 0
    warm_hits = 0

    for case in cases:
        retrieval_started = time.perf_counter()
        response = engine.retrieve(
            case["query"],
            mode="hybrid",
            top_k=evidence_top_k,
            candidate_k=candidate_k,
            rerank=True,
            rerank_candidate_k=rerank_candidate_k,
            rewrite=rewrite_enabled,
            multi_query=multi_query_enabled,
        )
        retrieval_latency = (time.perf_counter() - retrieval_started) * 1000.0
        ids_before = [hit.chunk_id for hit in response.hits]
        result = pipeline.assess(case["query"], response.hits, read_cache=False)
        ids_after = [hit.chunk_id for hit in response.hits]
        if ids_before != ids_after:
            hard_checks["retrieval_regression"] += 1

        available_ids = set(ids_before)
        validate_result(result, available_ids, hard_checks)
        warm_result = pipeline.assess(case["query"], response.hits, read_cache=True)
        if not warm_result.cache_hit or stable_judgment(result) != stable_judgment(warm_result):
            hard_checks["cache_inconsistency"] += 1
        else:
            warm_hits += 1

        api_calls += result.api_call_count
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        retrieval_latencies.append(retrieval_latency)
        judge_latencies.append(result.latency_ms)
        total_latencies.append(retrieval_latency + result.latency_ms)

        row = {
            "case_id": case["case_id"],
            "query": case["query"],
            "expected_answerable": case["answerable"],
            "predicted_answerable": result.answerable,
            "expected_support_level": case["support_level"],
            "predicted_support_level": result.support_level,
            "confidence": result.confidence,
            "reason": result.reason,
            "supporting_chunk_ids": list(result.supporting_chunk_ids),
            "retrieved_chunk_ids": ids_before,
            "retrieval_latency_ms": retrieval_latency,
            "judge_latency_ms": result.latency_ms,
            "total_latency_ms": retrieval_latency + result.latency_ms,
            "cache_hit": result.cache_hit,
            "warm_cache_hit": warm_result.cache_hit,
            "fallback_used": result.fallback_used,
            "fallback_reason": result.fallback_reason,
        }
        result_rows.append(row)
        if result.fallback_used:
            fallback_rows.append(row)
        if case["answerable"] != result.answerable or case["support_level"] != result.support_level:
            misclassified.append(row)

    answerability = classification_metrics(result_rows)
    support = support_metrics(result_rows)
    class_distribution = Counter(case["support_level"] for case in cases)
    hard_failed = any(hard_checks[key] for key in HARD_CHECK_KEYS)
    quality_passed = (
        answerability["accuracy"] >= 0.80
        and answerability["f1"] >= 0.80
        and support["accuracy"] >= 0.70
        and support["per_level"]["partial"]["recall"] >= 0.60
        and answerability["no_answer_false_positive_rate"] < 1.0
    )
    if hard_failed:
        conclusion = "FAIL"
    elif quality_passed:
        conclusion = "PASS"
    else:
        conclusion = "PASS_WITH_MINOR_ISSUES"

    latency = {
        "retrieval_average": statistics.mean(retrieval_latencies) if retrieval_latencies else 0.0,
        "retrieval_p95": percentile(retrieval_latencies, 0.95),
        "judge_average": statistics.mean(judge_latencies) if judge_latencies else 0.0,
        "judge_p95": percentile(judge_latencies, 0.95),
        "total_average": statistics.mean(total_latencies) if total_latencies else 0.0,
        "total_p95": percentile(total_latencies, 0.95),
    }
    examples: dict[str, Any] = {}
    for level in ("strong", "partial", "unsupported"):
        examples[level] = next(
            (
                row
                for row in result_rows
                if row["expected_support_level"] == level and row["predicted_support_level"] == level
            ),
            next((row for row in result_rows if row["expected_support_level"] == level), None),
        )

    summary = {
        "schema_version": "evidence_sufficiency_validation_v1",
        "conclusion": conclusion,
        "benchmark_path": str(benchmark_path),
        "benchmark_count": len(cases),
        "class_distribution": dict(sorted(class_distribution.items())),
        "model": settings.model,
        "temperature": settings.temperature,
        "prompt_version": settings.prompt_version,
        "config_version": settings.config_version,
        "retrieval_flow": {
            "mode": "hybrid",
            "rewrite_enabled": rewrite_enabled,
            "multi_query_enabled": multi_query_enabled,
            "rerank_enabled": True,
            "candidate_k": candidate_k,
            "rerank_candidate_k": rerank_candidate_k,
            "evidence_top_k": evidence_top_k,
        },
        "hard_checks": {key: int(hard_checks[key]) for key in HARD_CHECK_KEYS},
        "synthetic_tests": synthetic_tests,
        "benchmark_errors": benchmark_errors,
        "answerability_metrics": answerability,
        "support_level_metrics": support,
        "latency_ms": latency,
        "api": {
            "calls": api_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "cache": {
            "cold_hits": sum(1 for row in result_rows if row["cache_hit"]),
            "warm_hits": warm_hits,
            "warm_checks": len(result_rows),
            "warm_hit_rate": warm_hits / max(len(result_rows), 1),
        },
        "fallback_count": len(fallback_rows),
        "misclassified_count": len(misclassified),
        "quality_checks": {
            "accuracy_at_least_0_80": answerability["accuracy"] >= 0.80,
            "f1_at_least_0_80": answerability["f1"] >= 0.80,
            "support_accuracy_at_least_0_70": support["accuracy"] >= 0.70,
            "partial_recall_at_least_0_60": support["per_level"]["partial"]["recall"] >= 0.60,
            "no_answer_fpr_lower_than_retrieval_only_1_0": answerability["no_answer_false_positive_rate"] < 1.0,
        },
        "scope_boundary": {
            "answer_generation": False,
            "claim_extraction": False,
            "citation_audit": False,
            "retrieval_ranking_modified": False,
        },
    }

    write_json(output_dir / "evidence_sufficiency_validation_summary.json", summary)
    (output_dir / "evidence_sufficiency_validation_report.md").write_text(
        render_report(summary, examples),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "evidence_sufficiency_results.jsonl", result_rows)
    write_jsonl(output_dir / "misclassified_cases.jsonl", misclassified)
    write_jsonl(output_dir / "fallback_cases.jsonl", fallback_rows)
    write_json(output_dir / "synthetic_tests.json", synthetic_tests)
    write_json(output_dir / "latency_report.json", latency)
    write_json(
        output_dir / "api_cache_report.json",
        {"api": summary["api"], "cache": summary["cache"]},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if conclusion != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
