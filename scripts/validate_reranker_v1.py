# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_reranker_v1.py
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(r"C:\Users\18449\Desktop\researchguard_workspace")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.indexing.corpus_loader import load_yaml, read_jsonl, write_json, write_jsonl  # noqa: E402
from researchguard.retrieval import MetadataFilter, RetrievalEngine  # noqa: E402
from researchguard.retrieval.rerank_cache import RerankCache  # noqa: E402
from researchguard.retrieval.reranker import load_reranker_settings, render_rerank_document  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reranker_v1.yaml"
HARD_CHECK_KEYS = (
    "reranker_load_failure",
    "candidate_count_mismatch",
    "missing_chunk_id",
    "result_schema_failure",
    "non_deterministic_failure",
    "cache_consistency_failure",
    "benchmark_label_leakage",
    "retrieval_baseline_changed_unexpectedly",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Hybrid RRF + Cross-Encoder Reranker v1.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def dcg(binary_hits: list[int]) -> float:
    return sum(relevance / math.log2(rank + 1) for rank, relevance in enumerate(binary_hits, start=1))


def evaluate(results: dict[str, list[dict[str, Any]]], cases: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    answerable = [case for case in cases if case["relevant_chunk_ids"]]
    no_answer = [case for case in cases if not case["relevant_chunk_ids"]]
    metrics: dict[str, Any] = {}
    for k in ks:
        recall = 0
        document_hits = 0
        section_hits = 0
        for case in answerable:
            hits = results.get(case["query_id"], [])[:k]
            hit_ids = {str(hit["chunk_id"]) for hit in hits}
            recall += int(bool(hit_ids & set(case["relevant_chunk_ids"])))
            document_hits += int(bool({str(hit["doc_id"]) for hit in hits} & set(case["expected_doc_ids"])))
            section_hits += int(bool({str(hit["section"]) for hit in hits} & set(case["expected_sections"])))
        denominator = max(len(answerable), 1)
        metrics[f"recall@{k}"] = recall / denominator
        metrics[f"document_hit@{k}"] = document_hits / denominator
        metrics[f"section_hit@{k}"] = section_hits / denominator

    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    multi_any: list[float] = []
    multi_all: list[float] = []
    multi_coverage: list[float] = []
    for case in answerable:
        relevant = set(case["relevant_chunk_ids"])
        hits = results.get(case["query_id"], [])[:10]
        binary = [int(str(hit["chunk_id"]) in relevant) for hit in hits]
        first_rank = next((rank for rank, value in enumerate(binary, start=1) if value), None)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        ideal = dcg([1] * min(len(relevant), 10))
        ndcgs.append(dcg(binary) / ideal if ideal else 0.0)
        if len(relevant) > 1:
            found = {str(hit["chunk_id"]) for hit in hits} & relevant
            multi_any.append(float(bool(found)))
            multi_all.append(float(found == relevant))
            multi_coverage.append(len(found) / len(relevant))
    metrics["mrr@10"] = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    metrics["ndcg@10"] = statistics.mean(ndcgs) if ndcgs else 0.0
    metrics["multi_evidence_any_hit@10"] = statistics.mean(multi_any) if multi_any else 0.0
    metrics["multi_evidence_all_hit@10"] = statistics.mean(multi_all) if multi_all else 0.0
    metrics["multi_evidence_coverage@10"] = statistics.mean(multi_coverage) if multi_coverage else 0.0
    metrics["no_answer_false_positive_rate"] = (
        sum(int(bool(results.get(case["query_id"], []))) for case in no_answer) / len(no_answer)
        if no_answer
        else 0.0
    )
    return metrics


def first_relevant_rank(hits: list[dict[str, Any]], relevant_ids: list[str]) -> int | None:
    relevant = set(relevant_ids)
    return next((rank for rank, hit in enumerate(hits, start=1) if str(hit["chunk_id"]) in relevant), None)


def load_cases(path: Path, valid_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_number, row in enumerate(read_jsonl(path), start=1):
        query_id = str(row.get("query_id", "")).strip()
        query = str(row.get("query", "")).strip()
        relevant = [str(item) for item in row.get("relevant_chunk_ids", [])]
        if not query_id or query_id in seen or not query:
            errors.append({"type": "benchmark_schema", "row": row_number, "query_id": query_id})
        seen.add(query_id)
        invalid = [chunk_id for chunk_id in relevant if chunk_id not in valid_ids]
        if invalid:
            errors.append({"type": "invalid_relevant_chunk_id", "query_id": query_id, "ids": invalid})
        cases.append(
            {
                "query_id": query_id,
                "query": query,
                "query_type": str(row.get("query_type", "unknown")),
                "relevant_chunk_ids": relevant,
                "expected_doc_ids": [str(item) for item in row.get("expected_doc_ids", [])],
                "expected_sections": [str(item) for item in row.get("expected_sections", [])],
                "filters": MetadataFilter.from_mapping(row.get("filters", {})),
                "notes": str(row.get("notes", "")),
            }
        )
    return cases, errors


def load_previous_hybrid_results(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    previous: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        if row.get("mode") == "hybrid":
            previous[str(row.get("query_id"))] = [str(hit.get("chunk_id")) for hit in row.get("hits", [])[:10]]
    return previous


def public_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in hit.items() if key != "text"}


def render_report(summary: dict[str, Any]) -> str:
    baseline = summary["metrics"]["baseline"]
    reranked = summary["metrics"]["reranked"]
    lines = [
        "# Reranker v1 Validation",
        "",
        f"Conclusion: **{summary['conclusion']}**",
        "",
        "## Configuration",
        "",
        f"- backend: `{summary['reranker_backend']}`",
        f"- model: `{summary['reranker_model']}`",
        f"- device: `{summary['device']}`",
        f"- candidate_k: `{summary['rerank_candidate_k']}`",
        f"- final_top_k: `{summary['final_top_k']}`",
        "",
        "## Hard Checks",
        "",
    ]
    lines.extend(f"- {key}: `{summary['hard_checks'][key]}`" for key in HARD_CHECK_KEYS)
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Baseline | Reranked | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10", "document_hit@10", "section_hit@10", "multi_evidence_any_hit@10", "multi_evidence_all_hit@10"):
        lines.append(f"| {key} | {baseline[key]:.6f} | {reranked[key]:.6f} | {reranked[key] - baseline[key]:+.6f} |")
    lines.extend(
        [
            "",
            "## Latency",
            "",
            f"- baseline average / p95 ms: `{summary['latency']['baseline']['average_total_ms']:.4f}` / `{summary['latency']['baseline']['p95_total_ms']:.4f}`",
            f"- reranked cold average / p95 ms: `{summary['latency']['reranked_cold']['average_total_ms']:.4f}` / `{summary['latency']['reranked_cold']['p95_total_ms']:.4f}`",
            f"- reranked warm-cache average / p95 ms: `{summary['latency']['reranked_warm']['average_total_ms']:.4f}` / `{summary['latency']['reranked_warm']['p95_total_ms']:.4f}`",
            "",
            "Reranker does not perform answerability detection. No-answer queries remain retrieval candidates and are analyzed only by their maximum rerank score.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config, settings = load_reranker_settings(config_path)
    validation = config.get("validation", {}) or {}
    benchmark_path = Path(validation.get("benchmark_path", "data/eval/retrieval_v1_queries.jsonl"))
    retrieval_config_path = Path(validation.get("retrieval_config_path", "configs/retrieval_v1.yaml"))
    output_dir = Path(validation.get("output_directory", "outputs/reranker_validation_v1"))
    if not benchmark_path.is_absolute():
        benchmark_path = PROJECT_ROOT / benchmark_path
    if not retrieval_config_path.is_absolute():
        retrieval_config_path = PROJECT_ROOT / retrieval_config_path
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_candidate_k = int(validation.get("retrieval_candidate_k", 80))
    top_k_values = [int(item) for item in validation.get("top_k_values", [1, 3, 5, 10])]
    hard_checks: Counter[str] = Counter({key: 0 for key in HARD_CHECK_KEYS})
    failure_cases: list[dict[str, Any]] = []

    engine = RetrievalEngine.from_config(retrieval_config_path)
    cases, benchmark_errors = load_cases(benchmark_path, set(engine.bundle.document_by_id))
    if benchmark_errors:
        hard_checks["missing_chunk_id"] += len(benchmark_errors)
        failure_cases.extend(benchmark_errors)
    previous_hybrid = load_previous_hybrid_results(
        PROJECT_ROOT / "outputs" / "retrieval_validation_v1" / "retrieval_results.jsonl"
    )

    baseline_results: dict[str, list[dict[str, Any]]] = {}
    reranked_results: dict[str, list[dict[str, Any]]] = {}
    baseline_rows: list[dict[str, Any]] = []
    reranked_rows: list[dict[str, Any]] = []
    baseline_latencies: list[float] = []
    rerank_retrieval_latencies: list[float] = []
    rerank_inference_latencies: list[float] = []
    rerank_total_latencies: list[float] = []
    first_cache_hits = 0
    first_cache_misses = 0
    expected_cache_items = 0
    historical_baseline_drift: list[dict[str, Any]] = []

    for case in cases:
        query_id = case["query_id"]
        baseline_response = engine.retrieve(
            case["query"],
            mode="hybrid",
            top_k=settings.candidate_k,
            candidate_k=retrieval_candidate_k,
            filters=case["filters"],
            rerank=False,
        )
        baseline_hits = baseline_response.to_dict(include_text=False)["hits"]
        baseline_repeat = engine.retrieve(
            case["query"],
            mode="hybrid",
            top_k=settings.candidate_k,
            candidate_k=retrieval_candidate_k,
            filters=case["filters"],
            rerank=False,
        ).to_dict(include_text=False)["hits"]
        if [hit["chunk_id"] for hit in baseline_repeat] != [hit["chunk_id"] for hit in baseline_hits]:
            hard_checks["retrieval_baseline_changed_unexpectedly"] += 1
            failure_cases.append(
                {
                    "type": "retrieval_baseline_changed_unexpectedly",
                    "query_id": query_id,
                    "first": [hit["chunk_id"] for hit in baseline_hits],
                    "second": [hit["chunk_id"] for hit in baseline_repeat],
                }
            )
        baseline_results[query_id] = baseline_hits[: settings.final_top_k]
        baseline_latencies.append(float(baseline_response.total_latency_ms or baseline_response.latency_ms))
        baseline_rows.append(
            {
                "query_id": query_id,
                "query": case["query"],
                "query_type": case["query_type"],
                "relevant_chunk_ids": case["relevant_chunk_ids"],
                "latency_ms": baseline_response.total_latency_ms,
                "hits": [public_hit(hit) for hit in baseline_hits[: settings.final_top_k]],
            }
        )
        expected_previous = previous_hybrid.get(query_id)
        current_ids = [str(hit["chunk_id"]) for hit in baseline_hits[:10]]
        if expected_previous is not None and current_ids != expected_previous:
            historical_baseline_drift.append(
                {
                    "query_id": query_id,
                    "previous": expected_previous,
                    "current": current_ids,
                }
            )

        try:
            reranked_response = engine.retrieve(
                case["query"],
                mode="hybrid",
                top_k=settings.final_top_k,
                candidate_k=retrieval_candidate_k,
                filters=case["filters"],
                rerank=True,
                rerank_candidate_k=settings.candidate_k,
                rerank_read_cache=False,
            )
        except Exception as exc:
            hard_checks["reranker_load_failure"] += 1
            failure_cases.append(
                {"type": "reranker_load_failure", "query_id": query_id, "error": f"{type(exc).__name__}: {exc}"}
            )
            reranked_results[query_id] = []
            continue
        payload = reranked_response.to_dict(include_text=False)
        reranked_hits = payload["hits"]
        reranked_results[query_id] = reranked_hits
        rerank_retrieval_latencies.append(float(reranked_response.retrieval_latency_ms or 0.0))
        rerank_inference_latencies.append(float(reranked_response.trace["reranker"]["inference_latency_ms"]))
        rerank_total_latencies.append(float(reranked_response.total_latency_ms or reranked_response.latency_ms))
        first_cache_hits += int(reranked_response.trace["reranker"]["cache_hits"])
        first_cache_misses += int(reranked_response.trace["reranker"]["cache_misses"])

        expected_candidates = min(settings.candidate_k, len(baseline_hits))
        expected_cache_items += expected_candidates
        if reranked_response.trace["reranker"]["candidate_count"] != expected_candidates:
            hard_checks["candidate_count_mismatch"] += 1
        candidate_ids = reranked_response.trace["reranker"]["candidate_chunk_ids"]
        if candidate_ids != [str(hit["chunk_id"]) for hit in baseline_hits[:expected_candidates]]:
            hard_checks["candidate_count_mismatch"] += 1
        required_fields = (
            "chunk_id",
            "fusion_score",
            "fusion_rank",
            "rerank_score",
            "rerank_rank",
            "pre_rerank_rank",
            "reranker_backend",
            "reranker_model",
        )
        for hit in reranked_hits:
            if str(hit.get("chunk_id")) not in engine.bundle.document_by_id:
                hard_checks["missing_chunk_id"] += 1
            if any(hit.get(field) is None for field in required_fields):
                hard_checks["result_schema_failure"] += 1
        reranked_rows.append(
            {
                "query_id": query_id,
                "query": case["query"],
                "query_type": case["query_type"],
                "relevant_chunk_ids": case["relevant_chunk_ids"],
                "retrieval_latency_ms": reranked_response.retrieval_latency_ms,
                "rerank_latency_ms": reranked_response.rerank_latency_ms,
                "total_latency_ms": reranked_response.total_latency_ms,
                "cache_hits": reranked_response.trace["reranker"]["cache_hits"],
                "cache_misses": reranked_response.trace["reranker"]["cache_misses"],
                "hits": [public_hit(hit) for hit in reranked_hits],
            }
        )

    # A full warm-cache pass validates deterministic ordering and cache score consistency.
    warm_total_latencies: list[float] = []
    warm_rerank_latencies: list[float] = []
    warm_cache_hits = 0
    warm_cache_misses = 0
    for case in cases:
        if not reranked_results.get(case["query_id"]):
            continue
        response = engine.retrieve(
            case["query"],
            mode="hybrid",
            top_k=settings.final_top_k,
            candidate_k=retrieval_candidate_k,
            filters=case["filters"],
            rerank=True,
            rerank_candidate_k=settings.candidate_k,
        )
        hits = response.to_dict(include_text=False)["hits"]
        warm_total_latencies.append(float(response.total_latency_ms or response.latency_ms))
        warm_rerank_latencies.append(float(response.rerank_latency_ms))
        warm_cache_hits += int(response.trace["reranker"]["cache_hits"])
        warm_cache_misses += int(response.trace["reranker"]["cache_misses"])
        cold = reranked_results[case["query_id"]]
        if [hit["chunk_id"] for hit in hits] != [hit["chunk_id"] for hit in cold]:
            hard_checks["non_deterministic_failure"] += 1
        cold_scores = [float(hit["rerank_score"]) for hit in cold]
        warm_scores = [float(hit["rerank_score"]) for hit in hits]
        if cold_scores != warm_scores:
            hard_checks["cache_consistency_failure"] += 1
    if first_cache_hits or first_cache_misses != expected_cache_items:
        hard_checks["cache_consistency_failure"] += 1
    if warm_cache_misses or warm_cache_hits != expected_cache_items:
        hard_checks["cache_consistency_failure"] += 1

    # Verify the model input template cannot receive benchmark labels or internal identifiers.
    forbidden_labels = ("query_id", "relevant_chunk_ids", "content_hash", "chunk_id")
    for case in cases:
        for document in engine.bundle.documents[:3]:
            rendered = render_rerank_document(document)
            if any(label in rendered for label in forbidden_labels):
                hard_checks["benchmark_label_leakage"] += 1
            if case["query_id"] in rendered or any(chunk_id in rendered for chunk_id in case["relevant_chunk_ids"]):
                hard_checks["benchmark_label_leakage"] += 1

    # Cache keys must separate query, content, model/config, and template identity.
    first_document = engine.bundle.documents[0]
    cache_key = RerankCache.make_key(
        query=cases[0]["query"],
        content_hash=str(first_document["content_hash"]),
        metadata_hash=str(first_document["metadata_hash"]),
        settings=settings,
    )
    changed_query_key = RerankCache.make_key(
        query=cases[0]["query"] + " changed",
        content_hash=str(first_document["content_hash"]),
        metadata_hash=str(first_document["metadata_hash"]),
        settings=settings,
    )
    changed_content_key = RerankCache.make_key(
        query=cases[0]["query"],
        content_hash="changed",
        metadata_hash=str(first_document["metadata_hash"]),
        settings=settings,
    )
    changed_metadata_key = RerankCache.make_key(
        query=cases[0]["query"],
        content_hash=str(first_document["content_hash"]),
        metadata_hash="changed",
        settings=settings,
    )
    if len({cache_key, changed_query_key, changed_content_key, changed_metadata_key}) != 4:
        hard_checks["cache_consistency_failure"] += 1

    baseline_metrics = evaluate(baseline_results, cases, top_k_values)
    reranked_metrics = evaluate(reranked_results, cases, top_k_values)
    improved_cases: list[dict[str, Any]] = []
    regressed_cases: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []
    no_answer_scores: list[dict[str, Any]] = []
    for case in cases:
        baseline_hits = baseline_results.get(case["query_id"], [])
        reranked_hits = reranked_results.get(case["query_id"], [])
        baseline_rank = first_relevant_rank(baseline_hits, case["relevant_chunk_ids"])
        reranked_rank = first_relevant_rank(reranked_hits, case["relevant_chunk_ids"])
        row = {
            "query_id": case["query_id"],
            "query": case["query"],
            "query_type": case["query_type"],
            "relevant_chunk_ids": case["relevant_chunk_ids"],
            "baseline_first_relevant_rank": baseline_rank,
            "reranked_first_relevant_rank": reranked_rank,
            "rank_delta": (baseline_rank or 11) - (reranked_rank or 11),
            "baseline_top1": baseline_hits[0]["chunk_id"] if baseline_hits else None,
            "reranked_top1": reranked_hits[0]["chunk_id"] if reranked_hits else None,
            "reranked_top_score": reranked_hits[0].get("rerank_score") if reranked_hits else None,
            "dense_bm25_conflict": bool(
                baseline_hits
                and baseline_hits[0].get("dense_rank") != baseline_hits[0].get("sparse_rank")
            ),
            "special_block_query": any(
                hit.get("has_equation") or hit.get("has_table") or hit.get("has_caption")
                for hit in reranked_hits
            ),
            "multi_evidence": len(case["relevant_chunk_ids"]) > 1,
        }
        analysis_rows.append(row)
        if row["rank_delta"] > 0:
            improved_cases.append(row)
        elif row["rank_delta"] < 0:
            regressed_cases.append(row)
        if not case["relevant_chunk_ids"]:
            no_answer_scores.append(
                {
                    "query_id": case["query_id"],
                    "query": case["query"],
                    "highest_rerank_score": row["reranked_top_score"],
                    "top_chunk_id": row["reranked_top1"],
                }
            )

    query_type_metrics: dict[str, Any] = {}
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_type[case["query_type"]].append(case)
    for query_type, subset in sorted(by_type.items()):
        query_type_metrics[query_type] = {
            "baseline": evaluate(baseline_results, subset, top_k_values),
            "reranked": evaluate(reranked_results, subset, top_k_values),
        }

    latency = {
        "baseline": {
            "average_total_ms": statistics.mean(baseline_latencies) if baseline_latencies else 0.0,
            "p95_total_ms": percentile(baseline_latencies, 0.95),
        },
        "reranked_cold": {
            "average_retrieval_ms": statistics.mean(rerank_retrieval_latencies) if rerank_retrieval_latencies else 0.0,
            "average_inference_ms": statistics.mean(rerank_inference_latencies) if rerank_inference_latencies else 0.0,
            "average_total_ms": statistics.mean(rerank_total_latencies) if rerank_total_latencies else 0.0,
            "p95_total_ms": percentile(rerank_total_latencies, 0.95),
            "cache_hits": first_cache_hits,
            "cache_misses": first_cache_misses,
        },
        "reranked_warm": {
            "average_rerank_ms": statistics.mean(warm_rerank_latencies) if warm_rerank_latencies else 0.0,
            "average_total_ms": statistics.mean(warm_total_latencies) if warm_total_latencies else 0.0,
            "p95_total_ms": percentile(warm_total_latencies, 0.95),
            "cache_hits": warm_cache_hits,
            "cache_misses": warm_cache_misses,
        },
    }

    for key in HARD_CHECK_KEYS:
        if hard_checks[key]:
            failure_cases.append({"type": key, "count": hard_checks[key]})
    hard_failure = any(hard_checks[key] for key in HARD_CHECK_KEYS)
    quality_improved = (
        reranked_metrics["mrr@10"] > baseline_metrics["mrr@10"]
        or reranked_metrics["ndcg@10"] > baseline_metrics["ndcg@10"]
    )
    recall_not_lower = reranked_metrics["recall@10"] >= baseline_metrics["recall@10"]
    conclusion = "FAIL" if hard_failure else ("PASS" if quality_improved and recall_not_lower else "PASS_WITH_MINOR_ISSUES")
    summary = {
        "schema_version": "reranker_validation_v1",
        "conclusion": conclusion,
        "reranker_backend": "cross_encoder",
        "reranker_model": settings.model_identity,
        "device": settings.device,
        "benchmark_query_count": len(cases),
        "answerable_query_count": sum(int(bool(case["relevant_chunk_ids"])) for case in cases),
        "no_answer_query_count": sum(int(not case["relevant_chunk_ids"]) for case in cases),
        "retrieval_candidate_k": retrieval_candidate_k,
        "rerank_candidate_k": settings.candidate_k,
        "final_top_k": settings.final_top_k,
        "batch_size": settings.batch_size,
        "max_length": settings.max_length,
        "hard_checks": {key: int(hard_checks[key]) for key in HARD_CHECK_KEYS},
        "metrics": {"baseline": baseline_metrics, "reranked": reranked_metrics},
        "metric_delta": {
            key: reranked_metrics[key] - baseline_metrics[key]
            for key in baseline_metrics
            if isinstance(baseline_metrics[key], (int, float))
        },
        "latency": latency,
        "improved_case_count": len(improved_cases),
        "regressed_case_count": len(regressed_cases),
        "no_answer_highest_scores": no_answer_scores,
        "failure_case_count": len(failure_cases),
        "historical_baseline_drift_count": len(historical_baseline_drift),
        "model_input_fields": ["title", "section", "heading", "content"],
        "model_input_excludes": ["chunk_id", "content_hash", "query_id", "relevant_chunk_ids"],
    }

    write_json(output_dir / "reranker_validation_summary.json", summary)
    (output_dir / "reranker_validation_report.md").write_text(render_report(summary), encoding="utf-8")
    write_jsonl(output_dir / "baseline_results.jsonl", baseline_rows)
    write_jsonl(output_dir / "reranked_results.jsonl", reranked_rows)
    write_jsonl(output_dir / "improved_cases.jsonl", sorted(improved_cases, key=lambda row: -row["rank_delta"]))
    write_jsonl(output_dir / "regressed_cases.jsonl", sorted(regressed_cases, key=lambda row: row["rank_delta"]))
    write_jsonl(output_dir / "failure_cases.jsonl", failure_cases)
    write_json(output_dir / "query_type_metrics.json", query_type_metrics)
    write_json(output_dir / "latency_report.json", latency)
    write_jsonl(output_dir / "analysis_cases.jsonl", analysis_rows)
    write_jsonl(output_dir / "historical_baseline_drift.jsonl", historical_baseline_drift)
    print(json.dumps({"conclusion": conclusion, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 2 if conclusion == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
