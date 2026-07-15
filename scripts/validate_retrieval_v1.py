# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_retrieval_v1.py
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(r"C:\Users\18449\Desktop\researchguard_workspace")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.indexing.corpus_loader import corpus_fingerprint, read_jsonl, write_json, write_jsonl
from researchguard.retrieval import MetadataFilter, RetrievalEngine, RetrievalError
from researchguard.retrieval.filters import apply_metadata_filters
from researchguard.retrieval.index_loader import load_index_bundle, validate_loaded_index


DEFAULT_CONFIG = Path(r"C:\Users\18449\Desktop\researchguard_workspace\configs\retrieval_v1.yaml")
MODES = ("dense", "sparse", "hybrid")


@dataclass
class QueryCase:
    query_id: str
    query: str
    query_type: str
    relevant_chunk_ids: list[str]
    expected_doc_ids: list[str]
    expected_sections: list[str]
    filters: MetadataFilter
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Retrieval v1 with benchmark queries and hard checks.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to retrieval_v1.yaml.")
    return parser.parse_args()


def load_cases(path: Path, valid_chunk_ids: set[str]) -> tuple[list[QueryCase], dict[str, int], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    hard_checks = {
        "benchmark_invalid_chunk_id": 0,
        "benchmark_empty_query": 0,
        "benchmark_schema_error": 0,
    }
    errors: list[dict[str, Any]] = []
    cases: list[QueryCase] = []
    seen: set[str] = set()
    for row_no, row in enumerate(rows, start=1):
        query_id = str(row.get("query_id", "")).strip()
        query = str(row.get("query", "")).strip()
        relevant = [str(item) for item in row.get("relevant_chunk_ids", [])]
        if not query_id or query_id in seen:
            hard_checks["benchmark_schema_error"] += 1
            errors.append({"row": row_no, "type": "bad_query_id", "query_id": query_id})
        seen.add(query_id)
        if not query:
            hard_checks["benchmark_empty_query"] += 1
            errors.append({"row": row_no, "type": "empty_query", "query_id": query_id})
        invalid = [chunk_id for chunk_id in relevant if chunk_id not in valid_chunk_ids]
        if invalid:
            hard_checks["benchmark_invalid_chunk_id"] += len(invalid)
            errors.append({"row": row_no, "type": "invalid_relevant_chunk_id", "query_id": query_id, "invalid": invalid})
        try:
            filters = MetadataFilter.from_mapping(row.get("filters", {}))
        except Exception as exc:
            hard_checks["benchmark_schema_error"] += 1
            errors.append({"row": row_no, "type": "bad_filter", "query_id": query_id, "error": str(exc)})
            filters = MetadataFilter()
        cases.append(
            QueryCase(
                query_id=query_id,
                query=query,
                query_type=str(row.get("query_type", "unknown")),
                relevant_chunk_ids=relevant,
                expected_doc_ids=[str(item) for item in row.get("expected_doc_ids", [])],
                expected_sections=[str(item) for item in row.get("expected_sections", [])],
                filters=filters,
                notes=str(row.get("notes", "")),
            )
        )
    return cases, hard_checks, errors


def dcg(binary_hits: list[int]) -> float:
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(binary_hits, start=1))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * pct
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(sorted_values[lower])
    weight = pos - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def evaluate_mode(results: dict[str, list[dict[str, Any]]], cases: list[QueryCase], ks: list[int]) -> dict[str, Any]:
    answerable = [case for case in cases if case.relevant_chunk_ids]
    no_answer = [case for case in cases if not case.relevant_chunk_ids]
    metrics: dict[str, Any] = {}
    for k in ks:
        recall_hits = 0
        doc_hits = 0
        section_hits = 0
        for case in answerable:
            top = results[case.query_id][:k]
            hit_ids = {hit["chunk_id"] for hit in top}
            if hit_ids & set(case.relevant_chunk_ids):
                recall_hits += 1
            if case.expected_doc_ids and {hit["doc_id"] for hit in top} & set(case.expected_doc_ids):
                doc_hits += 1
            if case.expected_sections and {hit["section"] for hit in top} & set(case.expected_sections):
                section_hits += 1
        denom = max(len(answerable), 1)
        metrics[f"recall@{k}"] = recall_hits / denom
        metrics[f"document_hit@{k}"] = doc_hits / denom
        metrics[f"section_hit@{k}"] = section_hits / denom

    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    multi_coverages: list[float] = []
    for case in answerable:
        relevant = set(case.relevant_chunk_ids)
        hits = results[case.query_id][:10]
        rr = 0.0
        binary: list[int] = []
        for rank, hit in enumerate(hits, start=1):
            is_relevant = hit["chunk_id"] in relevant
            binary.append(1 if is_relevant else 0)
            if is_relevant and rr == 0.0:
                rr = 1.0 / rank
        reciprocal_ranks.append(rr)
        ideal_count = min(len(relevant), 10)
        ideal = dcg([1] * ideal_count)
        ndcgs.append(dcg(binary) / ideal if ideal else 0.0)
        if len(relevant) > 1:
            found = {hit["chunk_id"] for hit in hits} & relevant
            multi_coverages.append(len(found) / len(relevant))
    metrics["mrr@10"] = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    metrics["ndcg@10"] = statistics.mean(ndcgs) if ndcgs else 0.0
    metrics["multi_evidence_coverage@10"] = statistics.mean(multi_coverages) if multi_coverages else 0.0
    metrics["no_answer_false_positive_rate"] = (
        sum(1 for case in no_answer if results[case.query_id]) / len(no_answer) if no_answer else 0.0
    )
    return metrics


def synthetic_tests(engine: RetrievalEngine, bundle: Any, cases: list[QueryCase]) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []

    def record(name: str, passed: bool, details: dict[str, Any] | None = None) -> None:
        tests.append({"name": name, "passed": bool(passed), "details": details or {}})

    loaded = bundle.hard_checks
    record("index_loader_hard_checks_zero", all(value == 0 for value in loaded.values()), loaded)

    query = "How does ReAct combine reasoning traces with actions?"
    first = engine.retrieve(query, mode="hybrid", top_k=10, candidate_k=40).to_dict(include_text=False)["hits"]
    second = engine.retrieve(query, mode="hybrid", top_k=10, candidate_k=40).to_dict(include_text=False)["hits"]
    record(
        "deterministic_hybrid_ranking",
        [hit["chunk_id"] for hit in first] == [hit["chunk_id"] for hit in second],
        {"first": [hit["chunk_id"] for hit in first], "second": [hit["chunk_id"] for hit in second]},
    )

    dense_first = engine.retrieve(query, mode="dense", top_k=5, candidate_k=20).to_dict(include_text=False)["hits"]
    dense_second = engine.retrieve(query, mode="dense", top_k=5, candidate_k=20).to_dict(include_text=False)["hits"]
    record(
        "deterministic_dense_ranking",
        [hit["chunk_id"] for hit in dense_first] == [hit["chunk_id"] for hit in dense_second],
    )

    sparse_first = engine.retrieve("citation precision recall ALCE", mode="sparse", top_k=5, candidate_k=20).to_dict(include_text=False)["hits"]
    sparse_second = engine.retrieve("citation precision recall ALCE", mode="sparse", top_k=5, candidate_k=20).to_dict(include_text=False)["hits"]
    record(
        "deterministic_sparse_ranking",
        [hit["chunk_id"] for hit in sparse_first] == [hit["chunk_id"] for hit in sparse_second],
    )

    filtered = engine.retrieve(
        "citation precision recall ALCE",
        mode="hybrid",
        top_k=8,
        candidate_k=40,
        filters=MetadataFilter(doc_ids=("paper_citation",), sections=("results",)),
    ).to_dict(include_text=False)["hits"]
    record(
        "metadata_filter_doc_and_section",
        bool(filtered) and all(hit["doc_id"] == "paper_citation" and hit["section"] == "results" for hit in filtered),
        {"hits": [(hit["chunk_id"], hit["doc_id"], hit["section"]) for hit in filtered]},
    )

    non_refs = engine.retrieve(
        "References bibliography proceedings conference",
        mode="hybrid",
        top_k=10,
        candidate_k=60,
        filters=MetadataFilter(exclude_references=True),
    ).to_dict(include_text=False)["hits"]
    record("exclude_references_filter", all(hit["section"] != "references" for hit in non_refs))

    try:
        engine.retrieve("   ", mode="hybrid", top_k=5)
        record("empty_query_rejected", False)
    except RetrievalError:
        record("empty_query_rejected", True)

    bad_manifest = copy.deepcopy(bundle.manifest)
    bad_manifest["corpus_fingerprint"] = "bad"
    bad_checks = validate_loaded_index(
        manifest=bad_manifest,
        dense_manifest=bundle.dense_manifest,
        sparse_payload=bundle.sparse_payload,
        documents=bundle.documents,
        dense_index=bundle.dense_index,
        sparse_index=bundle.sparse_index,
    )
    record("fingerprint_mismatch_detected", bad_checks["fingerprint_mismatch"] > 0, bad_checks)

    rrf = engine.retrieve("SelfCheckGPT black-box hallucination detection", mode="hybrid", top_k=10, candidate_k=50)
    rrf_hits = rrf.to_dict(include_text=False)["hits"]
    record(
        "hybrid_schema_has_fusion_and_sources",
        bool(rrf_hits)
        and all(hit["fusion_score"] is not None and hit["retrieval_sources"] for hit in rrf_hits),
        {"first_hit": rrf_hits[0] if rrf_hits else None},
    )

    valid_case_count = len([case for case in cases if case.query and all(cid in bundle.document_by_id for cid in case.relevant_chunk_ids)])
    record("benchmark_cases_loadable", valid_case_count == len(cases) and len(cases) >= 30, {"case_count": len(cases)})
    return tests


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    bundle = load_index_bundle(config_path, strict=False)
    config = bundle.config
    validation_cfg = config.get("validation", {}) or {}
    benchmark_path = Path(validation_cfg.get("benchmark_path"))
    output_dir = Path(validation_cfg.get("output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    top_k_values = [int(value) for value in validation_cfg.get("top_k_values", [1, 3, 5, 10])]
    candidate_k = int(validation_cfg.get("candidate_k", 80))

    hard_checks = dict(bundle.hard_checks)
    hard_checks.setdefault("query_embedding_failure", 0)
    hard_checks.setdefault("non_deterministic_ranking", 0)
    hard_checks.setdefault("benchmark_invalid_chunk_id", 0)
    hard_checks.setdefault("benchmark_empty_query", 0)
    hard_checks.setdefault("result_schema_error", 0)

    cases, benchmark_checks, benchmark_errors = load_cases(benchmark_path, set(bundle.document_by_id))
    hard_checks.update({key: hard_checks.get(key, 0) + value for key, value in benchmark_checks.items()})

    engine = RetrievalEngine(bundle)
    synthetic = synthetic_tests(engine, bundle, cases)
    if any(not test["passed"] for test in synthetic):
        hard_checks["result_schema_error"] += sum(1 for test in synthetic if not test["passed"])

    all_results: dict[str, dict[str, list[dict[str, Any]]]] = {mode: {} for mode in MODES}
    latencies: dict[str, list[float]] = {mode: [] for mode in MODES}
    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for case in cases:
        for mode in MODES:
            try:
                started = time.perf_counter()
                response = engine.retrieve(
                    case.query,
                    mode=mode,
                    top_k=10,
                    candidate_k=candidate_k,
                    filters=case.filters,
                )
                elapsed = (time.perf_counter() - started) * 1000.0
                payload = response.to_dict(include_text=False)
                hits = payload["hits"]
                all_results[mode][case.query_id] = hits
                latencies[mode].append(elapsed)
                if len(hits) > 10 or any("chunk_id" not in hit or "doc_id" not in hit for hit in hits):
                    hard_checks["result_schema_error"] += 1
                result_rows.append(
                    {
                        "query_id": case.query_id,
                        "query_type": case.query_type,
                        "mode": mode,
                        "latency_ms": elapsed,
                        "filters": case.filters.to_dict(),
                        "relevant_chunk_ids": case.relevant_chunk_ids,
                        "hits": hits,
                    }
                )
            except Exception as exc:
                if mode in {"dense", "hybrid"}:
                    hard_checks["query_embedding_failure"] += 1
                else:
                    hard_checks["result_schema_error"] += 1
                all_results[mode][case.query_id] = []
                failure_rows.append(
                    {
                        "query_id": case.query_id,
                        "query_type": case.query_type,
                        "mode": mode,
                        "failure_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    for mode in MODES:
        probe_a = engine.retrieve("retrieval evaluator estimates document relevance", mode=mode, top_k=10, candidate_k=40)
        probe_b = engine.retrieve("retrieval evaluator estimates document relevance", mode=mode, top_k=10, candidate_k=40)
        ids_a = [hit.chunk_id for hit in probe_a.hits]
        ids_b = [hit.chunk_id for hit in probe_b.hits]
        if ids_a != ids_b:
            hard_checks["non_deterministic_ranking"] += 1

    metrics = {mode: evaluate_mode(all_results[mode], cases, top_k_values) for mode in MODES}
    for mode in MODES:
        metrics[mode]["average_latency_ms"] = statistics.mean(latencies[mode]) if latencies[mode] else 0.0
        metrics[mode]["p95_latency_ms"] = percentile(latencies[mode], 0.95)

    type_metrics: dict[str, dict[str, Any]] = {}
    by_type: dict[str, list[QueryCase]] = defaultdict(list)
    for case in cases:
        by_type[case.query_type].append(case)
    for query_type, subset in sorted(by_type.items()):
        type_metrics[query_type] = {mode: evaluate_mode(all_results[mode], subset, top_k_values) for mode in MODES}

    for case in cases:
        if not case.relevant_chunk_ids:
            for mode in MODES:
                hits = all_results[mode].get(case.query_id, [])
                if hits:
                    failure_rows.append(
                        {
                            "query_id": case.query_id,
                            "query_type": case.query_type,
                            "mode": mode,
                            "failure_type": "no_answer_retrieved",
                            "top_hit": hits[0]["chunk_id"],
                            "top_doc": hits[0]["doc_id"],
                        }
                    )
            continue
        relevant = set(case.relevant_chunk_ids)
        for mode in MODES:
            hits = all_results[mode].get(case.query_id, [])[:10]
            if not ({hit["chunk_id"] for hit in hits} & relevant):
                failure_rows.append(
                    {
                        "query_id": case.query_id,
                        "query_type": case.query_type,
                        "mode": mode,
                        "failure_type": "missed_relevant_at_10",
                        "relevant_chunk_ids": case.relevant_chunk_ids,
                        "top_hits": [hit["chunk_id"] for hit in hits[:5]],
                    }
                )

    hard_failed = any(value for value in hard_checks.values())
    hybrid = metrics["hybrid"]
    dense = metrics["dense"]
    sparse = metrics["sparse"]
    hybrid_not_worse_recall = hybrid["recall@10"] >= dense["recall@10"] and hybrid["recall@10"] >= sparse["recall@10"]
    hybrid_mrr_regression = hybrid["mrr@10"] + 0.05 < max(dense["mrr@10"], sparse["mrr@10"])
    if hard_failed:
        conclusion = "FAIL"
    elif not hybrid_not_worse_recall or hybrid_mrr_regression:
        conclusion = "PASS_WITH_MINOR_ISSUES"
    else:
        conclusion = "PASS"

    summary = {
        "schema_version": "retrieval_validation_v1",
        "conclusion": conclusion,
        "benchmark_path": str(benchmark_path),
        "query_count": len(cases),
        "answerable_query_count": sum(1 for case in cases if case.relevant_chunk_ids),
        "no_answer_query_count": sum(1 for case in cases if not case.relevant_chunk_ids),
        "corpus_fingerprint": corpus_fingerprint(bundle.documents),
        "index_manifest_fingerprint": bundle.manifest.get("corpus_fingerprint"),
        "hard_checks": hard_checks,
        "synthetic_tests": synthetic,
        "metrics": metrics,
        "benchmark_errors": benchmark_errors,
        "failure_case_count": len(failure_rows),
    }

    write_json(output_dir / "retrieval_validation_summary.json", summary)
    write_json(output_dir / "query_type_metrics.json", type_metrics)
    write_json(
        output_dir / "latency_report.json",
        {
            mode: {
                "average_latency_ms": metrics[mode]["average_latency_ms"],
                "p95_latency_ms": metrics[mode]["p95_latency_ms"],
                "samples": len(latencies[mode]),
            }
            for mode in MODES
        },
    )
    write_jsonl(output_dir / "retrieval_results.jsonl", result_rows)
    write_jsonl(output_dir / "failure_cases.jsonl", failure_rows)
    (output_dir / "retrieval_validation_report.md").write_text(render_report(summary, metrics, synthetic), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if conclusion != "FAIL" else 1


def render_report(summary: dict[str, Any], metrics: dict[str, Any], synthetic: list[dict[str, Any]]) -> str:
    lines = [
        "# Retrieval v1 Validation Report",
        "",
        f"Conclusion: `{summary['conclusion']}`",
        f"Benchmark queries: {summary['query_count']} ({summary['answerable_query_count']} answerable, {summary['no_answer_query_count']} no-answer)",
        f"Corpus fingerprint: `{summary['corpus_fingerprint']}`",
        "",
        "## Hard Checks",
        "",
        "| Check | Count |",
        "| --- | ---: |",
    ]
    for key, value in sorted(summary["hard_checks"].items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Metrics", ""])
    for mode, values in metrics.items():
        lines.extend([f"### {mode}", "", "| Metric | Value |", "| --- | ---: |"])
        for key, value in values.items():
            lines.append(f"| `{key}` | {value:.4f} |")
        lines.append("")
    lines.extend(["## Synthetic Tests", "", "| Test | Passed |", "| --- | --- |"])
    for test in synthetic:
        lines.append(f"| `{test['name']}` | {test['passed']} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `no_answer_false_positive_rate` is reported honestly for retrieval-only behavior: Retrieval v1 always returns nearest chunks when any lexical/vector match exists, and does not perform evidence sufficiency or answerability detection.",
            "- Query rewrite, LLM rerank, evidence sufficiency, retry, neighbor expansion, answer generation, citation audit, and Agentic RAG are intentionally out of scope for this validation.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
