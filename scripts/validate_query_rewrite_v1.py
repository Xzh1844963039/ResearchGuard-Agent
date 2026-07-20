# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_query_rewrite_v1.py
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(r"C:\Users\18449\Desktop\researchguard_workspace")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.indexing.corpus_loader import load_yaml, write_json, write_jsonl  # noqa: E402
from researchguard.retrieval import RetrievalEngine  # noqa: E402
from researchguard.retrieval.multi_query import build_query_variants  # noqa: E402
from researchguard.retrieval.query_rewrite_pipeline import QueryRewritePipeline  # noqa: E402
from researchguard.retrieval.query_rewriter import (  # noqa: E402
    BackendRewrite,
    QueryAnalysis,
    QueryRewriteBackend,
    QueryRewriteBackendError,
    QueryRewriteResult,
    OpenAIQueryRewriteBackend,
    analyze_query,
    build_rewrite_model_input,
    load_query_rewrite_settings,
    missing_constraints,
    missing_entities,
    normalize_query_text,
)
from researchguard.retrieval.rewrite_cache import QueryRewriteCache  # noqa: E402
from scripts.validate_reranker_v1 import (  # noqa: E402
    evaluate,
    first_relevant_rank,
    load_cases,
    percentile,
    public_hit,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "query_rewrite_v1.yaml"
HARD_CHECK_KEYS = (
    "empty_rewrite",
    "entity_preservation_failure",
    "invalid_json_failure",
    "benchmark_leakage",
    "fallback_failure",
    "duplicate_query_failure",
    "non_deterministic_cache_failure",
    "result_schema_failure",
)
MODE_NAMES = (
    "raw_hybrid",
    "raw_hybrid_reranked",
    "rewrite_hybrid",
    "rewrite_hybrid_reranked",
    "multi_query_hybrid_reranked",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Query Rewrite and Multi-query Retrieval v1.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


class MemoryRewriteCache:
    def __init__(self):
        self.values: dict[str, dict[str, Any]] = {}
        self.enabled = True

    make_key = staticmethod(QueryRewriteCache.make_key)

    def get(self, key: str) -> dict[str, Any] | None:
        value = self.values.get(key)
        return dict(value) if value is not None else None

    def put(self, key: str, result: dict[str, Any]) -> None:
        self.values[key] = dict(result)


class FixedBackend(QueryRewriteBackend):
    def __init__(self, normalized: str, expansions: tuple[str, ...] = ()):
        self.normalized = normalized
        self.expansions = expansions
        self.calls = 0

    def rewrite(self, analysis: QueryAnalysis) -> BackendRewrite:
        self.calls += 1
        return BackendRewrite(
            normalized_query=self.normalized,
            expansion_queries=self.expansions,
            api_call_count=1,
            input_tokens=10,
            output_tokens=10,
        )


class FailingBackend(QueryRewriteBackend):
    def __init__(self, reason: str):
        self.reason = reason

    def rewrite(self, analysis: QueryAnalysis) -> BackendRewrite:
        raise QueryRewriteBackendError(self.reason, api_call_count=1)


class InvalidJsonResponses:
    @staticmethod
    def create(**kwargs: Any) -> Any:
        return type("InvalidJsonResponse", (), {"output_text": "{not-json", "usage": None})()


class InvalidJsonClient:
    responses = InvalidJsonResponses()


def stable_rewrite_payload(result: QueryRewriteResult) -> dict[str, Any]:
    payload = result.to_dict()
    for key in ("cache_hit", "api_call_count", "input_tokens", "output_tokens", "latency_ms"):
        payload.pop(key, None)
    return payload


def run_synthetic_tests(settings: Any) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    query = "Compare GPT-4o and CRAG in Table 3 after 2024 without omitting 72.4%."

    valid_backend = FixedBackend(
        query + " scientific evidence",
        (
            query + " retrieval terminology",
            query + " experimental comparison",
        ),
    )
    cache = MemoryRewriteCache()
    pipeline = QueryRewritePipeline(settings, backend=valid_backend, cache=cache)  # type: ignore[arg-type]
    first = pipeline.rewrite(query, read_cache=True)
    second = pipeline.rewrite(query, read_cache=True)
    tests.append(
        {
            "name": "deterministic_cache_roundtrip",
            "passed": (
                not first.cache_hit
                and second.cache_hit
                and valid_backend.calls == 1
                and stable_rewrite_payload(first) == stable_rewrite_payload(second)
            ),
            "details": {"backend_calls": valid_backend.calls, "second_cache_hit": second.cache_hit},
        }
    )

    duplicate_backend = FixedBackend(query, (query, query + " unique expansion"))
    duplicate_result = QueryRewritePipeline(
        settings,
        backend=duplicate_backend,
        cache=MemoryRewriteCache(),  # type: ignore[arg-type]
    ).rewrite(query, read_cache=False)
    duplicate_variants, _ = build_query_variants(duplicate_result, multi_query=True)
    tests.append(
        {
            "name": "duplicate_queries_removed",
            "passed": len({item.query.casefold() for item in duplicate_variants}) == len(duplicate_variants),
            "details": {"queries": [item.query for item in duplicate_variants]},
        }
    )

    invalid_json_backend = OpenAIQueryRewriteBackend(replace(settings, max_retries=0))
    invalid_json_backend._client = InvalidJsonClient()  # type: ignore[assignment]
    fallback_cases = (
        ("empty", FixedBackend(""), "empty_rewrite"),
        ("entity", FixedBackend("generic retrieval query"), "entity_preservation_failure"),
        ("invalid_json", invalid_json_backend, "backend_failure"),
        ("api", FailingBackend("api_failure"), "backend_failure"),
    )
    for name, backend, expected_reason in fallback_cases:
        result = QueryRewritePipeline(
            settings,
            backend=backend,
            cache=MemoryRewriteCache(),  # type: ignore[arg-type]
        ).rewrite(query, read_cache=False)
        tests.append(
            {
                "name": f"{name}_fallback_to_original",
                "passed": (
                    result.fallback_used
                    and result.normalized_query == normalize_query_text(query)
                    and not result.expansion_queries
                    and expected_reason in str(result.fallback_reason)
                ),
                "details": result.to_dict(),
            }
        )

    analysis_keys = set(build_rewrite_model_input(analyze_query(query)))
    tests.append(
        {
            "name": "backend_input_excludes_benchmark_labels",
            "passed": not analysis_keys.intersection({"query_id", "relevant_chunk_ids", "expected_doc_ids"}),
            "details": {"analysis_keys": sorted(analysis_keys)},
        }
    )

    original_key = QueryRewriteCache.make_key(original_query=query, settings=settings)
    changed_query_key = QueryRewriteCache.make_key(original_query=query + " changed", settings=settings)
    changed_model_key = QueryRewriteCache.make_key(
        original_query=query,
        settings=replace(settings, model=settings.model + "-changed"),
    )
    changed_prompt_key = QueryRewriteCache.make_key(
        original_query=query,
        settings=replace(settings, prompt_version=settings.prompt_version + "-changed"),
    )
    tests.append(
        {
            "name": "cache_key_covers_query_model_prompt_and_config",
            "passed": len({original_key, changed_query_key, changed_model_key, changed_prompt_key}) == 4,
            "details": {},
        }
    )
    return tests


def validate_rewrite_result(
    result: QueryRewriteResult,
    case: dict[str, Any],
    hard_checks: Counter[str],
    failure_cases: list[dict[str, Any]],
) -> None:
    query_id = case["query_id"]
    variants = [result.normalized_query, *result.expansion_queries]
    if not result.normalized_query.strip():
        hard_checks["empty_rewrite"] += 1
    missing = {
        variant: missing_entities(variant, result.preserved_entities)
        for variant in variants
        if missing_entities(variant, result.preserved_entities)
    }
    missing_constraint_values = missing_constraints(result.normalized_query, result.preserved_constraints)
    if missing or missing_constraint_values:
        hard_checks["entity_preservation_failure"] += 1
        failure_cases.append(
            {
                "type": "entity_preservation_failure",
                "query_id": query_id,
                "missing_entities": missing,
                "missing_constraints": missing_constraint_values,
            }
        )
    forbidden_values = [case["query_id"], *case["relevant_chunk_ids"]]
    if any(value and value.casefold() in variant.casefold() for variant in variants for value in forbidden_values):
        hard_checks["benchmark_leakage"] += 1
    query_variants, _ = build_query_variants(result, multi_query=True)
    variant_texts = [item.query.casefold() for item in query_variants]
    if len(variant_texts) != len(set(variant_texts)) or len(query_variants) > 4:
        hard_checks["duplicate_query_failure"] += 1
    if result.fallback_used and (
        result.normalized_query != normalize_query_text(case["query"]) or result.expansion_queries
    ):
        hard_checks["fallback_failure"] += 1
    required_fields = {
        "original_query",
        "normalized_query",
        "expansion_queries",
        "preserved_entities",
        "preserved_constraints",
        "dropped_expansion_reasons",
        "fallback_used",
        "fallback_reason",
        "timestamp",
    }
    if not required_fields.issubset(result.to_dict()):
        hard_checks["result_schema_failure"] += 1


def mode_response(
    engine: RetrievalEngine,
    case: dict[str, Any],
    rewrite_result: QueryRewriteResult,
    mode_name: str,
    *,
    retrieval_candidate_k: int,
    rerank_candidate_k: int,
    final_top_k: int,
) -> tuple[Any, float]:
    common = {
        "mode": "hybrid",
        "top_k": final_top_k,
        "candidate_k": retrieval_candidate_k,
        "filters": case["filters"],
    }
    if mode_name == "raw_hybrid":
        response = engine.retrieve(case["query"], **common, rerank=False, rewrite=False)
        return response, float(response.total_latency_ms or response.latency_ms)
    if mode_name == "raw_hybrid_reranked":
        response = engine.retrieve(
            case["query"],
            **common,
            rerank=True,
            rerank_candidate_k=rerank_candidate_k,
            rewrite=False,
        )
        return response, float(response.total_latency_ms or response.latency_ms)
    if mode_name == "rewrite_hybrid":
        response = engine.retrieve(
            case["query"],
            **common,
            rerank=False,
            rewrite=True,
            multi_query=False,
            rewrite_result=rewrite_result,
        )
    elif mode_name == "rewrite_hybrid_reranked":
        response = engine.retrieve(
            case["query"],
            **common,
            rerank=True,
            rerank_candidate_k=rerank_candidate_k,
            rewrite=True,
            multi_query=False,
            rewrite_result=rewrite_result,
        )
    else:
        response = engine.retrieve(
            case["query"],
            **common,
            rerank=True,
            rerank_candidate_k=rerank_candidate_k,
            rewrite=True,
            multi_query=True,
            rewrite_result=rewrite_result,
        )
    logical_total = float(rewrite_result.latency_ms) + float(response.total_latency_ms or response.latency_ms)
    return response, logical_total


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Query Rewrite v1 Validation",
        "",
        f"Conclusion: **{summary['conclusion']}**",
        "",
        "## Configuration",
        "",
        f"- model: `{summary['model']}`",
        f"- prompt_version: `{summary['prompt_version']}`",
        f"- max_rewrites: `{summary['max_rewrites']}`",
        f"- benchmark queries: `{summary['benchmark_query_count']}`",
        "",
        "## Hard Checks",
        "",
    ]
    lines.extend(f"- {key}: `{summary['hard_checks'][key]}`" for key in HARD_CHECK_KEYS)
    lines.extend(["", "## Metrics", "", "| Mode | Recall@10 | MRR@10 | nDCG@10 | Multi evidence coverage |", "| --- | ---: | ---: | ---: | ---: |"])
    for mode_name in MODE_NAMES:
        metrics = summary["metrics"][mode_name]
        lines.append(
            f"| {mode_name} | {metrics['recall@10']:.4f} | {metrics['mrr@10']:.4f} | "
            f"{metrics['ndcg@10']:.4f} | {metrics['multi_evidence_coverage@10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Rewrite and Cache",
            "",
            f"- successful rewrites: `{summary['rewrite']['success_count']}`",
            f"- fallbacks: `{summary['rewrite']['fallback_count']}`",
            f"- dropped expansions: `{summary['rewrite']['dropped_expansion_count']}`",
            f"- original Top-10 misses improved: `{summary['original_miss_improved_count']}`",
            f"- cold API calls: `{summary['api']['rewrite_api_calls']}`",
            f"- warm cache hit rate: `{summary['cache']['warm_hit_rate']:.4f}`",
            "",
            "Query Rewrite improves query expression and candidate recall only. It does not perform answerability, evidence sufficiency, answer generation, or citation audit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config, settings = load_query_rewrite_settings(config_path)
    validation = config.get("validation", {}) or {}
    benchmark_path = Path(validation.get("benchmark_path", "data/eval/retrieval_v1_queries.jsonl"))
    retrieval_config_path = Path(validation.get("retrieval_config_path", "configs/retrieval_v1.yaml"))
    output_dir = Path(validation.get("output_directory", "outputs/query_rewrite_validation_v1"))
    if not benchmark_path.is_absolute():
        benchmark_path = PROJECT_ROOT / benchmark_path
    if not retrieval_config_path.is_absolute():
        retrieval_config_path = PROJECT_ROOT / retrieval_config_path
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_candidate_k = int(validation.get("retrieval_candidate_k", 80))
    rerank_candidate_k = int(validation.get("rerank_candidate_k", 20))
    final_top_k = int(validation.get("final_top_k", 10))
    top_k_values = [int(item) for item in validation.get("top_k_values", [1, 3, 5, 10])]

    hard_checks: Counter[str] = Counter({key: 0 for key in HARD_CHECK_KEYS})
    failure_cases: list[dict[str, Any]] = []
    synthetic_tests = run_synthetic_tests(settings)
    for test in synthetic_tests:
        if test["passed"]:
            continue
        if test["name"] in {"empty_fallback_to_original", "entity_fallback_to_original", "api_fallback_to_original"}:
            hard_checks["fallback_failure"] += 1
        elif test["name"] == "invalid_json_fallback_to_original":
            hard_checks["invalid_json_failure"] += 1
        elif test["name"] == "duplicate_queries_removed":
            hard_checks["duplicate_query_failure"] += 1
        elif test["name"] == "deterministic_cache_roundtrip":
            hard_checks["non_deterministic_cache_failure"] += 1
        elif test["name"] == "backend_input_excludes_benchmark_labels":
            hard_checks["benchmark_leakage"] += 1
        else:
            hard_checks["result_schema_failure"] += 1
        failure_cases.append({"type": "synthetic_test_failure", **test})

    engine = RetrievalEngine.from_config(retrieval_config_path)
    cases, benchmark_errors = load_cases(benchmark_path, set(engine.bundle.document_by_id))
    if benchmark_errors:
        hard_checks["result_schema_failure"] += len(benchmark_errors)
        failure_cases.extend(benchmark_errors)

    results: dict[str, dict[str, list[dict[str, Any]]]] = {name: {} for name in MODE_NAMES}
    result_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in MODE_NAMES}
    rewrite_rows: list[dict[str, Any]] = []
    latency_values: dict[str, dict[str, list[float]]] = {
        name: {"rewrite": [], "retrieval": [], "rerank": [], "total": []} for name in MODE_NAMES
    }
    rewrite_api_calls = 0
    rewrite_input_tokens = 0
    rewrite_output_tokens = 0
    embedding_api_calls = 0
    cold_cache_hits = 0
    warm_cache_hits = 0
    warm_cache_checks = 0

    for index, case in enumerate(cases, start=1):
        rewrite_result = engine.rewrite_query(case["query"], read_cache=False)
        validate_rewrite_result(rewrite_result, case, hard_checks, failure_cases)
        rewrite_api_calls += rewrite_result.api_call_count
        rewrite_input_tokens += rewrite_result.input_tokens
        rewrite_output_tokens += rewrite_result.output_tokens
        cold_cache_hits += int(rewrite_result.cache_hit)

        warm_result = engine.rewrite_query(case["query"], read_cache=True)
        warm_cache_checks += 1
        warm_cache_hits += int(warm_result.cache_hit)
        if not warm_result.cache_hit or stable_rewrite_payload(warm_result) != stable_rewrite_payload(rewrite_result):
            hard_checks["non_deterministic_cache_failure"] += 1
            failure_cases.append(
                {
                    "type": "non_deterministic_cache_failure",
                    "query_id": case["query_id"],
                    "cold": stable_rewrite_payload(rewrite_result),
                    "warm": stable_rewrite_payload(warm_result),
                }
            )

        query_variants, duplicate_count = build_query_variants(rewrite_result, multi_query=True)
        rewrite_rows.append(
            {
                "query_id": case["query_id"],
                "query_type": case["query_type"],
                **rewrite_result.to_dict(),
                "query_variants": [item.to_dict() for item in query_variants],
                "duplicate_queries_removed": duplicate_count,
            }
        )

        for mode_name in MODE_NAMES:
            response, logical_total = mode_response(
                engine,
                case,
                rewrite_result,
                mode_name,
                retrieval_candidate_k=retrieval_candidate_k,
                rerank_candidate_k=rerank_candidate_k,
                final_top_k=final_top_k,
            )
            payload = response.to_dict(include_text=False)
            hits = payload["hits"]
            results[mode_name][case["query_id"]] = hits
            embedding_api_calls += int(response.trace.get("query_embedding_api_calls", 0))
            latency_values[mode_name]["rewrite"].append(
                float(rewrite_result.latency_ms) if mode_name.startswith("rewrite") or mode_name.startswith("multi") else 0.0
            )
            latency_values[mode_name]["retrieval"].append(float(response.retrieval_latency_ms or 0.0))
            latency_values[mode_name]["rerank"].append(float(response.rerank_latency_ms or 0.0))
            latency_values[mode_name]["total"].append(logical_total)
            result_rows[mode_name].append(
                {
                    "query_id": case["query_id"],
                    "query": case["query"],
                    "query_type": case["query_type"],
                    "relevant_chunk_ids": case["relevant_chunk_ids"],
                    "rewrite": rewrite_result.to_dict() if response.trace.get("query_rewrite_enabled") else None,
                    "latency": {
                        "rewrite_ms": rewrite_result.latency_ms if response.trace.get("query_rewrite_enabled") else 0.0,
                        "retrieval_ms": response.retrieval_latency_ms,
                        "rerank_ms": response.rerank_latency_ms,
                        "total_ms": logical_total,
                    },
                    "trace": response.trace,
                    "hits": [public_hit(hit) for hit in hits],
                }
            )

            required_hit_fields = {"chunk_id", "doc_id", "section", "rank"}
            if any(not required_hit_fields.issubset(hit) for hit in hits):
                hard_checks["result_schema_failure"] += 1
            if mode_name == "multi_query_hybrid_reranked":
                multi_fields = {
                    "multi_query_fusion_score",
                    "multi_query_fusion_rank",
                    "query_variant_hits",
                    "original_query_recalled",
                    "rewrite_query_recalled",
                    "expansion_query_recalled",
                    "rerank_score",
                    "rerank_rank",
                }
                if any(any(hit.get(field) is None for field in multi_fields) for hit in hits):
                    hard_checks["result_schema_failure"] += 1
                if any(len({entry["variant_id"] for entry in hit["query_variant_hits"]}) != len(hit["query_variant_hits"]) for hit in hits):
                    hard_checks["duplicate_query_failure"] += 1

        print(f"validated {index}/{len(cases)} {case['query_id']}", flush=True)

    metrics = {name: evaluate(results[name], cases, top_k_values) for name in MODE_NAMES}
    latency: dict[str, Any] = {}
    for mode_name in MODE_NAMES:
        latency[mode_name] = {}
        for component, values in latency_values[mode_name].items():
            latency[mode_name][f"average_{component}_ms"] = statistics.mean(values) if values else 0.0
            latency[mode_name][f"p95_{component}_ms"] = percentile(values, 0.95)

    original_miss_rows: list[dict[str, Any]] = []
    improved_cases: list[dict[str, Any]] = []
    regressed_cases: list[dict[str, Any]] = []
    for case in cases:
        if not case["relevant_chunk_ids"]:
            continue
        ranks = {
            name: first_relevant_rank(results[name][case["query_id"]], case["relevant_chunk_ids"])
            for name in MODE_NAMES
        }
        raw_rank = ranks["raw_hybrid"]
        final_rank = ranks["multi_query_hybrid_reranked"]
        raw_reranked_relevant = [
            hit
            for hit in results["raw_hybrid_reranked"][case["query_id"]]
            if hit["chunk_id"] in case["relevant_chunk_ids"]
        ]
        multi_reranked_relevant = [
            hit
            for hit in results["multi_query_hybrid_reranked"][case["query_id"]]
            if hit["chunk_id"] in case["relevant_chunk_ids"]
        ]
        raw_pre_rerank_rank = min(
            (int(hit["pre_rerank_rank"]) for hit in raw_reranked_relevant if hit.get("pre_rerank_rank")),
            default=None,
        )
        multi_pre_rerank_rank = min(
            (int(hit["pre_rerank_rank"]) for hit in multi_reranked_relevant if hit.get("pre_rerank_rank")),
            default=None,
        )
        attributable_rewrite_improvement = bool(
            raw_rank is None
            and (
                ranks["rewrite_hybrid"] is not None
                or (
                    multi_pre_rerank_rank is not None
                    and multi_pre_rerank_rank <= 10
                    and (raw_pre_rerank_rank is None or raw_pre_rerank_rank > 10)
                )
            )
        )
        row = {
            "query_id": case["query_id"],
            "query": case["query"],
            "query_type": case["query_type"],
            "relevant_chunk_ids": case["relevant_chunk_ids"],
            "ranks": ranks,
            "rank_delta_vs_raw": (raw_rank or 11) - (final_rank or 11),
            "raw_rerank_pre_rerank_first_relevant_rank": raw_pre_rerank_rank,
            "multi_query_pre_rerank_first_relevant_rank": multi_pre_rerank_rank,
            "attributable_rewrite_improvement": attributable_rewrite_improvement,
            "rewrite": next(item for item in rewrite_rows if item["query_id"] == case["query_id"]),
            "final_top_doc_ids": [hit["doc_id"] for hit in results["multi_query_hybrid_reranked"][case["query_id"]][:3]],
            "introduced_wrong_top_document": bool(
                results["multi_query_hybrid_reranked"][case["query_id"]]
                and results["multi_query_hybrid_reranked"][case["query_id"]][0]["doc_id"]
                not in case["expected_doc_ids"]
            ),
        }
        if raw_rank is None:
            original_miss_rows.append(row)
        if row["rank_delta_vs_raw"] > 0:
            improved_cases.append(row)
        elif row["rank_delta_vs_raw"] < 0:
            regressed_cases.append(row)

    original_miss_improved = sum(int(row["attributable_rewrite_improvement"]) for row in original_miss_rows)

    query_type_metrics: dict[str, Any] = {}
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_type[case["query_type"]].append(case)
    for query_type, subset in sorted(by_type.items()):
        query_type_metrics[query_type] = {
            name: evaluate(results[name], subset, top_k_values) for name in MODE_NAMES
        }

    fallback_rows = [row for row in rewrite_rows if row["fallback_used"]]
    dropped_expansion_count = sum(len(row["dropped_expansion_reasons"]) for row in rewrite_rows)
    hard_failure = any(hard_checks[key] for key in HARD_CHECK_KEYS)
    recall_not_lower = (
        metrics["multi_query_hybrid_reranked"]["recall@10"]
        >= metrics["raw_hybrid"]["recall@10"]
    )
    ranking_not_materially_lower = (
        metrics["multi_query_hybrid_reranked"]["mrr@10"]
        >= metrics["raw_hybrid_reranked"]["mrr@10"] - 0.02
        and metrics["multi_query_hybrid_reranked"]["ndcg@10"]
        >= metrics["raw_hybrid_reranked"]["ndcg@10"] - 0.02
    )
    quality_pass = recall_not_lower and original_miss_improved > 0 and ranking_not_materially_lower
    conclusion = "FAIL" if hard_failure else ("PASS" if quality_pass else "PASS_WITH_MINOR_ISSUES")

    summary = {
        "schema_version": "query_rewrite_validation_v1",
        "conclusion": conclusion,
        "model": settings.model,
        "temperature": settings.temperature,
        "max_rewrites": settings.max_rewrites,
        "prompt_version": settings.prompt_version,
        "entity_rules_version": settings.entity_rules_version,
        "benchmark_query_count": len(cases),
        "answerable_query_count": sum(int(bool(case["relevant_chunk_ids"])) for case in cases),
        "no_answer_query_count": sum(int(not case["relevant_chunk_ids"]) for case in cases),
        "hard_checks": {key: int(hard_checks[key]) for key in HARD_CHECK_KEYS},
        "synthetic_tests": synthetic_tests,
        "metrics": metrics,
        "latency": latency,
        "api": {
            "rewrite_api_calls": rewrite_api_calls,
            "query_embedding_api_calls": embedding_api_calls,
            "rewrite_input_tokens": rewrite_input_tokens,
            "rewrite_output_tokens": rewrite_output_tokens,
        },
        "cache": {
            "cold_hits": cold_cache_hits,
            "cold_hit_rate": cold_cache_hits / max(len(cases), 1),
            "warm_hits": warm_cache_hits,
            "warm_checks": warm_cache_checks,
            "warm_hit_rate": warm_cache_hits / max(warm_cache_checks, 1),
        },
        "rewrite": {
            "success_count": len(cases) - len(fallback_rows),
            "fallback_count": len(fallback_rows),
            "dropped_expansion_count": dropped_expansion_count,
            "average_expansion_count": statistics.mean(len(row["expansion_queries"]) for row in rewrite_rows),
        },
        "original_miss_count": len(original_miss_rows),
        "original_miss_improved_count": original_miss_improved,
        "improved_case_count": len(improved_cases),
        "regressed_case_count": len(regressed_cases),
        "quality_checks": {
            "recall_at_10_not_lower_than_raw": recall_not_lower,
            "original_miss_improved": original_miss_improved > 0,
            "mrr_ndcg_not_materially_lower_than_raw_reranked": ranking_not_materially_lower,
        },
        "failure_case_count": len(failure_cases),
        "scope_boundary": {
            "improves": ["query expression", "recall coverage", "terminology expansion"],
            "does_not_determine": ["corpus answerability", "evidence sufficiency", "answer trustworthiness"],
        },
    }

    write_json(output_dir / "query_rewrite_validation_summary.json", summary)
    (output_dir / "query_rewrite_validation_report.md").write_text(render_report(summary), encoding="utf-8")
    write_jsonl(output_dir / "rewrite_audit.jsonl", rewrite_rows)
    for mode_name in MODE_NAMES:
        write_jsonl(output_dir / f"{mode_name}_results.jsonl", result_rows[mode_name])
    write_jsonl(output_dir / "original_miss_analysis.jsonl", original_miss_rows)
    write_jsonl(output_dir / "improved_cases.jsonl", sorted(improved_cases, key=lambda row: -row["rank_delta_vs_raw"]))
    write_jsonl(output_dir / "regressed_cases.jsonl", sorted(regressed_cases, key=lambda row: row["rank_delta_vs_raw"]))
    write_jsonl(output_dir / "fallback_cases.jsonl", fallback_rows)
    write_jsonl(output_dir / "failure_cases.jsonl", failure_cases)
    write_json(output_dir / "query_type_metrics.json", query_type_metrics)
    write_json(output_dir / "latency_report.json", latency)
    write_json(output_dir / "api_cache_report.json", {"api": summary["api"], "cache": summary["cache"]})
    print(json.dumps({"conclusion": conclusion, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 2 if conclusion == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
