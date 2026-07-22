# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_pipeline_v1.py
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.indexing.corpus_loader import load_yaml
from researchguard.pipeline import PipelineSettings, ResearchGuardPipeline, load_pipeline_settings
from researchguard.retrieval.answer_generator import (
    AnswerCitation,
    AnswerGenerationResult,
    utc_timestamp,
)
from researchguard.retrieval.citation_audit import CitationAuditResult
from researchguard.retrieval.evidence_judge import EvidenceSufficiencyResult
from researchguard.retrieval.models import MetadataFilter, RetrievalHit, RetrievalResponse
from researchguard.retrieval.retrieval_v1 import RetrievalEngine


HARD_CHECK_NAMES = (
    "pipeline_schema_failure",
    "module_failure",
    "evidence_gate_bypass",
    "unsupported_generation",
    "citation_missing",
    "cache_failure",
    "retrieval_regression",
    "result_schema_failure",
)
STAGES = (
    "rewrite",
    "retrieval",
    "reranking",
    "evidence_check",
    "answer_generation",
    "citation_audit",
)


def fake_hit() -> RetrievalHit:
    return RetrievalHit(
        rank=1,
        chunk_id="paper_demo_chunk_00001",
        doc_id="paper_demo",
        title="Demo Paper",
        section="method",
        section_heading="Method",
        heading_path=["Method"],
        chunk_type="text",
        page_start=2,
        page_end=2,
        source_block_ids=["block_1"],
        overlap_source_block_ids=[],
        content_types=["paragraph"],
        has_equation=False,
        has_table=False,
        has_caption=False,
        text="The method retrieves passages and conditions generation on their content.",
        fusion_score=0.03,
        rerank_score=0.9,
        rerank_rank=1,
        pre_rerank_rank=1,
        reranker_backend="synthetic",
        reranker_model="synthetic-reranker",
        multi_query_fusion_score=0.04,
        multi_query_fusion_rank=1,
        query_variant_hits=[{"variant_id": "original", "rank": 1}],
        original_query_recalled=True,
        retrieval_sources=["dense", "sparse"],
    )


class FakeRetrievalEngine:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.query_rewrite_settings = SimpleNamespace(model="synthetic-rewriter", prompt_version="synthetic-rewrite-v1")
        self.reranker_settings = SimpleNamespace(model_name="synthetic-reranker", config_version="synthetic-rerank-v1")
        self.embedding_provider = SimpleNamespace(config=SimpleNamespace(model="synthetic-embedding"))

    def retrieve(self, query: str, **kwargs: Any) -> RetrievalResponse:
        if self.fail:
            raise RuntimeError("synthetic retrieval failure")
        rewrite = {
            "original_query": query,
            "normalized_query": query,
            "expansion_queries": [],
            "fallback_used": False,
            "cache_hit": True,
            "variants": [{"variant_id": "original", "query": query, "variant_type": "original"}],
        }
        return RetrievalResponse(
            query=query,
            mode="hybrid",
            top_k=10,
            candidate_k=80,
            filters=MetadataFilter(),
            hits=[fake_hit()],
            latency_ms=3.0,
            retrieval_latency_ms=1.5,
            rewrite_latency_ms=0.5,
            rerank_latency_ms=1.0,
            total_latency_ms=3.0,
            trace={
                "query_rewrite": rewrite,
                "reranker": {
                    "model": "synthetic-reranker",
                    "cache_hits": 1,
                    "cache_misses": 0,
                },
            },
        )


class FakeEvidencePipeline:
    def __init__(self, support_level: str):
        self.support_level = support_level
        self.calls = 0
        self.settings = SimpleNamespace(model="synthetic-judge", config_version="synthetic-evidence-v1")

    def assess(self, query: str, hits: list[RetrievalHit], **kwargs: Any) -> EvidenceSufficiencyResult:
        self.calls += 1
        strong = self.support_level == "strong"
        supporting = (hits[0].chunk_id,) if self.support_level != "unsupported" else ()
        return EvidenceSufficiencyResult(
            answerable=strong,
            support_level=self.support_level,
            confidence=0.9,
            reason=f"Synthetic {self.support_level} judgment.",
            supporting_chunk_ids=supporting,
            model="synthetic-judge",
            prompt_version="synthetic",
            config_version="synthetic-evidence-v1",
            timestamp=utc_timestamp(),
            cache_hit=True,
            fallback_used=False,
            fallback_reason=None,
            api_call_count=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.1,
        )


class FakeAnswerPipeline:
    def __init__(self):
        self.calls = 0
        self.settings = SimpleNamespace(model="synthetic-answer", config_version="synthetic-answer-v1")

    def generate(self, query: str, hits: list[RetrievalHit], sufficiency: Any, **kwargs: Any) -> AnswerGenerationResult:
        self.calls += 1
        hit = hits[0]
        citation = AnswerCitation(hit.chunk_id, hit.doc_id, hit.section, hit.page_start)
        return AnswerGenerationResult(
            answer="The method retrieves passages and conditions generation on them.",
            citations=(citation,),
            confidence=0.9,
            refused=False,
            refusal_reason=None,
            evidence_chunk_ids=(hit.chunk_id,),
            model="synthetic-answer",
            prompt_version="synthetic",
            config_version="synthetic-answer-v1",
            timestamp=utc_timestamp(),
            cache_hit=True,
            fallback_used=False,
            fallback_reason=None,
            api_call_count=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.1,
        )


class FakeCitationAuditPipeline:
    def __init__(self):
        self.calls = 0
        self.settings = SimpleNamespace(model="synthetic-auditor", config_version="synthetic-audit-v1")

    def audit(self, answer: AnswerGenerationResult, hits: list[RetrievalHit], **kwargs: Any) -> CitationAuditResult:
        self.calls += 1
        return CitationAuditResult(
            answer=answer.answer,
            claims=(),
            overall_grounded=True,
            unsupported_claim_count=0,
            partial_claim_count=0,
            grounding_score=1.0,
            audit_completed=True,
            audit_reason=None,
            evidence_chunk_ids=answer.evidence_chunk_ids,
            model="synthetic-auditor",
            extraction_prompt_version="synthetic",
            verification_prompt_version="synthetic",
            config_version="synthetic-audit-v1",
            timestamp=utc_timestamp(),
            cache_hit=True,
            fallback_used=False,
            fallback_reason=None,
            api_call_count=0,
            extraction_api_calls=0,
            verification_api_calls=0,
            input_tokens=0,
            output_tokens=0,
            extraction_latency_ms=0.1,
            verification_latency_ms=0.1,
            latency_ms=0.2,
        )


def run_synthetic_tests(settings: PipelineSettings) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []

    def record(name: str, passed: bool, details: dict[str, Any] | None = None) -> None:
        tests.append({"name": name, "passed": bool(passed), "details": details or {}})

    def run_case(level: str) -> tuple[dict[str, Any], FakeAnswerPipeline, FakeCitationAuditPipeline]:
        answer = FakeAnswerPipeline()
        audit = FakeCitationAuditPipeline()
        pipeline = ResearchGuardPipeline(
            settings,
            retrieval_engine=FakeRetrievalEngine(),
            evidence_pipeline=FakeEvidencePipeline(level),
            answer_pipeline=answer,
            citation_audit_pipeline=audit,
        )
        return pipeline.run(f"Synthetic {level} query"), answer, audit

    sufficient, answer, audit = run_case("strong")
    record(
        "sufficient_runs_answer_and_audit",
        sufficient["final_status"] == "grounded"
        and answer.calls == 1
        and audit.calls == 1
        and sufficient["answer_generation"]["status"] == "completed"
        and sufficient["citation_audit"]["output"]["overall_grounded"],
    )

    partial, answer, audit = run_case("partial")
    record(
        "partial_is_rejected_before_generation",
        partial["final_status"] == "rejected"
        and answer.calls == 0
        and audit.calls == 0
        and partial["answer_generation"]["status"] == "skipped",
    )

    unsupported, answer, audit = run_case("unsupported")
    record(
        "unsupported_is_rejected_before_generation",
        unsupported["final_status"] == "rejected"
        and answer.calls == 0
        and audit.calls == 0
        and unsupported["citation_audit"]["status"] == "skipped",
    )

    answer_disabled = replace(settings, answer_generation_enabled=False, citation_audit_enabled=False)
    disabled_result = ResearchGuardPipeline(
        answer_disabled,
        retrieval_engine=FakeRetrievalEngine(),
        evidence_pipeline=FakeEvidencePipeline("strong"),
    ).run("Synthetic answer-disabled query")
    record(
        "answer_generation_can_be_disabled",
        disabled_result["final_status"] == "evidence_sufficient"
        and disabled_result["answer_generation"]["status"] == "disabled"
        and disabled_result["citation_audit"]["status"] == "disabled",
    )

    failed = ResearchGuardPipeline(settings, retrieval_engine=FakeRetrievalEngine(fail=True)).run("Synthetic failure")
    record(
        "module_exception_returns_failed_schema",
        failed["final_status"] == "failed"
        and failed["retrieval"]["status"] == "failed"
        and all(name in failed for name in STAGES),
    )

    first, _, _ = run_case("strong")
    second, _, _ = run_case("strong")
    record("same_input_has_stable_semantic_output", stable_payload(first) == stable_payload(second))
    return tests


def stable_payload(value: Any) -> Any:
    volatile = {
        "start_time",
        "end_time",
        "latency_ms",
        "timestamp",
        "cache_hit",
        "api_call_count",
        "input_tokens",
        "output_tokens",
        "extraction_api_calls",
        "verification_api_calls",
        "extraction_latency_ms",
        "verification_latency_ms",
        "inference_latency_ms",
        "backend_latency_ms",
        "total_latency_ms",
        "retrieval_latency_ms",
        "rewrite_latency_ms",
        "rerank_latency_ms",
        "query_embedding_api_calls",
    }
    if isinstance(value, dict):
        return {key: stable_payload(item) for key, item in sorted(value.items()) if key not in volatile}
    if isinstance(value, list):
        return [stable_payload(item) for item in value]
    return value


def schema_failures(result: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    required = {"query", *STAGES, "final_status", "pipeline"}
    missing = sorted(required - set(result))
    if missing:
        failures.append(f"missing_top_level:{','.join(missing)}")
    stage_required = {"status", "start_time", "end_time", "latency_ms", "model", "config_version", "reason", "output"}
    for name in STAGES:
        stage = result.get(name)
        if not isinstance(stage, dict):
            failures.append(f"invalid_stage:{name}")
            continue
        missing_stage = sorted(stage_required - set(stage))
        if missing_stage:
            failures.append(f"missing_{name}:{','.join(missing_stage)}")
    pipeline = result.get("pipeline", {})
    for key in ("schema_version", "config_version", "start_time", "end_time", "latency_ms"):
        if key not in pipeline:
            failures.append(f"missing_pipeline:{key}")
    return failures


def real_case_summary(result: dict[str, Any]) -> dict[str, Any]:
    evidence = result["evidence_check"].get("output") or {}
    answer = result["answer_generation"].get("output") or {}
    audit = result["citation_audit"].get("output") or {}
    hits = (result["retrieval"].get("output") or {}).get("hits", [])
    return {
        "query": result["query"],
        "final_status": result["final_status"],
        "retrieved_chunk_ids": [item.get("chunk_id") for item in hits],
        "support_level": evidence.get("support_level"),
        "answerable": evidence.get("answerable"),
        "answer_refused": answer.get("refused"),
        "answer_citation_count": len(answer.get("citations", [])),
        "audit_completed": audit.get("audit_completed"),
        "overall_grounded": audit.get("overall_grounded"),
        "stage_statuses": {name: result[name]["status"] for name in STAGES},
        "stage_latency_ms": {name: result[name]["latency_ms"] for name in STAGES},
        "total_latency_ms": result["pipeline"]["latency_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate unified ResearchGuard Pipeline v1.")
    parser.add_argument("--config", default="configs/pipeline_v1.yaml")
    args = parser.parse_args()

    raw_config, settings = load_pipeline_settings(args.config)
    synthetic_tests = run_synthetic_tests(settings)
    validation = raw_config.get("validation", {}) or {}
    queries = {
        "sufficient": str(validation["sufficient_query"]),
        "unsupported": str(validation["unsupported_query"]),
        "partial": str(validation["partial_query"]),
    }

    engine = RetrievalEngine.from_config(settings.retrieval_config_path)
    direct = engine.retrieve(
        queries["sufficient"],
        mode=settings.retrieval_mode,
        top_k=settings.retrieval_top_k,
        candidate_k=settings.retrieval_candidate_k,
        rerank=settings.reranker_enabled,
        rerank_candidate_k=settings.reranker_candidate_k,
        rewrite=settings.rewrite_enabled,
        multi_query=settings.multi_query_enabled,
    )
    pipeline = ResearchGuardPipeline(settings, retrieval_engine=engine)
    results = {name: pipeline.run(query) for name, query in queries.items()}
    repeated = pipeline.run(queries["sufficient"])

    hard_checks = {name: 0 for name in HARD_CHECK_NAMES}
    schema_issues = {name: schema_failures(result) for name, result in results.items()}
    schema_issues = {name: issues for name, issues in schema_issues.items() if issues}
    if schema_issues:
        hard_checks["pipeline_schema_failure"] = len(schema_issues)
        hard_checks["result_schema_failure"] = sum(len(items) for items in schema_issues.values())

    hard_checks["module_failure"] = sum(
        1
        for result in results.values()
        for name in STAGES
        if result[name]["status"] in {"failed", "fallback"}
    )
    for name in ("unsupported", "partial"):
        result = results[name]
        answer_output = result["answer_generation"].get("output")
        if result["answer_generation"]["status"] != "skipped" or answer_output is not None:
            hard_checks["evidence_gate_bypass"] += 1
        if answer_output and not answer_output.get("refused", False):
            hard_checks["unsupported_generation"] += 1

    sufficient = results["sufficient"]
    answer_output = sufficient["answer_generation"].get("output") or {}
    audit_output = sufficient["citation_audit"].get("output") or {}
    if (
        sufficient["final_status"] != "grounded"
        or not answer_output.get("citations")
        or not audit_output.get("audit_completed")
        or not audit_output.get("overall_grounded")
    ):
        hard_checks["citation_missing"] = 1

    stable_first = stable_payload(sufficient)
    stable_second = stable_payload(repeated)
    cache_flags = {
        "rewrite": bool((repeated["rewrite"].get("output") or {}).get("cache_hit")),
        "evidence_check": bool((repeated["evidence_check"].get("output") or {}).get("cache_hit")),
        "answer_generation": bool((repeated["answer_generation"].get("output") or {}).get("cache_hit")),
        "citation_audit": bool((repeated["citation_audit"].get("output") or {}).get("cache_hit")),
    }
    reranker_output = repeated["reranking"].get("output") or {}
    cache_flags["reranking"] = int(reranker_output.get("cache_misses", 1)) == 0
    if stable_first != stable_second or not all(cache_flags.values()):
        hard_checks["cache_failure"] = 1

    direct_ids = [hit.chunk_id for hit in direct.hits]
    pipeline_ids = [
        hit["chunk_id"] for hit in (sufficient["retrieval"].get("output") or {}).get("hits", [])
    ]
    if direct_ids != pipeline_ids:
        hard_checks["retrieval_regression"] = 1

    synthetic_failure_count = sum(1 for item in synthetic_tests if not item["passed"])
    if synthetic_failure_count:
        hard_checks["module_failure"] += synthetic_failure_count
    status = "PASS" if all(value == 0 for value in hard_checks.values()) else "FAIL"

    output_dir = Path(validation.get("output_directory", "outputs/pipeline_validation_v1"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config)),
        "synthetic_tests": synthetic_tests,
        "real_cases": {name: real_case_summary(result) for name, result in results.items()},
        "repeat_cache_hits": cache_flags,
        "retrieval_regression": {
            "direct_chunk_ids": direct_ids,
            "pipeline_chunk_ids": pipeline_ids,
            "identical": direct_ids == pipeline_ids,
        },
        "schema_issues": schema_issues,
        "hard_checks": hard_checks,
    }
    (output_dir / "pipeline_validation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Pipeline v1 Validation",
        "",
        f"Final status: **{status}**",
        "",
        "## Synthetic tests",
        "",
    ]
    lines.extend(f"- {'PASS' if item['passed'] else 'FAIL'}: `{item['name']}`" for item in synthetic_tests)
    lines.extend(["", "## Real E2E cases", ""])
    for name, result in report["real_cases"].items():
        lines.append(
            f"- `{name}`: final=`{result['final_status']}`, evidence=`{result['support_level']}`, "
            f"latency={result['total_latency_ms']:.2f} ms"
        )
    lines.extend(["", "## Hard checks", ""])
    lines.extend(f"- `{name}`: {value}" for name, value in hard_checks.items())
    (output_dir / "pipeline_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
