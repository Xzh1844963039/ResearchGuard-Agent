# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_agent_benchmark_v2.py
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.agent import BoundedResearchAgentController
from researchguard.evaluation import AgentEvaluationCase, AgentEvaluator
from researchguard.tools import (
    EvidenceBundle,
    GateDecision,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "eval" / "agent_benchmark_v2" / "cases.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "agent_benchmark_v2"
EVIDENCE_A = {
    "chunk_id": "paper-crag::chunk-7",
    "doc_id": "paper-crag",
    "section": "method",
    "page": 5,
    "content": "CRAG uses a retrieval evaluator before corrective generation.",
    "source": "paper_crag.pdf",
    "score": 0.94,
    "provenance": {"source_block_ids": ["p5-b2"]},
}
EVIDENCE_B = {
    "chunk_id": "paper-rag::chunk-4",
    "doc_id": "paper-rag",
    "section": "method",
    "page": 3,
    "content": "Standard RAG retrieves passages before generation.",
    "source": "paper_rag.pdf",
    "score": 0.90,
    "provenance": {"source_block_ids": ["p3-b4"]},
}
CANDIDATES = [
    {
        "schema_version": "researchguard.scholar_paper.v1",
        "title": "Corrective Retrieval Augmented Generation",
        "authors": ["Synthetic Author"],
        "year": 2024,
        "venue": "arXiv",
        "doi": None,
        "url": "https://example.org/crag",
        "abstract": "Metadata-only candidate.",
        "source": "arxiv",
        "paper_id": "arxiv:crag",
        "metadata": {},
        "source_type": "arxiv",
        "metadata_only": True,
        "retrieved_at": "2026-01-01T00:00:00+00:00",
    }
]


class ScenarioTool:
    version = "benchmark-v2"
    description = "Deterministic Tool used by the integration benchmark."

    def __init__(self, name: str, handler: Callable[..., Any]):
        self.name = name
        self.handler = handler

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema={},
        )

    def invoke(self, **kwargs: Any) -> Any:
        return self.handler(**kwargs)


def tool_result(
    name: str,
    *,
    status: str = "success",
    data: Mapping[str, Any] | None = None,
    reason: str | None = None,
    error: ToolError | None = None,
) -> ToolResult:
    return ToolResult.create(
        status=status,
        message=f"benchmark_{status}",
        reason=reason,
        tool_name=name,
        tool_version="benchmark-v2",
        latency_ms=1.0,
        data=dict(data or {}),
        error=error,
    )


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"Benchmark line {line_number} must be an object.")
        cases.append(value)
    if len(cases) < 30:
        raise ValueError("Agent Benchmark v2 requires at least 30 cases.")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Agent Benchmark v2 contains duplicate case_id values.")
    categories = Counter(str(case.get("category")) for case in cases)
    required = {"task_routing": 10, "replanning": 10, "safety_boundary": 10}
    if any(categories[name] < minimum for name, minimum in required.items()):
        raise ValueError(f"Benchmark category coverage is incomplete: {dict(categories)}")
    return cases


def build_registry(scenario: str) -> ToolRegistry:
    registry = ToolRegistry()
    counters = Counter()

    def retrieve(**kwargs: Any) -> ToolResult:
        counters["retrieve"] += 1
        attempt = counters["retrieve"]
        if scenario == "retrieval_error_then_success" and attempt == 1:
            return tool_result(
                "retrieve_evidence",
                status="failed",
                reason="retrieval_failed",
                error=ToolError(
                    code="retrieval_failed",
                    category="retrieval_failure",
                    message="Synthetic transient retrieval failure.",
                    retryable=True,
                ),
            )
        if scenario == "retrieval_empty_then_success" and attempt == 1:
            return tool_result("retrieve_evidence", data={"evidence": []})
        if scenario == "retrieval_empty_twice" and attempt <= 2:
            return tool_result("retrieve_evidence", data={"evidence": []})
        filters = kwargs.get("filters") or {}
        doc_ids = filters.get("doc_ids", []) if isinstance(filters, Mapping) else []
        if doc_ids == ["paper-crag"]:
            evidence = [EVIDENCE_A]
        elif doc_ids == ["paper-rag"]:
            evidence = [EVIDENCE_B]
        else:
            evidence = [EVIDENCE_A]
        bundle = EvidenceBundle.create(
            query=str(kwargs["query"]),
            evidence=evidence,
            retrieval_metadata={"mode": "hybrid", "attempt": attempt},
            provenance={"benchmark": "agent_benchmark_v2"},
        )
        return tool_result(
            "retrieve_evidence",
            data={
                "query": kwargs["query"],
                "evidence": evidence,
                "evidence_bundle": bundle.to_dict(),
                "retrieval": {"api_call_count": 0},
            },
        )

    def assess(**kwargs: Any) -> ToolResult:
        counters["assess"] += 1
        bundle = EvidenceBundle.from_mapping(kwargs["evidence_bundle"])
        if scenario == "unsupported":
            level = "unsupported"
        elif scenario == "partial_persistent":
            level = "partial"
        elif scenario == "partial_then_success" and counters["assess"] == 1:
            level = "partial"
        else:
            level = "strong"
        supporting = list(bundle.chunk_ids) if level in {"strong", "partial"} else []
        assessment = {
            "support_level": level,
            "answerable": level == "strong",
            "confidence": 0.9,
            "reason": f"benchmark_{level}",
            "supporting_chunk_ids": supporting,
            "model": "deterministic-benchmark",
            "prompt_version": "agent_benchmark_v2",
            "config_version": "agent_benchmark_v2.0",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "api_call_count": 0,
        }
        gate = GateDecision.from_assessment(
            evidence_bundle_id=bundle.bundle_id,
            assessment=assessment,
        )
        return tool_result(
            "assess_evidence",
            status="success" if level == "strong" else "rejected",
            reason=None if level == "strong" else "insufficient_evidence",
            data={
                "assessment": assessment,
                "gate_decision": gate.to_dict(),
                "evidence_bundle_id": bundle.bundle_id,
            },
        )

    def generate(**kwargs: Any) -> ToolResult:
        bundle = EvidenceBundle.from_mapping(kwargs["evidence_bundle"])
        if scenario == "answer_failure":
            return tool_result(
                "generate_grounded_answer",
                status="failed",
                reason="answer_generation_failed",
                error=ToolError(
                    code="answer_generation_failed",
                    category="api_failure",
                    message="Synthetic answer backend failure.",
                    retryable=False,
                ),
            )
        if scenario == "invalid_answer_output":
            return tool_result(
                "generate_grounded_answer",
                data={"evidence_bundle_id": bundle.bundle_id},
            )
        citations = [
            {
                "chunk_id": record.chunk_id,
                "doc_id": record.doc_id,
                "section": record.section,
                "page": record.page,
            }
            for record in bundle.evidence_records
        ]
        answer = {
            "answer": "The answer is grounded in the supplied canonical evidence.",
            "citations": citations,
            "confidence": 0.9,
            "refused": False,
            "refusal_reason": None,
            "evidence_chunk_ids": list(bundle.chunk_ids),
            "model": "deterministic-benchmark",
            "prompt_version": "agent_benchmark_v2",
            "config_version": "agent_benchmark_v2.0",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "cache_hit": False,
            "fallback_used": False,
            "fallback_reason": None,
            "api_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 1.0,
        }
        return tool_result(
            "generate_grounded_answer",
            data={
                "answer": answer,
                "evidence_bundle_id": bundle.bundle_id,
                "generation": {"api_call_count": 0},
            },
        )

    def audit(**kwargs: Any) -> Any:
        bundle = EvidenceBundle.from_mapping(kwargs["evidence_bundle"])
        if scenario == "invalid_audit_output":
            return {"invalid": "not-a-ToolResult"}
        grounded = scenario != "citation_failure"
        return tool_result(
            "audit_answer",
            status="success" if grounded else "rejected",
            reason=None if grounded else "answer_not_fully_grounded",
            data={
                "audit": {
                    "audit_completed": True,
                    "overall_grounded": grounded,
                    "claims": [],
                    "evidence_chunk_ids": list(bundle.chunk_ids),
                },
                "evidence_bundle_id": bundle.bundle_id,
            },
        )

    registry.register(ScenarioTool("retrieve_evidence", retrieve))
    registry.register(ScenarioTool("assess_evidence", assess))
    registry.register(ScenarioTool("generate_grounded_answer", generate))
    registry.register(ScenarioTool("audit_answer", audit))
    registry.register(
        ScenarioTool(
            "search_scholarly_sources",
            lambda **_: tool_result(
                "search_scholarly_sources",
                data={
                    "candidate_papers": CANDIDATES,
                    "metadata_only": True,
                    "evidence_eligible": False,
                },
            ),
        )
    )
    return registry


def workflow_input(case: Mapping[str, Any]) -> dict[str, Any]:
    if case.get("expected_workflow") == "paper_comparison":
        return {
            "papers": [
                {"name": "Paper A", "doc_id": "paper-crag"},
                {"name": "Paper B", "doc_id": "paper-rag"},
            ]
        }
    return {}


def evaluate_cases(cases: list[dict[str, Any]]) -> tuple[Any, list[tuple[Any, Any]]]:
    runs = []
    state_cases: list[tuple[Any, Any]] = []
    registered_tools: tuple[str, ...] = ()
    for value in cases:
        registry = build_registry(str(value["scenario"]))
        registered_tools = registry.names
        controller = BoundedResearchAgentController(
            registry=registry,
            memory_enabled=False,
        )
        state = controller.run(
            str(value["query"]),
            workflow_input=workflow_input(value),
        )
        relevant_ids: tuple[str, ...]
        if value["scenario"] == "retrieval_empty_twice":
            relevant_ids = ()
        elif value.get("expected_workflow") == "paper_comparison":
            relevant_ids = (EVIDENCE_A["chunk_id"], EVIDENCE_B["chunk_id"])
        else:
            relevant_ids = (EVIDENCE_A["chunk_id"],)
        case = AgentEvaluationCase(
            case_id=str(value["case_id"]),
            query=str(value["query"]),
            expected_task_type=str(value["expected_task_type"]),
            expected_workflow=value.get("expected_workflow"),
            expected_tools=tuple(str(item) for item in value["expected_tools"]),
            forbidden_tools=("generate_grounded_answer",)
            if value["scenario"] in {"unsupported", "partial_persistent", "retrieval_empty_twice"}
            else (),
            relevant_evidence_ids=relevant_ids,
            expected_status=str(value["expected_status"]),
            expected_plan_revisions=int(value["expected_revisions"]),
            metadata={
                "category": value["category"],
                "scenario": value["scenario"],
                "executes_real_controller": True,
            },
            version="2.0.0",
        )
        runs.append((state, case, None))
        state_cases.append((state, case))
    return AgentEvaluator(registered_tools).evaluate_many(runs), state_cases


def render_report(report: Any, state_cases: list[tuple[Any, Any]]) -> str:
    categories: dict[str, list[bool]] = {}
    for result, (_, case) in zip(report.results, state_cases):
        category = str(case.metadata["category"])
        categories.setdefault(category, []).append(result.passed)
    lines = [
        "# ResearchGuard Agent Benchmark v2 Report",
        "",
        f"- Cases: `{report.case_count}`",
        f"- Passed: `{report.passed_count}`",
        f"- Final status: `{'PASS' if report.passed else 'FAIL'}`",
        "- Execution: real Planner -> Controller -> Workflow/Tool Registry calls",
        "- LLM/API usage: deterministic integration doubles; API calls remain measured as 0",
        "",
        "## Category Coverage",
        "",
        "| Category | Cases | Pass rate |",
        "|---|---:|---:|",
    ]
    for category, values in sorted(categories.items()):
        lines.append(f"| {category} | {len(values)} | {sum(values) / len(values):.4f} |")
    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for name, value in sorted(report.aggregate_metrics.items()):
        rendered = "N/A" if value is None else (
            f"{value:.4f}" if isinstance(value, float) else str(value)
        )
        lines.append(f"| `{name}` | {rendered} |")
    lines.extend(["", "## Cases", ""])
    for result in report.results:
        observed = result.observed
        lines.append(
            f"- `{result.case_id}`: `{'PASS' if result.passed else 'FAIL'}`; "
            f"task=`{observed.get('task_type')}`; workflow=`{observed.get('workflow') or 'none'}`; "
            f"status=`{observed.get('status')}`; revisions=`{observed.get('plan_revision_count')}`; "
            f"tools=`{', '.join(observed.get('tools', []))}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 30-case ResearchGuard Agent Benchmark v2."
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases = load_cases(args.benchmark)
    report, state_cases = evaluate_cases(cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "agent_benchmark_v2_report.json"
    markdown_path = args.output_dir / "agent_benchmark_v2_report.md"
    json_path.write_text(report.to_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_report(report, state_cases),
        encoding="utf-8",
        newline="\n",
    )
    print(render_report(report, state_cases))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
