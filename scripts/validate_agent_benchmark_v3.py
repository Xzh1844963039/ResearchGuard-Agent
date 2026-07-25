# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_agent_benchmark_v3.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.agent import BoundedResearchAgentController  # noqa: E402
from researchguard.evaluation import (  # noqa: E402
    build_benchmark_v3_report,
    evaluate_benchmark_v3_run,
    load_benchmark_v3_cases,
    render_benchmark_v3_markdown,
)
from researchguard.tools import (  # noqa: E402
    CitationAuditTool,
    EvidenceTool,
    GuardedAnswerTool,
    RetrievalTool,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from researchguard.tools.scholarly_search_tool import (  # noqa: E402
    ScholarlySearchTool,
)


DEFAULT_CASES = (
    PROJECT_ROOT / "data" / "eval" / "agent_benchmark_v3" / "cases.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "agent_benchmark_v3"
DEFAULT_PIPELINE_CONFIG = PROJECT_ROOT / "configs" / "pipeline_v1.yaml"
DEFAULT_PLANNER_CONFIG = (
    PROJECT_ROOT / "configs" / "planner_deterministic.yaml"
)


class OfflineScholarlyTool:
    name = "search_scholarly_sources"
    version = "benchmark-offline-v1"
    description = (
        "Offline metadata boundary used to isolate Agent/Evidence Engine metrics "
        "from third-party scholarly API availability."
    )

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema={"query": "non-empty string"},
        )

    def invoke(self, **kwargs: Any) -> ToolResult:
        return ToolResult.create(
            status="success",
            message="Offline scholarly discovery boundary completed.",
            tool_name=self.name,
            tool_version=self.version,
            latency_ms=0.0,
            data={
                "query": str(kwargs.get("query", "")),
                "candidate_papers": [],
                "metadata_only": True,
                "evidence_eligible": False,
                "provider_mode": "offline_benchmark",
                "api_call_count": 0,
            },
        )


def build_real_evidence_registry(
    config_path: Path,
    *,
    online_discovery: bool,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(RetrievalTool(config_path=config_path))
    registry.register(EvidenceTool(config_path=config_path))
    registry.register(GuardedAnswerTool(config_path=config_path))
    registry.register(CitationAuditTool(config_path=config_path))
    registry.register(
        ScholarlySearchTool()
        if online_discovery
        else OfflineScholarlyTool()
    )
    return registry


def workflow_input(case: Any) -> dict[str, Any]:
    if case.expected_task_type != "paper_comparison":
        return {}
    return {
        "papers": [
            {"name": doc_id.replace("_", " "), "doc_id": doc_id}
            for doc_id in case.expected_doc_ids[:2]
        ],
        "dimensions": list(case.metadata.get("dimensions", ())),
        "candidate_limit": 3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Agent Benchmark v3 through the real Planner, Controller, "
            "Skill Executor and Evidence Engine."
        )
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=DEFAULT_PIPELINE_CONFIG,
    )
    parser.add_argument(
        "--planner-config",
        type=Path,
        default=DEFAULT_PLANNER_CONFIG,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-cost-per-million", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million", type=float, default=0.0)
    parser.add_argument(
        "--online-discovery",
        action="store_true",
        help=(
            "Call the live scholarly provider. By default metadata discovery is "
            "kept offline so third-party latency does not contaminate Agent metrics."
        ),
    )
    args = parser.parse_args()

    cases = load_benchmark_v3_cases(args.benchmark)
    results: list[dict[str, Any]] = []
    execution_exceptions = 0
    controller = BoundedResearchAgentController(
        registry=build_real_evidence_registry(
            args.pipeline_config,
            online_discovery=args.online_discovery,
        ),
        config_path=args.pipeline_config,
        planner_config_path=args.planner_config,
        memory_enabled=False,
    )
    for index, case in enumerate(cases, start=1):
        try:
            state = controller.run(
                case.query,
                workflow_input=workflow_input(case),
            )
            result = evaluate_benchmark_v3_run(
                state,
                case,
                input_cost_per_million=args.input_cost_per_million,
                output_cost_per_million=args.output_cost_per_million,
            )
        except Exception as exc:
            execution_exceptions += 1
            result = {
                "case_id": case.case_id,
                "category": case.category,
                "difficulty": case.difficulty,
                "passed": False,
                "expected": case.to_dict(),
                "observed": {},
                "metrics": {},
                "issues": ["execution_exception"],
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        results.append(result)
        print(
            f"[{index:02d}/{len(cases)}] {case.case_id}: "
            f"{'PASS' if result['passed'] else 'CHECK'}",
            flush=True,
        )

    report = build_benchmark_v3_report(results)
    report["execution_profile"] = {
        "planner_config": str(args.planner_config),
        "evidence_engine": "real",
        "scholarly_discovery": (
            "online" if args.online_discovery else "offline_boundary"
        ),
    }
    report["hard_checks"]["execution_exception"] = execution_exceptions
    if execution_exceptions:
        report["conclusion"] = "FAIL"
    markdown = render_benchmark_v3_markdown(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "agent_benchmark_v3_report.json"
    markdown_path = args.output_dir / "agent_benchmark_v3_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    print(markdown)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    if report["conclusion"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
