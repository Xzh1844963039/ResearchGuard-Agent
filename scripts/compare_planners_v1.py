# scripts/compare_planners_v1.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.agent import BoundedResearchAgentController  # noqa: E402
from researchguard.evaluation import (  # noqa: E402
    build_planner_comparison_report,
    evaluate_planner_outcome,
    load_benchmark_v3_cases,
    render_planner_comparison_markdown,
)


DEFAULT_CASES = (
    PROJECT_ROOT / "data" / "eval" / "agent_benchmark_v3" / "cases.jsonl"
)
DEFAULT_PIPELINE = PROJECT_ROOT / "configs" / "pipeline_v1.yaml"
DEFAULT_DETERMINISTIC = (
    PROJECT_ROOT / "configs" / "planner_deterministic.yaml"
)
DEFAULT_HYBRID = PROJECT_ROOT / "configs" / "planner_v2.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "planner_comparison_v1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare deterministic and Hybrid LLM planners."
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument(
        "--deterministic-config",
        type=Path,
        default=DEFAULT_DETERMINISTIC,
    )
    parser.add_argument("--hybrid-config", type=Path, default=DEFAULT_HYBRID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of paired labeled cases to compare.",
    )
    args = parser.parse_args()

    cases = load_benchmark_v3_cases(args.benchmark)[: max(1, args.limit)]
    deterministic_controller = BoundedResearchAgentController(
        config_path=args.pipeline_config,
        planner_config_path=args.deterministic_config,
        memory_enabled=False,
    )
    hybrid_controller = BoundedResearchAgentController(
        config_path=args.pipeline_config,
        planner_config_path=args.hybrid_config,
        memory_enabled=False,
    )
    deterministic_rows = []
    hybrid_rows = []
    for index, case in enumerate(cases, start=1):
        deterministic = deterministic_controller.planner.generate_plan(
            case.query
        )
        hybrid = hybrid_controller.planner.generate_plan(case.query)
        deterministic_rows.append(
            evaluate_planner_outcome(case, deterministic)
        )
        hybrid_rows.append(evaluate_planner_outcome(case, hybrid))
        print(
            f"[{index:02d}/{len(cases)}] {case.case_id}: "
            f"deterministic={deterministic.mode}, hybrid={hybrid.mode}",
            flush=True,
        )

    report = build_planner_comparison_report(
        deterministic_rows,
        hybrid_rows,
    )
    markdown = render_planner_comparison_markdown(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "planner_comparison_report.json"
    markdown_path = args.output_dir / "planner_comparison_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    print(markdown)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
