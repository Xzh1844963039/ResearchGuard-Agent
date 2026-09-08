# C:\Users\18449\Desktop\researchguard_workspace\scripts\compare_planners_v2.py
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.agent import BoundedResearchAgentController  # noqa: E402
from researchguard.evaluation.planner_comparison_v2 import (  # noqa: E402
    build_open_goal_comparison,
    evaluate_open_goal_state,
    load_open_goal_cases,
    render_open_goal_comparison,
)


DEFAULT_CASES = PROJECT_ROOT / "data" / "eval" / "planner_v2" / "open_goal_cases.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "planner_comparison_v2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run live deterministic vs Hybrid Planner open-goal A/B v2."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    cases = load_open_goal_cases(args.cases)[: max(1, args.limit)]
    controllers = {
        "deterministic": BoundedResearchAgentController(
            planner_config_path=PROJECT_ROOT / "configs" / "planner_deterministic.yaml",
            memory_enabled=False,
        ),
        "hybrid": BoundedResearchAgentController(
            planner_config_path=PROJECT_ROOT / "configs" / "planner_v2.yaml",
            memory_enabled=False,
        ),
    }
    rows = {"deterministic": [], "hybrid": []}
    for index, case in enumerate(cases, start=1):
        for mode in ("deterministic", "hybrid"):
            started = time.perf_counter()
            state = controllers[mode].run(case.query, task_type=case.task_type)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            rows[mode].append(
                evaluate_open_goal_state(case, state, latency_ms=elapsed_ms)
            )
            print(
                f"[{index}/{len(cases)}] {case.case_id} {mode}: "
                f"status={state.status}, subgoals={len(state.subgoals)}, "
                f"planner={state.planner_metadata.get('mode')}",
                flush=True,
            )

    report = build_open_goal_comparison(rows["deterministic"], rows["hybrid"])
    markdown = render_open_goal_comparison(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "planner_comparison_v2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "planner_comparison_v2_report.md").write_text(
        markdown,
        encoding="utf-8",
        newline="\n",
    )
    print(markdown)
    if report["conclusion"] == "HYBRID_UNSAFE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
