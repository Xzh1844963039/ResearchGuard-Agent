# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_agent_capability_v2.py
from __future__ import annotations

import argparse
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "agent_capability_v2"
TEST_MODULE = "tests.agent.test_goal_decomposition"
CAPABILITY_BY_TEST = {
    "test_subgoal_schema_round_trip_is_json_stable": "versioned_goal_decomposition",
    "test_hybrid_planner_compiles_valid_open_goal_without_fallback": "validated_open_goal_plan",
    "test_validator_rejects_unnecessary_single_subgoal": "unnecessary_decomposition_guard",
    "test_hybrid_planner_collapses_single_subgoal_to_simple_plan": "simple_plan_canonicalization",
    "test_validator_rejects_repeated_retrieval_for_same_subgoal": "subgoal_cycle_guard",
    "test_missing_subgoal_is_retried_without_rerunning_completed_goal": "targeted_retrieval_recovery",
    "test_subgoal_gate_uses_completion_criterion_and_refines_only_that_query": "subgoal_aware_evidence_gate",
    "test_partial_subgoal_cannot_be_hidden_by_aggregate_answer": "partial_subgoal_fail_closed",
    "test_aggregate_gate_gap_inserts_bounded_runtime_subgoal": "structural_gap_subgoal_insertion",
    "test_metadata_fallback_cannot_enter_canonical_evidence": "metadata_evidence_isolation",
    "test_identical_observations_produce_identical_execution_shape": "deterministic_bounded_adaptation",
    "test_internal_skill_calls_cannot_exceed_tool_budget": "strict_internal_tool_budget",
}


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.passed_test_names: set[str] = set()

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self.passed_test_names.add(test._testMethodName)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Agent Capability Validation v2",
        "",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Tests: `{report['metrics']['passed']}/{report['metrics']['total']}`",
        "",
        "| Capability | Verified |",
        "|---|---|",
    ]
    for item in report["capabilities"]:
        lines.append(f"| `{item['capability']}` | `{item['verified']}` |")
    lines.extend(["", "## Hard Checks", ""])
    for name, value in report["hard_checks"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(
        [
            "",
            "This suite validates Agent-layer subgoal reasoning and bounded plan adaptation. "
            "It does not claim autonomous corpus acquisition or unrestricted web evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate subgoal-aware Gate and structural replanning invariants."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.loadTestsFromName(TEST_MODULE)
    result = unittest.TextTestRunner(
        verbosity=2,
        resultclass=RecordingResult,
    ).run(suite)
    passed = result.passed_test_names
    capabilities = [
        {
            "capability": capability,
            "verified": test_name in passed,
            "test": test_name,
        }
        for test_name, capability in CAPABILITY_BY_TEST.items()
    ]
    hard_checks = {
        "partial_subgoal_hidden": int(
            "test_partial_subgoal_cannot_be_hidden_by_aggregate_answer" not in passed
        ),
        "gate_or_audit_bypass": int(
            "test_subgoal_gate_uses_completion_criterion_and_refines_only_that_query"
            not in passed
        ),
        "metadata_evidence_leak": int(
            "test_metadata_fallback_cannot_enter_canonical_evidence" not in passed
        ),
        "unbounded_or_nondeterministic_revision": int(
            "test_identical_observations_produce_identical_execution_shape" not in passed
        ),
    }
    conclusion = (
        "PASS"
        if result.wasSuccessful()
        and all(item["verified"] for item in capabilities)
        and not any(hard_checks.values())
        else "FAIL"
    )
    report = {
        "schema_version": "researchguard.agent_capability_validation.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conclusion": conclusion,
        "metrics": {
            "total": result.testsRun,
            "passed": len(passed),
            "failed": len(result.failures),
            "errors": len(result.errors),
        },
        "capabilities": capabilities,
        "hard_checks": hard_checks,
        "scope": {
            "subgoal_aware_evidence_reasoning": True,
            "bounded_structural_replanning": True,
            "canonical_corpus_expansion": False,
        },
    }
    markdown = render_markdown(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "agent_capability_v2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "agent_capability_v2_report.md").write_text(
        markdown,
        encoding="utf-8",
        newline="\n",
    )
    print(markdown)
    if conclusion != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
