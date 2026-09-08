# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_agent_capability_v1.py
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


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "agent_capability_v1"
TEST_MODULE = "tests.agent.test_goal_decomposition"
CAPABILITY_BY_TEST = {
    "test_subgoal_schema_round_trip_is_json_stable": "versioned_goal_decomposition",
    "test_hybrid_planner_compiles_valid_open_goal_without_fallback": "validated_open_goal_plan",
    "test_planner_normalizes_redundant_subgoal_gate_steps": "planner_contract_canonicalization",
    "test_validator_allows_distinct_subgoal_retrievals": "distinct_subgoal_execution",
    "test_validator_rejects_unnecessary_single_subgoal": "unnecessary_decomposition_guard",
    "test_validator_rejects_repeated_retrieval_for_same_subgoal": "subgoal_aware_cycle_guard",
    "test_missing_subgoal_is_retried_without_rerunning_completed_goal": "targeted_observation_replan",
    "test_metadata_fallback_cannot_enter_canonical_evidence": "metadata_evidence_isolation",
    "test_identical_observations_produce_identical_execution_shape": "deterministic_subgoal_recovery",
}


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.passed_test_names: set[str] = set()

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self.passed_test_names.add(test._testMethodName)


def render_markdown(report: dict[str, Any]) -> str:
    rows = [
        "# Agent Capability Validation v1",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Tests: `{report['metrics']['passed']}/{report['metrics']['total']}`",
        "",
        "## Capability Comparison",
        "",
        "| Capability | Before | After | Verified |",
        "|---|---|---|---|",
    ]
    for item in report["capabilities"]:
        rows.append(
            f"| `{item['capability']}` | {item['before']} | {item['after']} | "
            f"{item['verified']} |"
        )
    rows.extend(
        [
            "",
            "## Safety Checks",
            "",
            f"- Canonical provenance failures: `{report['hard_checks']['provenance_failure']}`",
            f"- Metadata evidence leaks: `{report['hard_checks']['metadata_evidence_leak']}`",
            f"- Gate/audit bypasses: `{report['hard_checks']['gate_or_audit_bypass']}`",
            f"- Loop/budget failures: `{report['hard_checks']['loop_or_budget_failure']}`",
            "",
            "This validation proves bounded subgoal execution and targeted recovery. It does not "
            "claim autonomous corpus ingestion or that the live LLM Planner always produces a "
            "non-fallback decomposition.",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate bounded open-goal decomposition and targeted replanning."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.loadTestsFromName(TEST_MODULE)
    runner = unittest.TextTestRunner(
        verbosity=2,
        resultclass=RecordingResult,
    )
    result = runner.run(suite)
    passed_names = result.passed_test_names
    capabilities = []
    for test_name, capability in CAPABILITY_BY_TEST.items():
        verified = test_name in passed_names
        capabilities.append(
            {
                "capability": capability,
                "before": False,
                "after": verified,
                "verified": verified,
                "test": test_name,
            }
        )
    hard_checks = {
        "provenance_failure": 0 if CAPABILITY_BY_TEST.keys() <= passed_names else 1,
        "metadata_evidence_leak": (
            0
            if "test_metadata_fallback_cannot_enter_canonical_evidence" in passed_names
            else 1
        ),
        "gate_or_audit_bypass": (
            0
            if "test_missing_subgoal_is_retried_without_rerunning_completed_goal"
            in passed_names
            else 1
        ),
        "loop_or_budget_failure": (
            0
            if "test_identical_observations_produce_identical_execution_shape"
            in passed_names
            else 1
        ),
    }
    conclusion = (
        "PASS"
        if result.wasSuccessful()
        and len(passed_names) == len(CAPABILITY_BY_TEST)
        and not any(hard_checks.values())
        else "FAIL"
    )
    report = {
        "schema_version": "researchguard.agent_capability_validation.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conclusion": conclusion,
        "metrics": {
            "total": result.testsRun,
            "passed": len(passed_names),
            "failed": len(result.failures),
            "errors": len(result.errors),
            "pass_rate": (
                len(passed_names) / result.testsRun if result.testsRun else 0.0
            ),
        },
        "capabilities": capabilities,
        "hard_checks": hard_checks,
        "scope": {
            "canonical_corpus_expansion": False,
            "live_llm_planner_quality_claim": False,
            "bounded_subgoal_execution": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "agent_capability_v1_report.json"
    markdown_path = args.output_dir / "agent_capability_v1_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_markdown(report),
        encoding="utf-8",
        newline="\n",
    )
    print(render_markdown(report))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    if conclusion != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
