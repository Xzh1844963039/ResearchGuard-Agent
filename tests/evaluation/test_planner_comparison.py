# C:\Users\18449\Desktop\researchguard_workspace\tests\evaluation\test_planner_comparison.py
from __future__ import annotations

import unittest

from researchguard.evaluation import (
    build_planner_comparison_report,
    render_planner_comparison_markdown,
)


def row(
    case_id: str,
    *,
    routing: float,
    skills: float,
    fallback: bool = False,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "observed_skills": ["retrieve_evidence", "generate_report"],
        "task_routing_accuracy": routing,
        "skill_sequence_quality": skills,
        "plan_completeness": skills,
        "tool_efficiency": 0.5,
        "plan_validity": 1.0,
        "latency_ms": 1.0,
        "api_call_count": int(not fallback),
        "input_tokens": 10 if not fallback else 0,
        "output_tokens": 5 if not fallback else 0,
        "fallback_used": fallback,
        "mode": "deterministic_fallback" if fallback else "hybrid_llm",
    }


class PlannerComparisonTests(unittest.TestCase):
    def test_reports_hybrid_gain_only_for_non_fallback_improvement(self) -> None:
        report = build_planner_comparison_report(
            [row("one", routing=0.0, skills=0.0)],
            [row("one", routing=1.0, skills=1.0)],
        )

        self.assertEqual(report["conclusion"], "HYBRID_BETTER")
        self.assertEqual(
            report["hybrid_lift"]["task_routing_accuracy"],
            1.0,
        )
        self.assertIn(
            "HYBRID_BETTER",
            render_planner_comparison_markdown(report),
        )

    def test_all_fallback_runs_are_inconclusive(self) -> None:
        report = build_planner_comparison_report(
            [row("one", routing=1.0, skills=1.0)],
            [row("one", routing=1.0, skills=1.0, fallback=True)],
        )

        self.assertEqual(report["conclusion"], "INCONCLUSIVE_FALLBACK_ONLY")

    def test_rejects_unpaired_case_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "case order"):
            build_planner_comparison_report(
                [row("one", routing=1.0, skills=1.0)],
                [row("two", routing=1.0, skills=1.0)],
            )


if __name__ == "__main__":
    unittest.main()
