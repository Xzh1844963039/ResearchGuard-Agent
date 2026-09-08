# C:\Users\18449\Desktop\researchguard_workspace\tests\evaluation\test_planner_comparison_v2.py
from __future__ import annotations

import unittest
from types import SimpleNamespace

from researchguard.evaluation.planner_comparison_v2 import (
    OpenGoalCase,
    build_open_goal_comparison,
    evaluate_open_goal_state,
)


def state_with_subgoals(*, subgoals: list[dict[str, str]]) -> SimpleNamespace:
    return SimpleNamespace(
        subgoals=subgoals,
        task_type="qa",
        status="completed",
        reason=None,
        planner_metadata={
            "mode": "hybrid_llm",
            "fallback_used": False,
            "validation": {"valid": True},
            "api_call_count": 1,
            "input_tokens": 100,
            "output_tokens": 30,
        },
        plan_revisions=[],
        tool_history=[],
        skill_history=[],
        evidence=[],
        gate_decision=None,
        answer=None,
        audit_result=None,
    )


class PlannerComparisonV2Tests(unittest.TestCase):
    def test_compound_goal_requires_distinct_concept_covering_subgoals(self) -> None:
        case = OpenGoalCase(
            case_id="compound",
            query="Explain mechanism and experiments.",
            task_type="qa",
            complexity="compound",
            min_subgoals=2,
            max_subgoals=3,
            required_concept_groups=(("mechanism",), ("experiment",)),
        )
        state = state_with_subgoals(
            subgoals=[
                {
                    "question": "What is the mechanism?",
                    "objective": "Explain mechanism",
                    "success_criterion": "method evidence",
                },
                {
                    "question": "What experiments support it?",
                    "objective": "Find experiment results",
                    "success_criterion": "result evidence",
                },
            ]
        )

        row = evaluate_open_goal_state(case, state, latency_ms=10.0)

        self.assertEqual(row["decomposition_quality"], 1.0)
        self.assertEqual(row["concept_coverage"], 1.0)

    def test_simple_goal_penalizes_unnecessary_decomposition(self) -> None:
        case = OpenGoalCase(
            case_id="simple",
            query="What is CRAG?",
            task_type="qa",
            complexity="simple",
            min_subgoals=0,
            max_subgoals=0,
            required_concept_groups=(),
        )
        row = evaluate_open_goal_state(
            case,
            state_with_subgoals(
                subgoals=[
                    {
                        "question": "What is CRAG?",
                        "objective": "Define it",
                        "success_criterion": "definition",
                    }
                ]
            ),
            latency_ms=10.0,
        )

        self.assertEqual(row["unnecessary_decomposition"], 1)
        self.assertEqual(row["decomposition_quality"], 0.0)

    def test_hybrid_gain_requires_safety_and_non_degraded_completion(self) -> None:
        deterministic = {
            "case_id": "one",
            "complexity": "compound",
            "completed": 1.0,
            "decomposition_quality": 0.0,
            "unnecessary_decomposition": 0,
            "plan_validity": 1.0,
            "fallback_used": True,
            "tool_call_count": 4,
            "skill_call_count": 2,
            "latency_ms": 10.0,
            "api_call_count": 0,
            "planner_api_calls": 0,
            "execution_api_calls": 0,
            "planner_input_tokens": 0,
            "planner_output_tokens": 0,
            "gate_bypass": 0,
            "provenance_failures": 0,
            "metadata_evidence_leaks": 0,
        }
        hybrid = {
            **deterministic,
            "decomposition_quality": 1.0,
            "fallback_used": False,
            "tool_call_count": 7,
            "api_call_count": 3,
            "planner_api_calls": 1,
            "execution_api_calls": 2,
            "planner_input_tokens": 100,
            "planner_output_tokens": 30,
        }

        report = build_open_goal_comparison([deterministic], [hybrid])

        self.assertEqual(report["conclusion"], "HYBRID_STRUCTURAL_GAIN_ONLY")
        self.assertFalse(any(report["hard_checks"].values()))


if __name__ == "__main__":
    unittest.main()
