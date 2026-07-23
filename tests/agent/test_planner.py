# C:\Users\18449\Desktop\researchguard_workspace\tests\agent\test_planner.py
from __future__ import annotations

import unittest

from researchguard.agent.planner import BoundedPlanner, PlannerError
from researchguard.tools import ToolRegistry, ToolResult, ToolSpec


class DummyTool:
    version = "test"
    description = "Synthetic planner tool."

    def __init__(self, name: str):
        self.name = name

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema={},
        )

    def invoke(self, **kwargs: object) -> ToolResult:
        del kwargs
        return ToolResult.create(
            status="success",
            message="ok",
            tool_name=self.name,
            tool_version=self.version,
            latency_ms=0,
        )


def _registry(*, include_audit: bool = True) -> ToolRegistry:
    registry = ToolRegistry()
    names = [
        "retrieve_evidence",
        "assess_evidence",
        "generate_grounded_answer",
    ]
    if include_audit:
        names.append("audit_answer")
    for name in names:
        registry.register(DummyTool(name))
    return registry


class BoundedPlannerTests(unittest.TestCase):
    def test_comparison_query_selects_registered_workflow(self) -> None:
        registry = _registry()
        planner = BoundedPlanner(
            registry,
            max_steps=6,
            workflow_names=("paper_comparison",),
        )

        plan = planner.create_plan("Compare CRAG and Self-RAG")

        self.assertEqual(plan.task_type, "paper_comparison")
        self.assertEqual(plan.workflow, "paper_comparison")
        self.assertEqual(plan.steps, ())

    def test_qa_query_uses_same_guarded_sequence(self) -> None:
        planner = BoundedPlanner(_registry())

        plan = planner.create_plan("How does CRAG reduce hallucination?")

        self.assertEqual(plan.task_type, "qa")
        self.assertEqual(plan.steps[-1].tool, "audit_answer")

    def test_audit_requires_answer_artifact(self) -> None:
        planner = BoundedPlanner(_registry())

        with self.assertRaises(PlannerError):
            planner.create_plan("Audit this answer and its citations", task_type="audit")

        plan = planner.create_plan(
            "Audit this answer and its citations",
            task_type="audit",
            has_answer=True,
            has_evidence=True,
        )
        self.assertEqual([step.tool for step in plan.steps], ["audit_answer"])

    def test_unregistered_tool_prevents_plan_creation(self) -> None:
        planner = BoundedPlanner(_registry(include_audit=False))

        with self.assertRaises(PlannerError):
            planner.create_plan("How does CRAG work?")


if __name__ == "__main__":
    unittest.main()
