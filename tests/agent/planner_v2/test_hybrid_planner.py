# C:\Users\18449\Desktop\researchguard_workspace\tests\agent\planner_v2\test_hybrid_planner.py
from __future__ import annotations

import json
import unittest

from researchguard.agent import (
    AgentPlan,
    AgentPolicy,
    BoundedReplanner,
    BoundedResearchAgentController,
    DeterministicPlanner,
    HybridPlanner,
    HybridPlannerSettings,
    PlanBudget,
    PlanStep,
    PlanValidationResult,
    PlannerBackendResponse,
    PlannerOutcome,
    StructuredPlan,
    StructuredPlanStep,
)
from researchguard.agent.state import ResearchAgentState, utc_timestamp
from researchguard.skills import build_default_skill_registry
from researchguard.tools import ToolError, ToolRegistry, ToolResult, ToolSpec


class StaticTool:
    version = "test-v1"
    description = "Planner v2 test tool."

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
        data = {"candidate_papers": []} if self.name == "search_scholarly_sources" else {}
        return ToolResult.create(
            status="success",
            message="ok",
            tool_name=self.name,
            tool_version=self.version,
            latency_ms=0.0,
            data=data,
        )


class FakeBackend:
    def __init__(self, payload: object):
        self.payload = payload
        self.calls = 0

    def propose(self, **kwargs: object) -> PlannerBackendResponse:
        self.calls += 1
        return PlannerBackendResponse(
            payload=self.payload,
            api_call_count=1,
            input_tokens=10,
            output_tokens=20,
        )


class FailingBackend:
    def propose(self, **kwargs: object) -> PlannerBackendResponse:
        raise RuntimeError("planner service unavailable")


def full_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for name in (
        "retrieve_evidence",
        "assess_evidence",
        "generate_grounded_answer",
        "audit_answer",
        "search_scholarly_sources",
    ):
        registry.register(StaticTool(name))
    return registry


def settings() -> HybridPlannerSettings:
    return HybridPlannerSettings(
        enabled=True,
        backend="openai",
        model="planner-test-model",
        temperature=0,
        timeout=5,
        max_tokens=400,
        max_steps=6,
        fallback_enabled=True,
        prompt_version="planner-test-v1",
        config_version="planner-test-v1",
    )


def valid_payload() -> dict[str, object]:
    return {
        "task_type": "qa",
        "goal": "Explain CRAG with canonical evidence.",
        "steps": [
            {
                "skill": "retrieve_evidence",
                "purpose": "Retrieve local evidence.",
                "expected_observation": "EvidenceBundle",
                "max_retry": 1,
            },
            {
                "skill": "assess_evidence",
                "purpose": "Assess the exact bundle.",
                "expected_observation": "GateDecision",
                "max_retry": 1,
            },
            {
                "skill": "generate_report",
                "purpose": "Generate through the guarded answer tool.",
                "expected_observation": "AnswerArtifact",
                "max_retry": 0,
            },
            {
                "skill": "audit_claims",
                "purpose": "Audit the generated claims.",
                "expected_observation": "CitationAuditResult",
                "max_retry": 1,
            },
        ],
        "budget": {
            "max_steps": 6,
            "max_tool_calls": 10,
            "max_retries": 2,
            "max_plan_revisions": 2,
        },
        "reasoning_summary": "Use the guarded evidence-first QA sequence.",
        "planner_version": "planner-test-v1",
    }


def planner_for(payload: object) -> HybridPlanner:
    registry = full_registry()
    skills = build_default_skill_registry(registry)
    policy = AgentPolicy()
    deterministic = DeterministicPlanner(
        registry=registry,
        skills=skills,
        policy=policy,
    )
    return HybridPlanner(
        registry=registry,
        skills=skills,
        policy=policy,
        deterministic=deterministic,
        settings=settings(),
        backend=FakeBackend(payload),
    )


def planner_with_backend(backend: object) -> HybridPlanner:
    registry = full_registry()
    skills = build_default_skill_registry(registry)
    policy = AgentPolicy()
    return HybridPlanner(
        registry=registry,
        skills=skills,
        policy=policy,
        deterministic=DeterministicPlanner(
            registry=registry,
            skills=skills,
            policy=policy,
        ),
        settings=settings(),
        backend=backend,
    )


class InterfacePlanner:
    def generate_plan(self, query: str, **kwargs: object) -> PlannerOutcome:
        structured = StructuredPlan(
            task_type="literature_search",
            goal=query,
            steps=(
                StructuredPlanStep(
                    skill="search_scholarly_sources",
                    purpose="Discover metadata-only candidates.",
                    expected_observation="ScholarPaperRecordList",
                    max_retry=0,
                ),
            ),
            budget=PlanBudget(6, 10, 2, 2),
            reasoning_summary="Use one bounded metadata discovery skill.",
            planner_version="interface-test-v1",
        )
        return PlannerOutcome(
            structured_plan=structured,
            executable_plan=AgentPlan(
                task_type="literature_search",
                steps=(
                    PlanStep(
                        1,
                        "search_scholarly_sources",
                        "Discover metadata-only candidates.",
                        max_retry=0,
                    ),
                ),
                created_at=utc_timestamp(),
            ),
            mode="test_interface",
            fallback_used=False,
            fallback_reason=None,
            validation=PlanValidationResult(True, None, ()),
            planner_model="test",
            prompt_version="test",
            latency_ms=0.0,
            api_call_count=0,
        )


class HybridPlannerTests(unittest.TestCase):
    def test_llm_planner_accepts_valid_strict_json(self) -> None:
        planner = planner_for(json.dumps(valid_payload()))
        outcome = planner.generate_plan("How does CRAG work?")

        self.assertEqual(outcome.mode, "hybrid_llm")
        self.assertFalse(outcome.fallback_used)
        self.assertEqual(outcome.structured_plan.steps[2].skill, "generate_report")
        self.assertEqual(
            [step.tool for step in outcome.executable_plan.steps],
            [
                "retrieve_evidence",
                "assess_evidence",
                "generate_grounded_answer",
                "audit_answer",
            ],
        )
        self.assertEqual(outcome.api_call_count, 1)
        self.assertEqual(
            json.loads(outcome.structured_plan.to_json())["task_type"],
            "qa",
        )

    def test_invalid_json_falls_back_to_deterministic_planner(self) -> None:
        outcome = planner_for("{not-json").generate_plan("How does CRAG work?")

        self.assertEqual(outcome.mode, "deterministic_fallback")
        self.assertTrue(outcome.fallback_used)
        self.assertEqual(outcome.fallback_reason, "invalid_json")
        self.assertEqual(outcome.api_call_count, 1)

    def test_planner_api_failure_falls_back_to_deterministic_planner(self) -> None:
        outcome = planner_with_backend(FailingBackend()).generate_plan(
            "How does CRAG work?"
        )

        self.assertEqual(outcome.mode, "deterministic_fallback")
        self.assertTrue(outcome.fallback_used)
        self.assertEqual(
            outcome.fallback_reason,
            "planner_failure:RuntimeError",
        )

    def test_unknown_skill_is_rejected_and_falls_back(self) -> None:
        payload = valid_payload()
        payload["steps"] = list(payload["steps"])
        payload["steps"][0] = {
            **payload["steps"][0],
            "skill": "invent_evidence",
        }
        outcome = planner_for(payload).generate_plan("How does CRAG work?")

        self.assertTrue(outcome.fallback_used)
        self.assertIn("unknown_skill", outcome.fallback_reason or "")

    def test_plan_over_policy_budget_is_rejected(self) -> None:
        payload = valid_payload()
        payload["budget"] = {
            **payload["budget"],
            "max_steps": 99,
        }
        outcome = planner_for(payload).generate_plan("How does CRAG work?")

        self.assertTrue(outcome.fallback_used)
        self.assertIn("plan_budget_exceeds_policy_steps", outcome.fallback_reason or "")

    def test_plan_cannot_bypass_evidence_gate(self) -> None:
        payload = valid_payload()
        payload["steps"] = [
            payload["steps"][2],
            payload["steps"][3],
        ]
        outcome = planner_for(payload).generate_plan("How does CRAG work?")

        self.assertTrue(outcome.fallback_used)
        self.assertIn("generation_without_evidence_gate", outcome.fallback_reason or "")

    def test_plan_cannot_generate_again_after_citation_audit(self) -> None:
        payload = valid_payload()
        payload["steps"] = [
            *payload["steps"],
            payload["steps"][2],
        ]
        outcome = planner_for(payload).generate_plan("How does CRAG work?")

        self.assertTrue(outcome.fallback_used)
        self.assertIn(
            "answering_plan_requires_single_generation",
            outcome.fallback_reason or "",
        )

    def test_controller_consumes_planner_interface(self) -> None:
        registry = ToolRegistry()
        registry.register(StaticTool("search_scholarly_sources"))
        controller = BoundedResearchAgentController(
            registry=registry,
            planner=InterfacePlanner(),
            memory_enabled=False,
        )

        state = controller.run("Find papers about corrective retrieval")

        self.assertEqual(state.status, "completed")
        self.assertEqual(state.planner_metadata["mode"], "test_interface")
        self.assertEqual(
            state.planner_plan["steps"][0]["skill"],
            "search_scholarly_sources",
        )
        self.assertEqual(state.tool_history[0]["tool_name"], "search_scholarly_sources")

    def test_deterministic_fallback_preserves_guarded_sequence(self) -> None:
        outcome = planner_for("[]").generate_plan("How does CRAG work?")

        self.assertTrue(outcome.fallback_used)
        self.assertEqual(
            [step.skill for step in outcome.structured_plan.steps],
            [
                "retrieve_evidence",
                "assess_evidence",
                "generate_report",
                "audit_claims",
            ],
        )
        self.assertEqual(
            [step.max_retry for step in outcome.executable_plan.steps],
            [1, 1, 0, 1],
        )

    def test_plan_revision_records_structured_error_observation(self) -> None:
        state = ResearchAgentState(
            "How does CRAG work?",
            plan=[{"step_id": 1, "tool": "retrieve_evidence"}],
            status="running",
        )
        result = ToolResult.create(
            status="failed",
            message="timeout",
            reason="retrieval_timeout",
            tool_name="retrieve_evidence",
            tool_version="test",
            latency_ms=1,
            error=ToolError(
                code="retrieval_timeout",
                category="timeout",
                message="timeout",
                retryable=True,
            ),
        )

        revision = BoundedReplanner().revise(
            state,
            tool_name="retrieve_evidence",
            result=result,
            available_tools=(
                "retrieve_evidence",
                "assess_evidence",
                "generate_grounded_answer",
                "audit_answer",
                "search_scholarly_sources",
            ),
        )

        self.assertIsNotNone(revision)
        assert revision is not None
        self.assertEqual(revision.observation["error_type"], "timeout")
        self.assertEqual(revision.observation["error_code"], "retrieval_timeout")
        self.assertTrue(revision.observation["retryable"])


if __name__ == "__main__":
    unittest.main()
