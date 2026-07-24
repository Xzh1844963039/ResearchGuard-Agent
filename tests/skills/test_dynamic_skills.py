# C:\Users\18449\Desktop\researchguard_workspace\tests\skills\test_dynamic_skills.py
from __future__ import annotations

import json
import unittest
from collections import Counter
from typing import Any, Mapping

from researchguard.agent import (
    AgentPolicy,
    BoundedResearchAgentController,
    PlanBudget,
    PlannerValidator,
    StructuredPlan,
    StructuredPlanStep,
)
from researchguard.agent.state import ResearchAgentState
from researchguard.skills import (
    SkillContext,
    SkillExecutor,
    SkillResult,
    build_default_skill_registry,
)
from researchguard.tools import (
    EvidenceBundle,
    GateDecision,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from researchguard.tracing import TraceCollector


EVIDENCE = {
    "chunk_id": "paper-crag::chunk-7",
    "doc_id": "paper-crag",
    "section": "method",
    "page": 5,
    "content": "CRAG uses a retrieval evaluator before corrective generation.",
    "source": "paper_crag.pdf",
    "score": 0.94,
    "provenance": {"source_block_ids": ["p5-b2"]},
}


class StaticTool:
    version = "skill-test-v1"
    description = "Synthetic dynamic skill tool."

    def __init__(self, name: str, handler: Any):
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

    def invoke(self, **kwargs: Any) -> ToolResult:
        return self.handler(**kwargs)


def result(
    name: str,
    *,
    status: str = "success",
    data: Mapping[str, Any] | None = None,
    reason: str | None = None,
) -> ToolResult:
    return ToolResult.create(
        status=status,
        message=f"skill_test_{status}",
        reason=reason,
        tool_name=name,
        tool_version="skill-test-v1",
        latency_ms=0.1,
        data=dict(data or {}),
    )


def make_registry(
    *,
    unsupported: bool = False,
    empty_first: bool = False,
) -> tuple[ToolRegistry, Counter[str]]:
    registry = ToolRegistry()
    calls: Counter[str] = Counter()

    def retrieve(**kwargs: Any) -> ToolResult:
        calls["retrieve_evidence"] += 1
        if empty_first and calls["retrieve_evidence"] == 1:
            return result("retrieve_evidence", data={"evidence": []})
        bundle = EvidenceBundle.create(
            query=str(kwargs["query"]),
            evidence=[EVIDENCE],
            retrieval_metadata={"attempt": calls["retrieve_evidence"]},
        )
        return result(
            "retrieve_evidence",
            data={
                "evidence": [EVIDENCE],
                "evidence_bundle": bundle.to_dict(),
                "retrieval": {"coverage": 1.0},
            },
        )

    def assess(**kwargs: Any) -> ToolResult:
        calls["assess_evidence"] += 1
        bundle = EvidenceBundle.from_mapping(kwargs["evidence_bundle"])
        level = "unsupported" if unsupported else "strong"
        assessment = {
            "support_level": level,
            "answerable": level == "strong",
            "confidence": 0.9,
            "reason": f"skill_test_{level}",
            "supporting_chunk_ids": list(bundle.chunk_ids) if level == "strong" else [],
        }
        gate = GateDecision.from_assessment(
            evidence_bundle_id=bundle.bundle_id,
            assessment=assessment,
        )
        return result(
            "assess_evidence",
            status="success" if level == "strong" else "rejected",
            reason=None if level == "strong" else "insufficient_evidence",
            data={
                "assessment": assessment,
                "gate_decision": gate.to_dict(),
            },
        )

    def generate(**kwargs: Any) -> ToolResult:
        calls["generate_grounded_answer"] += 1
        bundle = EvidenceBundle.from_mapping(kwargs["evidence_bundle"])
        answer = {
            "answer": "CRAG uses a retrieval evaluator.",
            "citations": [
                {
                    "chunk_id": EVIDENCE["chunk_id"],
                    "doc_id": EVIDENCE["doc_id"],
                    "section": EVIDENCE["section"],
                    "page": EVIDENCE["page"],
                }
            ],
            "confidence": 0.9,
            "refused": False,
            "refusal_reason": None,
            "evidence_chunk_ids": list(bundle.chunk_ids),
            "model": "skill-test",
            "prompt_version": "skill-test-v1",
            "config_version": "skill-test-v1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "cache_hit": False,
            "fallback_used": False,
            "fallback_reason": None,
            "api_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.1,
        }
        return result(
            "generate_grounded_answer",
            data={
                "answer": answer,
                "evidence_bundle_id": bundle.bundle_id,
            },
        )

    def audit(**kwargs: Any) -> ToolResult:
        calls["audit_answer"] += 1
        return result(
            "audit_answer",
            data={
                "audit": {
                    "audit_completed": True,
                    "overall_grounded": True,
                    "claims": [],
                }
            },
        )

    handlers = {
        "retrieve_evidence": retrieve,
        "assess_evidence": assess,
        "generate_grounded_answer": generate,
        "audit_answer": audit,
        "search_scholarly_sources": lambda **_: result(
            "search_scholarly_sources",
            data={"candidate_papers": []},
        ),
    }
    for name, handler in handlers.items():
        registry.register(StaticTool(name, handler))
    return registry, calls


class DynamicSkillTests(unittest.TestCase):
    def test_skill_registry_exposes_runtime_contracts(self) -> None:
        tools, _ = make_registry()
        skills = build_default_skill_registry(tools)

        self.assertIn("search_papers", skills.names)
        self.assertIn("analyze_claims", skills.names)
        self.assertEqual(
            skills.get("generate_report").required_tools,
            (
                "assess_evidence",
                "generate_grounded_answer",
                "audit_answer",
            ),
        )
        self.assertEqual(skills.runtime("retrieve_evidence").name, "retrieve_evidence")

    def test_skill_result_schema_is_json_serializable(self) -> None:
        value = SkillResult.create(
            skill_name="analyze_claims",
            skill_version="1.0.0",
            status="success",
            output={"claims": []},
            evidence=(EVIDENCE,),
        )

        payload = json.loads(value.to_json())

        self.assertEqual(payload["schema_version"], "researchguard.skill_result.v1")
        self.assertEqual(payload["evidence"][0]["chunk_id"], EVIDENCE["chunk_id"])

    def test_skill_executor_runs_retrieval_skill(self) -> None:
        tools, _ = make_registry()
        skills = build_default_skill_registry(tools)
        state = ResearchAgentState("How does CRAG work?")

        value = SkillExecutor(skills).execute(
            "retrieve_evidence",
            SkillContext(
                query=state.query,
                task_type="qa",
                state=state,
                inputs={"query": state.query},
                invoke_tool=lambda name, inputs: tools.invoke(name, **dict(inputs)),
            ),
        )

        self.assertEqual(value.status, "success")
        self.assertEqual(value.output["evidence_count"], 1)
        self.assertEqual(value.evidence[0]["chunk_id"], EVIDENCE["chunk_id"])

    def test_unknown_skill_fails_closed(self) -> None:
        tools, _ = make_registry()
        state = ResearchAgentState("Question")
        value = SkillExecutor(build_default_skill_registry(tools)).execute(
            "invent_skill",
            SkillContext(
                query=state.query,
                task_type="qa",
                state=state,
                inputs={},
                invoke_tool=lambda name, inputs: tools.invoke(name, **dict(inputs)),
            ),
        )

        self.assertEqual(value.status, "failed")
        self.assertEqual(value.reason, "skill_execution_error")

    def test_skill_budget_is_validated(self) -> None:
        tools, _ = make_registry()
        skills = build_default_skill_registry(tools)
        plan = StructuredPlan(
            task_type="qa",
            goal="Answer from evidence.",
            steps=(
                StructuredPlanStep(
                    "retrieve_evidence",
                    "Retrieve.",
                    "EvidenceBundle",
                    0,
                ),
                StructuredPlanStep(
                    "generate_report",
                    "Generate and audit.",
                    "GroundedReport",
                    0,
                ),
            ),
            budget=PlanBudget(1, 10, 1, 1),
            reasoning_summary="Bounded plan.",
            planner_version="skill-test",
        )

        validation = PlannerValidator(
            skills=skills,
            tools=tools,
            policy=AgentPolicy(),
        ).validate(plan)

        self.assertFalse(validation.valid)
        self.assertIn("plan_budget_max_steps_exceeded", validation.errors)

    def test_duplicate_skill_cycle_is_rejected(self) -> None:
        tools, _ = make_registry()
        skills = build_default_skill_registry(tools)
        plan = StructuredPlan(
            task_type="qa",
            goal="Answer from evidence.",
            steps=(
                StructuredPlanStep(
                    "retrieve_evidence",
                    "Retrieve once.",
                    "EvidenceBundle",
                    0,
                ),
                StructuredPlanStep(
                    "retrieve_evidence",
                    "Retrieve again.",
                    "EvidenceBundle",
                    0,
                ),
                StructuredPlanStep(
                    "generate_report",
                    "Generate and audit.",
                    "GroundedReport",
                    0,
                ),
            ),
            budget=PlanBudget(6, 10, 1, 1),
            reasoning_summary="Invalid cyclic plan.",
            planner_version="skill-test",
        )

        validation = PlannerValidator(
            skills=skills,
            tools=tools,
            policy=AgentPolicy(),
        ).validate(plan)

        self.assertFalse(validation.valid)
        self.assertTrue(
            any(
                item.startswith("skill_cycle_or_duplicate")
                for item in validation.errors
            )
        )

    def test_generate_report_cannot_bypass_evidence_gate(self) -> None:
        tools, calls = make_registry(unsupported=True)
        bundle = EvidenceBundle.create(
            query="Question",
            evidence=[EVIDENCE],
        )
        state = ResearchAgentState(
            "Question",
            evidence=[EVIDENCE],
            evidence_bundle=bundle.to_dict(),
        )
        value = SkillExecutor(build_default_skill_registry(tools)).execute(
            "generate_report",
            SkillContext(
                query=state.query,
                task_type="qa",
                state=state,
                inputs={"evidence_bundle": bundle.to_dict()},
                invoke_tool=lambda name, inputs: tools.invoke(name, **dict(inputs)),
            ),
        )

        self.assertEqual(value.status, "rejected")
        self.assertEqual(value.reason, "insufficient_evidence")
        self.assertEqual(calls["generate_grounded_answer"], 0)
        self.assertEqual(calls["audit_answer"], 0)

    def test_controller_executes_dynamic_skill_plan(self) -> None:
        tools, _ = make_registry()
        state = BoundedResearchAgentController(
            registry=tools,
            memory_enabled=False,
        ).run("How does CRAG work?")

        self.assertEqual(state.status, "completed")
        self.assertEqual(
            [item["skill_name"] for item in state.skill_history],
            ["retrieve_evidence", "generate_report"],
        )
        self.assertEqual(
            [item["tool_name"] for item in state.tool_history],
            [
                "retrieve_evidence",
                "assess_evidence",
                "generate_grounded_answer",
                "audit_answer",
            ],
        )
        self.assertTrue(state.audit_result["overall_grounded"])
        trace = TraceCollector().collect(state)
        self.assertEqual(trace.skill_calls[0]["skill_name"], "retrieve_evidence")

    def test_skill_replanning_is_bounded_and_observation_driven(self) -> None:
        tools, _ = make_registry(empty_first=True)
        state = BoundedResearchAgentController(
            registry=tools,
            memory_enabled=False,
        ).run("How does CRAG work?")

        self.assertEqual(state.status, "completed")
        self.assertEqual(len(state.plan_revisions), 1)
        revision = state.plan_revisions[0]
        self.assertEqual(revision["previous_skill"], "retrieve_evidence")
        self.assertEqual(
            revision["new_skill_sequence"][-2:],
            ["retrieve_evidence", "generate_report"],
        )
        self.assertEqual(
            [item["skill_name"] for item in state.skill_history],
            ["retrieve_evidence", "retrieve_evidence", "generate_report"],
        )


if __name__ == "__main__":
    unittest.main()
