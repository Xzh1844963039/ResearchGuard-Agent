# C:\Users\18449\Desktop\researchguard_workspace\tests\agent\test_goal_decomposition.py
from __future__ import annotations

import json
import unittest
from collections import Counter
from typing import Any, Mapping

from researchguard.agent import (
    AgentPlan,
    AgentPolicy,
    BoundedResearchAgentController,
    DeterministicPlanner,
    HybridPlanner,
    HybridPlannerSettings,
    PlanBudget,
    PlanStep,
    PlanValidationResult,
    PlannerOutcome,
    PlannerBackendResponse,
    PlannerValidator,
    ResearchSubgoal,
    StructuredPlan,
    StructuredPlanStep,
)
from researchguard.agent.state import utc_timestamp
from researchguard.skills import build_default_skill_registry
from researchguard.tools import (
    EvidenceBundle,
    GateDecision,
    ScholarPaperRecord,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from researchguard.tracing import TraceCollector


METHOD_QUERY = "What mechanism does CRAG use to correct weak retrieval?"
EXPERIMENT_QUERY = "What experiments evaluate CRAG retrieval correction?"


def evidence(chunk_id: str, content: str, page: int) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": "paper-crag",
        "section": "method" if page == 5 else "experiments",
        "page": page,
        "content": content,
        "source": "paper_crag.pdf",
        "score": 0.9,
        "provenance": {"source_block_ids": [f"p{page}-b1"]},
    }


METHOD_EVIDENCE = evidence(
    "paper-crag::method",
    "CRAG evaluates retrieval quality before corrective retrieval.",
    5,
)
EXPERIMENT_EVIDENCE = evidence(
    "paper-crag::experiment",
    "CRAG reports experiments comparing correction against RAG baselines.",
    8,
)


class StaticTool:
    version = "goal-decomposition-test-v1"
    description = "Synthetic tool for bounded goal-decomposition tests."

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


def tool_result(
    name: str,
    *,
    status: str = "success",
    data: Mapping[str, Any] | None = None,
    reason: str | None = None,
) -> ToolResult:
    return ToolResult.create(
        status=status,
        message=f"goal_decomposition_{status}",
        reason=reason,
        tool_name=name,
        tool_version="goal-decomposition-test-v1",
        latency_ms=0.1,
        data=dict(data or {}),
    )


class OpenGoalPlanner:
    def __init__(self, plan: StructuredPlan, registry: ToolRegistry):
        self.plan = plan
        self.registry = registry

    def generate_plan(self, query: str, **_: Any) -> PlannerOutcome:
        skills = build_default_skill_registry(self.registry)
        validation = PlannerValidator(
            skills=skills,
            tools=self.registry,
            policy=AgentPolicy(),
        ).validate(self.plan)
        if not validation.valid:
            raise AssertionError(validation.errors)
        executable = AgentPlan(
            task_type=self.plan.task_type,
            steps=tuple(
                PlanStep(
                    step_id=index,
                    tool="",
                    skill=step.skill,
                    purpose=step.purpose,
                    parameters=step.parameters,
                    max_retry=step.max_retry,
                    subgoal_id=step.subgoal_id,
                )
                for index, step in enumerate(self.plan.steps, start=1)
            ),
            created_at=utc_timestamp(),
        )
        return PlannerOutcome(
            structured_plan=self.plan,
            executable_plan=executable,
            mode="open_goal_test",
            fallback_used=False,
            fallback_reason=None,
            validation=PlanValidationResult(True, None, ()),
            planner_model="scripted-open-goal",
            prompt_version="open-goal-test-v1",
            latency_ms=0.0,
            api_call_count=0,
        )


class StaticPlannerBackend:
    def __init__(self, payload: Mapping[str, Any]):
        self.payload = payload

    def propose(self, **_: Any) -> PlannerBackendResponse:
        return PlannerBackendResponse(
            payload=self.payload,
            api_call_count=1,
            input_tokens=10,
            output_tokens=10,
        )


def open_goal_plan() -> StructuredPlan:
    subgoals = (
        ResearchSubgoal(
            "sg-mechanism",
            METHOD_QUERY,
            "Identify the corrective retrieval mechanism.",
            "At least one canonical method passage is retrieved.",
        ),
        ResearchSubgoal(
            "sg-experiments",
            EXPERIMENT_QUERY,
            "Identify experimental evidence for the mechanism.",
            "At least one canonical experiment passage is retrieved.",
        ),
    )
    return StructuredPlan(
        task_type="qa",
        goal=(
            "Explain how CRAG corrects weak retrieval and what experiments "
            "support the mechanism."
        ),
        subgoals=subgoals,
        steps=(
            StructuredPlanStep(
                "retrieve_evidence",
                "Collect mechanism evidence.",
                "EvidenceBundle",
                1,
                subgoal_id="sg-mechanism",
                parameters={"query": METHOD_QUERY},
            ),
            StructuredPlanStep(
                "retrieve_evidence",
                "Collect experiment evidence.",
                "EvidenceBundle",
                1,
                subgoal_id="sg-experiments",
                parameters={"query": EXPERIMENT_QUERY},
            ),
            StructuredPlanStep(
                "generate_report",
                "Synthesize only from the aggregated canonical evidence.",
                "GroundedReport",
                0,
            ),
        ),
        budget=PlanBudget(6, 10, 2, 2),
        reasoning_summary="Decompose mechanism and experiment evidence needs.",
        planner_version="open-goal-test-v1",
    )


def registry_for(
    *,
    experiment_failures: int,
) -> tuple[ToolRegistry, list[dict[str, Any]], Counter[str]]:
    registry = ToolRegistry()
    retrieval_calls: list[dict[str, Any]] = []
    calls: Counter[str] = Counter()

    def retrieve(**kwargs: Any) -> ToolResult:
        calls["retrieve_evidence"] += 1
        retrieval_calls.append(dict(kwargs))
        query = str(kwargs["query"])
        is_experiment = query == EXPERIMENT_QUERY
        experiment_attempt = sum(
            1 for item in retrieval_calls if item.get("query") == EXPERIMENT_QUERY
        )
        if is_experiment and experiment_attempt <= experiment_failures:
            return tool_result(
                "retrieve_evidence",
                data={"evidence": [], "retrieval": {"coverage": 0.0}},
            )
        record = EXPERIMENT_EVIDENCE if is_experiment else METHOD_EVIDENCE
        bundle = EvidenceBundle.create(
            query=query,
            evidence=[record],
            retrieval_metadata={"coverage": 1.0},
            provenance={"source": "synthetic_local_corpus"},
        )
        return tool_result(
            "retrieve_evidence",
            data={
                "evidence": [record],
                "evidence_bundle": bundle.to_dict(),
                "retrieval": {"coverage": 1.0},
            },
        )

    def assess(**kwargs: Any) -> ToolResult:
        calls["assess_evidence"] += 1
        bundle = EvidenceBundle.from_mapping(kwargs["evidence_bundle"])
        assessment = {
            "support_level": "strong",
            "answerable": True,
            "confidence": 0.95,
            "reason": "Both bounded subgoals have canonical evidence.",
            "supporting_chunk_ids": list(bundle.chunk_ids),
        }
        gate = GateDecision.from_assessment(
            evidence_bundle_id=bundle.bundle_id,
            assessment=assessment,
        )
        return tool_result(
            "assess_evidence",
            data={"assessment": assessment, "gate_decision": gate.to_dict()},
        )

    def generate(**kwargs: Any) -> ToolResult:
        calls["generate_grounded_answer"] += 1
        bundle = EvidenceBundle.from_mapping(kwargs["evidence_bundle"])
        citations = [
            {
                "chunk_id": record.chunk_id,
                "doc_id": record.doc_id,
                "section": record.section,
                "page": record.page,
            }
            for record in bundle.evidence_records
        ]
        return tool_result(
            "generate_grounded_answer",
            data={
                "answer": {
                    "answer": "CRAG uses corrective retrieval and evaluates it experimentally.",
                    "citations": citations,
                    "confidence": 0.9,
                    "refused": False,
                    "refusal_reason": None,
                    "evidence_chunk_ids": list(bundle.chunk_ids),
                    "model": "synthetic",
                    "prompt_version": "synthetic-v1",
                    "config_version": "synthetic-v1",
                    "timestamp": utc_timestamp(),
                    "cache_hit": False,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "api_call_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0.1,
                },
                "evidence_bundle_id": bundle.bundle_id,
            },
        )

    def audit(**_: Any) -> ToolResult:
        calls["audit_answer"] += 1
        return tool_result(
            "audit_answer",
            data={
                "audit": {
                    "audit_completed": True,
                    "overall_grounded": True,
                    "claims": [],
                }
            },
        )

    def search(**kwargs: Any) -> ToolResult:
        calls["search_scholarly_sources"] += 1
        return tool_result(
            "search_scholarly_sources",
            data={
                "query": kwargs.get("query"),
                "candidate_papers": [
                    ScholarPaperRecord(
                        paper_id="external-candidate",
                        title="Metadata-only CRAG follow-up",
                        authors=("Example Author",),
                        year=2025,
                        venue="Example Venue",
                        doi=None,
                        source="openalex",
                        source_type="metadata",
                        url="https://example.org/paper",
                        abstract="This abstract is not canonical evidence.",
                    ).to_dict()
                ],
                "metadata_only": True,
                "evidence_eligible": False,
            },
        )

    handlers = {
        "retrieve_evidence": retrieve,
        "assess_evidence": assess,
        "generate_grounded_answer": generate,
        "audit_answer": audit,
        "search_scholarly_sources": search,
    }
    for name, handler in handlers.items():
        registry.register(StaticTool(name, handler))
    return registry, retrieval_calls, calls


class GoalDecompositionTests(unittest.TestCase):
    def test_subgoal_schema_round_trip_is_json_stable(self) -> None:
        plan = open_goal_plan()
        restored = StructuredPlan.from_mapping(
            json.loads(plan.to_json())
        )

        self.assertEqual(restored, plan)
        self.assertEqual(len(restored.subgoals), 2)
        self.assertEqual(restored.steps[0].subgoal_id, "sg-mechanism")

    def test_hybrid_planner_compiles_valid_open_goal_without_fallback(self) -> None:
        registry, _, _ = registry_for(experiment_failures=0)
        skills = build_default_skill_registry(registry)
        policy = AgentPolicy()
        planner = HybridPlanner(
            registry=registry,
            skills=skills,
            policy=policy,
            deterministic=DeterministicPlanner(
                registry=registry,
                skills=skills,
                policy=policy,
            ),
            settings=HybridPlannerSettings(
                enabled=True,
                backend="openai",
                model="synthetic-planner",
                temperature=0,
                timeout=5,
                max_tokens=500,
                max_steps=6,
                fallback_enabled=True,
                prompt_version="open-goal-test-v1",
                config_version="open-goal-test-v1",
            ),
            backend=StaticPlannerBackend(open_goal_plan().to_dict()),
        )

        outcome = planner.generate_plan(open_goal_plan().goal)

        self.assertFalse(outcome.fallback_used)
        self.assertEqual(outcome.mode, "hybrid_llm")
        self.assertEqual(
            [step.subgoal_id for step in outcome.executable_plan.steps],
            ["sg-mechanism", "sg-experiments", None],
        )

    def test_planner_normalizes_redundant_subgoal_gate_steps(self) -> None:
        registry, _, _ = registry_for(experiment_failures=0)
        skills = build_default_skill_registry(registry)
        policy = AgentPolicy()
        payload = open_goal_plan().to_dict()
        noisy_steps: list[dict[str, Any]] = []
        for step in payload["skills"][:2]:
            noisy_retrieval = dict(step)
            noisy_retrieval["parameters"] = {"query": "drifted query"}
            noisy_steps.extend(
                [
                    noisy_retrieval,
                    {
                        "skill": "assess_evidence",
                        "purpose": "Redundant subgoal gate.",
                        "expected_observation": "wrong-value",
                        "max_retry": 0,
                        "subgoal_id": step["subgoal_id"],
                        "parameters": {"query": step["parameters"]["query"]},
                    },
                ]
            )
        noisy_steps.append(payload["skills"][-1])
        payload["skills"] = noisy_steps
        planner = HybridPlanner(
            registry=registry,
            skills=skills,
            policy=policy,
            deterministic=DeterministicPlanner(
                registry=registry,
                skills=skills,
                policy=policy,
            ),
            settings=HybridPlannerSettings(
                enabled=True,
                backend="openai",
                model="synthetic-planner",
                temperature=0,
                timeout=5,
                max_tokens=500,
                max_steps=6,
                fallback_enabled=True,
                prompt_version="open-goal-test-v1",
                config_version="open-goal-test-v1",
            ),
            backend=StaticPlannerBackend(payload),
        )

        outcome = planner.generate_plan(open_goal_plan().goal)

        self.assertFalse(outcome.fallback_used)
        self.assertEqual(
            [step.skill for step in outcome.executable_plan.steps],
            ["retrieve_evidence", "retrieve_evidence", "generate_report"],
        )

    def test_validator_allows_distinct_subgoal_retrievals(self) -> None:
        registry, _, _ = registry_for(experiment_failures=0)
        result = PlannerValidator(
            skills=build_default_skill_registry(registry),
            tools=registry,
            policy=AgentPolicy(),
        ).validate(open_goal_plan())

        self.assertTrue(result.valid, result.errors)

    def test_validator_rejects_unnecessary_single_subgoal(self) -> None:
        registry, _, _ = registry_for(experiment_failures=0)
        plan = open_goal_plan()
        single = StructuredPlan(
            task_type=plan.task_type,
            goal=plan.goal,
            subgoals=plan.subgoals[:1],
            steps=(plan.steps[0], plan.steps[-1]),
            budget=plan.budget,
            reasoning_summary=plan.reasoning_summary,
            planner_version=plan.planner_version,
        )
        result = PlannerValidator(
            skills=build_default_skill_registry(registry),
            tools=registry,
            policy=AgentPolicy(),
        ).validate(single)

        self.assertFalse(result.valid)
        self.assertIn("single_subgoal_is_unnecessary", result.errors)

    def test_validator_rejects_repeated_retrieval_for_same_subgoal(self) -> None:
        registry, _, _ = registry_for(experiment_failures=0)
        plan = open_goal_plan()
        duplicate = StructuredPlan(
            task_type=plan.task_type,
            goal=plan.goal,
            subgoals=plan.subgoals,
            steps=(plan.steps[0], plan.steps[0], plan.steps[-1]),
            budget=plan.budget,
            reasoning_summary=plan.reasoning_summary,
            planner_version=plan.planner_version,
        )
        result = PlannerValidator(
            skills=build_default_skill_registry(registry),
            tools=registry,
            policy=AgentPolicy(),
        ).validate(duplicate)

        self.assertFalse(result.valid)
        self.assertIn(
            "skill_cycle_or_duplicate:retrieve_evidence",
            result.errors,
        )
        self.assertIn("subgoal_without_retrieval:sg-experiments", result.errors)

    def test_missing_subgoal_is_retried_without_rerunning_completed_goal(self) -> None:
        registry, retrieval_calls, calls = registry_for(experiment_failures=1)
        state = BoundedResearchAgentController(
            registry=registry,
            planner=OpenGoalPlanner(open_goal_plan(), registry),
            memory_enabled=False,
        ).run(open_goal_plan().goal)

        self.assertEqual(state.status, "completed")
        self.assertEqual(
            [item["query"] for item in retrieval_calls],
            [METHOD_QUERY, EXPERIMENT_QUERY, EXPERIMENT_QUERY],
        )
        self.assertNotIn("rewrite", retrieval_calls[0])
        self.assertTrue(retrieval_calls[2]["rewrite"])
        self.assertEqual(state.subgoal_results["sg-mechanism"]["attempts"], 1)
        self.assertEqual(state.subgoal_results["sg-experiments"]["attempts"], 2)
        self.assertEqual(
            {item["chunk_id"] for item in state.evidence},
            {METHOD_EVIDENCE["chunk_id"], EXPERIMENT_EVIDENCE["chunk_id"]},
        )
        self.assertEqual(
            state.evidence_bundle["provenance"]["source"],
            "controller_subgoal_aggregation",
        )
        evidence_by_id = {item["chunk_id"]: item for item in state.evidence}
        self.assertEqual(
            evidence_by_id[METHOD_EVIDENCE["chunk_id"]]["provenance"],
            METHOD_EVIDENCE["provenance"],
        )
        self.assertEqual(
            evidence_by_id[EXPERIMENT_EVIDENCE["chunk_id"]]["provenance"],
            EXPERIMENT_EVIDENCE["provenance"],
        )
        self.assertEqual(len(state.plan_revisions), 1)
        revision = state.plan_revisions[0]
        self.assertEqual(revision["observation"]["subgoal_id"], "sg-experiments")
        self.assertEqual(
            [step.get("subgoal_id") for step in revision["new_plan"]],
            ["sg-mechanism", "sg-experiments", "sg-experiments", None],
        )
        self.assertEqual(calls["search_scholarly_sources"], 0)
        self.assertEqual(calls["generate_grounded_answer"], 1)
        self.assertEqual(calls["audit_answer"], 1)

        trace = TraceCollector().collect(state).to_dict()
        self.assertEqual(len(trace["subgoals"]), 2)
        self.assertEqual(
            trace["subgoal_results"]["sg-experiments"]["status"],
            "satisfied",
        )

    def test_metadata_fallback_cannot_enter_canonical_evidence(self) -> None:
        registry, retrieval_calls, calls = registry_for(experiment_failures=99)
        state = BoundedResearchAgentController(
            registry=registry,
            planner=OpenGoalPlanner(open_goal_plan(), registry),
            memory_enabled=False,
        ).run(open_goal_plan().goal)

        self.assertEqual(state.status, "rejected")
        self.assertEqual(
            state.reason,
            "no_corpus_evidence_scholarly_candidates_only",
        )
        self.assertEqual(len(retrieval_calls), 3)
        self.assertEqual(calls["search_scholarly_sources"], 1)
        self.assertEqual(calls["generate_grounded_answer"], 0)
        self.assertIsNone(state.answer)
        self.assertEqual(
            [item["chunk_id"] for item in state.evidence],
            [METHOD_EVIDENCE["chunk_id"]],
        )
        self.assertNotIn("external-candidate", json.dumps(state.evidence))
        self.assertEqual(len(state.plan_revisions), 2)

    def test_identical_observations_produce_identical_execution_shape(self) -> None:
        shapes = []
        for _ in range(2):
            registry, retrieval_calls, _ = registry_for(experiment_failures=1)
            state = BoundedResearchAgentController(
                registry=registry,
                planner=OpenGoalPlanner(open_goal_plan(), registry),
                memory_enabled=False,
            ).run(open_goal_plan().goal)
            shapes.append(
                {
                    "queries": [item["query"] for item in retrieval_calls],
                    "skills": [item["skill_name"] for item in state.skill_history],
                    "revision_reasons": [
                        item["reason"] for item in state.plan_revisions
                    ],
                    "evidence_ids": [item["chunk_id"] for item in state.evidence],
                    "status": state.status,
                }
            )

        self.assertEqual(shapes[0], shapes[1])


if __name__ == "__main__":
    unittest.main()
