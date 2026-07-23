# C:\Users\18449\Desktop\researchguard_workspace\tests\agent\test_controller.py
from __future__ import annotations

import time
import unittest
from collections.abc import Callable

from researchguard.agent.controller import BoundedResearchAgentController
from researchguard.agent.planner import AgentPlan, PlanStep
from researchguard.agent.policy import AgentPolicy
from researchguard.tools import ToolError, ToolRegistry, ToolResult, ToolSpec


EVIDENCE = {
    "chunk_id": "paper-crag::chunk-7",
    "doc_id": "paper-crag",
    "section": "method",
    "page": 5,
    "page_end": 5,
    "content": "CRAG uses a retrieval evaluator to estimate retrieval quality.",
    "source": "paper_crag.pdf",
    "score": 0.94,
    "rank": 1,
    "provenance": {
        "title": "Corrective Retrieval Augmented Generation",
        "source_block_ids": ["p5-b2"],
        "content_types": ["paragraph"],
    },
}

ANSWER_ARTIFACT = {
    "answer": "CRAG uses a retrieval evaluator [paper-crag::chunk-7].",
    "citations": [
        {
            "chunk_id": "paper-crag::chunk-7",
            "doc_id": "paper-crag",
            "section": "method",
            "page": 5,
        }
    ],
    "confidence": 0.92,
    "refused": False,
    "refusal_reason": None,
    "evidence_chunk_ids": ["paper-crag::chunk-7"],
    "model": "synthetic",
    "prompt_version": "test",
    "config_version": "test",
    "timestamp": "2026-01-01T00:00:00+00:00",
    "cache_hit": False,
    "fallback_used": False,
    "fallback_reason": None,
    "api_call_count": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "latency_ms": 0.1,
}


class FakeTool:
    version = "test"
    description = "Synthetic controller tool."

    def __init__(self, name: str, handler: Callable[..., ToolResult]):
        self.name = name
        self.handler = handler
        self.call_count = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema={},
        )

    def invoke(self, **kwargs: object) -> ToolResult:
        self.call_count += 1
        return self.handler(**kwargs)


def _success(name: str, data: dict[str, object]) -> ToolResult:
    return ToolResult.create(
        status="success",
        message="ok",
        tool_name=name,
        tool_version="test",
        latency_ms=1,
        data=data,
    )


def _grounded_pipeline_result() -> dict[str, object]:
    retrieval_hit = {
        **EVIDENCE,
        "page_start": EVIDENCE["page"],
        "text": EVIDENCE["content"],
        "title": EVIDENCE["provenance"]["title"],
    }
    audit = {
        "audit_completed": True,
        "overall_grounded": True,
        "claims": [],
        "evidence_chunk_ids": ["paper-crag::chunk-7"],
    }
    return {
        "final_status": "grounded",
        "retrieval": {"output": {"hits": [retrieval_hit]}},
        "answer_generation": {"output": ANSWER_ARTIFACT},
        "citation_audit": {"output": audit},
    }


def _complete_registry(*, unsupported: bool = False) -> tuple[ToolRegistry, dict[str, FakeTool]]:
    registry = ToolRegistry()
    tools: dict[str, FakeTool] = {}
    tools["retrieve_evidence"] = FakeTool(
        "retrieve_evidence",
        lambda **_: _success("retrieve_evidence", {"evidence": [EVIDENCE]}),
    )
    if unsupported:
        tools["assess_evidence"] = FakeTool(
            "assess_evidence",
            lambda **_: ToolResult.create(
                status="rejected",
                message="Evidence support level: unsupported.",
                reason="no support",
                tool_name="assess_evidence",
                tool_version="test",
                latency_ms=1,
                data={"assessment": {"support_level": "unsupported", "answerable": False}},
            ),
        )
    else:
        tools["assess_evidence"] = FakeTool(
            "assess_evidence",
            lambda **_: _success(
                "assess_evidence",
                {"assessment": {"support_level": "strong", "answerable": True}},
            ),
        )
    tools["generate_grounded_answer"] = FakeTool(
        "generate_grounded_answer",
        lambda **_: _success(
            "generate_grounded_answer",
            {"pipeline_result": _grounded_pipeline_result()},
        ),
    )
    tools["audit_answer"] = FakeTool(
        "audit_answer",
        lambda **_: _success(
            "audit_answer",
            {"audit": {"audit_completed": True, "overall_grounded": True}},
        ),
    )
    for tool in tools.values():
        registry.register(tool)
    return registry, tools


class RepeatingPlanner:
    def __init__(self, count: int):
        self.count = count

    def create_plan(self, query: str, **kwargs: object) -> AgentPlan:
        del query, kwargs
        return AgentPlan(
            task_type="qa",
            steps=tuple(
                PlanStep(
                    step_id=index,
                    tool="retrieve_evidence",
                    purpose="Synthetic repeated step.",
                )
                for index in range(1, self.count + 1)
            ),
            created_at="2026-01-01T00:00:00+00:00",
        )


class ControllerTests(unittest.TestCase):
    def test_successful_controller_uses_registry_and_records_trace(self) -> None:
        registry, tools = _complete_registry()
        controller = BoundedResearchAgentController(registry=registry)

        state = controller.run("How does CRAG reduce hallucination?")

        self.assertEqual(state.status, "completed")
        self.assertEqual(
            [entry["tool_name"] for entry in state.tool_history],
            [
                "retrieve_evidence",
                "assess_evidence",
                "generate_grounded_answer",
                "audit_answer",
            ],
        )
        self.assertEqual(state.answer["evidence_chunk_ids"], ["paper-crag::chunk-7"])
        self.assertTrue(state.audit_result["overall_grounded"])
        self.assertEqual(state.evidence[0]["chunk_id"], "paper-crag::chunk-7")
        self.assertTrue(all(entry["trace_id"] for entry in state.tool_history))
        self.assertTrue(all(tool.call_count == 1 for tool in tools.values()))

    def test_unsupported_evidence_stops_before_answer(self) -> None:
        registry, tools = _complete_registry(unsupported=True)
        controller = BoundedResearchAgentController(registry=registry)

        state = controller.run("Does the corpus cover quantum error correction?")

        self.assertEqual(state.status, "rejected")
        self.assertEqual(state.reason, "insufficient_evidence")
        self.assertEqual(tools["generate_grounded_answer"].call_count, 0)
        self.assertEqual(tools["audit_answer"].call_count, 0)
        self.assertEqual(
            [entry["tool_name"] for entry in state.tool_history],
            ["retrieve_evidence", "assess_evidence"],
        )

    def test_policy_stops_repeated_steps_at_tool_call_limit(self) -> None:
        registry = ToolRegistry()
        tool = FakeTool(
            "retrieve_evidence",
            lambda **_: _success("retrieve_evidence", {"evidence": [EVIDENCE]}),
        )
        registry.register(tool)
        controller = BoundedResearchAgentController(
            registry=registry,
            planner=RepeatingPlanner(4),
            policy=AgentPolicy(max_steps=6, max_tool_calls=2, max_retry=0, timeout=10),
        )

        state = controller.run("Synthetic repeated task")

        self.assertEqual(state.status, "failed")
        self.assertEqual(state.reason, "max_tool_calls_exceeded")
        self.assertEqual(tool.call_count, 2)

    def test_retry_limit_stops_retryable_tool_failure(self) -> None:
        registry = ToolRegistry()
        tool = FakeTool(
            "retrieve_evidence",
            lambda **_: ToolResult.create(
                status="failed",
                message="temporary failure",
                reason="retrieval_failed",
                tool_name="retrieve_evidence",
                tool_version="test",
                latency_ms=1,
                error=ToolError(
                    code="retrieval_failed",
                    category="retrieval_failure",
                    message="temporary failure",
                    retryable=True,
                ),
            ),
        )
        registry.register(tool)
        controller = BoundedResearchAgentController(
            registry=registry,
            planner=RepeatingPlanner(1),
            policy=AgentPolicy(max_steps=6, max_tool_calls=10, max_retry=2, timeout=10),
        )

        state = controller.run("Synthetic retry task")

        self.assertEqual(state.status, "failed")
        self.assertEqual(state.reason, "tool_error")
        self.assertEqual(tool.call_count, 3)
        self.assertEqual(state.retry_counts["0"], 2)
        restored_attempts = tool.call_count
        controller.resume(state)
        self.assertEqual(tool.call_count, restored_attempts)

    def test_unhandled_tool_exception_becomes_failed_state(self) -> None:
        registry = ToolRegistry()

        def raise_error(**_: object) -> ToolResult:
            raise RuntimeError("synthetic unhandled error")

        tool = FakeTool("retrieve_evidence", raise_error)
        registry.register(tool)
        controller = BoundedResearchAgentController(
            registry=registry,
            planner=RepeatingPlanner(1),
        )

        state = controller.run("Synthetic exception task")

        self.assertEqual(state.status, "failed")
        self.assertEqual(state.reason, "tool_error")
        self.assertEqual(state.observations[0]["error"]["category"], "execution_failure")

    def test_timeout_is_checked_after_tool_returns(self) -> None:
        registry = ToolRegistry()

        def delayed_success(**_: object) -> ToolResult:
            time.sleep(0.02)
            return _success("retrieve_evidence", {"evidence": [EVIDENCE]})

        tool = FakeTool("retrieve_evidence", delayed_success)
        registry.register(tool)
        controller = BoundedResearchAgentController(
            registry=registry,
            planner=RepeatingPlanner(1),
            policy=AgentPolicy(max_steps=6, max_tool_calls=10, max_retry=0, timeout=0.005),
        )

        state = controller.run("Synthetic timeout task")

        self.assertEqual(state.status, "failed")
        self.assertEqual(state.reason, "timeout_exceeded")
        self.assertEqual(tool.call_count, 1)

    def test_audit_task_uses_supplied_artifacts(self) -> None:
        registry, tools = _complete_registry()
        controller = BoundedResearchAgentController(registry=registry)

        state = controller.run(
            "Audit this answer and its citations",
            task_type="audit",
            answer_artifact=ANSWER_ARTIFACT,
            evidence=[EVIDENCE],
        )

        self.assertEqual(state.status, "completed")
        self.assertEqual([entry["tool_name"] for entry in state.tool_history], ["audit_answer"])
        self.assertEqual(tools["retrieve_evidence"].call_count, 0)
        self.assertEqual(tools["audit_answer"].call_count, 1)


if __name__ == "__main__":
    unittest.main()
