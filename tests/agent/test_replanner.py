# C:\Users\18449\Desktop\researchguard_workspace\tests\agent\test_replanner.py
from __future__ import annotations

import json
import unittest
from collections.abc import Callable
from typing import Any

from researchguard.agent import BoundedResearchAgentController
from researchguard.tools import (
    EvidenceBundle,
    GateDecision,
    ToolError,
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
    "content": "CRAG uses a retrieval evaluator before generation.",
    "source": "paper_crag.pdf",
    "provenance": {"source_block_ids": ["p5-b2"]},
}
ANSWER = {
    "answer": "CRAG evaluates retrieval quality.",
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
    "evidence_chunk_ids": [EVIDENCE["chunk_id"]],
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
    description = "Synthetic adaptive-agent tool."

    def __init__(self, name: str, handler: Callable[..., ToolResult]):
        self.name = name
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema={},
        )

    def invoke(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        return self.handler(**kwargs)


def result(
    name: str,
    *,
    status: str = "success",
    data: dict[str, Any] | None = None,
    reason: str | None = None,
    error: ToolError | None = None,
) -> ToolResult:
    return ToolResult.create(
        status=status,
        message=status,
        reason=reason,
        tool_name=name,
        tool_version="test",
        latency_ms=1,
        data=data or {},
        error=error,
    )


def registry_for(
    retrieval_modes: list[str],
    *,
    assessment_modes: list[str] | None = None,
) -> tuple[ToolRegistry, dict[str, FakeTool]]:
    registry = ToolRegistry()
    tools: dict[str, FakeTool] = {}
    retrieval_count = 0
    assessment_count = 0

    def retrieve(**kwargs: Any) -> ToolResult:
        nonlocal retrieval_count
        mode = retrieval_modes[min(retrieval_count, len(retrieval_modes) - 1)]
        retrieval_count += 1
        if mode == "error":
            return result(
                "retrieve_evidence",
                status="failed",
                reason="retrieval_failed",
                error=ToolError(
                    code="retrieval_failed",
                    category="retrieval_failure",
                    message="synthetic failure",
                    retryable=True,
                ),
            )
        if mode == "empty":
            return result("retrieve_evidence", data={"evidence": []})
        bundle = EvidenceBundle.create(
            query=str(kwargs["query"]),
            evidence=[EVIDENCE],
        )
        return result(
            "retrieve_evidence",
            data={
                "evidence": [EVIDENCE],
                "evidence_bundle": bundle.to_dict(),
            },
        )

    def assess(**kwargs: Any) -> ToolResult:
        nonlocal assessment_count
        bundle = EvidenceBundle.from_mapping(kwargs["evidence_bundle"])
        modes = assessment_modes or ["strong"]
        mode = modes[min(assessment_count, len(modes) - 1)]
        assessment_count += 1
        supporting = list(bundle.chunk_ids) if mode in {"strong", "partial"} else []
        assessment = {
            "support_level": mode,
            "answerable": mode == "strong",
            "reason": f"synthetic_{mode}",
            "confidence": 0.9,
            "supporting_chunk_ids": supporting,
        }
        gate = GateDecision.from_assessment(
            evidence_bundle_id=bundle.bundle_id,
            assessment=assessment,
        )
        return result(
            "assess_evidence",
            status="success" if mode == "strong" else "rejected",
            reason=None if mode == "strong" else "insufficient_evidence",
            data={"assessment": assessment, "gate_decision": gate.to_dict()},
        )

    def generate(**kwargs: Any) -> ToolResult:
        bundle = EvidenceBundle.from_mapping(kwargs["evidence_bundle"])
        return result(
            "generate_grounded_answer",
            data={"answer": ANSWER, "evidence_bundle_id": bundle.bundle_id},
        )

    tools["retrieve_evidence"] = FakeTool("retrieve_evidence", retrieve)
    tools["assess_evidence"] = FakeTool("assess_evidence", assess)
    tools["generate_grounded_answer"] = FakeTool(
        "generate_grounded_answer", generate
    )
    tools["audit_answer"] = FakeTool(
        "audit_answer",
        lambda **_: result(
            "audit_answer",
            data={"audit": {"audit_completed": True, "overall_grounded": True}},
        ),
    )
    tools["search_scholarly_sources"] = FakeTool(
        "search_scholarly_sources",
        lambda **_: result(
            "search_scholarly_sources",
            data={
                "candidate_papers": [],
                "metadata_only": True,
                "evidence_eligible": False,
            },
        ),
    )
    for tool in tools.values():
        registry.register(tool)
    return registry, tools


class ReplannerTests(unittest.TestCase):
    def test_empty_retrieval_expands_then_completes(self) -> None:
        registry, tools = registry_for(["empty", "success"])
        controller = BoundedResearchAgentController(
            registry=registry,
            memory_enabled=False,
        )

        state = controller.run("How does CRAG reduce hallucination?")

        self.assertEqual(state.status, "completed")
        self.assertEqual(len(state.plan_revisions), 1)
        self.assertEqual(
            state.plan_revisions[0]["reason"],
            "expanded_retrieval_after_miss",
        )
        self.assertEqual(len(tools["retrieve_evidence"].calls), 2)
        self.assertTrue(tools["retrieve_evidence"].calls[1]["rewrite"])
        self.assertEqual(tools["retrieve_evidence"].calls[1]["candidate_k"], 160)

    def test_second_retrieval_miss_discovers_metadata_then_rejects(self) -> None:
        registry, tools = registry_for(["empty", "empty"])
        controller = BoundedResearchAgentController(
            registry=registry,
            memory_enabled=False,
        )

        state = controller.run("How does an absent method work?")

        self.assertEqual(state.status, "rejected")
        self.assertEqual(
            state.reason,
            "no_corpus_evidence_scholarly_candidates_only",
        )
        self.assertEqual(len(state.plan_revisions), 2)
        self.assertEqual(len(tools["search_scholarly_sources"].calls), 1)
        self.assertEqual(len(tools["generate_grounded_answer"].calls), 0)

    def test_partial_evidence_triggers_expanded_retrieval(self) -> None:
        registry, tools = registry_for(
            ["success", "success"],
            assessment_modes=["partial", "strong"],
        )
        controller = BoundedResearchAgentController(
            registry=registry,
            memory_enabled=False,
        )

        state = controller.run("Explain the supported mechanism and its limitation.")

        self.assertEqual(state.status, "completed")
        self.assertEqual(
            state.plan_revisions[0]["reason"],
            "expanded_retrieval_after_partial_evidence",
        )
        self.assertEqual(len(tools["assess_evidence"].calls), 2)

    def test_retrieval_tool_failure_recovers_with_bounded_revision(self) -> None:
        registry, tools = registry_for(["error", "success"])
        controller = BoundedResearchAgentController(
            registry=registry,
            memory_enabled=False,
        )

        state = controller.run("How does CRAG work?")

        self.assertEqual(state.status, "completed")
        self.assertEqual(len(state.plan_revisions), 1)
        self.assertEqual(len(tools["retrieve_evidence"].calls), 2)

    def test_trace_serializes_previous_observation_and_new_plan(self) -> None:
        registry, _ = registry_for(["empty", "success"])
        state = BoundedResearchAgentController(
            registry=registry,
            memory_enabled=False,
        ).run("How does CRAG work?")

        trace = TraceCollector().collect(state)
        payload = json.loads(trace.to_json())

        self.assertEqual(len(payload["plan_revisions"]), 1)
        revision = payload["plan_revisions"][0]
        self.assertTrue(revision["previous_plan"])
        self.assertEqual(revision["observation"]["evidence_count"], 0)
        self.assertTrue(revision["new_plan"])
        self.assertIn(
            "replan",
            [event["stage"] for event in payload["timeline"]],
        )


if __name__ == "__main__":
    unittest.main()
