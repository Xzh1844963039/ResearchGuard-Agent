# C:\Users\18449\Desktop\researchguard_workspace\tests\workflows\test_workflows.py
from __future__ import annotations

import json
import unittest
from collections.abc import Callable
from typing import Any

from researchguard.agent import BoundedPlanner, BoundedResearchAgentController, ResearchAgentState
from researchguard.tools import ToolRegistry, ToolResult, ToolSpec
from researchguard.workflows import (
    ClaimAuditWorkflow,
    LiteratureReviewWorkflow,
    PaperComparisonWorkflow,
    WorkflowRegistry,
    build_default_workflow_registry,
)


EVIDENCE_A = {
    "chunk_id": "paper-a::chunk-1",
    "doc_id": "paper-a",
    "section": "method",
    "page": 3,
    "page_end": 3,
    "content": "Paper A uses retrieval correction before generation.",
    "source": "paper_a.pdf",
    "score": 0.93,
    "rank": 1,
    "provenance": {"source_block_ids": ["a-p3-b1"], "content_types": ["paragraph"]},
}
EVIDENCE_B = {
    "chunk_id": "paper-b::chunk-2",
    "doc_id": "paper-b",
    "section": "results",
    "page": 6,
    "page_end": 6,
    "content": "Paper B reports a measured retrieval improvement.",
    "source": "paper_b.pdf",
    "score": 0.89,
    "rank": 2,
    "provenance": {"source_block_ids": ["b-p6-b2"], "content_types": ["paragraph"]},
}
CANDIDATES = [
    {
        "schema_version": "researchguard.scholar_paper.v1",
        "title": "Paper A",
        "authors": ["Author A"],
        "year": 2024,
        "venue": "arXiv",
        "doi": None,
        "url": "https://example.org/a",
        "abstract": "METADATA ONLY: this abstract is not answer evidence.",
        "source": "arxiv",
        "paper_id": "arxiv:a",
        "metadata": {},
        "source_type": "arxiv",
        "metadata_only": True,
        "retrieved_at": "2026-01-01T00:00:00+00:00",
    },
    {
        "schema_version": "researchguard.scholar_paper.v1",
        "title": "Paper B",
        "authors": ["Author B"],
        "year": 2025,
        "venue": "Conference",
        "doi": "10.1000/paper-b",
        "url": "https://example.org/b",
        "abstract": "Second metadata-only candidate.",
        "source": "openalex",
        "paper_id": "doi:10.1000/paper-b",
        "metadata": {},
        "source_type": "conference",
        "metadata_only": True,
        "retrieved_at": "2026-01-01T00:00:00+00:00",
    },
]


class FakeTool:
    version = "test"
    description = "Synthetic workflow tool."

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


def success(name: str, data: dict[str, Any]) -> ToolResult:
    return ToolResult.create(
        status="success",
        message="ok",
        tool_name=name,
        tool_version="test",
        latency_ms=1,
        data=data,
    )


def rejected_assessment() -> ToolResult:
    return ToolResult.create(
        status="rejected",
        message="unsupported",
        reason="insufficient_evidence",
        tool_name="assess_evidence",
        tool_version="test",
        latency_ms=1,
        data={
            "assessment": {
                "support_level": "unsupported",
                "answerable": False,
                "supporting_chunk_ids": [],
            }
        },
    )


def answer_artifact() -> dict[str, Any]:
    return {
        "answer": "Paper A corrects retrieval, while Paper B reports an improvement.",
        "citations": [
            {
                "chunk_id": EVIDENCE_A["chunk_id"],
                "doc_id": EVIDENCE_A["doc_id"],
                "section": EVIDENCE_A["section"],
                "page": EVIDENCE_A["page"],
            },
            {
                "chunk_id": EVIDENCE_B["chunk_id"],
                "doc_id": EVIDENCE_B["doc_id"],
                "section": EVIDENCE_B["section"],
                "page": EVIDENCE_B["page"],
            },
        ],
        "confidence": 0.91,
        "refused": False,
        "refusal_reason": None,
        "evidence_chunk_ids": [EVIDENCE_A["chunk_id"], EVIDENCE_B["chunk_id"]],
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
        "latency_ms": 1.0,
    }


def guarded_pipeline_result() -> dict[str, Any]:
    return {
        "final_status": "grounded",
        "retrieval": {"output": {"hits": [EVIDENCE_A, EVIDENCE_B]}},
        "answer_generation": {"output": answer_artifact()},
        "citation_audit": {
            "output": {
                "audit_completed": True,
                "overall_grounded": True,
                "evidence_chunk_ids": [EVIDENCE_A["chunk_id"], EVIDENCE_B["chunk_id"]],
            }
        },
    }


def make_registry(*, unsupported: bool = False) -> tuple[ToolRegistry, dict[str, FakeTool]]:
    registry = ToolRegistry()
    tools: dict[str, FakeTool] = {}
    tools["search_scholarly_sources"] = FakeTool(
        "search_scholarly_sources",
        lambda **_: success(
            "search_scholarly_sources",
            {
                "candidate_papers": CANDIDATES,
                "metadata_only": True,
                "evidence_eligible": False,
            },
        ),
    )

    def retrieve(**kwargs: Any) -> ToolResult:
        filters = kwargs.get("filters") or {}
        doc_ids = filters.get("doc_ids") or []
        if doc_ids == ["paper-a"]:
            evidence = [EVIDENCE_A]
        elif doc_ids == ["paper-b"]:
            evidence = [EVIDENCE_B]
        else:
            evidence = [EVIDENCE_A, EVIDENCE_B]
        return success("retrieve_evidence", {"evidence": evidence})

    tools["retrieve_evidence"] = FakeTool("retrieve_evidence", retrieve)
    tools["assess_evidence"] = FakeTool(
        "assess_evidence",
        (lambda **_: rejected_assessment())
        if unsupported
        else (
            lambda **_: success(
                "assess_evidence",
                {
                    "assessment": {
                        "support_level": "strong",
                        "answerable": True,
                        "confidence": 0.94,
                        "supporting_chunk_ids": [
                            EVIDENCE_A["chunk_id"],
                            EVIDENCE_B["chunk_id"],
                        ],
                    }
                },
            )
        ),
    )
    tools["generate_grounded_answer"] = FakeTool(
        "generate_grounded_answer",
        lambda **_: success(
            "generate_grounded_answer",
            {"pipeline_result": guarded_pipeline_result()},
        ),
    )
    tools["audit_answer"] = FakeTool(
        "audit_answer",
        lambda **_: success(
            "audit_answer",
            {
                "audit": {
                    "audit_completed": True,
                    "overall_grounded": True,
                    "evidence_chunk_ids": [
                        EVIDENCE_A["chunk_id"],
                        EVIDENCE_B["chunk_id"],
                    ],
                }
            },
        ),
    )
    for tool in tools.values():
        registry.register(tool)
    return registry, tools


class WorkflowTests(unittest.TestCase):
    def test_workflow_registry_exposes_stable_specs(self) -> None:
        registry, _ = make_registry()
        workflows = build_default_workflow_registry(registry)

        self.assertEqual(
            workflows.names,
            ("literature_review", "paper_comparison", "claim_audit"),
        )
        serialized = json.dumps(workflows.specs())
        self.assertIn("required_tools", serialized)
        self.assertIn("researchguard.workflow_spec.v1", serialized)

    def test_literature_review_keeps_candidate_metadata_out_of_evidence(self) -> None:
        registry, tools = make_registry()
        state = ResearchAgentState(
            query="Review retrieval correction methods",
            workflow_name="literature_review",
        )

        result = LiteratureReviewWorkflow(registry).run(state)

        self.assertEqual(result.status, "success")
        self.assertEqual(len(tools["search_scholarly_sources"].calls), 1)
        self.assertTrue(all(item["metadata_only"] for item in result.output["papers"]))
        self.assertEqual(
            {item["chunk_id"] for item in result.output["evidence"]},
            {EVIDENCE_A["chunk_id"], EVIDENCE_B["chunk_id"]},
        )
        self.assertNotIn("METADATA ONLY", json.dumps(result.output["evidence"]))
        self.assertIsNotNone(result.output["summary"])
        self.assertEqual(len(tools["generate_grounded_answer"].calls), 1)
        self.assertEqual(len(tools["audit_answer"].calls), 1)
        json.dumps(result.to_dict())

    def test_literature_review_rejects_before_generation_when_unsupported(self) -> None:
        registry, tools = make_registry(unsupported=True)
        state = ResearchAgentState(
            query="Review an unsupported topic",
            workflow_name="literature_review",
        )

        result = LiteratureReviewWorkflow(registry).run(state)

        self.assertEqual(result.status, "rejected")
        self.assertIsNone(result.output["summary"])
        self.assertEqual(tools["generate_grounded_answer"].calls, [])
        self.assertEqual(tools["audit_answer"].calls, [])

    def test_success_status_cannot_bypass_non_strong_evidence_gate(self) -> None:
        registry, tools = make_registry()
        tools["assess_evidence"].handler = lambda **_: success(
            "assess_evidence",
            {
                "assessment": {
                    "support_level": "partial",
                    "answerable": False,
                    "supporting_chunk_ids": [EVIDENCE_A["chunk_id"]],
                }
            },
        )
        state = ResearchAgentState(
            query="Review a partially supported topic",
            workflow_name="literature_review",
        )

        result = LiteratureReviewWorkflow(registry).run(state)

        self.assertEqual(result.status, "rejected")
        self.assertEqual(tools["generate_grounded_answer"].calls, [])
        self.assertEqual(tools["audit_answer"].calls, [])

    def test_paper_comparison_separates_evidence_and_preserves_audit_provenance(self) -> None:
        registry, tools = make_registry()
        state = ResearchAgentState(
            query="Compare Paper A and Paper B",
            task_type="paper_comparison",
            workflow_name="paper_comparison",
            workflow_input={
                "papers": [
                    {"name": "Paper A", "doc_id": "paper-a"},
                    {"name": "Paper B", "doc_id": "paper-b"},
                ],
                "comparison_dimensions": ["method", "metric"],
            },
        )

        result = PaperComparisonWorkflow(registry).run(state)

        self.assertEqual(result.status, "success")
        table = result.output["evidence_table"]
        self.assertEqual(
            [row["doc_id_filter"] for row in table],
            ["paper-a", "paper-b"],
        )
        self.assertEqual(
            [[item["doc_id"] for item in row["evidence"]] for row in table],
            [["paper-a"], ["paper-b"]],
        )
        retrieval_calls = tools["retrieve_evidence"].calls
        self.assertEqual(
            [call["filters"]["doc_ids"] for call in retrieval_calls],
            [["paper-a"], ["paper-b"]],
        )
        audit_call = tools["audit_answer"].calls[0]
        self.assertEqual(
            {item["chunk_id"] for item in audit_call["evidence"]},
            set(audit_call["answer"]["evidence_chunk_ids"]),
        )

    def test_claim_audit_rejects_unsupported_claim_before_audit(self) -> None:
        registry, tools = make_registry(unsupported=True)
        state = ResearchAgentState(
            query="A claim unsupported by the corpus.",
            workflow_name="claim_audit",
        )

        result = ClaimAuditWorkflow(registry).run(state)

        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.output["support_level"], "unsupported")
        self.assertEqual(tools["audit_answer"].calls, [])

    def test_claim_audit_passes_canonical_citations_to_audit(self) -> None:
        registry, tools = make_registry()
        state = ResearchAgentState(
            query="Paper A uses retrieval correction.",
            workflow_name="claim_audit",
        )

        result = ClaimAuditWorkflow(registry).run(state)

        self.assertEqual(result.status, "success")
        audit_call = tools["audit_answer"].calls[0]
        citation_ids = {item["chunk_id"] for item in audit_call["answer"]["citations"]}
        evidence_ids = {item["chunk_id"] for item in audit_call["evidence"]}
        self.assertEqual(citation_ids, evidence_ids)
        self.assertIn(EVIDENCE_A["chunk_id"], citation_ids)
        self.assertEqual(result.output["citations"][0]["page"], EVIDENCE_A["page"])

    def test_planner_and_controller_select_workflow_without_internal_steps(self) -> None:
        registry, _ = make_registry()
        workflows = build_default_workflow_registry(registry)
        planner = BoundedPlanner(registry, workflow_names=workflows.names)
        self.assertEqual(
            planner.create_plan("Review RAG hallucination mitigation").workflow,
            "literature_review",
        )
        self.assertEqual(
            planner.create_plan("Compare Paper A and Paper B").workflow,
            "paper_comparison",
        )
        self.assertEqual(
            planner.create_plan("Verify this claim: Paper A uses correction").workflow,
            "claim_audit",
        )

        controller = BoundedResearchAgentController(
            registry=registry,
            workflow_registry=workflows,
        )
        state = controller.run(
            "Compare Paper A and Paper B",
            workflow_input={
                "papers": [
                    {"name": "Paper A", "doc_id": "paper-a"},
                    {"name": "Paper B", "doc_id": "paper-b"},
                ]
            },
        )

        self.assertEqual(state.status, "completed")
        self.assertEqual(state.workflow_name, "paper_comparison")
        self.assertEqual(state.plan, [])
        self.assertEqual(state.workflow_result["status"], "success")
        self.assertEqual(len(state.workflow_steps), 6)
        self.assertEqual(state.current_step, 6)

    def test_invalid_workflow_input_returns_structured_failure(self) -> None:
        registry, _ = make_registry()
        state = ResearchAgentState(
            query="Review a topic",
            workflow_name="literature_review",
            workflow_input={"candidate_limit": "not-an-integer"},
        )

        result = LiteratureReviewWorkflow(registry).run(state)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.trace, ())
        json.dumps(result.to_dict())

    def test_registry_rejects_duplicate_workflow(self) -> None:
        registry, _ = make_registry()
        workflows = WorkflowRegistry()
        workflow = ClaimAuditWorkflow(registry)
        workflows.register(workflow)

        with self.assertRaises(ValueError):
            workflows.register(workflow)


if __name__ == "__main__":
    unittest.main()
