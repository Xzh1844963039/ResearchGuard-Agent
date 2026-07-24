# C:\Users\18449\Desktop\researchguard_workspace\tests\memory\test_memory.py
from __future__ import annotations

import json
import tempfile
import unittest
from typing import Any

from researchguard.agent import BoundedResearchAgentController, ResearchAgentState
from researchguard.memory import (
    EvidenceLedger,
    FailureStore,
    ResearchMemory,
    ResearchRunStore,
    RunRecord,
)
from researchguard.memory.schemas import content_hash
from researchguard.tools import ToolRegistry, ToolResult, ToolSpec
from researchguard.workflows import ClaimAuditWorkflow, WorkflowRegistry


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
        "source_block_ids": ["p5-b2"],
        "content_types": ["paragraph"],
    },
}


class FakeTool:
    version = "test"
    description = "Synthetic memory integration tool."

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


def success(name: str, data: dict[str, Any]) -> ToolResult:
    return ToolResult.create(
        status="success",
        message="ok",
        tool_name=name,
        tool_version="test",
        latency_ms=1,
        data=data,
    )


def tool_registry(*, unsupported: bool = False) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        FakeTool(
            "retrieve_evidence",
            lambda **_: success("retrieve_evidence", {"evidence": [EVIDENCE]}),
        )
    )
    if unsupported:
        assessment = FakeTool(
            "assess_evidence",
            lambda **_: ToolResult.create(
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
            ),
        )
    else:
        assessment = FakeTool(
            "assess_evidence",
            lambda **_: success(
                "assess_evidence",
                {
                    "assessment": {
                        "support_level": "strong",
                        "answerable": True,
                        "confidence": 0.95,
                        "supporting_chunk_ids": [EVIDENCE["chunk_id"]],
                    }
                },
            ),
        )
    registry.register(assessment)
    registry.register(
        FakeTool(
            "audit_answer",
            lambda **_: success(
                "audit_answer",
                {
                    "audit": {
                        "audit_completed": True,
                        "overall_grounded": True,
                        "claims": [
                            {
                                "id": "claim-1",
                                "text": "CRAG uses a retrieval evaluator.",
                                "support_level": "supported",
                                "citations": [
                                    {
                                        "chunk_id": EVIDENCE["chunk_id"],
                                        "doc_id": EVIDENCE["doc_id"],
                                        "section": EVIDENCE["section"],
                                        "page": EVIDENCE["page"],
                                    }
                                ],
                            }
                        ],
                    }
                },
            ),
        )
    )
    return registry


def workflow_registry(registry: ToolRegistry) -> WorkflowRegistry:
    workflows = WorkflowRegistry()
    workflows.register(ClaimAuditWorkflow(registry))
    return workflows


class MemorySchemaTests(unittest.TestCase):
    def test_run_record_json_roundtrip(self) -> None:
        state = ResearchAgentState(
            query="Review RAG hallucination mitigation",
            task_type="literature_review",
            workflow_name="literature_review",
            status="completed",
            evidence=[EVIDENCE],
            answer={"answer": "A grounded summary."},
            audit_result={"overall_grounded": True},
        )
        record = RunRecord.from_state(
            state,
            latency_ms=12.5,
            claim_ids=("claim-1",),
        )

        restored = RunRecord.from_dict(json.loads(record.to_json()))

        self.assertEqual(restored, record)
        self.assertEqual(restored.evidence_ids, (EVIDENCE["chunk_id"],))
        self.assertEqual(restored.schema_version, "researchguard.memory.run_record.v1")

    def test_evidence_ledger_preserves_canonical_provenance_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state = ResearchAgentState(
                query="Audit CRAG claim",
                task_type="claim_audit",
                workflow_name="claim_audit",
                status="completed",
                evidence=[EVIDENCE],
                workflow_result={
                    "output": {
                        "claim": "CRAG uses a retrieval evaluator.",
                        "support_level": "strong",
                        "citations": [
                            {
                                "chunk_id": EVIDENCE["chunk_id"],
                                "doc_id": EVIDENCE["doc_id"],
                                "section": EVIDENCE["section"],
                                "page": EVIDENCE["page"],
                            }
                        ],
                    }
                },
            )
            ledger = EvidenceLedger(temp_dir)
            records = ledger.build_from_state(state)
            ledger.add(records[0])
            restored = EvidenceLedger(temp_dir).for_run(state.run_id)[0]

        evidence_ref = restored.evidence_refs[0]
        self.assertEqual(evidence_ref.chunk_id, EVIDENCE["chunk_id"])
        self.assertEqual(evidence_ref.doc_id, EVIDENCE["doc_id"])
        self.assertEqual(evidence_ref.section, EVIDENCE["section"])
        self.assertEqual(evidence_ref.page, EVIDENCE["page"])
        self.assertEqual(evidence_ref.hash, content_hash(EVIDENCE["content"]))

    def test_each_workflow_result_can_create_a_ledger_record(self) -> None:
        workflow_outputs = {
            "literature_review": {
                "summary": "CRAG uses a retrieval evaluator.",
                "citations": [{"chunk_id": EVIDENCE["chunk_id"]}],
            },
            "paper_comparison": {
                "summary": "The compared paper uses retrieval correction.",
                "citations": [{"chunk_id": EVIDENCE["chunk_id"]}],
            },
            "claim_audit": {
                "claim": "CRAG uses a retrieval evaluator.",
                "support_level": "strong",
                "citations": [{"chunk_id": EVIDENCE["chunk_id"]}],
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = EvidenceLedger(temp_dir)
            records = []
            for workflow_name, output in workflow_outputs.items():
                state = ResearchAgentState(
                    query=f"Run {workflow_name}",
                    task_type=workflow_name,
                    workflow_name=workflow_name,
                    status="completed",
                    evidence=[EVIDENCE],
                    workflow_result={"output": output},
                )
                record = ledger.build_from_state(state)[0]
                ledger.add(record)
                records.append(record)

        self.assertEqual(len(records), 3)
        self.assertEqual(
            {record.source for record in records},
            {f"workflow:{name}" for name in workflow_outputs},
        )
        self.assertTrue(
            all(record.evidence_refs[0].chunk_id == EVIDENCE["chunk_id"] for record in records)
        )

    def test_supported_claim_without_canonical_evidence_is_not_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state = ResearchAgentState(
                query="Unsupported provenance",
                workflow_name="claim_audit",
                status="completed",
                evidence=[EVIDENCE],
                workflow_result={
                    "output": {
                        "claim": "A claim with an unknown citation.",
                        "support_level": "strong",
                        "citations": [{"chunk_id": "missing::chunk"}],
                    }
                },
            )

            records = EvidenceLedger(temp_dir).build_from_state(state)

        self.assertEqual(records, [])

    def test_failure_types_cover_no_evidence_tool_and_rejected_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FailureStore(temp_dir)
            no_evidence = ResearchAgentState(query="No evidence", status="rejected")
            no_evidence.reason = "not_found"
            tool_failure = ResearchAgentState(query="Tool failure", status="failed")
            tool_failure.reason = "tool_error"
            rejected_answer = ResearchAgentState(
                query="Rejected answer",
                status="rejected",
                evidence=[EVIDENCE],
            )
            rejected_answer.reason = "answer_not_grounded"

            self.assertEqual(store.from_state(no_evidence).failure_type, "no_evidence")
            self.assertEqual(store.from_state(tool_failure).failure_type, "tool_failure")
            self.assertEqual(
                store.from_state(rejected_answer).failure_type,
                "rejected_answer",
            )


class MemoryPersistenceTests(unittest.TestCase):
    def test_run_persists_and_reloads_from_new_store_instance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ResearchMemory(temp_dir)
            state = ResearchAgentState(query="Review retrieval correction")
            memory.start_run(state)
            state.set_status("completed")
            memory.complete_run(state)

            restored = ResearchMemory(temp_dir).show(state.run_id)

        self.assertIsNotNone(restored)
        self.assertEqual(restored["run"]["status"], "completed")
        self.assertEqual(restored["run"]["query"], state.query)

    def test_failure_memory_records_insufficient_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ResearchMemory(temp_dir)
            state = ResearchAgentState(
                query="Unsupported research question",
                workflow_name="claim_audit",
            )
            memory.start_run(state)
            state.set_status("rejected", "insufficient_evidence")
            memory.complete_run(state)

            failures = ResearchMemory(temp_dir).failures.for_run(state.run_id)

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].failure_type, "insufficient_evidence")
        self.assertEqual(failures[0].workflow_name, "claim_audit")

    def test_runs_are_isolated_and_searchable_by_keyword_and_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ResearchMemory(temp_dir)
            first = ResearchAgentState(
                query="Review RAG hallucination mitigation",
                workflow_name="literature_review",
            )
            second = ResearchAgentState(
                query="Audit a citation claim",
                workflow_name="claim_audit",
            )
            for state in (first, second):
                memory.start_run(state)
                state.set_status("completed")
                memory.complete_run(state)

            store = ResearchRunStore(temp_dir)
            review_matches = memory.find_previous_runs(
                "RAG hallucination",
                workflow="literature_review",
            )
            audit_matches = store.find_previous_runs(
                "citation",
                workflow="claim_audit",
            )

        self.assertEqual([item.run_id for item in review_matches], [first.run_id])
        self.assertEqual([item.run_id for item in audit_matches], [second.run_id])
        self.assertNotEqual(review_matches[0].run_id, audit_matches[0].run_id)

    def test_search_context_returns_bounded_advisory_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ResearchMemory(temp_dir)
            previous = ResearchAgentState(
                query="Review CRAG hallucination mitigation",
                workflow_name="literature_review",
                candidate_papers=[
                    {
                        "paper_id": "paper-crag",
                        "title": "Corrective Retrieval Augmented Generation",
                        "source": "semantic_scholar",
                    }
                ],
            )
            memory.start_run(previous)
            previous.set_status("rejected", "insufficient_evidence")
            memory.complete_run(previous)

            context = memory.search_context("CRAG hallucination", limit=3)

        self.assertEqual(context["matched_run_ids"], [previous.run_id])
        self.assertEqual(context["previous_workflows"], ["literature_review"])
        self.assertEqual(context["previous_papers"][0]["paper_id"], "paper-crag")
        self.assertEqual(
            context["previous_failures"][0]["failure_type"],
            "insufficient_evidence",
        )
        self.assertEqual(context["schema_version"], "researchguard.memory_context.v1")


class ControllerMemoryTests(unittest.TestCase):
    def test_controller_persists_run_and_evidence_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = tool_registry()
            memory = ResearchMemory(temp_dir)
            controller = BoundedResearchAgentController(
                registry=registry,
                workflow_registry=workflow_registry(registry),
                memory=memory,
            )

            state = controller.run(
                "Verify this claim: CRAG uses a retrieval evaluator.",
                task_type="claim_audit",
            )
            restored = ResearchMemory(temp_dir).show(state.run_id)

        self.assertEqual(state.status, "completed")
        self.assertTrue(state.memory_status["persisted"])
        self.assertEqual(restored["run"]["workflow_name"], "claim_audit")
        self.assertEqual(restored["run"]["tool_trace"][0]["tool_name"], "retrieve_evidence")
        self.assertEqual(
            restored["evidence_ledger"][0]["evidence_refs"][0]["chunk_id"],
            EVIDENCE["chunk_id"],
        )

    def test_unsupported_controller_run_writes_failure_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = tool_registry(unsupported=True)
            memory = ResearchMemory(temp_dir)
            controller = BoundedResearchAgentController(
                registry=registry,
                workflow_registry=workflow_registry(registry),
                memory=memory,
            )

            state = controller.run(
                "Verify this claim: unsupported claim.",
                task_type="claim_audit",
            )
            restored = ResearchMemory(temp_dir).show(state.run_id)

        self.assertEqual(state.status, "rejected")
        self.assertTrue(state.memory_status["failure_recorded"])
        self.assertEqual(restored["evidence_ledger"], [])
        self.assertEqual(
            restored["failures"][0]["failure_type"],
            "insufficient_evidence",
        )

    def test_controller_uses_previous_run_as_bounded_planner_advisory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = tool_registry()
            memory = ResearchMemory(temp_dir)
            controller = BoundedResearchAgentController(
                registry=registry,
                workflow_registry=workflow_registry(registry),
                memory=memory,
            )
            first = controller.run(
                "Verify this claim: CRAG uses a retrieval evaluator.",
                task_type="claim_audit",
            )
            second = controller.run(
                "Verify this claim: CRAG uses a retrieval evaluator.",
                task_type="claim_audit",
            )

        self.assertIn(first.run_id, second.memory_context["matched_run_ids"])
        self.assertNotIn(second.run_id, second.memory_context["matched_run_ids"])
        self.assertEqual(second.task_type, "claim_audit")
        self.assertEqual(second.workflow_name, "claim_audit")

    def test_memory_failure_does_not_change_successful_agent_result(self) -> None:
        class BrokenMemory:
            def start_run(self, state: Any) -> None:
                del state
                raise OSError("synthetic start failure")

            def complete_run(self, state: Any) -> dict[str, Any]:
                del state
                raise OSError("synthetic completion failure")

        registry = tool_registry()
        controller = BoundedResearchAgentController(
            registry=registry,
            workflow_registry=workflow_registry(registry),
            memory=BrokenMemory(),  # type: ignore[arg-type]
        )

        state = controller.run(
            "Verify this claim: CRAG uses a retrieval evaluator.",
            task_type="claim_audit",
        )

        self.assertEqual(state.status, "completed")
        self.assertFalse(state.memory_status["persisted"])
        self.assertEqual(len(state.memory_status["errors"]), 2)


if __name__ == "__main__":
    unittest.main()
