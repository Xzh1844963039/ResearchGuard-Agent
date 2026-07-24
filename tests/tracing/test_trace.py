# C:\Users\18449\Desktop\researchguard_workspace\tests\tracing\test_trace.py
from __future__ import annotations

import json
import unittest

from researchguard.agent import ResearchAgentState
from researchguard.tracing import (
    TraceCollector,
    format_trace_json,
    format_trace_markdown,
)


class AgentTraceTests(unittest.TestCase):
    def test_trace_contains_complete_serializable_chain(self) -> None:
        state = ResearchAgentState(
            query="Compare CRAG and standard RAG.",
            task_type="paper_comparison",
            plan=[{"step_id": 1, "tool": "retrieve_evidence"}],
            workflow_name="paper_comparison",
            workflow_steps=[
                {
                    "step": 1,
                    "tool_name": "retrieve_evidence",
                    "status": "success",
                    "trace_id": "trace-1",
                }
            ],
            tool_history=[
                {
                    "tool_name": "retrieve_evidence",
                    "output_status": "success",
                    "latency_ms": 2.0,
                    "trace_id": "trace-1",
                }
            ],
            observations=[{"tool_name": "retrieve_evidence", "status": "success"}],
            evidence=[
                {
                    "chunk_id": "paper-crag::chunk-7",
                    "doc_id": "paper-crag",
                    "section": "method",
                    "page": 5,
                    "content": "Canonical evidence.",
                    "source": "paper_crag.pdf",
                }
            ],
            answer={"answer": "A grounded comparison."},
            audit_result={"overall_grounded": True},
            memory_status={"enabled": True, "persisted": True},
            memory_context={"matched_run_ids": ["agent-old"]},
            status="completed",
        )
        snapshot = {
            "run": {"run_id": state.run_id},
            "evidence_ledger": [{"claim_id": "claim-1"}],
            "failures": [],
        }

        trace = TraceCollector().collect(state, memory_snapshot=snapshot)
        payload = json.loads(trace.to_json())

        self.assertEqual(payload["query"], state.query)
        self.assertEqual(payload["workflow_name"], "paper_comparison")
        self.assertEqual(payload["tool_calls"][0]["trace_id"], "trace-1")
        self.assertEqual(payload["evidence"][0]["chunk_id"], "paper-crag::chunk-7")
        self.assertEqual(payload["memory"]["snapshot"]["evidence_ledger"][0]["claim_id"], "claim-1")
        self.assertEqual(payload["memory_context"]["matched_run_ids"], ["agent-old"])
        self.assertEqual(payload["timeline"][-1]["stage"], "final")

    def test_cli_formatters_do_not_drop_provenance(self) -> None:
        state = ResearchAgentState(
            query="Audit a claim.",
            evidence=[
                {
                    "chunk_id": "doc::chunk-1",
                    "doc_id": "doc",
                    "section": "results",
                    "page": 4,
                    "content": "Result evidence.",
                    "source": "doc.pdf",
                }
            ],
            status="rejected",
            reason="insufficient_evidence",
        )
        trace = TraceCollector().collect(state)

        self.assertIn("doc::chunk-1", format_trace_json(trace))
        self.assertIn("insufficient_evidence", format_trace_markdown(trace))

    def test_workflow_selection_is_preserved_as_plan_summary(self) -> None:
        state = ResearchAgentState(
            query="Review RAG literature.",
            task_type="literature_review",
            workflow_name="literature_review",
            status="completed",
        )

        trace = TraceCollector().collect(state)

        self.assertEqual(trace.plan[0]["mode"], "bounded_workflow")
        self.assertEqual(trace.plan[0]["workflow"], "literature_review")


if __name__ == "__main__":
    unittest.main()
