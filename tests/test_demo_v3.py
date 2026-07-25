# C:\Users\18449\Desktop\researchguard_workspace\tests\test_demo_v3.py
from __future__ import annotations

import unittest

from demo.utils import (
    agent_answer_view,
    agent_audit_view,
    agent_plan_rows,
    evidence_bundle_view,
    memory_trace_view,
    skill_timeline_rows,
    tool_call_rows,
)


class DemoV3ViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "query": "A question",
            "plan": [{"skill": "retrieve_evidence", "purpose": "Retrieve"}],
            "skill_history": [
                {
                    "skill_name": "retrieve_evidence",
                    "output_status": "success",
                    "evidence_count": 2,
                }
            ],
            "tool_history": [
                {
                    "tool_name": "retrieve_evidence",
                    "output_status": "success",
                    "api_call_count": 1,
                }
            ],
            "evidence_bundle": {
                "bundle_id": "bundle-1",
                "query": "A question",
                "evidence_records": [{}, {}],
            },
            "answer": {
                "answer": "Grounded.",
                "citations": [{"chunk_id": "c1"}],
            },
            "audit_result": {
                "audit_completed": True,
                "claims": [{"text": "Grounded.", "status": "supported"}],
            },
            "memory_status": {"persisted": True},
            "memory_context": {"matched_run_ids": ["r1"]},
            "memory_snapshot": {"run": {"run_id": "r2"}},
        }

    def test_agent_trace_views_are_separate(self) -> None:
        self.assertEqual("retrieve_evidence", agent_plan_rows(self.state)[0]["skill"])
        self.assertEqual("retrieve_evidence", skill_timeline_rows(self.state)[0]["skill"])
        self.assertEqual("retrieve_evidence", tool_call_rows(self.state)[0]["tool"])

    def test_evidence_answer_and_audit_views(self) -> None:
        self.assertEqual(2, evidence_bundle_view(self.state)["evidence_count"])
        self.assertTrue(agent_answer_view(self.state)["generated"])
        self.assertTrue(agent_audit_view(self.state)["completed"])

    def test_memory_trace_preserves_sections(self) -> None:
        trace = memory_trace_view(self.state)
        self.assertTrue(trace["status"]["persisted"])
        self.assertEqual(["r1"], trace["context"]["matched_run_ids"])
        self.assertEqual("r2", trace["snapshot"]["run"]["run_id"])


if __name__ == "__main__":
    unittest.main()
