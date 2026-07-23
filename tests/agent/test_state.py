# C:\Users\18449\Desktop\researchguard_workspace\tests\agent\test_state.py
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from researchguard.agent.state import AGENT_STATE_SCHEMA_VERSION, ResearchAgentState


class ResearchAgentStateTests(unittest.TestCase):
    def test_state_is_json_serializable_and_restorable(self) -> None:
        state = ResearchAgentState(
            query="Compare CRAG and Self-RAG",
            task_type="paper_comparison",
            workflow_name="paper_comparison",
            workflow_input={"comparison_dimensions": ["method", "limitation"]},
            workflow_steps=[{"step": 1, "tool_name": "search_scholarly_sources"}],
            workflow_result={"status": "success"},
            memory_status={"enabled": True, "persisted": True, "errors": []},
            evidence=[
                {
                    "chunk_id": "paper::chunk-1",
                    "doc_id": "paper",
                    "section": "method",
                    "page": 2,
                    "content": "Evidence text.",
                    "source": "paper.pdf",
                    "score": 0.9,
                    "provenance": {},
                }
            ],
            status="planned",
        )

        payload = json.loads(state.to_json())

        self.assertEqual(payload["schema_version"], AGENT_STATE_SCHEMA_VERSION)
        self.assertEqual(payload["task_type"], "paper_comparison")
        self.assertEqual(payload["workflow_name"], "paper_comparison")
        self.assertEqual(payload["workflow_input"]["comparison_dimensions"], ["method", "limitation"])
        self.assertEqual(payload["workflow_result"]["status"], "success")
        self.assertTrue(payload["memory_status"]["persisted"])
        self.assertEqual(payload["evidence"][0]["chunk_id"], "paper::chunk-1")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "agent_state.json"
            state.save(path)
            restored = ResearchAgentState.load(path)

        self.assertEqual(restored.to_dict(), state.to_dict())

    def test_unknown_schema_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ResearchAgentState.from_dict(
                {
                    "schema_version": "future.v99",
                    "query": "Question",
                }
            )


if __name__ == "__main__":
    unittest.main()
