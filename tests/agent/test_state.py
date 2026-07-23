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
            task_type="comparison",
            plan=[{"step_id": 1, "tool": "retrieve_evidence"}],
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
        self.assertEqual(payload["task_type"], "comparison")
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
