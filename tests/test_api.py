# C:\Users\18449\Desktop\researchguard_workspace\tests\test_api.py
from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from api.app import create_app


class StubController:
    def run(
        self,
        query: str,
        *,
        task_type: str | None = None,
        workflow_input: dict[str, Any] | None = None,
    ) -> Any:
        del workflow_input
        value = {
            "run_id": "test-run",
            "query": query,
            "task_type": task_type or "qa",
            "status": "rejected",
            "reason": "insufficient_evidence",
            "plan": [{"skill": "retrieve_evidence"}],
            "planner_plan": {},
            "planner_metadata": {},
            "skill_history": [],
            "tool_history": [],
            "evidence_bundle": None,
            "evidence": [],
            "answer": None,
            "audit_result": None,
            "memory_status": {"enabled": False},
            "memory_context": {},
            "plan_revisions": [],
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:01+00:00",
        }
        return SimpleNamespace(to_dict=lambda: value)


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(lambda: StubController()))

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(200, response.status_code)
        self.assertEqual("ok", response.json()["status"])

    def test_research_calls_controller_and_returns_trace_schema(self) -> None:
        response = self.client.post(
            "/research",
            json={"query": "Audit this claim.", "workflow": "claim_audit"},
        )
        self.assertEqual(200, response.status_code)
        result = response.json()
        self.assertEqual("claim_audit", result["task_type"])
        self.assertIn("plan", result)
        self.assertIn("skills", result)
        self.assertIn("tool_calls", result)
        self.assertIn("evidence", result)
        self.assertIn("audit", result)
        self.assertIn("memory", result)

    def test_rejects_unknown_workflow(self) -> None:
        response = self.client.post(
            "/research",
            json={"query": "A question.", "workflow": "autonomous"},
        )
        self.assertEqual(422, response.status_code)


if __name__ == "__main__":
    unittest.main()
