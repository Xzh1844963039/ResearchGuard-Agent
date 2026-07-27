# scripts/validate_api_v1.py
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import create_app  # noqa: E402


class SmokeController:
    def run(
        self,
        query: str,
        *,
        task_type: str | None = None,
        workflow_input: dict[str, Any] | None = None,
    ) -> Any:
        del workflow_input
        value = {
            "run_id": "api-smoke-run",
            "query": query,
            "task_type": task_type or "qa",
            "status": "completed",
            "reason": None,
            "plan": [{"skill": "retrieve_evidence"}],
            "planner_plan": {"task_type": task_type or "qa"},
            "planner_metadata": {"mode": "smoke"},
            "skill_history": [{"skill_name": "retrieve_evidence"}],
            "tool_history": [{"tool_name": "retrieve_evidence"}],
            "evidence_bundle": {"evidence_records": []},
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


def main() -> None:
    client = TestClient(create_app(lambda: SmokeController()))
    health = client.get("/health")
    research = client.post(
        "/research",
        json={
            "query": "How does evidence-grounded retrieval work?",
            "workflow": "literature_review",
        },
    )
    invalid = client.post("/research", json={"query": "   "})
    hard_checks = {
        "health_failure": int(
            health.status_code != 200
            or health.json().get("status") != "ok"
        ),
        "controller_route_failure": int(
            research.status_code != 200
            or research.json().get("run_id") != "api-smoke-run"
            or research.json().get("task_type") != "literature_review"
        ),
        "input_validation_failure": int(invalid.status_code != 422),
    }
    report = {
        "status": "PASS" if not sum(hard_checks.values()) else "FAIL",
        "hard_checks": hard_checks,
        "health": health.json(),
        "research_schema_keys": sorted(research.json()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
