# C:\Users\18449\Desktop\researchguard_workspace\researchguard\agent\state.py
from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


AGENT_STATE_SCHEMA_VERSION = "researchguard.agent_state.v1"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ResearchAgentState:
    query: str
    task_type: str = "unknown"
    plan: list[dict[str, Any]] = field(default_factory=list)
    workflow_name: str | None = None
    workflow_input: dict[str, Any] = field(default_factory=dict)
    workflow_steps: list[dict[str, Any]] = field(default_factory=list)
    workflow_result: dict[str, Any] | None = None
    current_step: int = 0
    retry_counts: dict[str, int] = field(default_factory=dict)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    candidate_papers: list[dict[str, Any]] = field(default_factory=list)
    answer: dict[str, Any] | None = None
    audit_result: dict[str, Any] | None = None
    status: str = "created"
    reason: str | None = None
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)
    run_id: str = field(default_factory=lambda: f"agent-{uuid.uuid4().hex}")
    schema_version: str = AGENT_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.query = str(self.query).strip()
        if not self.query:
            raise ValueError("ResearchAgentState.query must not be empty.")
        if self.current_step < 0:
            raise ValueError("ResearchAgentState.current_step must not be negative.")
        if any(value < 0 for value in self.retry_counts.values()):
            raise ValueError("ResearchAgentState.retry_counts must not contain negative values.")
        if self.status not in {
            "created",
            "planned",
            "running",
            "completed",
            "rejected",
            "failed",
        }:
            raise ValueError(f"Unsupported agent status: {self.status}")

    def touch(self) -> None:
        self.updated_at = utc_timestamp()

    def set_status(self, status: str, reason: str | None = None) -> None:
        if status not in {"planned", "running", "completed", "rejected", "failed"}:
            raise ValueError(f"Unsupported agent status: {status}")
        self.status = status
        self.reason = reason
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "query": self.query,
            "task_type": self.task_type,
            "plan": copy.deepcopy(self.plan),
            "workflow_name": self.workflow_name,
            "workflow_input": copy.deepcopy(self.workflow_input),
            "workflow_steps": copy.deepcopy(self.workflow_steps),
            "workflow_result": copy.deepcopy(self.workflow_result),
            "current_step": self.current_step,
            "retry_counts": copy.deepcopy(self.retry_counts),
            "tool_history": copy.deepcopy(self.tool_history),
            "observations": copy.deepcopy(self.observations),
            "evidence": copy.deepcopy(self.evidence),
            "candidate_papers": copy.deepcopy(self.candidate_papers),
            "answer": copy.deepcopy(self.answer),
            "audit_result": copy.deepcopy(self.audit_result),
            "status": self.status,
            "reason": self.reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchAgentState":
        schema_version = str(value.get("schema_version", ""))
        if schema_version != AGENT_STATE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported agent state schema: {schema_version or 'missing'}")
        return cls(
            query=str(value.get("query", "")),
            task_type=str(value.get("task_type", "unknown")),
            plan=copy.deepcopy(list(value.get("plan", []))),
            workflow_name=(
                str(value["workflow_name"])
                if value.get("workflow_name") is not None
                else None
            ),
            workflow_input=copy.deepcopy(dict(value.get("workflow_input", {}))),
            workflow_steps=copy.deepcopy(list(value.get("workflow_steps", []))),
            workflow_result=copy.deepcopy(value.get("workflow_result")),
            current_step=int(value.get("current_step", 0)),
            retry_counts={
                str(key): int(item)
                for key, item in dict(value.get("retry_counts", {})).items()
            },
            tool_history=copy.deepcopy(list(value.get("tool_history", []))),
            observations=copy.deepcopy(list(value.get("observations", []))),
            evidence=copy.deepcopy(list(value.get("evidence", []))),
            candidate_papers=copy.deepcopy(list(value.get("candidate_papers", []))),
            answer=copy.deepcopy(value.get("answer")),
            audit_result=copy.deepcopy(value.get("audit_result")),
            status=str(value.get("status", "created")),
            reason=value.get("reason"),
            created_at=str(value.get("created_at", utc_timestamp())),
            updated_at=str(value.get("updated_at", utc_timestamp())),
            run_id=str(value.get("run_id", f"agent-{uuid.uuid4().hex}")),
            schema_version=schema_version,
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "ResearchAgentState":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("Saved agent state must be a JSON object.")
        return cls.from_dict(value)
