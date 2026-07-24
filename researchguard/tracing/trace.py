# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tracing\trace.py
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping


AGENT_TRACE_SCHEMA_VERSION = "researchguard.agent_trace.v2"
AGENT_TRACE_VERSION = "2.0.0"


@dataclass(frozen=True)
class AgentTrace:
    run_id: str
    query: str
    task_type: str
    plan: tuple[Mapping[str, Any], ...]
    plan_revisions: tuple[Mapping[str, Any], ...]
    workflow_name: str | None
    workflow_steps: tuple[Mapping[str, Any], ...]
    tool_calls: tuple[Mapping[str, Any], ...]
    observations: tuple[Mapping[str, Any], ...]
    evidence: tuple[Mapping[str, Any], ...]
    answer: Mapping[str, Any] | None
    audit: Mapping[str, Any] | None
    memory: Mapping[str, Any]
    memory_context: Mapping[str, Any]
    status: str
    reason: str | None
    created_at: str
    completed_at: str
    timeline: tuple[Mapping[str, Any], ...]
    version: str = AGENT_TRACE_VERSION
    schema_version: str = AGENT_TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.query.strip():
            raise ValueError("AgentTrace requires run_id and query.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "run_id": self.run_id,
            "query": self.query,
            "task_type": self.task_type,
            "plan": copy.deepcopy(list(self.plan)),
            "plan_revisions": copy.deepcopy(list(self.plan_revisions)),
            "workflow_name": self.workflow_name,
            "workflow_steps": copy.deepcopy(list(self.workflow_steps)),
            "tool_calls": copy.deepcopy(list(self.tool_calls)),
            "observations": copy.deepcopy(list(self.observations)),
            "evidence": copy.deepcopy(list(self.evidence)),
            "answer": copy.deepcopy(dict(self.answer)) if self.answer else None,
            "audit": copy.deepcopy(dict(self.audit)) if self.audit else None,
            "memory": copy.deepcopy(dict(self.memory)),
            "memory_context": copy.deepcopy(dict(self.memory_context)),
            "status": self.status,
            "reason": self.reason,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "timeline": copy.deepcopy(list(self.timeline)),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent)
