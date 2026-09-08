# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tracing\trace.py
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping


AGENT_TRACE_SCHEMA_VERSION = "researchguard.agent_trace.v4"
AGENT_TRACE_VERSION = "4.0.0"


@dataclass(frozen=True)
class AgentTrace:
    run_id: str
    query: str
    task_type: str
    plan: tuple[Mapping[str, Any], ...]
    planner_plan: Mapping[str, Any] | None
    planner_metadata: Mapping[str, Any]
    subgoals: tuple[Mapping[str, Any], ...]
    subgoal_results: Mapping[str, Mapping[str, Any]]
    objective_status: tuple[Mapping[str, Any], ...]
    completed_objectives: tuple[Mapping[str, Any], ...]
    missing_objectives: tuple[Mapping[str, Any], ...]
    plan_revisions: tuple[Mapping[str, Any], ...]
    workflow_name: str | None
    workflow_steps: tuple[Mapping[str, Any], ...]
    skill_calls: tuple[Mapping[str, Any], ...]
    tool_calls: tuple[Mapping[str, Any], ...]
    observations: tuple[Mapping[str, Any], ...]
    evidence: tuple[Mapping[str, Any], ...]
    evidence_bundle: Mapping[str, Any] | None
    gate_decision: Mapping[str, Any] | None
    answer: Mapping[str, Any] | None
    audit: Mapping[str, Any] | None
    memory: Mapping[str, Any]
    memory_context: Mapping[str, Any]
    status: str
    reason: str | None
    created_at: str
    completed_at: str
    timeline: tuple[Mapping[str, Any], ...]
    decision_summary: Mapping[str, Any]
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
            "planner_plan": (
                copy.deepcopy(dict(self.planner_plan))
                if self.planner_plan
                else None
            ),
            "planner_metadata": copy.deepcopy(dict(self.planner_metadata)),
            "subgoals": copy.deepcopy(list(self.subgoals)),
            "subgoal_results": copy.deepcopy(dict(self.subgoal_results)),
            "objective_status": copy.deepcopy(list(self.objective_status)),
            "completed_objectives": copy.deepcopy(list(self.completed_objectives)),
            "missing_objectives": copy.deepcopy(list(self.missing_objectives)),
            "plan_revisions": copy.deepcopy(list(self.plan_revisions)),
            "workflow_name": self.workflow_name,
            "workflow_steps": copy.deepcopy(list(self.workflow_steps)),
            "skill_calls": copy.deepcopy(list(self.skill_calls)),
            "tool_calls": copy.deepcopy(list(self.tool_calls)),
            "observations": copy.deepcopy(list(self.observations)),
            "evidence": copy.deepcopy(list(self.evidence)),
            "evidence_bundle": (
                copy.deepcopy(dict(self.evidence_bundle))
                if self.evidence_bundle
                else None
            ),
            "gate_decision": (
                copy.deepcopy(dict(self.gate_decision))
                if self.gate_decision
                else None
            ),
            "answer": copy.deepcopy(dict(self.answer)) if self.answer else None,
            "audit": copy.deepcopy(dict(self.audit)) if self.audit else None,
            "memory": copy.deepcopy(dict(self.memory)),
            "memory_context": copy.deepcopy(dict(self.memory_context)),
            "status": self.status,
            "reason": self.reason,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "timeline": copy.deepcopy(list(self.timeline)),
            "decision_summary": copy.deepcopy(dict(self.decision_summary)),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent)
