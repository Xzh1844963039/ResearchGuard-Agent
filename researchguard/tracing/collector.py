# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tracing\collector.py
from __future__ import annotations

import copy
from typing import Any, Mapping

from researchguard.tracing.trace import AgentTrace


class TraceCollector:
    def collect(
        self,
        state: Any,
        *,
        memory_snapshot: Mapping[str, Any] | None = None,
    ) -> AgentTrace:
        memory = {
            "status": copy.deepcopy(state.memory_status),
            "snapshot": copy.deepcopy(dict(memory_snapshot)) if memory_snapshot else None,
        }
        plan = list(copy.deepcopy(state.plan))
        if not plan and state.workflow_name:
            plan = [
                {
                    "mode": "bounded_workflow",
                    "task_type": state.task_type,
                    "workflow": state.workflow_name,
                }
            ]
        return AgentTrace(
            run_id=state.run_id,
            query=state.query,
            task_type=state.task_type,
            plan=tuple(plan),
            planner_plan=copy.deepcopy(
                getattr(state, "planner_plan", None)
            ),
            planner_metadata=copy.deepcopy(
                getattr(state, "planner_metadata", {})
            ),
            plan_revisions=tuple(
                copy.deepcopy(getattr(state, "plan_revisions", ()))
            ),
            workflow_name=state.workflow_name,
            workflow_steps=tuple(copy.deepcopy(state.workflow_steps)),
            tool_calls=tuple(copy.deepcopy(state.tool_history)),
            observations=tuple(copy.deepcopy(state.observations)),
            evidence=tuple(copy.deepcopy(state.evidence)),
            answer=copy.deepcopy(state.answer),
            audit=copy.deepcopy(state.audit_result),
            memory=memory,
            memory_context=copy.deepcopy(getattr(state, "memory_context", {})),
            status=state.status,
            reason=state.reason,
            created_at=state.created_at,
            completed_at=state.updated_at,
            timeline=tuple(self._timeline(state)),
        )

    @staticmethod
    def _timeline(state: Any) -> list[dict[str, Any]]:
        timeline: list[dict[str, Any]] = [
            {
                "stage": "query",
                "status": "received",
                "timestamp": state.created_at,
                "summary": state.query[:160],
            },
            {
                "stage": "plan",
                "status": str(
                    getattr(state, "planner_metadata", {}).get(
                        "mode", "selected"
                    )
                ),
                "timestamp": state.created_at,
                "summary": state.workflow_name or state.task_type,
            },
        ]
        for call in state.tool_history:
            timeline.append(
                {
                    "stage": "tool",
                    "status": str(call.get("output_status", "unknown")),
                    "timestamp": call.get("timestamp"),
                    "summary": str(call.get("tool_name", "unknown")),
                    "trace_id": call.get("trace_id"),
                    "latency_ms": call.get("latency_ms"),
                }
            )
        for revision in getattr(state, "plan_revisions", ()):
            timeline.append(
                {
                    "stage": "replan",
                    "status": "revised",
                    "timestamp": revision.get("created_at"),
                    "summary": revision.get("reason", "plan_revised"),
                    "revision_id": revision.get("revision_id"),
                }
            )
        timeline.extend(
            [
                {
                    "stage": "evidence",
                    "status": "collected",
                    "timestamp": state.updated_at,
                    "summary": f"{len(state.evidence)} canonical evidence records",
                },
                {
                    "stage": "memory",
                    "status": (
                        "persisted"
                        if state.memory_status.get("persisted")
                        else "not_persisted"
                    ),
                    "timestamp": state.updated_at,
                    "summary": str(state.memory_status.get("run_id") or state.run_id),
                },
                {
                    "stage": "final",
                    "status": state.status,
                    "timestamp": state.updated_at,
                    "summary": state.reason or state.status,
                },
            ]
        )
        return timeline
