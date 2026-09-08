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
        objective_status = self._objective_status(state)
        completed_objectives = tuple(
            item for item in objective_status if item["status"] == "satisfied"
        )
        missing_objectives = tuple(
            item for item in objective_status if item["status"] != "satisfied"
        )
        evidence_bundle = copy.deepcopy(
            getattr(state, "evidence_bundle", None)
        )
        gate_decision = copy.deepcopy(
            getattr(state, "gate_decision", None)
        )
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
            subgoals=tuple(copy.deepcopy(getattr(state, "subgoals", ()))),
            subgoal_results=copy.deepcopy(
                getattr(state, "subgoal_results", {})
            ),
            objective_status=objective_status,
            completed_objectives=completed_objectives,
            missing_objectives=missing_objectives,
            plan_revisions=tuple(
                copy.deepcopy(getattr(state, "plan_revisions", ()))
            ),
            workflow_name=state.workflow_name,
            workflow_steps=tuple(copy.deepcopy(state.workflow_steps)),
            skill_calls=tuple(
                copy.deepcopy(getattr(state, "skill_history", ()))
            ),
            tool_calls=tuple(copy.deepcopy(state.tool_history)),
            observations=tuple(copy.deepcopy(state.observations)),
            evidence=tuple(copy.deepcopy(state.evidence)),
            evidence_bundle=evidence_bundle,
            gate_decision=gate_decision,
            answer=copy.deepcopy(state.answer),
            audit=copy.deepcopy(state.audit_result),
            memory=memory,
            memory_context=copy.deepcopy(getattr(state, "memory_context", {})),
            status=state.status,
            reason=state.reason,
            created_at=state.created_at,
            completed_at=state.updated_at,
            timeline=tuple(self._timeline(state)),
            decision_summary={
                "original_goal": state.query,
                "current_plan_skills": [
                    step.get("skill") or step.get("tool") for step in plan
                ],
                "completed_objective_ids": [
                    item["subgoal_id"] for item in completed_objectives
                ],
                "missing_objective_ids": [
                    item["subgoal_id"] for item in missing_objectives
                ],
                "evidence_bundle_id": (
                    evidence_bundle.get("bundle_id")
                    if isinstance(evidence_bundle, Mapping)
                    else None
                ),
                "gate_status": (
                    gate_decision.get("status")
                    if isinstance(gate_decision, Mapping)
                    else None
                ),
                "final_status": state.status,
                "final_reason": state.reason,
            },
        )

    @staticmethod
    def _objective_status(state: Any) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        results = getattr(state, "subgoal_results", {})
        for subgoal in getattr(state, "subgoals", ()):
            subgoal_id = str(subgoal.get("subgoal_id", ""))
            result = results.get(subgoal_id, {})
            gate = result.get("gate_decision")
            rows.append(
                {
                    "subgoal_id": subgoal_id,
                    "question": subgoal.get("question"),
                    "objective": subgoal.get("objective"),
                    "success_criterion": subgoal.get("success_criterion"),
                    "status": result.get("status", "pending"),
                    "attempts": result.get("attempts", 0),
                    "evidence_count": result.get("evidence_count", 0),
                    "completion_criterion_met": bool(
                        result.get("completion_criterion_met", False)
                    ),
                    "gate_status": (
                        gate.get("status") if isinstance(gate, Mapping) else None
                    ),
                    "reason": result.get("reason"),
                    "source_bundle_id": result.get("source_bundle_id"),
                    "assessment_bundle_id": result.get("assessment_bundle_id"),
                }
            )
        return tuple(rows)

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
        for call in getattr(state, "skill_history", ()):
            timeline.append(
                {
                    "stage": "skill",
                    "status": str(call.get("output_status", "unknown")),
                    "timestamp": call.get("timestamp"),
                    "summary": str(call.get("skill_name", "unknown")),
                    "trace_id": call.get("trace_id"),
                    "latency_ms": call.get("latency_ms"),
                }
            )
        for objective in TraceCollector._objective_status(state):
            timeline.append(
                {
                    "stage": "objective",
                    "status": str(objective.get("status", "pending")),
                    "timestamp": state.updated_at,
                    "summary": str(objective.get("subgoal_id", "unknown")),
                    "gate_status": objective.get("gate_status"),
                    "reason": objective.get("reason"),
                }
            )
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
