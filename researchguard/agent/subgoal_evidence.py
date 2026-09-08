# C:\Users\18449\Desktop\researchguard_workspace\researchguard\agent\subgoal_evidence.py
from __future__ import annotations

from typing import Any, Mapping

from researchguard.agent.state import utc_timestamp
from researchguard.tools import EvidenceBundle, GateDecision


SUBGOAL_GATE_VERSION = "1.0.0"
MAX_OBSERVATION_QUERY_CHARS = 480


def build_subgoal_assessment_bundle(
    source_bundle: EvidenceBundle,
    *,
    subgoal_id: str,
    question: str,
    objective: str,
    success_criterion: str,
) -> EvidenceBundle:
    assessment_query = (
        f"Research question: {question.strip()}\n"
        f"Research objective: {objective.strip()}\n"
        f"Required completion evidence: {success_criterion.strip()}"
    )
    return EvidenceBundle.create(
        query=assessment_query,
        evidence=source_bundle.evidence_records,
        retrieval_metadata={
            "source_bundle_id": source_bundle.bundle_id,
            "source_query": source_bundle.query,
        },
        provenance={
            "source": "controller_subgoal_assessment",
            "subgoal_id": subgoal_id,
            "source_bundle_id": source_bundle.bundle_id,
            "objective": objective,
            "success_criterion": success_criterion,
        },
    )


def build_incomplete_aggregate_gate(
    aggregate_bundle: EvidenceBundle,
    subgoal_results: Mapping[str, Mapping[str, Any]],
) -> GateDecision | None:
    if not subgoal_results:
        return None
    rows = [dict(value) for value in subgoal_results.values()]
    if rows and all(row.get("status") == "satisfied" for row in rows):
        return None

    available_ids = set(aggregate_bundle.chunk_ids)
    supporting_ids: list[str] = []
    confidences: list[float] = []
    incomplete: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for row in rows:
        gate_value = row.get("gate_decision")
        gate: GateDecision | None = None
        if isinstance(gate_value, Mapping):
            try:
                gate = GateDecision.from_mapping(gate_value)
            except (TypeError, ValueError):
                gate = None
        if gate is not None:
            supporting_ids.extend(
                chunk_id
                for chunk_id in gate.supporting_chunk_ids
                if chunk_id in available_ids
            )
            confidences.append(gate.confidence)
        decisions.append(
            {
                "subgoal_id": row.get("subgoal_id"),
                "status": row.get("status"),
                "gate_status": gate.status if gate is not None else None,
                "criterion_met": bool(row.get("completion_criterion_met", False)),
                "assessment_bundle_id": row.get("assessment_bundle_id"),
            }
        )
        if row.get("status") != "satisfied":
            incomplete.append(
                {
                    "subgoal_id": row.get("subgoal_id"),
                    "question": row.get("question"),
                    "objective": row.get("objective"),
                    "success_criterion": row.get("success_criterion"),
                    "status": row.get("status"),
                    "reason": row.get("reason"),
                }
            )

    unique_supporting = tuple(dict.fromkeys(supporting_ids))
    status = "partial" if unique_supporting else "unsupported"
    labels = [
        f"{row.get('subgoal_id')}: {row.get('success_criterion')}"
        for row in incomplete
    ]
    reason = "Incomplete research objectives: " + "; ".join(labels)
    assessment = {
        "source": "subgoal_gate_aggregation",
        "model": "deterministic_subgoal_aggregator",
        "prompt_version": "subgoal_gate_aggregation_v1",
        "config_version": "subgoal_gate_aggregation_v1.0",
        "timestamp": utc_timestamp(),
        "support_level": status,
        "answerable": False,
        "confidence": min(confidences) if confidences else 0.0,
        "reason": reason,
        "supporting_chunk_ids": list(unique_supporting),
        "subgoal_decisions": decisions,
        "missing_objectives": incomplete,
        "api_call_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0.0,
    }
    return GateDecision.from_assessment(
        evidence_bundle_id=aggregate_bundle.bundle_id,
        assessment=assessment,
    )


def refined_subgoal_query(subgoal: Mapping[str, Any]) -> str:
    question = " ".join(str(subgoal.get("question", "")).split()).strip()
    criterion = " ".join(
        str(subgoal.get("success_criterion", "")).split()
    ).strip()
    gate = subgoal.get("gate_decision")
    reason = ""
    if isinstance(gate, Mapping):
        reason = " ".join(str(gate.get("reason", "")).split()).strip()
        if len(reason) > MAX_OBSERVATION_QUERY_CHARS:
            reason = reason[:MAX_OBSERVATION_QUERY_CHARS].rsplit(" ", 1)[0]
    details = [question]
    if criterion:
        details.append(f"Find canonical evidence that satisfies: {criterion}")
    if reason:
        details.append(f"Resolve this observed evidence gap: {reason}")
    return " ".join(details)
