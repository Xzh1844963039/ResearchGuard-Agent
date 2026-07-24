# C:\Users\18449\Desktop\researchguard_workspace\researchguard\evaluation\agent_metrics.py
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any, Iterable, Mapping

from researchguard.evaluation.schemas import MetricValue


PROVENANCE_FIELDS = ("chunk_id", "doc_id", "section", "page", "content")


def planning_metrics(
    *,
    observed_task_type: str,
    observed_workflow: str | None,
    expected_task_type: str | None,
    expected_workflow: str | None,
    tool_names: Iterable[str],
    registered_tools: Iterable[str],
) -> dict[str, MetricValue]:
    registered = set(registered_tools)
    called = list(tool_names)
    invalid = [name for name in called if name not in registered]
    return {
        "task_classification_accuracy": MetricValue(
            "task_classification_accuracy",
            "planning",
            None if expected_task_type is None else float(observed_task_type == expected_task_type),
            None if expected_task_type is None else observed_task_type == expected_task_type,
            {"expected": expected_task_type, "observed": observed_task_type},
        ),
        "workflow_selection_accuracy": MetricValue(
            "workflow_selection_accuracy",
            "planning",
            None if expected_task_type is None else float(observed_workflow == expected_workflow),
            None if expected_task_type is None else observed_workflow == expected_workflow,
            {"expected": expected_workflow, "observed": observed_workflow},
        ),
        "invalid_tool_rate": MetricValue(
            "invalid_tool_rate",
            "planning",
            len(invalid) / len(called) if called else 0.0,
            not invalid,
            {"invalid_tools": invalid, "registered_tool_count": len(registered)},
        ),
    }


def tool_metrics(
    tool_calls: Iterable[Mapping[str, Any]],
    *,
    expected_tools: Iterable[str] = (),
    forbidden_tools: Iterable[str] = (),
    allow_failed_calls: bool = False,
) -> dict[str, MetricValue]:
    calls = list(tool_calls)
    names = [str(call.get("tool_name", call.get("tool", ""))) for call in calls]
    successes = sum(
        str(call.get("output_status", call.get("status", ""))).lower()
        in {"ok", "success", "completed", "rejected"}
        for call in calls
    )
    expected = Counter(expected_tools)
    observed = Counter(names)
    extra = list((observed - expected).elements()) if expected else []
    forbidden = sorted(set(names).intersection(forbidden_tools))
    unnecessary = extra + forbidden
    return {
        "tool_success_rate": MetricValue(
            "tool_success_rate",
            "tool",
            successes / len(calls) if calls else 1.0,
            successes == len(calls) or allow_failed_calls,
            {"success_count": successes, "tool_call_count": len(calls)},
        ),
        "tool_call_count": MetricValue(
            "tool_call_count",
            "tool",
            len(calls),
            None,
            {"tools": names},
        ),
        "unnecessary_tool_calls": MetricValue(
            "unnecessary_tool_calls",
            "tool",
            len(unnecessary),
            not unnecessary,
            {"calls": unnecessary},
        ),
    }


def evidence_metrics(
    evidence: Iterable[Mapping[str, Any]],
    *,
    relevant_evidence_ids: Iterable[str] = (),
    audit: Mapping[str, Any] | None = None,
) -> dict[str, MetricValue]:
    records = list(evidence)
    invalid_records = [
        index
        for index, record in enumerate(records)
        if any(record.get(field) in (None, "") for field in PROVENANCE_FIELDS)
    ]
    relevant = set(relevant_evidence_ids)
    observed_ids = {str(item.get("chunk_id", "")) for item in records}
    coverage = len(relevant.intersection(observed_ids)) / len(relevant) if relevant else None
    unsupported, claim_count = _unsupported_claim_counts(audit)
    unsupported_rate = unsupported / claim_count if claim_count else None
    return {
        "provenance_validity": MetricValue(
            "provenance_validity",
            "evidence",
            1.0 - (len(invalid_records) / len(records)) if records else 1.0,
            not invalid_records,
            {"invalid_record_indexes": invalid_records, "evidence_count": len(records)},
        ),
        "evidence_coverage": MetricValue(
            "evidence_coverage",
            "evidence",
            coverage,
            None if not relevant else coverage == 1.0,
            {"relevant_ids": sorted(relevant), "observed_ids": sorted(observed_ids)},
        ),
        "unsupported_claim_rate": MetricValue(
            "unsupported_claim_rate",
            "evidence",
            unsupported_rate,
            unsupported == 0 if claim_count else None,
            {"unsupported_claims": unsupported, "claim_count": claim_count},
        ),
    }


def efficiency_metrics(state: Any) -> dict[str, MetricValue]:
    tool_calls = list(getattr(state, "tool_history", ()))
    retries = sum(int(value) for value in getattr(state, "retry_counts", {}).values())
    planner_metadata = getattr(state, "planner_metadata", {})
    planner_api_calls = (
        int(planner_metadata.get("api_call_count", 0) or 0)
        if isinstance(planner_metadata, Mapping)
        else 0
    )
    tool_api_calls = sum(_api_calls(call) for call in tool_calls)
    return {
        "latency_ms": MetricValue(
            "latency_ms",
            "efficiency",
            _elapsed_ms(getattr(state, "created_at", ""), getattr(state, "updated_at", "")),
            None,
        ),
        "step_count": MetricValue(
            "step_count",
            "efficiency",
            max(
                int(getattr(state, "current_step", 0)),
                len(getattr(state, "workflow_steps", ())),
            ),
            None,
        ),
        "retry_count": MetricValue(
            "retry_count",
            "efficiency",
            retries,
            None,
        ),
        "api_call_count": MetricValue(
            "api_call_count",
            "efficiency",
            planner_api_calls + tool_api_calls,
            None,
            {
                "planner_api_calls": planner_api_calls,
                "tool_api_calls": tool_api_calls,
            },
        ),
    }


def intelligence_metrics(
    state: Any,
    *,
    expected_plan_revisions: int | None = None,
) -> dict[str, MetricValue]:
    calls = list(getattr(state, "tool_history", ()))
    revisions = list(getattr(state, "plan_revisions", ()))
    final_status = str(getattr(state, "status", ""))
    signatures: list[str] = []
    duplicates = 0
    for call in calls:
        signature = json.dumps(
            {
                "tool_name": call.get("tool_name", call.get("tool", "")),
                "input_summary": call.get("input_summary", {}),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in signatures:
            duplicates += 1
        signatures.append(signature)

    source_bundle_ids = {
        str(call.get("evidence_bundle_id"))
        for call in calls
        if str(call.get("tool_name", "")) in {"retrieve_evidence", "assess_evidence"}
        and call.get("evidence_bundle_id")
    }
    answer_calls = [
        call
        for call in calls
        if str(call.get("tool_name", ""))
        in {"generate_grounded_answer", "audit_answer"}
    ]
    reused = sum(
        bool(call.get("evidence_bundle_id"))
        and str(call.get("evidence_bundle_id")) in source_bundle_ids
        for call in answer_calls
    )
    return {
        "replanning_rate": MetricValue(
            "replanning_rate",
            "agent_intelligence",
            float(bool(revisions)),
            None,
            {"revision_count": len(revisions)},
        ),
        "successful_recovery_rate": MetricValue(
            "successful_recovery_rate",
            "agent_intelligence",
            float(final_status == "completed") if revisions else None,
            None,
            {"applicable": bool(revisions), "final_status": final_status},
        ),
        "average_plan_revision": MetricValue(
            "average_plan_revision",
            "agent_intelligence",
            len(revisions),
            None,
        ),
        "plan_revision_accuracy": MetricValue(
            "plan_revision_accuracy",
            "agent_intelligence",
            (
                None
                if expected_plan_revisions is None
                else float(len(revisions) == expected_plan_revisions)
            ),
            (
                None
                if expected_plan_revisions is None
                else len(revisions) == expected_plan_revisions
            ),
            {
                "expected": expected_plan_revisions,
                "observed": len(revisions),
            },
        ),
        "duplicate_tool_call_rate": MetricValue(
            "duplicate_tool_call_rate",
            "agent_intelligence",
            duplicates / len(calls) if calls else 0.0,
            duplicates == 0,
            {"duplicate_count": duplicates, "tool_call_count": len(calls)},
        ),
        "evidence_reuse_rate": MetricValue(
            "evidence_reuse_rate",
            "agent_intelligence",
            reused / len(answer_calls) if answer_calls else None,
            reused == len(answer_calls) if answer_calls else None,
            {
                "reused_calls": reused,
                "answer_and_audit_calls": len(answer_calls),
                "source_bundle_ids": sorted(source_bundle_ids),
            },
        ),
    }


def memory_metrics(
    memory_status: Mapping[str, Any],
    memory_snapshot: Mapping[str, Any] | None,
    *,
    final_status: str,
) -> dict[str, MetricValue]:
    enabled = bool(memory_status.get("enabled"))
    persisted = bool(memory_status.get("persisted")) if enabled else False
    ledger = list((memory_snapshot or {}).get("evidence_ledger", ()))
    failures = list((memory_snapshot or {}).get("failures", ()))
    expected_ledger = final_status == "completed"
    expected_failure = final_status in {"failed", "rejected"}
    return {
        "memory_persistence_success": MetricValue(
            "memory_persistence_success",
            "memory",
            persisted if enabled else None,
            persisted if enabled else None,
            {"enabled": enabled, "errors": list(memory_status.get("errors", ()))},
        ),
        "ledger_completeness": MetricValue(
            "ledger_completeness",
            "memory",
            len(ledger),
            (bool(ledger) if expected_ledger else True) if enabled else None,
            {"ledger_record_count": len(ledger), "expected": expected_ledger},
        ),
        "failure_recording": MetricValue(
            "failure_recording",
            "memory",
            len(failures),
            (bool(failures) if expected_failure else True) if enabled else None,
            {"failure_count": len(failures), "expected": expected_failure},
        ),
    }


def _elapsed_ms(start: str, end: str) -> float:
    try:
        start_at = datetime.fromisoformat(start)
        end_at = datetime.fromisoformat(end)
        return max(0.0, (end_at - start_at).total_seconds() * 1000.0)
    except (TypeError, ValueError):
        return 0.0


def _api_calls(value: Any) -> int:
    if isinstance(value, Mapping):
        direct = value.get("api_call_count")
        if isinstance(direct, int) and not isinstance(direct, bool):
            return max(0, direct)
        return sum(_api_calls(item) for item in value.values())
    if isinstance(value, list):
        return sum(_api_calls(item) for item in value)
    return 0


def _unsupported_claim_counts(audit: Mapping[str, Any] | None) -> tuple[int, int]:
    if not audit:
        return 0, 0
    claims = audit.get("claims")
    if not isinstance(claims, list):
        claims = audit.get("claim_results")
    if not isinstance(claims, list):
        return 0, 0
    statuses = [
        str(
            item.get(
                "status",
                item.get("verdict", item.get("support_level", "")),
            )
        ).lower()
        for item in claims
        if isinstance(item, Mapping)
    ]
    unsupported = sum(status in {"unsupported", "contradicted", "invalid"} for status in statuses)
    return unsupported, len(statuses)
