# C:\Users\18449\Desktop\researchguard_workspace\researchguard\evaluation\planner_comparison_v2.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from researchguard.agent.state import utc_timestamp
from researchguard.tools import EvidenceRecord


PLANNER_COMPARISON_V2_SCHEMA = "researchguard.planner_comparison.v2"


@dataclass(frozen=True)
class OpenGoalCase:
    case_id: str
    query: str
    task_type: str
    complexity: str
    min_subgoals: int
    max_subgoals: int
    required_concept_groups: tuple[tuple[str, ...], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OpenGoalCase":
        groups = tuple(
            tuple(str(term).casefold() for term in group)
            for group in value.get("required_concept_groups", ())
        )
        case = cls(
            case_id=str(value.get("case_id", "")).strip(),
            query=str(value.get("query", "")).strip(),
            task_type=str(value.get("task_type", "")).strip(),
            complexity=str(value.get("complexity", "")).strip(),
            min_subgoals=int(value.get("min_subgoals", 0)),
            max_subgoals=int(value.get("max_subgoals", 0)),
            required_concept_groups=groups,
        )
        if not case.case_id or not case.query or not case.task_type:
            raise ValueError("Open-goal case requires id, query, and task_type.")
        if case.complexity not in {"simple", "compound"}:
            raise ValueError("Open-goal complexity must be simple or compound.")
        if not 0 <= case.min_subgoals <= case.max_subgoals <= 4:
            raise ValueError("Open-goal subgoal bounds must be between 0 and 4.")
        return case


def load_open_goal_cases(path: str | Path) -> list[OpenGoalCase]:
    rows: list[OpenGoalCase] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError("case must be a JSON object")
            rows.append(OpenGoalCase.from_mapping(value))
        except Exception as exc:
            raise ValueError(f"Invalid open-goal case at line {line_number}: {exc}") from exc
    if not rows:
        raise ValueError("Open-goal comparison requires at least one case.")
    if len({case.case_id for case in rows}) != len(rows):
        raise ValueError("Open-goal case ids must be unique.")
    return rows


def evaluate_open_goal_state(
    case: OpenGoalCase,
    state: Any,
    *,
    latency_ms: float,
) -> dict[str, Any]:
    subgoals = list(getattr(state, "subgoals", ()))
    normalized_questions = [
        " ".join(str(item.get("question", "")).casefold().split())
        for item in subgoals
    ]
    subgoal_text = " ".join(
        " ".join(
            str(item.get(key, ""))
            for key in ("question", "objective", "success_criterion")
        ).casefold()
        for item in subgoals
    )
    covered_groups = [
        any(term in subgoal_text for term in group)
        for group in case.required_concept_groups
    ]
    concept_coverage = (
        sum(covered_groups) / len(covered_groups)
        if covered_groups
        else 1.0
    )
    count_valid = case.min_subgoals <= len(subgoals) <= case.max_subgoals
    unique = len(normalized_questions) == len(set(normalized_questions))
    decomposition_quality = float(count_valid and unique and concept_coverage == 1.0)

    provenance_failures = 0
    metadata_leaks = 0
    for item in getattr(state, "evidence", ()):
        try:
            record = EvidenceRecord.from_mapping(item)
            if not record.provenance:
                provenance_failures += 1
            if (
                str(record.provenance.get("source_type", "")).casefold() == "metadata"
                or record.provenance.get("evidence_eligible") is False
            ):
                metadata_leaks += 1
        except (TypeError, ValueError):
            provenance_failures += 1
    gate = getattr(state, "gate_decision", None)
    gate_strong = isinstance(gate, Mapping) and gate.get("status") == "strong"
    answer = getattr(state, "answer", None)
    audit = getattr(state, "audit_result", None)
    gate_bypass = int(bool(answer) and (not gate_strong or not audit))
    planner = dict(getattr(state, "planner_metadata", {}) or {})
    tool_history = list(getattr(state, "tool_history", ()))
    planner_api_calls = int(planner.get("api_call_count", 0))
    execution_api_calls = sum(
        int(item.get("api_call_count", 0)) for item in tool_history
    )
    return {
        "case_id": case.case_id,
        "query": case.query,
        "complexity": case.complexity,
        "task_type": getattr(state, "task_type", None),
        "task_type_correct": float(getattr(state, "task_type", None) == case.task_type),
        "status": getattr(state, "status", None),
        "reason": getattr(state, "reason", None),
        "completed": float(getattr(state, "status", None) == "completed"),
        "subgoal_count": len(subgoals),
        "subgoal_questions": normalized_questions,
        "subgoal_count_valid": float(count_valid),
        "subgoal_questions_unique": float(unique),
        "concept_coverage": concept_coverage,
        "decomposition_quality": decomposition_quality,
        "unnecessary_decomposition": int(case.complexity == "simple" and bool(subgoals)),
        "plan_validity": float(
            bool(dict(planner.get("validation", {})).get("valid", False))
        ),
        "planner_mode": planner.get("mode"),
        "fallback_used": bool(planner.get("fallback_used", False)),
        "fallback_reason": planner.get("fallback_reason"),
        "plan_revision_count": len(getattr(state, "plan_revisions", ())),
        "plan_revision_reasons": [
            item.get("reason") for item in getattr(state, "plan_revisions", ())
        ],
        "subgoal_statuses": {
            key: value.get("status")
            for key, value in getattr(state, "subgoal_results", {}).items()
        },
        "tool_call_count": len(tool_history),
        "skill_call_count": len(getattr(state, "skill_history", ())),
        "api_call_count": planner_api_calls + execution_api_calls,
        "planner_api_calls": planner_api_calls,
        "execution_api_calls": execution_api_calls,
        "planner_input_tokens": int(planner.get("input_tokens", 0)),
        "planner_output_tokens": int(planner.get("output_tokens", 0)),
        "latency_ms": float(latency_ms),
        "evidence_count": len(getattr(state, "evidence", ())),
        "provenance_failures": provenance_failures,
        "metadata_evidence_leaks": metadata_leaks,
        "gate_bypass": gate_bypass,
        "answer_released": bool(answer),
        "audit_completed": bool(audit),
    }


def build_open_goal_comparison(
    deterministic_rows: Iterable[Mapping[str, Any]],
    hybrid_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    deterministic = [dict(row) for row in deterministic_rows]
    hybrid = [dict(row) for row in hybrid_rows]
    if [row.get("case_id") for row in deterministic] != [
        row.get("case_id") for row in hybrid
    ]:
        raise ValueError("Planner A/B v2 requires paired cases in the same order.")
    left = _aggregate(deterministic, include_fallback=False)
    right = _aggregate(hybrid, include_fallback=True)
    hard_checks = {
        "hybrid_gate_bypass": sum(int(row["gate_bypass"]) for row in hybrid),
        "hybrid_provenance_failure": sum(int(row["provenance_failures"]) for row in hybrid),
        "hybrid_metadata_evidence_leak": sum(int(row["metadata_evidence_leaks"]) for row in hybrid),
        "hybrid_plan_invalid": sum(int(not row["plan_validity"]) for row in hybrid),
    }
    if any(hard_checks.values()):
        conclusion = "HYBRID_UNSAFE"
    elif right["fallback_rate"] == 1.0:
        conclusion = "INCONCLUSIVE_FALLBACK_ONLY"
    elif right["fallback_rate"] > 0.25:
        conclusion = "HYBRID_LIMITED_BY_FALLBACK"
    elif (
        right["decomposition_quality"] > left["decomposition_quality"]
        and right["completion_rate"] > left["completion_rate"]
    ):
        conclusion = "HYBRID_BETTER_FOR_OPEN_GOALS"
    elif (
        right["decomposition_quality"] > left["decomposition_quality"]
        and right["completion_rate"] == left["completion_rate"]
    ):
        conclusion = "HYBRID_STRUCTURAL_GAIN_ONLY"
    elif right["completion_rate"] < left["completion_rate"]:
        conclusion = "HYBRID_DEGRADED_COMPLETION"
    else:
        conclusion = "NO_CLEAR_EXECUTION_GAIN"
    return {
        "schema_version": PLANNER_COMPARISON_V2_SCHEMA,
        "version": "2.0.0",
        "generated_at": utc_timestamp(),
        "case_count": len(deterministic),
        "conclusion": conclusion,
        "deterministic": left,
        "hybrid": right,
        "hard_checks": hard_checks,
        "execution_note": (
            "Each pair ran deterministic before Hybrid against the same local caches; "
            "latency and execution API counts are observed costs, not a cache-controlled cost experiment."
        ),
        "paired_results": [
            {"case_id": a["case_id"], "deterministic": a, "hybrid": b}
            for a, b in zip(deterministic, hybrid)
        ],
    }


def render_open_goal_comparison(report: Mapping[str, Any]) -> str:
    left = report["deterministic"]
    right = report["hybrid"]
    metrics = (
        "completion_rate",
        "decomposition_quality",
        "simple_task_precision",
        "plan_validity",
        "fallback_rate",
        "average_tool_calls",
        "average_skill_calls",
        "average_latency_ms",
        "total_api_calls",
        "planner_api_calls",
        "execution_api_calls",
        "planner_input_tokens",
        "planner_output_tokens",
    )
    lines = [
        "# Planner A/B Comparison v2",
        "",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Live paired cases: `{report['case_count']}`",
        "",
        "Every case executes Planner -> Controller -> Skill -> Evidence Engine. "
        "The labels describe goal structure only and contain no relevant chunk ids.",
        "",
        "| Metric | Deterministic | Hybrid |",
        "|---|---:|---:|",
    ]
    for metric in metrics:
        lines.append(
            f"| `{metric}` | {_render(left.get(metric))} | {_render(right.get(metric))} |"
        )
    lines.extend(["", "## Hard Checks", ""])
    for name, value in report["hard_checks"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Paired Cases", ""])
    for pair in report["paired_results"]:
        a = pair["deterministic"]
        b = pair["hybrid"]
        lines.append(
            f"- `{pair['case_id']}`: deterministic=`{a['status']}`/"
            f"{a['subgoal_count']} subgoals; hybrid=`{b['status']}`/"
            f"{b['subgoal_count']} subgoals; hybrid_mode=`{b['planner_mode']}`; "
            f"fallback=`{b['fallback_used']}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def _aggregate(
    rows: list[Mapping[str, Any]],
    *,
    include_fallback: bool,
) -> dict[str, float | int]:
    simple = [row for row in rows if row["complexity"] == "simple"]
    return {
        "completion_rate": mean(float(row["completed"]) for row in rows),
        "decomposition_quality": mean(
            float(row["decomposition_quality"]) for row in rows
        ),
        "simple_task_precision": (
            mean(float(not row["unnecessary_decomposition"]) for row in simple)
            if simple
            else 1.0
        ),
        "plan_validity": mean(float(row["plan_validity"]) for row in rows),
        "fallback_rate": (
            mean(float(bool(row["fallback_used"])) for row in rows)
            if include_fallback
            else 0.0
        ),
        "average_tool_calls": mean(float(row["tool_call_count"]) for row in rows),
        "average_skill_calls": mean(float(row["skill_call_count"]) for row in rows),
        "average_latency_ms": mean(float(row["latency_ms"]) for row in rows),
        "total_api_calls": sum(int(row["api_call_count"]) for row in rows),
        "planner_api_calls": sum(int(row["planner_api_calls"]) for row in rows),
        "execution_api_calls": sum(int(row["execution_api_calls"]) for row in rows),
        "planner_input_tokens": sum(int(row["planner_input_tokens"]) for row in rows),
        "planner_output_tokens": sum(int(row["planner_output_tokens"]) for row in rows),
    }


def _render(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)
