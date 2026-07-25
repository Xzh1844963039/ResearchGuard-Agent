# C:\Users\18449\Desktop\researchguard_workspace\researchguard\evaluation\planner_comparison.py
from __future__ import annotations

import copy
from statistics import mean
from typing import Any, Iterable, Mapping

from researchguard.evaluation.benchmark_v3 import (
    AgentBenchmarkCaseV3,
    utc_timestamp,
)


PLANNER_COMPARISON_SCHEMA_VERSION = "researchguard.planner_comparison.v1"


def evaluate_planner_outcome(
    case: AgentBenchmarkCaseV3,
    outcome: Any,
) -> dict[str, Any]:
    plan = outcome.structured_plan
    skills = [step.skill for step in plan.steps]
    expected = list(case.expected_skill_pattern)
    expected_set = set(expected)
    observed_set = set(skills)
    completeness = (
        len(expected_set.intersection(observed_set)) / len(expected_set)
        if expected_set
        else 1.0
    )
    unnecessary = [name for name in skills if name not in expected_set]
    required_tool_calls = sum(
        len(
            outcome.executable_plan.steps[index].parameters.get(
                "required_tools",
                (),
            )
        )
        if outcome.executable_plan.steps[index].parameters
        else 0
        for index in range(len(outcome.executable_plan.steps))
    )
    if not required_tool_calls:
        required_tool_calls = len(skills)
    return {
        "case_id": case.case_id,
        "category": case.category,
        "expected_task_type": case.expected_task_type,
        "observed_task_type": plan.task_type,
        "expected_skills": expected,
        "observed_skills": skills,
        "task_routing_accuracy": float(
            plan.task_type == case.expected_task_type
        ),
        "skill_sequence_quality": float(skills == expected),
        "plan_completeness": completeness,
        "tool_efficiency": 1.0 / max(1, required_tool_calls),
        "unnecessary_skill_count": len(unnecessary),
        "plan_validity": float(outcome.validation.valid),
        "latency_ms": float(outcome.latency_ms),
        "api_call_count": int(outcome.api_call_count),
        "input_tokens": int(outcome.input_tokens),
        "output_tokens": int(outcome.output_tokens),
        "mode": str(outcome.mode),
        "fallback_used": bool(outcome.fallback_used),
        "fallback_reason": outcome.fallback_reason,
    }


def build_planner_comparison_report(
    deterministic_rows: Iterable[Mapping[str, Any]],
    hybrid_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    deterministic = [copy.deepcopy(dict(row)) for row in deterministic_rows]
    hybrid = [copy.deepcopy(dict(row)) for row in hybrid_rows]
    if len(deterministic) != len(hybrid):
        raise ValueError("Planner comparison requires paired runs.")
    if [row.get("case_id") for row in deterministic] != [
        row.get("case_id") for row in hybrid
    ]:
        raise ValueError("Planner comparison case order does not match.")
    deterministic_metrics = _aggregate(deterministic)
    hybrid_metrics = _aggregate(hybrid)
    lift = {
        name: (
            None
            if deterministic_metrics.get(name) is None
            or hybrid_metrics.get(name) is None
            else float(hybrid_metrics[name]) - float(deterministic_metrics[name])
        )
        for name in (
            "task_routing_accuracy",
            "skill_sequence_quality",
            "plan_completeness",
            "tool_efficiency",
        )
    }
    fallback_rate = float(hybrid_metrics.get("fallback_rate") or 0.0)
    quality_lift = sum(
        value or 0.0
        for name, value in lift.items()
        if name != "tool_efficiency"
    )
    if fallback_rate == 1.0:
        conclusion = "INCONCLUSIVE_FALLBACK_ONLY"
    elif quality_lift > 0:
        conclusion = "HYBRID_BETTER"
    elif quality_lift < 0:
        conclusion = "HYBRID_DEGRADED"
    else:
        conclusion = "NO_MEASURED_GAIN"
    return {
        "schema_version": PLANNER_COMPARISON_SCHEMA_VERSION,
        "version": "1.0.0",
        "generated_at": utc_timestamp(),
        "case_count": len(deterministic),
        "conclusion": conclusion,
        "deterministic": deterministic_metrics,
        "hybrid": hybrid_metrics,
        "hybrid_lift": lift,
        "paired_results": [
            {
                "case_id": deterministic_row.get("case_id"),
                "deterministic": deterministic_row,
                "hybrid": hybrid_row,
            }
            for deterministic_row, hybrid_row in zip(
                deterministic,
                hybrid,
            )
        ],
    }


def render_planner_comparison_markdown(
    report: Mapping[str, Any],
) -> str:
    deterministic = report.get("deterministic", {})
    hybrid = report.get("hybrid", {})
    lift = report.get("hybrid_lift", {})
    names = (
        "task_routing_accuracy",
        "skill_sequence_quality",
        "plan_completeness",
        "tool_efficiency",
        "plan_validity",
        "average_latency_ms",
        "average_skill_count",
        "fallback_rate",
        "api_call_count",
        "input_tokens",
        "output_tokens",
    )
    lines = [
        "# Planner A/B Comparison Report",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Paired cases: `{report.get('case_count')}`",
        f"- Conclusion: `{report.get('conclusion')}`",
        "",
        "The report only claims Hybrid Planner benefit when a non-fallback "
        "run improves labeled routing or skill-plan quality.",
        "",
        "## Aggregate Comparison",
        "",
        "| Metric | Deterministic | Hybrid | Hybrid lift |",
        "|---|---:|---:|---:|",
    ]
    for name in names:
        left = _render_value(deterministic.get(name))
        right = _render_value(hybrid.get(name))
        delta = _render_value(lift.get(name))
        lines.append(f"| `{name}` | {left} | {right} | {delta} |")
    lines.extend(["", "## Paired Cases", ""])
    for pair in report.get("paired_results", ()):
        left = pair.get("deterministic", {})
        right = pair.get("hybrid", {})
        lines.append(
            f"- `{pair.get('case_id')}`: deterministic="
            f"`{', '.join(left.get('observed_skills', ()))}`; hybrid="
            f"`{', '.join(right.get('observed_skills', ()))}`; "
            f"hybrid_mode=`{right.get('mode')}`; "
            f"fallback=`{right.get('fallback_used')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, float | int | None]:
    metric_names = (
        "task_routing_accuracy",
        "skill_sequence_quality",
        "plan_completeness",
        "tool_efficiency",
        "plan_validity",
    )
    result: dict[str, float | int | None] = {
        name: mean(float(row[name]) for row in rows) if rows else None
        for name in metric_names
    }
    result.update(
        {
            "average_latency_ms": (
                mean(float(row.get("latency_ms", 0.0)) for row in rows)
                if rows
                else None
            ),
            "average_skill_count": (
                mean(len(row.get("observed_skills", ())) for row in rows)
                if rows
                else None
            ),
            "fallback_rate": (
                mean(float(bool(row.get("fallback_used"))) for row in rows)
                if rows
                else None
            ),
            "api_call_count": sum(
                int(row.get("api_call_count", 0)) for row in rows
            ),
            "input_tokens": sum(
                int(row.get("input_tokens", 0)) for row in rows
            ),
            "output_tokens": sum(
                int(row.get("output_tokens", 0)) for row in rows
            ),
        }
    )
    return result


def _render_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
