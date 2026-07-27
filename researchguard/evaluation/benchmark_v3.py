# C:\Users\18449\Desktop\researchguard_workspace\researchguard\evaluation\benchmark_v3.py
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


BENCHMARK_V3_SCHEMA_VERSION = "researchguard.agent_benchmark.v3"
VALID_CATEGORIES = {
    "literature_research",
    "paper_comparison",
    "claim_audit",
}
VALID_BEHAVIORS = {
    "metadata_only",
    "answer_or_refuse",
    "audit_or_refuse",
}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
PROVENANCE_FIELDS = ("chunk_id", "doc_id", "section", "page", "content")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AgentBenchmarkCaseV3:
    case_id: str
    category: str
    query: str
    expected_task_type: str
    expected_skill_pattern: tuple[str, ...]
    expected_behavior: str
    difficulty: str
    expected_doc_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = "3.0.0"
    schema_version: str = BENCHMARK_V3_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.query.strip():
            raise ValueError("Benchmark case_id and query must not be empty.")
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"Unsupported benchmark category: {self.category}")
        if self.expected_behavior not in VALID_BEHAVIORS:
            raise ValueError(
                f"Unsupported expected behavior: {self.expected_behavior}"
            )
        if self.difficulty not in VALID_DIFFICULTIES:
            raise ValueError(f"Unsupported difficulty: {self.difficulty}")
        if not self.expected_skill_pattern:
            raise ValueError("expected_skill_pattern must not be empty.")
        if len(self.expected_skill_pattern) != len(
            set(self.expected_skill_pattern)
        ):
            raise ValueError("expected_skill_pattern must not contain cycles.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AgentBenchmarkCaseV3":
        allowed = {
            "schema_version",
            "version",
            "case_id",
            "category",
            "query",
            "expected_task_type",
            "expected_skill_pattern",
            "expected_behavior",
            "difficulty",
            "expected_doc_ids",
            "metadata",
        }
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise ValueError(
                "Unknown Agent Benchmark v3 fields: " + ", ".join(unknown)
            )
        schema_version = str(
            value.get("schema_version", BENCHMARK_V3_SCHEMA_VERSION)
        )
        if schema_version != BENCHMARK_V3_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Agent Benchmark v3 schema: {schema_version}"
            )
        return cls(
            case_id=str(value.get("case_id", "")),
            category=str(value.get("category", "")),
            query=str(value.get("query", "")),
            expected_task_type=str(value.get("expected_task_type", "")),
            expected_skill_pattern=tuple(
                str(item) for item in value.get("expected_skill_pattern", ())
            ),
            expected_behavior=str(value.get("expected_behavior", "")),
            difficulty=str(value.get("difficulty", "")),
            expected_doc_ids=tuple(
                str(item) for item in value.get("expected_doc_ids", ())
            ),
            metadata=copy.deepcopy(dict(value.get("metadata", {}))),
            version=str(value.get("version", "3.0.0")),
            schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "case_id": self.case_id,
            "category": self.category,
            "query": self.query,
            "expected_task_type": self.expected_task_type,
            "expected_skill_pattern": list(self.expected_skill_pattern),
            "expected_behavior": self.expected_behavior,
            "difficulty": self.difficulty,
            "expected_doc_ids": list(self.expected_doc_ids),
            "metadata": copy.deepcopy(dict(self.metadata)),
        }


def load_benchmark_v3_cases(path: str | Path) -> tuple[AgentBenchmarkCaseV3, ...]:
    source = Path(path)
    cases: list[AgentBenchmarkCaseV3] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("case must be a JSON object")
            cases.append(AgentBenchmarkCaseV3.from_mapping(value))
        except Exception as exc:
            raise ValueError(
                f"Invalid Agent Benchmark v3 line {line_number}: {exc}"
            ) from exc
    if len(cases) < 50:
        raise ValueError("Agent Benchmark v3 requires at least 50 cases.")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Agent Benchmark v3 case_id values must be unique.")
    category_counts = {
        category: sum(case.category == category for case in cases)
        for category in VALID_CATEGORIES
    }
    required = {
        "literature_research": 20,
        "paper_comparison": 15,
        "claim_audit": 15,
    }
    if any(category_counts[name] < count for name, count in required.items()):
        raise ValueError(
            f"Agent Benchmark v3 category coverage is incomplete: {category_counts}"
        )
    return tuple(cases)


def evaluate_benchmark_v3_run(
    state: Any,
    case: AgentBenchmarkCaseV3,
    *,
    input_cost_per_million: float = 0.0,
    output_cost_per_million: float = 0.0,
) -> dict[str, Any]:
    planned_skills = _planned_skills(state)
    executed_skills = [
        str(item.get("skill_name", ""))
        for item in getattr(state, "skill_history", ())
        if str(item.get("skill_name", "")).strip()
    ]
    evidence = [
        item
        for item in getattr(state, "evidence", ())
        if isinstance(item, Mapping)
    ]
    observed_docs = {str(item.get("doc_id", "")) for item in evidence}
    expected_docs = set(case.expected_doc_ids)
    coverage = (
        len(observed_docs.intersection(expected_docs)) / len(expected_docs)
        if expected_docs
        else None
    )
    invalid_provenance = [
        index
        for index, item in enumerate(evidence)
        if any(item.get(field) in (None, "") for field in PROVENANCE_FIELDS)
    ]
    plan_validation = getattr(state, "planner_metadata", {}).get(
        "validation",
        {},
    )
    plan_valid = bool(
        isinstance(plan_validation, Mapping)
        and plan_validation.get("valid", False)
    )
    task_correct = (
        str(getattr(state, "task_type", "")) == case.expected_task_type
    )
    skill_exact = tuple(planned_skills) == case.expected_skill_pattern
    expected_skill_set = set(case.expected_skill_pattern)
    unnecessary = [
        name for name in executed_skills if name not in expected_skill_set
    ]
    gate_bypass = _gate_bypass_attempts(state)
    refusal_correct = _refusal_correctness(state, case.expected_behavior)
    unsupported, claim_count = _unsupported_claim_counts(
        getattr(state, "audit_result", None)
    )
    input_tokens = _sum_key(getattr(state, "planner_metadata", {}), "input_tokens")
    output_tokens = _sum_key(
        getattr(state, "planner_metadata", {}),
        "output_tokens",
    )
    for item in getattr(state, "tool_history", ()):
        input_tokens += _sum_key(item, "input_tokens")
        output_tokens += _sum_key(item, "output_tokens")
    token_cost = (
        input_tokens * max(0.0, input_cost_per_million)
        + output_tokens * max(0.0, output_cost_per_million)
    ) / 1_000_000
    provenance_valid = not invalid_provenance
    evidence_covered = coverage in (None, 1.0)
    passed = all(
        (
            task_correct,
            skill_exact,
            plan_valid,
            provenance_valid,
            evidence_covered,
            gate_bypass == 0,
            refusal_correct,
            str(getattr(state, "status", "")) != "failed",
        )
    )
    return {
        "case_id": case.case_id,
        "category": case.category,
        "difficulty": case.difficulty,
        "passed": passed,
        "expected": case.to_dict(),
        "observed": {
            "task_type": str(getattr(state, "task_type", "")),
            "planned_skills": planned_skills,
            "executed_skills": executed_skills,
            "status": str(getattr(state, "status", "")),
            "reason": getattr(state, "reason", None),
            "planner_mode": getattr(state, "planner_metadata", {}).get("mode"),
            "planner_fallback": bool(
                getattr(state, "planner_metadata", {}).get(
                    "fallback_used",
                    False,
                )
            ),
            "evidence_doc_ids": sorted(observed_docs),
        },
        "metrics": {
            "task_classification_accuracy": float(task_correct),
            "skill_selection_accuracy": float(skill_exact),
            "plan_validity": float(plan_valid),
            "successful_completion": float(
                str(getattr(state, "status", "")) == "completed"
            ),
            "unnecessary_skill_call_rate": (
                len(unnecessary) / len(executed_skills)
                if executed_skills
                else 0.0
            ),
            "skill_count": len(executed_skills),
            "evidence_coverage": coverage,
            "provenance_validity": (
                1.0 - len(invalid_provenance) / len(evidence)
                if evidence
                else 1.0
            ),
            "unsupported_claim_rate": (
                unsupported / claim_count if claim_count else None
            ),
            "latency_ms": _elapsed_ms(
                getattr(state, "created_at", ""),
                getattr(state, "updated_at", ""),
            ),
            "tool_call_count": len(getattr(state, "tool_history", ())),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "token_cost_usd": token_cost,
            "gate_bypass_attempts": gate_bypass,
            "refusal_correctness": float(refusal_correct),
        },
        "issues": [
            name
            for name, failed in (
                ("task_classification", not task_correct),
                ("skill_selection", not skill_exact),
                ("plan_validity", not plan_valid),
                ("evidence_coverage", not evidence_covered),
                ("provenance", not provenance_valid),
                ("gate_bypass", gate_bypass > 0),
                ("refusal_correctness", not refusal_correct),
                (
                    "execution_failure",
                    str(getattr(state, "status", "")) == "failed",
                ),
            )
            if failed
        ],
    }


def build_benchmark_v3_report(
    results: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [copy.deepcopy(dict(item)) for item in results]
    metric_names = sorted(
        {
            name
            for row in rows
            for name in dict(row.get("metrics", {}))
        }
    )
    aggregates: dict[str, float | int | None] = {
        "case_count": len(rows),
        "case_pass_rate": (
            sum(bool(row.get("passed")) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
    }
    for name in metric_names:
        values = [
            row["metrics"][name]
            for row in rows
            if isinstance(row.get("metrics"), Mapping)
            and isinstance(row["metrics"].get(name), (int, float))
        ]
        aggregates[name] = mean(values) if values else None
    hard_checks = {
        "result_schema_failure": sum(
            not isinstance(row.get("metrics"), Mapping) for row in rows
        ),
        "plan_invalid": sum(
            row.get("metrics", {}).get("plan_validity") != 1.0
            for row in rows
        ),
        "gate_bypass_attempt": int(
            sum(
                float(row.get("metrics", {}).get("gate_bypass_attempts", 0))
                for row in rows
            )
        ),
        "provenance_failure": sum(
            row.get("metrics", {}).get("provenance_validity", 1.0) < 1.0
            for row in rows
        ),
        "execution_exception": sum(
            "execution_exception" in row.get("issues", ()) for row in rows
        ),
    }
    if sum(hard_checks.values()):
        conclusion = "FAIL"
    elif float(aggregates["case_pass_rate"] or 0.0) >= 0.90:
        conclusion = "PASS"
    else:
        conclusion = "PASS_WITH_LIMITATIONS"
    categories = {}
    for category in sorted(VALID_CATEGORIES):
        selected = [row for row in rows if row.get("category") == category]
        categories[category] = {
            "case_count": len(selected),
            "pass_rate": (
                sum(bool(row.get("passed")) for row in selected) / len(selected)
                if selected
                else 0.0
            ),
        }
    return {
        "schema_version": BENCHMARK_V3_SCHEMA_VERSION,
        "version": "3.0.0",
        "generated_at": utc_timestamp(),
        "conclusion": conclusion,
        "aggregate_metrics": aggregates,
        "category_metrics": categories,
        "hard_checks": hard_checks,
        "results": rows,
    }


def render_benchmark_v3_markdown(report: Mapping[str, Any]) -> str:
    aggregates = report.get("aggregate_metrics", {})
    lines = [
        "# ResearchGuard Agent Benchmark v3 Report",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Cases: `{aggregates.get('case_count', 0)}`",
        f"- Conclusion: `{report.get('conclusion')}`",
        "- Execution: real Planner -> Controller -> Skill Executor -> Evidence Engine",
        "- Benchmark cases contain expected behavior and provenance targets, not answers.",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in sorted(dict(aggregates).items()):
        rendered = (
            "N/A"
            if value is None
            else f"{value:.4f}"
            if isinstance(value, float)
            else str(value)
        )
        lines.append(f"| `{name}` | {rendered} |")
    lines.extend(
        [
            "",
            "## Category Metrics",
            "",
            "| Category | Cases | Pass rate |",
            "|---|---:|---:|",
        ]
    )
    for category, metrics in sorted(
        dict(report.get("category_metrics", {})).items()
    ):
        lines.append(
            f"| {category} | {metrics['case_count']} | "
            f"{metrics['pass_rate']:.4f} |"
        )
    lines.extend(["", "## Hard Checks", "", "| Check | Count |", "|---|---:|"])
    for name, value in sorted(dict(report.get("hard_checks", {})).items()):
        lines.append(f"| `{name}` | {value} |")
    lines.extend(["", "## Cases", ""])
    for row in report.get("results", ()):
        observed = row.get("observed", {})
        lines.append(
            f"- `{row.get('case_id')}`: "
            f"`{'PASS' if row.get('passed') else 'CHECK'}`; "
            f"task=`{observed.get('task_type')}`; "
            f"status=`{observed.get('status')}`; "
            f"skills=`{', '.join(observed.get('planned_skills', ()))}`; "
            f"issues=`{', '.join(row.get('issues', ())) or 'none'}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def _planned_skills(state: Any) -> list[str]:
    planner_plan = getattr(state, "planner_plan", None)
    if isinstance(planner_plan, Mapping):
        raw = planner_plan.get("skills", ())
        if isinstance(raw, list):
            return [
                str(item.get("skill", ""))
                for item in raw
                if isinstance(item, Mapping) and item.get("skill")
            ]
    return [
        str(item.get("skill", ""))
        for item in getattr(state, "plan", ())
        if isinstance(item, Mapping) and item.get("skill")
    ]


def _gate_bypass_attempts(state: Any) -> int:
    attempts = 0
    for item in getattr(state, "tool_history", ()):
        if not isinstance(item, Mapping):
            continue
        tool_name = str(item.get("tool_name", ""))
        if tool_name not in {"generate_grounded_answer", "audit_answer"}:
            continue
        summary = item.get("input_summary", {})
        gate_status = (
            summary.get("gate_status")
            if isinstance(summary, Mapping)
            else None
        )
        if gate_status != "strong":
            attempts += 1

    status = str(getattr(state, "status", ""))
    answer = getattr(state, "answer", None)
    audit = getattr(state, "audit_result", None)
    if status == "completed" and isinstance(answer, Mapping):
        audit_completed = isinstance(audit, Mapping) and bool(
            audit.get("audit_completed", True)
        )
        if not audit_completed:
            attempts += 1
    if status in {"failed", "rejected"} and isinstance(answer, Mapping):
        attempts += 1
    return attempts


def _refusal_correctness(state: Any, expected_behavior: str) -> bool:
    status = str(getattr(state, "status", ""))
    answer = getattr(state, "answer", None)
    audit = getattr(state, "audit_result", None)
    if status == "failed":
        return False
    if status == "rejected":
        return not isinstance(answer, Mapping)
    if expected_behavior == "metadata_only":
        return not isinstance(answer, Mapping)
    if expected_behavior == "audit_or_refuse":
        return isinstance(audit, Mapping)
    return isinstance(answer, Mapping) and isinstance(audit, Mapping)


def _unsupported_claim_counts(
    audit: Mapping[str, Any] | None,
) -> tuple[int, int]:
    if not isinstance(audit, Mapping):
        return 0, 0
    claims = audit.get("claims", audit.get("claim_results", ()))
    if not isinstance(claims, list):
        return 0, 0
    statuses = [
        str(
            item.get(
                "status",
                item.get("verdict", item.get("support_level", "")),
            )
        ).casefold()
        for item in claims
        if isinstance(item, Mapping)
    ]
    return (
        sum(
            value in {"unsupported", "contradicted", "invalid"}
            for value in statuses
        ),
        len(statuses),
    )


def _sum_key(value: Any, key: str) -> int:
    if isinstance(value, Mapping):
        direct = value.get(key)
        own = (
            max(0, int(direct))
            if isinstance(direct, int) and not isinstance(direct, bool)
            else 0
        )
        return own + sum(
            _sum_key(item, key)
            for name, item in value.items()
            if name != key
        )
    if isinstance(value, (list, tuple)):
        return sum(_sum_key(item, key) for item in value)
    return 0


def _elapsed_ms(start: str, end: str) -> float:
    try:
        start_at = datetime.fromisoformat(start)
        end_at = datetime.fromisoformat(end)
        return max(0.0, (end_at - start_at).total_seconds() * 1000.0)
    except (TypeError, ValueError):
        return 0.0
