# C:\Users\18449\Desktop\researchguard_workspace\researchguard\evaluation\holdout.py
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from researchguard.evaluation.benchmark_v3 import utc_timestamp


HOLDOUT_SCHEMA_VERSION = "researchguard.holdout_evaluation.v1"
VALID_TASK_TYPES = {"qa", "paper_comparison", "claim_audit"}


@dataclass(frozen=True)
class HoldoutQuestion:
    question_id: str
    query: str
    expected_task_type: str
    expected_skill_pattern: tuple[str, ...]
    expected_doc_ids: tuple[str, ...]
    difficulty: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HoldoutQuestion":
        question = cls(
            question_id=str(value.get("question_id", "")).strip(),
            query=str(value.get("query", "")).strip(),
            expected_task_type=str(value.get("expected_task_type", "")).strip(),
            expected_skill_pattern=tuple(
                str(item).strip()
                for item in value.get("expected_skill_pattern", ())
                if str(item).strip()
            ),
            expected_doc_ids=tuple(
                str(item).strip()
                for item in value.get("expected_doc_ids", ())
                if str(item).strip()
            ),
            difficulty=str(value.get("difficulty", "")).strip(),
        )
        if not question.question_id or not question.query:
            raise ValueError("Holdout question id and query must not be empty.")
        if question.expected_task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"Unsupported holdout task type: {question.expected_task_type}"
            )
        if not question.expected_skill_pattern:
            raise ValueError("Holdout question must label a skill pattern.")
        if not question.expected_doc_ids:
            raise ValueError("Holdout question must label at least one document.")
        return question

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "query": self.query,
            "expected_task_type": self.expected_task_type,
            "expected_skill_pattern": list(self.expected_skill_pattern),
            "expected_doc_ids": list(self.expected_doc_ids),
            "difficulty": self.difficulty,
        }


def load_holdout_questions(path: str | Path) -> list[HoldoutQuestion]:
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Holdout line {line_number} must contain a JSON object."
            )
        rows.append(HoldoutQuestion.from_mapping(value))
    if not 20 <= len(rows) <= 30:
        raise ValueError("Holdout v1 must contain between 20 and 30 questions.")
    if len({row.question_id for row in rows}) != len(rows):
        raise ValueError("Holdout question IDs must be unique.")
    return rows


def evaluate_holdout_run(
    state: Any,
    question: HoldoutQuestion,
) -> dict[str, Any]:
    planned_skills = [
        str(item.get("skill", ""))
        for item in getattr(state, "plan", ())
        if isinstance(item, Mapping) and item.get("skill")
    ]
    evidence = [
        item
        for item in getattr(state, "evidence", ())
        if isinstance(item, Mapping)
    ]
    observed_docs = {
        str(item.get("doc_id", "")).strip()
        for item in evidence
        if str(item.get("doc_id", "")).strip()
    }
    expected_docs = set(question.expected_doc_ids)
    matched_docs = observed_docs.intersection(expected_docs)
    invalid_provenance = [
        item
        for item in evidence
        if not all(
            (
                str(item.get("chunk_id", "")).strip(),
                str(item.get("doc_id", "")).strip(),
                str(item.get("section", "")).strip(),
                item.get("page") is not None,
                str(item.get("content", "")).strip(),
            )
        )
    ]
    status = str(getattr(state, "status", ""))
    answer = getattr(state, "answer", None)
    audit = getattr(state, "audit_result", None)
    gate = getattr(state, "gate_decision", None)
    gate_status = (
        str(gate.get("status", "")).casefold()
        if isinstance(gate, Mapping)
        else ""
    )
    generated = isinstance(answer, Mapping)
    audited = isinstance(audit, Mapping)
    gate_bypass = int(generated and gate_status != "strong")
    unsupported, claim_count = _unsupported_claim_counts(audit)
    task_correct = str(getattr(state, "task_type", "")) == question.expected_task_type
    skills_correct = tuple(planned_skills) == question.expected_skill_pattern
    document_hit = bool(matched_docs)
    document_coverage = len(matched_docs) / len(expected_docs)
    completion_ok = status in {"completed", "rejected"}
    answer_behavior_ok = (
        not generated if status == "rejected" else generated and audited
    )
    passed = all(
        (
            task_correct,
            skills_correct,
            document_hit,
            not invalid_provenance,
            completion_ok,
            answer_behavior_ok,
            gate_bypass == 0,
        )
    )
    return {
        "question_id": question.question_id,
        "passed": passed,
        "expected": question.to_dict(),
        "observed": {
            "task_type": str(getattr(state, "task_type", "")),
            "planned_skills": planned_skills,
            "status": status,
            "reason": getattr(state, "reason", None),
            "evidence_doc_ids": sorted(observed_docs),
            "gate_status": gate_status or None,
            "answer_generated": generated,
            "audit_completed": audited,
        },
        "metrics": {
            "task_classification_accuracy": float(task_correct),
            "skill_sequence_accuracy": float(skills_correct),
            "document_hit_at_10": float(document_hit),
            "document_coverage_at_10": document_coverage,
            "provenance_validity": (
                1.0 - len(invalid_provenance) / len(evidence)
                if evidence
                else 1.0
            ),
            "completion_rate": float(completion_ok),
            "answer_or_refusal_correctness": float(answer_behavior_ok),
            "unsupported_claim_rate": (
                unsupported / claim_count if claim_count else None
            ),
            "gate_bypass_attempts": gate_bypass,
            "tool_call_count": len(getattr(state, "tool_history", ())),
            "skill_count": len(getattr(state, "skill_history", ())),
        },
        "limitations": [
            label
            for label, active in (
                ("task_routing_mismatch", not task_correct),
                ("skill_sequence_mismatch", not skills_correct),
                ("expected_document_not_retrieved", not document_hit),
                ("partial_document_coverage", document_coverage < 1.0),
                ("invalid_provenance", bool(invalid_provenance)),
                ("execution_failed", not completion_ok),
                ("answer_behavior_mismatch", not answer_behavior_ok),
                ("gate_bypass", gate_bypass > 0),
            )
            if active
        ],
    }


def build_holdout_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    corpus_status: Mapping[str, Any],
) -> dict[str, Any]:
    results = [copy.deepcopy(dict(row)) for row in rows]
    metric_names = sorted(
        {
            name
            for row in results
            for name in dict(row.get("metrics", {}))
        }
    )
    aggregate: dict[str, Any] = {
        "question_count": len(results),
        "pass_rate": (
            sum(bool(row.get("passed")) for row in results) / len(results)
            if results
            else 0.0
        ),
    }
    for name in metric_names:
        values = [
            float(row["metrics"][name])
            for row in results
            if isinstance(row.get("metrics"), Mapping)
            and isinstance(row["metrics"].get(name), (int, float))
        ]
        aggregate[name] = mean(values) if values else None
    hard_checks = {
        "empty_result": int(not results),
        "gate_bypass_attempt": int(
            sum(
                float(row.get("metrics", {}).get("gate_bypass_attempts", 0))
                for row in results
            )
        ),
        "provenance_failure": sum(
            float(row.get("metrics", {}).get("provenance_validity", 1.0)) < 1.0
            for row in results
        ),
        "execution_exception": sum(
            "execution_exception" in row.get("limitations", ())
            for row in results
        ),
    }
    if not corpus_status.get("ready"):
        conclusion = "NOT_RUN"
    elif sum(hard_checks.values()):
        conclusion = "FAIL"
    elif float(aggregate.get("pass_rate") or 0.0) >= 0.80:
        conclusion = "PASS"
    else:
        conclusion = "PASS_WITH_LIMITATIONS"
    return {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "version": "1.0.0",
        "generated_at": utc_timestamp(),
        "conclusion": conclusion,
        "corpus_status": copy.deepcopy(dict(corpus_status)),
        "aggregate_metrics": aggregate,
        "hard_checks": hard_checks,
        "results": results,
    }


def render_holdout_markdown(report: Mapping[str, Any]) -> str:
    aggregate = report.get("aggregate_metrics", {})
    corpus = report.get("corpus_status", {})
    lines = [
        "# Holdout Evaluation v1",
        "",
        f"- Conclusion: `{report.get('conclusion')}`",
        f"- Corpus ready: `{corpus.get('ready', False)}`",
        f"- Papers: `{corpus.get('paper_count', 0)}`",
        f"- Blind questions executed: `{aggregate.get('question_count', 0)}`",
        "- Labels contain routing and document targets only; no answers or chunk IDs.",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in sorted(dict(aggregate).items()):
        rendered = (
            "N/A"
            if value is None
            else f"{value:.4f}"
            if isinstance(value, float)
            else str(value)
        )
        lines.append(f"| `{name}` | {rendered} |")
    lines.extend(["", "## Limitations", ""])
    limitations = sorted(
        {
            limitation
            for row in report.get("results", ())
            for limitation in row.get("limitations", ())
        }
    )
    if not corpus.get("ready"):
        limitations.append(
            str(corpus.get("reason") or "Dedicated holdout index is not ready.")
        )
    for item in limitations or ["No structural limitation recorded."]:
        lines.append(f"- {item}")
    lines.extend(["", "## Questions", ""])
    for row in report.get("results", ()):
        observed = row.get("observed", {})
        lines.append(
            f"- `{row.get('question_id')}`: "
            f"`{'PASS' if row.get('passed') else 'CHECK'}`; "
            f"task=`{observed.get('task_type')}`; "
            f"status=`{observed.get('status')}`; "
            f"documents=`{', '.join(observed.get('evidence_doc_ids', ()))}`; "
            f"limitations=`{', '.join(row.get('limitations', ())) or 'none'}`"
        )
    return "\n".join(lines).rstrip() + "\n"


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
