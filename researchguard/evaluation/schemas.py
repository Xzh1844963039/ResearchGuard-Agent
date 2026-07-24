# C:\Users\18449\Desktop\researchguard_workspace\researchguard\evaluation\schemas.py
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


EVALUATION_SCHEMA_VERSION = "researchguard.agent_evaluation.v1"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AgentEvaluationCase:
    case_id: str
    query: str
    expected_task_type: str
    expected_workflow: str | None
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    relevant_evidence_ids: tuple[str, ...] = ()
    expected_status: str = "completed"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("AgentEvaluationCase.case_id must not be empty.")
        if not self.query.strip():
            raise ValueError("AgentEvaluationCase.query must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "case_id": self.case_id,
            "query": self.query,
            "expected_task_type": self.expected_task_type,
            "expected_workflow": self.expected_workflow,
            "expected_tools": list(self.expected_tools),
            "forbidden_tools": list(self.forbidden_tools),
            "relevant_evidence_ids": list(self.relevant_evidence_ids),
            "expected_status": self.expected_status,
            "metadata": copy.deepcopy(dict(self.metadata)),
        }


@dataclass(frozen=True)
class MetricValue:
    name: str
    category: str
    value: float | int | bool | None
    passed: bool | None
    details: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "category": self.category,
            "value": self.value,
            "passed": self.passed,
            "details": copy.deepcopy(dict(self.details)),
        }


@dataclass(frozen=True)
class AgentEvaluationResult:
    case_id: str
    passed: bool
    metrics: Mapping[str, MetricValue]
    issues: tuple[str, ...]
    expected: Mapping[str, Any]
    observed: Mapping[str, Any]
    trace: Mapping[str, Any]
    evaluated_at: str = field(default_factory=utc_timestamp)
    version: str = "1.0.0"
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "case_id": self.case_id,
            "passed": self.passed,
            "metrics": {
                name: metric.to_dict() for name, metric in sorted(self.metrics.items())
            },
            "issues": list(self.issues),
            "expected": copy.deepcopy(dict(self.expected)),
            "observed": copy.deepcopy(dict(self.observed)),
            "trace": copy.deepcopy(dict(self.trace)),
            "evaluated_at": self.evaluated_at,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent)


@dataclass(frozen=True)
class AgentEvaluationReport:
    results: tuple[AgentEvaluationResult, ...]
    aggregate_metrics: Mapping[str, float | int | None]
    passed: bool
    generated_at: str = field(default_factory=utc_timestamp)
    version: str = "1.0.0"
    schema_version: str = EVALUATION_SCHEMA_VERSION

    @property
    def case_count(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(result.passed for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "generated_at": self.generated_at,
            "passed": self.passed,
            "case_count": self.case_count,
            "passed_count": self.passed_count,
            "aggregate_metrics": copy.deepcopy(dict(self.aggregate_metrics)),
            "results": [result.to_dict() for result in self.results],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent)
