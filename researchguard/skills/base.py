# C:\Users\18449\Desktop\researchguard_workspace\researchguard\skills\base.py
from __future__ import annotations

import copy
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping


SKILL_SPEC_SCHEMA_VERSION = "researchguard.skill_spec.v2"
SKILL_RESULT_SCHEMA_VERSION = "researchguard.skill_result.v1"
VALID_RISK_LEVELS = {"low", "medium", "high"}
VALID_SKILL_STATUSES = {"success", "rejected", "failed"}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    )


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    required_tools: tuple[str, ...]
    risk_level: str
    version: str = "1.0.0"
    schema_version: str = SKILL_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("SkillSpec.name must not be empty.")
        if not self.description.strip():
            raise ValueError("SkillSpec.description must not be empty.")
        if self.risk_level not in VALID_RISK_LEVELS:
            raise ValueError(f"Unsupported skill risk level: {self.risk_level}")
        if len(self.required_tools) != len(set(self.required_tools)):
            raise ValueError("SkillSpec.required_tools must be unique.")
        if self.input_schema.get("type") != "object":
            raise ValueError("SkillSpec.input_schema must describe an object.")
        if self.output_schema.get("type") != "object":
            raise ValueError("SkillSpec.output_schema must describe an object.")

    @property
    def required_inputs(self) -> tuple[str, ...]:
        return tuple(
            str(item)
            for item in self.input_schema.get("required", ())
        )

    @property
    def output_type(self) -> str:
        return str(
            self.output_schema.get("title")
            or self.output_schema.get("result_type")
            or "SkillOutput"
        )

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return self.required_tools

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "input_schema": _json_copy(dict(self.input_schema)),
            "output_schema": _json_copy(dict(self.output_schema)),
            "required_tools": list(self.required_tools),
            "risk_level": self.risk_level,
        }

    def copy_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.to_dict())


@dataclass(frozen=True)
class SkillResult:
    skill_name: str
    skill_version: str
    status: str
    output: Mapping[str, Any]
    evidence: tuple[Mapping[str, Any], ...]
    trace_id: str
    latency_ms: float
    reason: str | None
    timestamp: str
    version: str = "1.0.0"
    schema_version: str = SKILL_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.skill_name.strip() or not self.skill_version.strip():
            raise ValueError("SkillResult requires skill name and version.")
        if self.status not in VALID_SKILL_STATUSES:
            raise ValueError(f"Unsupported skill status: {self.status}")
        if self.status != "success" and not str(self.reason or "").strip():
            raise ValueError("Rejected or failed SkillResult requires a reason.")
        if self.latency_ms < 0:
            raise ValueError("SkillResult.latency_ms must not be negative.")
        _json_copy(self.to_dict())

    @classmethod
    def create(
        cls,
        *,
        skill_name: str,
        skill_version: str,
        status: str,
        output: Mapping[str, Any] | None = None,
        evidence: tuple[Mapping[str, Any], ...] = (),
        reason: str | None = None,
        trace_id: str | None = None,
        latency_ms: float = 0.0,
    ) -> "SkillResult":
        return cls(
            skill_name=skill_name,
            skill_version=skill_version,
            status=status,
            output=copy.deepcopy(dict(output or {})),
            evidence=tuple(copy.deepcopy(dict(item)) for item in evidence),
            trace_id=trace_id or f"skill-{uuid.uuid4().hex}",
            latency_ms=max(0.0, float(latency_ms)),
            reason=reason,
            timestamp=utc_timestamp(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "skill_name": self.skill_name,
            "skill_version": self.skill_version,
            "status": self.status,
            "output": _json_copy(dict(self.output)),
            "evidence": _json_copy(list(self.evidence)),
            "trace_id": self.trace_id,
            "latency_ms": self.latency_ms,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )


@dataclass(frozen=True)
class SkillContext:
    query: str
    task_type: str
    state: Any
    inputs: Mapping[str, Any]
    invoke_tool: Callable[[str, Mapping[str, Any]], Any]


class ResearchSkill(ABC):
    name: str
    version: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    required_tools: tuple[str, ...]
    risk_level: str

    @property
    def spec(self) -> SkillSpec:
        return SkillSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            required_tools=self.required_tools,
            risk_level=self.risk_level,
            version=self.version,
        )

    @abstractmethod
    def run(self, context: SkillContext) -> SkillResult:
        raise NotImplementedError

    def _result(
        self,
        started: float,
        *,
        status: str,
        output: Mapping[str, Any] | None = None,
        evidence: tuple[Mapping[str, Any], ...] = (),
        reason: str | None = None,
    ) -> SkillResult:
        return SkillResult.create(
            skill_name=self.name,
            skill_version=self.version,
            status=status,
            output=output,
            evidence=evidence,
            reason=reason,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
