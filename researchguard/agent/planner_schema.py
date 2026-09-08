# C:\Users\18449\Desktop\researchguard_workspace\researchguard\agent\planner_schema.py
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


STRUCTURED_PLAN_SCHEMA_VERSION = "researchguard.structured_plan.v2"
SUPPORTED_PLAN_SCHEMA_VERSIONS = {
    "researchguard.structured_plan.v1",
    STRUCTURED_PLAN_SCHEMA_VERSION,
}


class PlanSchemaError(ValueError):
    pass


def _strict_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    label: str,
) -> None:
    missing = sorted(required.difference(value))
    unknown = sorted(set(value).difference(required | optional))
    if missing:
        raise PlanSchemaError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise PlanSchemaError(f"{label} contains unknown fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class PlanBudget:
    max_steps: int
    max_tool_calls: int
    max_retries: int
    max_plan_revisions: int

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.max_tool_calls < 1:
            raise PlanSchemaError("Plan budget step and tool-call limits must be positive.")
        if self.max_retries < 0 or self.max_plan_revisions < 0:
            raise PlanSchemaError("Plan budget retry and revision limits must not be negative.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PlanBudget":
        _strict_keys(
            value,
            required={
                "max_steps",
                "max_tool_calls",
                "max_retries",
                "max_plan_revisions",
            },
            label="budget",
        )
        for key in value:
            if isinstance(value[key], bool) or not isinstance(value[key], int):
                raise PlanSchemaError(f"budget.{key} must be an integer.")
        return cls(
            max_steps=int(value["max_steps"]),
            max_tool_calls=int(value["max_tool_calls"]),
            max_retries=int(value["max_retries"]),
            max_plan_revisions=int(value["max_plan_revisions"]),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_retries": self.max_retries,
            "max_plan_revisions": self.max_plan_revisions,
        }


@dataclass(frozen=True)
class ResearchSubgoal:
    subgoal_id: str
    question: str
    objective: str
    success_criterion: str

    def __post_init__(self) -> None:
        for label, value in (
            ("subgoal_id", self.subgoal_id),
            ("question", self.question),
            ("objective", self.objective),
            ("success_criterion", self.success_criterion),
        ):
            if not str(value).strip():
                raise PlanSchemaError(f"Subgoal {label} must not be empty.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResearchSubgoal":
        _strict_keys(
            value,
            required={
                "subgoal_id",
                "question",
                "objective",
                "success_criterion",
            },
            label="research subgoal",
        )
        return cls(
            subgoal_id=str(value["subgoal_id"]).strip(),
            question=str(value["question"]).strip(),
            objective=str(value["objective"]).strip(),
            success_criterion=str(value["success_criterion"]).strip(),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "subgoal_id": self.subgoal_id,
            "question": self.question,
            "objective": self.objective,
            "success_criterion": self.success_criterion,
        }


@dataclass(frozen=True)
class StructuredPlanStep:
    skill: str
    purpose: str
    expected_observation: str
    max_retry: int
    subgoal_id: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.skill.strip():
            raise PlanSchemaError("Plan step skill must not be empty.")
        if not self.purpose.strip():
            raise PlanSchemaError("Plan step purpose must not be empty.")
        if not self.expected_observation.strip():
            raise PlanSchemaError("Plan step expected_observation must not be empty.")
        if isinstance(self.max_retry, bool) or not isinstance(self.max_retry, int):
            raise PlanSchemaError("Plan step max_retry must be an integer.")
        if self.max_retry < 0:
            raise PlanSchemaError("Plan step max_retry must not be negative.")
        normalized_subgoal_id = (
            str(self.subgoal_id).strip() if self.subgoal_id is not None else None
        )
        object.__setattr__(self, "subgoal_id", normalized_subgoal_id or None)
        if not isinstance(self.parameters, Mapping):
            raise PlanSchemaError("Plan step parameters must be an object.")
        normalized_parameters = {
            key: copy.deepcopy(value)
            for key, value in dict(self.parameters).items()
            if value is not None
        }
        if any(not isinstance(key, str) for key in normalized_parameters):
            raise PlanSchemaError("Plan step parameter names must be strings.")
        try:
            json.dumps(normalized_parameters, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise PlanSchemaError(
                "Plan step parameters must be JSON serializable."
            ) from exc
        object.__setattr__(self, "parameters", normalized_parameters)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StructuredPlanStep":
        _strict_keys(
            value,
            required={"skill", "purpose", "expected_observation", "max_retry"},
            optional={"subgoal_id", "parameters"},
            label="plan step",
        )
        return cls(
            skill=str(value["skill"]).strip(),
            purpose=str(value["purpose"]).strip(),
            expected_observation=str(value["expected_observation"]).strip(),
            max_retry=value["max_retry"],
            subgoal_id=(
                str(value["subgoal_id"]).strip()
                if value.get("subgoal_id") is not None
                else None
            ),
            parameters=dict(value.get("parameters", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "purpose": self.purpose,
            "expected_observation": self.expected_observation,
            "max_retry": self.max_retry,
            "subgoal_id": self.subgoal_id,
            "parameters": copy.deepcopy(dict(self.parameters)),
        }


@dataclass(frozen=True)
class StructuredPlan:
    task_type: str
    goal: str
    steps: tuple[StructuredPlanStep, ...]
    budget: PlanBudget
    reasoning_summary: str
    planner_version: str
    subgoals: tuple[ResearchSubgoal, ...] = ()
    schema_version: str = STRUCTURED_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.task_type.strip():
            raise PlanSchemaError("Plan task_type must not be empty.")
        if not self.goal.strip():
            raise PlanSchemaError("Plan goal must not be empty.")
        if not self.steps:
            raise PlanSchemaError("Plan must contain at least one step.")
        if not self.reasoning_summary.strip():
            raise PlanSchemaError("Plan reasoning_summary must not be empty.")
        if not self.planner_version.strip():
            raise PlanSchemaError("Plan planner_version must not be empty.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StructuredPlan":
        _strict_keys(
            value,
            required={
                "task_type",
                "goal",
                "skills",
                "budget",
                "reasoning_summary",
                "planner_version",
            },
            optional={"schema_version", "subgoals"},
            label="structured plan",
        )
        raw_steps = value["skills"]
        if not isinstance(raw_steps, list):
            raise PlanSchemaError("Plan skills must be a list.")
        if any(not isinstance(item, Mapping) for item in raw_steps):
            raise PlanSchemaError("Every plan skill must be an object.")
        raw_budget = value["budget"]
        if not isinstance(raw_budget, Mapping):
            raise PlanSchemaError("Plan budget must be an object.")
        schema_version = str(
            value.get("schema_version", STRUCTURED_PLAN_SCHEMA_VERSION)
        )
        if schema_version not in SUPPORTED_PLAN_SCHEMA_VERSIONS:
            raise PlanSchemaError(f"Unsupported plan schema: {schema_version}")
        raw_subgoals = value.get("subgoals", [])
        if not isinstance(raw_subgoals, list):
            raise PlanSchemaError("Plan subgoals must be a list.")
        if any(not isinstance(item, Mapping) for item in raw_subgoals):
            raise PlanSchemaError("Every plan subgoal must be an object.")
        return cls(
            task_type=str(value["task_type"]).strip(),
            goal=str(value["goal"]).strip(),
            steps=tuple(
                StructuredPlanStep.from_mapping(item)
                for item in raw_steps
            ),
            budget=PlanBudget.from_mapping(raw_budget),
            reasoning_summary=str(value["reasoning_summary"]).strip(),
            planner_version=str(value["planner_version"]).strip(),
            subgoals=tuple(
                ResearchSubgoal.from_mapping(item)
                for item in raw_subgoals
            ),
            schema_version=STRUCTURED_PLAN_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "goal": self.goal,
            "subgoals": [subgoal.to_dict() for subgoal in self.subgoals],
            "skills": [step.to_dict() for step in self.steps],
            "budget": self.budget.to_dict(),
            "reasoning_summary": self.reasoning_summary,
            "planner_version": self.planner_version,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    def copy_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.to_dict())
