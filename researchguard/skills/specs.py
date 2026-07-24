# C:\Users\18449\Desktop\researchguard_workspace\researchguard\skills\specs.py
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


SKILL_SPEC_SCHEMA_VERSION = "researchguard.skill_spec.v1"
VALID_RISK_LEVELS = {"low", "medium", "high"}


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    required_inputs: tuple[str, ...]
    output_type: str
    allowed_tools: tuple[str, ...]
    risk_level: str
    version: str = "1.0.0"
    schema_version: str = SKILL_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("SkillSpec.name must not be empty.")
        if not self.description.strip():
            raise ValueError("SkillSpec.description must not be empty.")
        if not self.output_type.strip():
            raise ValueError("SkillSpec.output_type must not be empty.")
        if not self.allowed_tools:
            raise ValueError("SkillSpec.allowed_tools must not be empty.")
        if self.risk_level not in VALID_RISK_LEVELS:
            raise ValueError(f"Unsupported skill risk level: {self.risk_level}")
        if len(self.required_inputs) != len(set(self.required_inputs)):
            raise ValueError("SkillSpec.required_inputs must be unique.")
        if len(self.allowed_tools) != len(set(self.allowed_tools)):
            raise ValueError("SkillSpec.allowed_tools must be unique.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "required_inputs": list(self.required_inputs),
            "output_type": self.output_type,
            "allowed_tools": list(self.allowed_tools),
            "risk_level": self.risk_level,
        }

    def copy_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.to_dict())
