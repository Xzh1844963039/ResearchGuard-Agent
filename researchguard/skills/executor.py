# C:\Users\18449\Desktop\researchguard_workspace\researchguard\skills\executor.py
from __future__ import annotations

import time
from typing import Any

from researchguard.skills.base import SkillContext, SkillResult
from researchguard.skills.registry import SkillRegistry


class SkillExecutor:
    version = "1.0.0"

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def execute(self, skill_name: str, context: SkillContext) -> SkillResult:
        started = time.perf_counter()
        try:
            skill = self.registry.runtime(skill_name)
            self._validate_input(skill.spec.input_schema, context.inputs)
            result = skill.run(context)
            if not isinstance(result, SkillResult):
                raise TypeError(
                    f"Skill {skill_name!r} returned {type(result).__name__}, "
                    "expected SkillResult."
                )
            if result.skill_name != skill_name:
                raise ValueError("SkillResult.skill_name does not match the requested skill.")
            self._validate_output(skill.spec.output_schema, result)
            return result
        except Exception as exc:
            version = (
                self.registry.get(skill_name).version
                if skill_name in self.registry.names
                else self.version
            )
            return SkillResult.create(
                skill_name=skill_name or "unknown_skill",
                skill_version=version,
                status="failed",
                reason="skill_execution_error",
                output={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "retryable": False,
                },
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

    @staticmethod
    def _validate_input(schema: Any, values: Any) -> None:
        required = tuple(schema.get("required", ()))
        missing = [name for name in required if values.get(name) is None]
        if missing:
            raise ValueError(
                "Missing required skill inputs: " + ", ".join(missing)
            )

    @staticmethod
    def _validate_output(schema: Any, result: SkillResult) -> None:
        if result.status != "success" and any(
            isinstance(result.output.get(name), dict)
            for name in ("answer", "answer_artifact")
        ):
            raise ValueError(
                "A failed or rejected skill must not release an answer artifact."
            )
        if result.status != "success":
            return
        required = tuple(schema.get("required", ()))
        missing = [name for name in required if name not in result.output]
        if missing:
            raise ValueError(
                "Skill output is missing fields: " + ", ".join(missing)
            )
