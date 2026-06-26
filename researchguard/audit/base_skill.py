# C:\Users\18449\Desktop\researchguard_workspace\researchguard\audit\base_skill.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from researchguard.memory.memory_store import MemoryStore
from researchguard.text_utils_v2 import compact_text


@dataclass
class SkillSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


class BaseSkill:
    name = "base_skill"
    description = "Base skill"
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    @property
    def spec(self) -> SkillSpec:
        return SkillSpec(self.name, self.description, self.input_schema, self.output_schema)

    def execute(self, case_id: str, payload: dict[str, Any], cached: bool = False) -> dict[str, Any]:
        """Run a registered skill with structured trace and error handling."""
        with self.memory.trace_skill(case_id, self.name, compact_text(str(payload), 220), cached=cached) as trace:
            try:
                result = self.run(case_id, payload)
                result.setdefault("ok", True)
                summary = compact_text(str(result), 240)
                if result.get("status") == "skipped":
                    trace.skipped(summary)
                elif result.get("status") == "failed":
                    trace.failed(summary, str(result.get("error_message") or "skill returned failed status"))
                else:
                    trace.success(summary)
                return result
            except Exception as exc:
                self.memory.append_failure(case_id, f"{self.name}_failed", str(exc), {"input": payload})
                trace.failed("failed", str(exc))
                return {"ok": False, "error_message": str(exc), "skill_name": self.name}

    def run(self, case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
