# C:\Users\18449\Desktop\researchguard_workspace\researchguard\skills\registry.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from researchguard.skills.base import ResearchSkill, SkillSpec


class SkillRegistry:
    version = "2.0.0"

    def __init__(self, *, available_tools: Iterable[str] = ()):
        self._specs: dict[str, SkillSpec] = {}
        self._runtimes: dict[str, ResearchSkill] = {}
        self._available_tools = frozenset(str(name) for name in available_tools)

    def register(self, skill: SkillSpec | ResearchSkill) -> None:
        spec = skill if isinstance(skill, SkillSpec) else skill.spec
        if spec.name in self._specs:
            raise ValueError(f"Skill already registered: {spec.name}")
        missing = sorted(
            set(spec.required_tools).difference(self._available_tools)
        )
        if self._available_tools and missing:
            raise ValueError(
                f"Skill {spec.name!r} references unavailable tools: "
                + ", ".join(missing)
            )
        self._specs[spec.name] = spec
        if isinstance(skill, ResearchSkill):
            self._runtimes[spec.name] = skill

    def get(self, name: str) -> SkillSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown skill: {name}")
        return self._specs[name]

    def runtime(self, name: str) -> ResearchSkill:
        if name not in self._runtimes:
            if name not in self._specs:
                raise KeyError(f"Unknown skill: {name}")
            raise KeyError(f"Skill has no runtime implementation: {name}")
        return self._runtimes[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._specs[name].copy_dict() for name in self.names)


def build_default_skill_registry(tool_registry: Any) -> SkillRegistry:
    from researchguard.skills.runtime import (
        AnalyzeClaimsSkill,
        AssessEvidenceSkill,
        AuditClaimsSkill,
        CompareEvidenceSkill,
        GenerateReportSkill,
        RetrieveEvidenceSkill,
        SearchPapersSkill,
    )

    registry = SkillRegistry()
    for skill in (
        SearchPapersSkill(),
        SearchPapersSkill(name="search_scholarly_sources"),
        RetrieveEvidenceSkill(),
        AssessEvidenceSkill(),
        AnalyzeClaimsSkill(),
        CompareEvidenceSkill(),
        AuditClaimsSkill(),
        GenerateReportSkill(),
    ):
        registry.register(skill)
    return registry
