# C:\Users\18449\Desktop\researchguard_workspace\researchguard\skills\registry.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from researchguard.skills.specs import SkillSpec


class SkillRegistry:
    version = "1.0.0"

    def __init__(self, *, available_tools: Iterable[str] = ()):
        self._skills: dict[str, SkillSpec] = {}
        self._available_tools = frozenset(str(name) for name in available_tools)

    def register(self, skill: SkillSpec) -> None:
        if skill.name in self._skills:
            raise ValueError(f"Skill already registered: {skill.name}")
        missing = sorted(set(skill.allowed_tools).difference(self._available_tools))
        if self._available_tools and missing:
            raise ValueError(
                f"Skill {skill.name!r} references unavailable tools: {', '.join(missing)}"
            )
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillSpec:
        if name not in self._skills:
            raise KeyError(f"Unknown skill: {name}")
        return self._skills[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._skills)

    def specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._skills[name].copy_dict() for name in self.names)


def build_default_skill_registry(tool_registry: Any) -> SkillRegistry:
    # Declare the complete capability catalog. PlannerValidator checks whether
    # a selected skill is executable by the current (possibly partial) ToolRegistry.
    registry = SkillRegistry()
    registry.register(
        SkillSpec(
            name="retrieve_evidence",
            description="Retrieve ranked canonical evidence from the local corpus.",
            required_inputs=("query",),
            output_type="EvidenceBundle",
            allowed_tools=("retrieve_evidence",),
            risk_level="low",
        )
    )
    registry.register(
        SkillSpec(
            name="search_scholarly_sources",
            description="Discover metadata-only scholarly candidates outside the corpus.",
            required_inputs=("query",),
            output_type="ScholarPaperRecordList",
            allowed_tools=("search_scholarly_sources",),
            risk_level="low",
        )
    )
    registry.register(
        SkillSpec(
            name="assess_evidence",
            description="Classify an EvidenceBundle as strong, partial, or unsupported.",
            required_inputs=("evidence_bundle",),
            output_type="GateDecision",
            allowed_tools=("assess_evidence",),
            risk_level="medium",
        )
    )
    registry.register(
        SkillSpec(
            name="compare_evidence",
            description="Collect and separate canonical evidence for a bounded paper comparison.",
            required_inputs=("query", "papers"),
            output_type="ComparisonEvidence",
            allowed_tools=("retrieve_evidence", "assess_evidence"),
            risk_level="medium",
        )
    )
    registry.register(
        SkillSpec(
            name="audit_claims",
            description="Audit answer or claim citations against the originating EvidenceBundle.",
            required_inputs=("answer_artifact", "evidence_bundle"),
            output_type="CitationAuditResult",
            allowed_tools=("audit_answer",),
            risk_level="high",
        )
    )
    registry.register(
        SkillSpec(
            name="generate_report",
            description="Generate a grounded answer only from a strong GateDecision and its bundle.",
            required_inputs=("evidence_bundle", "gate_decision"),
            output_type="AnswerArtifact",
            allowed_tools=("generate_grounded_answer",),
            risk_level="high",
        )
    )
    return registry
