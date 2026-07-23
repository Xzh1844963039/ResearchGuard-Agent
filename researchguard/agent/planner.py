# C:\Users\18449\Desktop\researchguard_workspace\researchguard\agent\planner.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from researchguard.agent.state import utc_timestamp
from researchguard.tools import ToolRegistry


PLAN_SCHEMA_VERSION = "researchguard.agent_plan.v1"
SUPPORTED_TASK_TYPES = ("qa", "comparison", "audit", "literature_search")
COMPARISON_RE = re.compile(
    r"\b(compare|comparison|contrast|difference|different|versus|vs\.?|distinguish)\b",
    re.IGNORECASE,
)
AUDIT_RE = re.compile(
    r"\b(audit|verify|validate|check)\b.{0,32}\b(citation|claim|answer|response)\b",
    re.IGNORECASE,
)
LITERATURE_SEARCH_RE = re.compile(
    r"\b(find|search|discover|identify|locate|recommend)\b.{0,48}"
    r"\b(papers?|literature|articles?|publications?|preprints?)\b",
    re.IGNORECASE,
)


class PlannerError(ValueError):
    pass


@dataclass(frozen=True)
class PlanStep:
    step_id: int
    tool: str
    purpose: str
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool": self.tool,
            "purpose": self.purpose,
            "optional": self.optional,
        }


@dataclass(frozen=True)
class AgentPlan:
    task_type: str
    steps: tuple[PlanStep, ...]
    created_at: str
    schema_version: str = PLAN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "created_at": self.created_at,
            "steps": [step.to_dict() for step in self.steps],
        }


class BoundedPlanner:
    _PURPOSES = {
        "retrieve_evidence": "Retrieve ranked canonical evidence.",
        "assess_evidence": "Check whether the retrieved evidence is sufficient.",
        "generate_grounded_answer": "Generate only through the guarded evidence pipeline.",
        "audit_answer": "Verify answer claims and citation provenance.",
        "search_scholarly_sources": "Discover external candidate paper metadata.",
    }

    def __init__(self, registry: ToolRegistry, *, max_steps: int = 6):
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        self.registry = registry
        self.max_steps = max_steps

    def create_plan(
        self,
        query: str,
        *,
        task_type: str | None = None,
        has_evidence: bool = False,
        has_answer: bool = False,
    ) -> AgentPlan:
        normalized_query = " ".join(str(query).split()).strip()
        if not normalized_query:
            raise PlannerError("Query must not be empty.")
        selected_type = task_type or self.classify_task(normalized_query)
        if selected_type not in SUPPORTED_TASK_TYPES:
            raise PlannerError(f"Unsupported task type: {selected_type}")

        if selected_type == "literature_search":
            tool_names = ["search_scholarly_sources"]
        elif selected_type == "audit":
            if not has_answer:
                raise PlannerError("Audit tasks require a provenance-bearing answer artifact.")
            tool_names = ([] if has_evidence else ["retrieve_evidence"]) + ["audit_answer"]
        else:
            tool_names = [
                "retrieve_evidence",
                "assess_evidence",
                "generate_grounded_answer",
                "audit_answer",
            ]

        if len(tool_names) > self.max_steps:
            raise PlannerError(
                f"Plan requires {len(tool_names)} steps, exceeding max_steps={self.max_steps}."
            )
        unknown_tools = [name for name in tool_names if name not in self.registry.names]
        if unknown_tools:
            raise PlannerError(f"Plan references unregistered tools: {', '.join(unknown_tools)}")

        steps = tuple(
            PlanStep(
                step_id=index,
                tool=name,
                purpose=self._PURPOSES[name],
                optional=selected_type == "audit" and name == "retrieve_evidence",
            )
            for index, name in enumerate(tool_names, start=1)
        )
        return AgentPlan(
            task_type=selected_type,
            steps=steps,
            created_at=utc_timestamp(),
        )

    @staticmethod
    def classify_task(query: str) -> str:
        if AUDIT_RE.search(query):
            return "audit"
        if LITERATURE_SEARCH_RE.search(query):
            return "literature_search"
        if COMPARISON_RE.search(query):
            return "comparison"
        return "qa"
