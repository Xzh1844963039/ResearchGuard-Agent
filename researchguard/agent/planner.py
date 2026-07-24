# C:\Users\18449\Desktop\researchguard_workspace\researchguard\agent\planner.py
from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Mapping

from researchguard.agent.state import utc_timestamp
from researchguard.tools import ToolRegistry


PLAN_SCHEMA_VERSION = "researchguard.agent_plan.v1"
SUPPORTED_TASK_TYPES = (
    "qa",
    "comparison",
    "audit",
    "literature_search",
    "literature_review",
    "paper_comparison",
    "claim_audit",
)
WORKFLOW_TASK_TYPES = {
    "literature_review": "literature_review",
    "paper_comparison": "paper_comparison",
    "claim_audit": "claim_audit",
}
COMPARISON_RE = re.compile(
    r"\b(compare|comparison|contrast|difference|different|versus|vs\.?|distinguish)\b",
    re.IGNORECASE,
)
CLAIM_AUDIT_RE = re.compile(
    r"\b(audit|verify|validate|check|fact-check)\b.{0,40}"
    r"\b(claim|statement|assertion)\b",
    re.IGNORECASE,
)
LITERATURE_REVIEW_RE = re.compile(
    r"^\s*(review|survey|synthesize)\b|"
    r"\b(literature|research)\s+(review|survey|synthesis)\b",
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
    workflow: str | None = None
    memory_context: Mapping[str, Any] | None = None
    schema_version: str = PLAN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "created_at": self.created_at,
            "workflow": self.workflow,
            "memory_context": copy.deepcopy(dict(self.memory_context or {})),
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

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_steps: int = 6,
        workflow_names: Iterable[str] = (),
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        self.registry = registry
        self.max_steps = max_steps
        self.workflow_names = tuple(dict.fromkeys(str(name) for name in workflow_names))

    def create_plan(
        self,
        query: str,
        *,
        task_type: str | None = None,
        has_evidence: bool = False,
        has_answer: bool = False,
        memory_context: Mapping[str, Any] | None = None,
    ) -> AgentPlan:
        normalized_query = " ".join(str(query).split()).strip()
        if not normalized_query:
            raise PlannerError("Query must not be empty.")
        selected_type = task_type or self.classify_task(normalized_query)
        if selected_type not in SUPPORTED_TASK_TYPES:
            raise PlannerError(f"Unsupported task type: {selected_type}")

        workflow = WORKFLOW_TASK_TYPES.get(selected_type)
        if workflow is not None:
            if workflow not in self.workflow_names:
                raise PlannerError(f"Workflow is not registered: {workflow}")
            tool_names: list[str] = []
        elif selected_type == "literature_search":
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
            workflow=workflow,
            memory_context=self._advisory_memory(memory_context),
        )

    @staticmethod
    def _advisory_memory(
        memory_context: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if not memory_context:
            return {}
        return {
            "schema_version": str(memory_context.get("schema_version", "")),
            "matched_run_ids": [
                str(item)
                for item in list(memory_context.get("matched_run_ids", ()))[:5]
            ],
            "previous_workflows": [
                str(item)
                for item in list(memory_context.get("previous_workflows", ()))[:5]
            ],
            "previous_papers": copy.deepcopy(
                list(memory_context.get("previous_papers", ()))[:10]
            ),
            "previous_failures": copy.deepcopy(
                list(memory_context.get("previous_failures", ()))[:5]
            ),
        }

    @staticmethod
    def classify_task(query: str) -> str:
        if CLAIM_AUDIT_RE.search(query):
            return "claim_audit"
        if AUDIT_RE.search(query):
            return "audit"
        if LITERATURE_REVIEW_RE.search(query):
            return "literature_review"
        if LITERATURE_SEARCH_RE.search(query):
            return "literature_search"
        if COMPARISON_RE.search(query):
            return "paper_comparison"
        return "qa"
