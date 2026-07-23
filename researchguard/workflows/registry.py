# C:\Users\18449\Desktop\researchguard_workspace\researchguard\workflows\registry.py
from __future__ import annotations

from typing import Any

from researchguard.tools import ToolRegistry
from researchguard.workflows.base import ResearchWorkflow, WorkflowLimits, WorkflowResult


class WorkflowRegistry:
    version = "1.0.0"

    def __init__(self) -> None:
        self._workflows: dict[str, ResearchWorkflow] = {}

    def register(self, workflow: ResearchWorkflow) -> None:
        name = str(workflow.workflow_name).strip()
        if not name:
            raise ValueError("Registered workflows must define workflow_name.")
        if name in self._workflows:
            raise ValueError(f"Workflow already registered: {name}")
        self._workflows[name] = workflow

    def get(self, name: str) -> ResearchWorkflow:
        if name not in self._workflows:
            raise KeyError(f"Unknown workflow: {name}")
        return self._workflows[name]

    def run(self, name: str, state: Any) -> WorkflowResult:
        result = self.get(name).run(state)
        if not isinstance(result, WorkflowResult):
            raise TypeError(f"Workflow {name!r} must return WorkflowResult.")
        return result

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._workflows)

    def specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._workflows[name].spec.to_dict() for name in self.names)


def build_default_workflow_registry(
    tool_registry: ToolRegistry,
    *,
    limits: WorkflowLimits | None = None,
) -> WorkflowRegistry:
    from researchguard.workflows.claim_audit import ClaimAuditWorkflow
    from researchguard.workflows.literature_review import LiteratureReviewWorkflow
    from researchguard.workflows.paper_comparison import PaperComparisonWorkflow

    registry = WorkflowRegistry()
    registry.register(LiteratureReviewWorkflow(tool_registry, limits=limits))
    registry.register(PaperComparisonWorkflow(tool_registry, limits=limits))
    registry.register(ClaimAuditWorkflow(tool_registry, limits=limits))
    return registry
