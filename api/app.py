# api/app.py
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Mapping

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from researchguard.agent import BoundedResearchAgentController


API_VERSION = "1.0.0"
WORKFLOW_TASK_TYPES = {
    "literature_review": "literature_review",
    "paper_comparison": "paper_comparison",
    "claim_audit": "claim_audit",
}


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    workflow: str | None = None
    workflow_input: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized

    @field_validator("workflow")
    @classmethod
    def validate_workflow(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().casefold()
        if normalized not in WORKFLOW_TASK_TYPES:
            allowed = ", ".join(sorted(WORKFLOW_TASK_TYPES))
            raise ValueError(f"workflow must be one of: {allowed}")
        return normalized


class ResearchResponse(BaseModel):
    schema_version: str = "researchguard.api_response.v1"
    api_version: str = API_VERSION
    run_id: str
    query: str
    task_type: str
    status: str
    reason: str | None
    plan: list[dict[str, Any]]
    planner: dict[str, Any]
    skills: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    evidence_bundle: dict[str, Any] | None
    evidence: list[dict[str, Any]]
    answer: dict[str, Any] | None
    audit: dict[str, Any] | None
    memory: dict[str, Any]
    plan_revisions: list[dict[str, Any]]
    created_at: str
    updated_at: str


@lru_cache(maxsize=1)
def get_controller() -> BoundedResearchAgentController:
    return BoundedResearchAgentController()


def _response_from_state(state: Any) -> ResearchResponse:
    value = state.to_dict()
    return ResearchResponse(
        run_id=str(value["run_id"]),
        query=str(value["query"]),
        task_type=str(value["task_type"]),
        status=str(value["status"]),
        reason=value.get("reason"),
        plan=list(value.get("plan", ())),
        planner={
            "structured_plan": value.get("planner_plan"),
            "metadata": value.get("planner_metadata", {}),
        },
        skills=list(value.get("skill_history", ())),
        tool_calls=list(value.get("tool_history", ())),
        evidence_bundle=value.get("evidence_bundle"),
        evidence=list(value.get("evidence", ())),
        answer=value.get("answer"),
        audit=value.get("audit_result"),
        memory={
            "status": value.get("memory_status", {}),
            "context": value.get("memory_context", {}),
        },
        plan_revisions=list(value.get("plan_revisions", ())),
        created_at=str(value["created_at"]),
        updated_at=str(value["updated_at"]),
    )


def create_app(
    controller_factory: Callable[[], Any] | None = None,
) -> FastAPI:
    application = FastAPI(
        title="ResearchGuard-Agent API",
        description=(
            "Bounded, evidence-grounded research execution through the existing "
            "ResearchGuard controller."
        ),
        version=API_VERSION,
    )
    resolve_controller = controller_factory or get_controller

    @application.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "researchguard-agent",
            "version": API_VERSION,
        }

    @application.post("/research", response_model=ResearchResponse)
    async def research(request: ResearchRequest) -> ResearchResponse:
        task_type = (
            WORKFLOW_TASK_TYPES[request.workflow]
            if request.workflow is not None
            else None
        )
        try:
            controller = resolve_controller()
            state = await run_in_threadpool(
                controller.run,
                request.query,
                task_type=task_type,
                workflow_input=request.workflow_input,
            )
            return _response_from_state(state)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid research request: {exc}",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "ResearchGuard could not complete the request. "
                    f"Failure type: {type(exc).__name__}."
                ),
            ) from exc

    return application


app = create_app()
