# C:\Users\18449\Desktop\researchguard_workspace\researchguard\agent\hybrid_planner.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Protocol

import yaml
from openai import OpenAI

from researchguard.agent.planner import (
    AgentPlan,
    BoundedPlanner,
    PlanStep,
    PlannerError,
    SUPPORTED_TASK_TYPES,
    WORKFLOW_TASK_TYPES,
)
from researchguard.agent.planner_schema import (
    PlanBudget,
    PlanSchemaError,
    StructuredPlan,
    StructuredPlanStep,
)
from researchguard.agent.planner_validator import (
    PlanValidationResult,
    PlannerValidator,
)
from researchguard.agent.policy import AgentPolicy
from researchguard.agent.state import utc_timestamp
from researchguard.skills import SkillRegistry
from researchguard.tools import ToolRegistry


DEFAULT_PLANNER_CONFIG = Path("configs/planner_v2.yaml")
SKILL_TO_TOOL = {
    "retrieve_evidence": "retrieve_evidence",
    "search_scholarly_sources": "search_scholarly_sources",
    "assess_evidence": "assess_evidence",
    "generate_report": "generate_grounded_answer",
    "audit_claims": "audit_answer",
}


@dataclass(frozen=True)
class HybridPlannerSettings:
    enabled: bool
    backend: str
    model: str
    temperature: float
    timeout: float
    max_tokens: int
    max_steps: int
    fallback_enabled: bool
    prompt_version: str
    config_version: str

    def __post_init__(self) -> None:
        if self.backend != "openai":
            raise ValueError(f"Unsupported planner backend: {self.backend}")
        if self.temperature != 0:
            raise ValueError("Hybrid Planner temperature must be 0.")
        if self.timeout <= 0 or self.max_tokens < 1 or self.max_steps < 1:
            raise ValueError("Hybrid Planner limits must be positive.")


@dataclass(frozen=True)
class PlannerBackendResponse:
    payload: Mapping[str, Any] | str
    api_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class PlannerOutcome:
    structured_plan: StructuredPlan
    executable_plan: AgentPlan
    mode: str
    fallback_used: bool
    fallback_reason: str | None
    validation: PlanValidationResult
    planner_model: str
    prompt_version: str
    latency_ms: float
    api_call_count: int
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: str = field(default_factory=utc_timestamp)
    version: str = "1.0.0"

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "validation": self.validation.to_dict(),
            "planner_model": self.planner_model,
            "prompt_version": self.prompt_version,
            "latency_ms": self.latency_ms,
            "api_call_count": self.api_call_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "created_at": self.created_at,
        }


class PlannerInterface(Protocol):
    def generate_plan(
        self,
        query: str,
        *,
        task_type: str | None = None,
        has_evidence: bool = False,
        has_answer: bool = False,
        memory_context: Mapping[str, Any] | None = None,
    ) -> PlannerOutcome:
        ...


def load_hybrid_planner_settings(
    path: str | Path = DEFAULT_PLANNER_CONFIG,
) -> HybridPlannerSettings:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Planner config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    planner = raw.get("planner", {}) or {}
    return HybridPlannerSettings(
        enabled=bool(planner.get("enabled", True)),
        backend=str(planner.get("backend", "openai")),
        model=str(planner.get("model", "gpt-4.1-mini")),
        temperature=float(planner.get("temperature", 0)),
        timeout=max(1.0, float(planner.get("timeout", 30))),
        max_tokens=max(1, int(planner.get("max_tokens", 800))),
        max_steps=max(1, int(planner.get("max_steps", 6))),
        fallback_enabled=bool(planner.get("fallback_enabled", True)),
        prompt_version=str(planner.get("prompt_version", "hybrid_planner_v1.0")),
        config_version=str(planner.get("config_version", "planner_v2.0")),
    )


class OpenAIPlannerBackend:
    def __init__(self, settings: HybridPlannerSettings):
        self.settings = settings
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing.")
        self._client = OpenAI(
            api_key=api_key,
            timeout=self.settings.timeout,
            max_retries=0,
        )
        return self._client

    def propose(
        self,
        *,
        query: str,
        skills: SkillRegistry,
        policy: AgentPolicy,
        task_type: str | None,
        has_evidence: bool,
        has_answer: bool,
        memory_context: Mapping[str, Any] | None,
    ) -> PlannerBackendResponse:
        schema = self._json_schema(skills)
        instructions = (
            "You propose a bounded research plan; you do not execute tools and do not answer the query. "
            "Use only the supplied skill names. Plans that generate a report must retrieve evidence, "
            "assess the exact EvidenceBundle, generate through generate_report, and then use audit_claims. "
            "search_scholarly_sources returns metadata only and cannot support an answer. "
            "claim_audit must not generate a report. Keep reasoning_summary to one concise sentence; "
            "do not reveal private chain-of-thought. Respect every supplied budget exactly or more strictly."
        )
        payload = {
            "query": query,
            "requested_task_type": task_type,
            "has_canonical_evidence": has_evidence,
            "has_answer_artifact": has_answer,
            "available_skills": list(skills.specs()),
            "policy_budget": {
                "max_steps": min(policy.max_steps, self.settings.max_steps),
                "max_tool_calls": policy.max_tool_calls,
                "max_retries": policy.max_retry,
                "max_plan_revisions": policy.max_plan_revisions,
            },
            "advisory_memory": dict(memory_context or {}),
        }
        response = self._get_client().responses.create(
            model=self.settings.model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "researchguard_structured_plan_v1",
                    "schema": schema,
                    "strict": True,
                }
            },
            store=False,
        )
        usage = getattr(response, "usage", None)
        return PlannerBackendResponse(
            payload=response.output_text,
            api_call_count=1,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )

    def _json_schema(self, skills: SkillRegistry) -> dict[str, Any]:
        budget_properties = {
            key: {"type": "integer", "minimum": minimum}
            for key, minimum in (
                ("max_steps", 1),
                ("max_tool_calls", 1),
                ("max_retries", 0),
                ("max_plan_revisions", 0),
            )
        }
        return {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": list(SUPPORTED_TASK_TYPES),
                },
                "goal": {"type": "string", "minLength": 1},
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": self.settings.max_steps,
                    "items": {
                        "type": "object",
                        "properties": {
                            "skill": {
                                "type": "string",
                                "enum": list(skills.names),
                            },
                            "purpose": {"type": "string", "minLength": 1},
                            "expected_observation": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "max_retry": {
                                "type": "integer",
                                "minimum": 0,
                            },
                        },
                        "required": [
                            "skill",
                            "purpose",
                            "expected_observation",
                            "max_retry",
                        ],
                        "additionalProperties": False,
                    },
                },
                "budget": {
                    "type": "object",
                    "properties": budget_properties,
                    "required": list(budget_properties),
                    "additionalProperties": False,
                },
                "reasoning_summary": {"type": "string", "minLength": 1},
                "planner_version": {"type": "string", "minLength": 1},
            },
            "required": [
                "task_type",
                "goal",
                "steps",
                "budget",
                "reasoning_summary",
                "planner_version",
            ],
            "additionalProperties": False,
        }


class DeterministicPlanner:
    version = "deterministic_planner_v2.0"

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        skills: SkillRegistry,
        policy: AgentPolicy,
        workflow_names: tuple[str, ...] = (),
    ):
        self.registry = registry
        self.skills = skills
        self.policy = policy
        self.legacy = BoundedPlanner(
            registry,
            max_steps=policy.max_steps,
            workflow_names=workflow_names,
        )
        self.validator = PlannerValidator(
            skills=skills,
            tools=registry,
            policy=policy,
        )

    def generate_plan(
        self,
        query: str,
        *,
        task_type: str | None = None,
        has_evidence: bool = False,
        has_answer: bool = False,
        memory_context: Mapping[str, Any] | None = None,
    ) -> PlannerOutcome:
        started = time.perf_counter()
        executable = self.legacy.create_plan(
            query,
            task_type=task_type,
            has_evidence=has_evidence,
            has_answer=has_answer,
            memory_context=memory_context,
        )
        structured = self._structured_plan(
            query,
            executable.task_type,
            has_evidence=has_evidence,
        )
        validation = self.validator.validate(
            structured,
            has_evidence=has_evidence,
            has_answer=has_answer,
        )
        if not validation.valid:
            raise PlannerError(
                "Deterministic plan failed validation: "
                + ", ".join(validation.errors)
            )
        if executable.workflow is None:
            executable = replace(
                executable,
                steps=tuple(
                    replace(
                        step,
                        max_retry=structured.steps[index].max_retry,
                    )
                    for index, step in enumerate(executable.steps)
                ),
            )
        return PlannerOutcome(
            structured_plan=structured,
            executable_plan=executable,
            mode="deterministic",
            fallback_used=False,
            fallback_reason=None,
            validation=validation,
            planner_model="deterministic_rules",
            prompt_version=self.version,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            api_call_count=0,
        )

    def _structured_plan(
        self,
        query: str,
        task_type: str,
        *,
        has_evidence: bool,
    ) -> StructuredPlan:
        templates = {
            "qa": (
                "retrieve_evidence",
                "assess_evidence",
                "generate_report",
                "audit_claims",
            ),
            "comparison": (
                "retrieve_evidence",
                "assess_evidence",
                "generate_report",
                "audit_claims",
            ),
            "literature_search": ("search_scholarly_sources",),
            "literature_review": (
                "search_scholarly_sources",
                "retrieve_evidence",
                "assess_evidence",
                "generate_report",
                "audit_claims",
            ),
            "paper_comparison": (
                "search_scholarly_sources",
                "compare_evidence",
                "assess_evidence",
                "generate_report",
                "audit_claims",
            ),
            "claim_audit": (
                "retrieve_evidence",
                "assess_evidence",
                "audit_claims",
            ),
            "audit": (
                ("audit_claims",)
                if has_evidence
                else ("retrieve_evidence", "audit_claims")
            ),
        }
        names = templates[task_type]
        steps = tuple(
            StructuredPlanStep(
                skill=name,
                purpose=self.skills.get(name).description,
                expected_observation=self.skills.get(name).output_type,
                max_retry=(
                    0
                    if name == "generate_report"
                    else min(1, self.policy.max_retry)
                ),
            )
            for name in names
        )
        return StructuredPlan(
            task_type=task_type,
            goal=" ".join(query.split()).strip(),
            steps=steps,
            budget=PlanBudget(
                max_steps=self.policy.max_steps,
                max_tool_calls=self.policy.max_tool_calls,
                max_retries=self.policy.max_retry,
                max_plan_revisions=self.policy.max_plan_revisions,
            ),
            reasoning_summary=(
                "Use the smallest deterministic evidence-first plan for the selected task."
            ),
            planner_version=self.version,
        )


class HybridPlanner:
    version = "hybrid_planner_v1.0"

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        skills: SkillRegistry,
        policy: AgentPolicy,
        deterministic: DeterministicPlanner,
        settings: HybridPlannerSettings,
        backend: Any | None = None,
        workflow_names: tuple[str, ...] = (),
    ):
        self.registry = registry
        self.skills = skills
        self.policy = policy
        self.deterministic = deterministic
        self.settings = settings
        self.backend = backend or OpenAIPlannerBackend(settings)
        self.workflow_names = tuple(workflow_names)
        self.validator = PlannerValidator(
            skills=skills,
            tools=registry,
            policy=policy,
        )

    def generate_plan(
        self,
        query: str,
        *,
        task_type: str | None = None,
        has_evidence: bool = False,
        has_answer: bool = False,
        memory_context: Mapping[str, Any] | None = None,
    ) -> PlannerOutcome:
        started = time.perf_counter()
        backend_response: PlannerBackendResponse | None = None
        if not self.settings.enabled:
            return self._fallback(
                query,
                task_type=task_type,
                has_evidence=has_evidence,
                has_answer=has_answer,
                memory_context=memory_context,
                reason="planner_disabled",
                started=started,
                backend_response=None,
            )
        try:
            response = self.backend.propose(
                query=query,
                skills=self.skills,
                policy=self.policy,
                task_type=task_type,
                has_evidence=has_evidence,
                has_answer=has_answer,
                memory_context=memory_context,
            )
            if not isinstance(response, PlannerBackendResponse):
                response = PlannerBackendResponse(payload=response)
            backend_response = response
            payload = self._parse_payload(response.payload)
            structured = StructuredPlan.from_mapping(payload)
            if task_type is not None and structured.task_type != task_type:
                raise PlanSchemaError(
                    "Planner changed the explicitly requested task_type."
                )
            validation = self.validator.validate(
                structured,
                has_evidence=has_evidence,
                has_answer=has_answer,
            )
            if not validation.valid:
                raise PlannerError(
                    "invalid_plan:" + ",".join(validation.errors)
                )
            executable = self._compile_plan(
                structured,
                has_evidence=has_evidence,
                memory_context=memory_context,
            )
            return PlannerOutcome(
                structured_plan=structured,
                executable_plan=executable,
                mode="hybrid_llm",
                fallback_used=False,
                fallback_reason=None,
                validation=validation,
                planner_model=self.settings.model,
                prompt_version=self.settings.prompt_version,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                api_call_count=response.api_call_count,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        except Exception as exc:
            if not self.settings.fallback_enabled:
                raise PlannerError(
                    f"Hybrid planner failed without fallback: {type(exc).__name__}: {exc}"
                ) from exc
            return self._fallback(
                query,
                task_type=task_type,
                has_evidence=has_evidence,
                has_answer=has_answer,
                memory_context=memory_context,
                reason=self._fallback_reason(exc),
                started=started,
                backend_response=backend_response,
            )

    @staticmethod
    def _parse_payload(value: Mapping[str, Any] | str) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        if not isinstance(value, str):
            raise PlanSchemaError("Planner output must be a JSON object or JSON string.")
        parsed = json.loads(value)
        if not isinstance(parsed, Mapping):
            raise PlanSchemaError("Planner JSON output must be an object.")
        return parsed

    def _compile_plan(
        self,
        plan: StructuredPlan,
        *,
        has_evidence: bool,
        memory_context: Mapping[str, Any] | None,
    ) -> AgentPlan:
        workflow = WORKFLOW_TASK_TYPES.get(plan.task_type)
        if workflow is not None:
            if workflow not in self.workflow_names:
                raise PlannerError(f"Workflow is not registered: {workflow}")
            steps: tuple[PlanStep, ...] = ()
        else:
            executable: list[PlanStep] = []
            for index, step in enumerate(plan.steps, start=1):
                if step.skill == "compare_evidence":
                    raise PlannerError(
                        "compare_evidence is only executable through paper_comparison."
                    )
                tool = SKILL_TO_TOOL.get(step.skill)
                if tool is None:
                    raise PlannerError(f"No execution adapter for skill: {step.skill}")
                if (
                    plan.task_type == "audit"
                    and has_evidence
                    and step.skill == "retrieve_evidence"
                ):
                    continue
                executable.append(
                    PlanStep(
                        step_id=len(executable) + 1,
                        tool=tool,
                        purpose=step.purpose,
                        optional=False,
                        max_retry=step.max_retry,
                    )
                )
            steps = tuple(executable)
        return AgentPlan(
            task_type=plan.task_type,
            steps=steps,
            created_at=utc_timestamp(),
            workflow=workflow,
            memory_context=BoundedPlanner._advisory_memory(memory_context),
        )

    def _fallback(
        self,
        query: str,
        *,
        task_type: str | None,
        has_evidence: bool,
        has_answer: bool,
        memory_context: Mapping[str, Any] | None,
        reason: str,
        started: float,
        backend_response: PlannerBackendResponse | None,
    ) -> PlannerOutcome:
        outcome = self.deterministic.generate_plan(
            query,
            task_type=task_type,
            has_evidence=has_evidence,
            has_answer=has_answer,
            memory_context=memory_context,
        )
        return PlannerOutcome(
            structured_plan=outcome.structured_plan,
            executable_plan=outcome.executable_plan,
            mode="deterministic_fallback",
            fallback_used=True,
            fallback_reason=reason,
            validation=outcome.validation,
            planner_model=self.settings.model,
            prompt_version=self.settings.prompt_version,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            api_call_count=(
                backend_response.api_call_count
                if backend_response is not None
                else 0
            ),
            input_tokens=(
                backend_response.input_tokens
                if backend_response is not None
                else 0
            ),
            output_tokens=(
                backend_response.output_tokens
                if backend_response is not None
                else 0
            ),
        )

    @staticmethod
    def _fallback_reason(exc: Exception) -> str:
        if isinstance(exc, json.JSONDecodeError):
            return "invalid_json"
        if isinstance(exc, PlanSchemaError):
            return "invalid_schema"
        if isinstance(exc, PlannerError) and str(exc).startswith("invalid_plan:"):
            return str(exc)
        if isinstance(exc, (TimeoutError,)):
            return "planner_timeout"
        text = str(exc)
        if "OPENAI_API_KEY" in text:
            return "planner_api_unavailable"
        return f"planner_failure:{type(exc).__name__}"
