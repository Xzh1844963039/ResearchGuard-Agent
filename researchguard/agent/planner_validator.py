# C:\Users\18449\Desktop\researchguard_workspace\researchguard\agent\planner_validator.py
from __future__ import annotations

from dataclasses import dataclass

from researchguard.agent.planner import SUPPORTED_TASK_TYPES
from researchguard.agent.planner_schema import StructuredPlan
from researchguard.agent.policy import AgentPolicy
from researchguard.skills import SkillRegistry
from researchguard.tools import ToolRegistry


@dataclass(frozen=True)
class PlanValidationResult:
    valid: bool
    reason: str | None
    errors: tuple[str, ...]
    validator_version: str = "1.0.0"

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "errors": list(self.errors),
            "validator_version": self.validator_version,
        }


class PlannerValidator:
    version = "1.0.0"

    def __init__(
        self,
        *,
        skills: SkillRegistry,
        tools: ToolRegistry,
        policy: AgentPolicy,
    ):
        self.skills = skills
        self.tools = tools
        self.policy = policy

    def validate(
        self,
        plan: StructuredPlan,
        *,
        has_evidence: bool = False,
        has_answer: bool = False,
    ) -> PlanValidationResult:
        errors: list[str] = []
        if plan.task_type not in SUPPORTED_TASK_TYPES:
            errors.append(f"unsupported_task_type:{plan.task_type}")
        if len(plan.steps) > self.policy.max_steps:
            errors.append("policy_max_steps_exceeded")
        if len(plan.steps) > plan.budget.max_steps:
            errors.append("plan_budget_max_steps_exceeded")
        if plan.budget.max_steps > self.policy.max_steps:
            errors.append("plan_budget_exceeds_policy_steps")
        if plan.budget.max_tool_calls > self.policy.max_tool_calls:
            errors.append("plan_budget_exceeds_policy_tool_calls")
        if plan.budget.max_retries > self.policy.max_retry:
            errors.append("plan_budget_exceeds_policy_retries")
        if plan.budget.max_plan_revisions > self.policy.max_plan_revisions:
            errors.append("plan_budget_exceeds_policy_revisions")

        skill_names = [step.skill for step in plan.steps]
        estimated_tool_calls = 0
        for index, step in enumerate(plan.steps, start=1):
            if step.skill not in self.skills.names:
                errors.append(f"unknown_skill:{index}:{step.skill}")
                continue
            spec = self.skills.get(step.skill)
            estimated_tool_calls += len(spec.allowed_tools)
            if step.expected_observation != spec.output_type:
                errors.append(f"unexpected_observation_type:{index}:{step.skill}")
            missing_tools = sorted(set(spec.allowed_tools).difference(self.tools.names))
            if missing_tools:
                errors.append(
                    f"skill_tool_unavailable:{index}:{','.join(missing_tools)}"
                )
            if step.max_retry > plan.budget.max_retries:
                errors.append(f"step_retry_exceeds_plan_budget:{index}")
            if step.max_retry > self.policy.max_retry:
                errors.append(f"step_retry_exceeds_policy:{index}")
        if estimated_tool_calls > plan.budget.max_tool_calls:
            errors.append("estimated_tool_calls_exceed_plan_budget")

        if "generate_grounded_answer" in skill_names:
            errors.append("raw_answer_generator_is_not_a_skill")
        if "audit_answer" in skill_names:
            errors.append("raw_audit_tool_is_not_a_skill")

        self._validate_evidence_first(
            plan.task_type,
            skill_names,
            has_evidence=has_evidence,
            has_answer=has_answer,
            errors=errors,
        )
        return PlanValidationResult(
            valid=not errors,
            reason=None if not errors else "invalid_plan",
            errors=tuple(dict.fromkeys(errors)),
        )

    @staticmethod
    def _validate_evidence_first(
        task_type: str,
        skills: list[str],
        *,
        has_evidence: bool,
        has_answer: bool,
        errors: list[str],
    ) -> None:
        if task_type == "literature_search":
            if skills != ["search_scholarly_sources"]:
                errors.append("literature_search_must_be_metadata_only")
            return

        generates = "generate_report" in skills
        audits = "audit_claims" in skills
        assesses = "assess_evidence" in skills
        retrieves = (
            "retrieve_evidence" in skills
            or "compare_evidence" in skills
            or has_evidence
        )
        evidence_indexes = [
            index
            for index, skill in enumerate(skills)
            if skill in {"retrieve_evidence", "compare_evidence"}
        ]
        if assesses and not has_evidence:
            assess_index = skills.index("assess_evidence")
            if not evidence_indexes:
                errors.append("evidence_gate_without_retrieval")
            elif min(evidence_indexes) > assess_index:
                errors.append("retrieval_after_evidence_gate")

        if generates:
            generate_indexes = [
                index
                for index, skill in enumerate(skills)
                if skill == "generate_report"
            ]
            audit_indexes = [
                index
                for index, skill in enumerate(skills)
                if skill == "audit_claims"
            ]
            generate_index = generate_indexes[0]
            if len(generate_indexes) != 1:
                errors.append("answering_plan_requires_single_generation")
            if len(audit_indexes) != 1:
                errors.append("answering_plan_requires_single_citation_audit")
            if not retrieves:
                errors.append("generation_without_evidence")
            if not assesses:
                errors.append("generation_without_evidence_gate")
            elif skills.index("assess_evidence") > generate_index:
                errors.append("evidence_gate_after_generation")
            if not audits:
                errors.append("generation_without_citation_audit")
            elif skills.index("audit_claims") < generate_index:
                errors.append("citation_audit_before_generation")
            elif audit_indexes[-1] != len(skills) - 1:
                errors.append("citation_audit_must_be_final_step")

        if task_type in {"qa", "comparison", "literature_review", "paper_comparison"}:
            if not generates:
                errors.append("answering_plan_missing_generate_report")
            if not audits:
                errors.append("answering_plan_missing_audit_claims")

        if task_type == "claim_audit":
            if not retrieves or not assesses or not audits:
                errors.append("claim_audit_requires_evidence_gate_and_audit")
            if generates:
                errors.append("claim_audit_must_not_generate_report")
            if assesses and audits and skills.index("audit_claims") < skills.index(
                "assess_evidence"
            ):
                errors.append("claim_audit_before_evidence_gate")

        if task_type == "audit":
            if not has_answer:
                errors.append("audit_requires_answer_artifact")
            if not retrieves:
                errors.append("audit_requires_evidence")
            if not audits:
                errors.append("audit_plan_missing_audit_claims")
