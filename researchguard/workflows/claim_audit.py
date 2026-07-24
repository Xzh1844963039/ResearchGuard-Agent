# C:\Users\18449\Desktop\researchguard_workspace\researchguard\workflows\claim_audit.py
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Mapping

from researchguard.workflows.base import (
    ResearchWorkflow,
    WorkflowExecutionError,
    WorkflowLimitError,
    WorkflowResult,
    utc_timestamp,
)


@dataclass(frozen=True)
class ClaimAuditResult:
    claim: str
    evidence: tuple[Mapping[str, Any], ...]
    support_level: str
    citations: tuple[Mapping[str, Any], ...]
    audit_result: Mapping[str, Any] | None
    trace: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_type": "claim_audit",
            "claim": self.claim,
            "evidence": copy.deepcopy(list(self.evidence)),
            "support_level": self.support_level,
            "citations": copy.deepcopy(list(self.citations)),
            "audit_result": copy.deepcopy(dict(self.audit_result)) if self.audit_result else None,
            "trace": copy.deepcopy(list(self.trace)),
        }


class ClaimAuditWorkflow(ResearchWorkflow):
    workflow_name = "claim_audit"
    version = "1.0.0"
    description = "Retrieve evidence and verify a user-supplied claim without generating a new answer."
    required_tools = ("retrieve_evidence", "assess_evidence", "audit_answer")
    input_schema = {"claim": "optional claim override; defaults to state.query"}
    output_schema = {
        "type": "ClaimAuditResult",
        "fields": ["claim", "evidence", "support_level", "citations", "audit_result", "trace"],
    }

    def run(self, state: Any) -> WorkflowResult:
        started = time.perf_counter()
        started_at = utc_timestamp()
        trace: list[dict[str, Any]] = []
        workflow_input = state.workflow_input if isinstance(state.workflow_input, Mapping) else {}
        claim = " ".join(str(workflow_input.get("claim") or state.query).split()).strip()
        evidence: list[dict[str, Any]] = []
        support_level = "unknown"
        citations: list[dict[str, Any]] = []
        try:
            retrieval_result = self._invoke(
                state,
                trace,
                started,
                "retrieve_evidence",
                query=claim,
            )
            if retrieval_result.status == "failed":
                return self._failure(
                    retrieval_result.reason or "tool_error",
                    claim,
                    evidence,
                    support_level,
                    citations,
                    trace,
                    started_at,
                    started,
                )
            bundle = self._bundle_from_result(retrieval_result, query=claim)
            evidence = [record.to_dict() for record in bundle.evidence_records]
            state.evidence = copy.deepcopy(evidence)
            state.evidence_bundle = bundle.to_dict()
            assessment_result = self._invoke(
                state,
                trace,
                started,
                "assess_evidence",
                evidence_bundle=bundle.to_dict(),
            )
            assessment = assessment_result.data.get("assessment")
            assessment = dict(assessment) if isinstance(assessment, Mapping) else {}
            gate = self._gate_from_result(assessment_result, bundle=bundle)
            state.gate_decision = gate.to_dict()
            support_level = gate.status
            if assessment_result.status == "failed":
                return self._failure(
                    assessment_result.reason or "tool_error",
                    claim,
                    evidence,
                    support_level,
                    citations,
                    trace,
                    started_at,
                    started,
                )
            if (
                assessment_result.status == "rejected"
                or gate.status != "strong"
                or not gate.answerable
            ):
                return self._finish(
                    status="rejected",
                    message="Claim is not sufficiently supported by the current corpus.",
                    reason="insufficient_evidence",
                    output=ClaimAuditResult(
                        claim=claim,
                        evidence=tuple(evidence),
                        support_level=support_level,
                        citations=(),
                        audit_result=None,
                        trace=tuple(trace),
                    ).to_dict(),
                    trace=trace,
                    started_at=started_at,
                    started=started,
                )

            supporting_ids = list(gate.supporting_chunk_ids)
            evidence_by_id = {str(item["chunk_id"]): item for item in evidence}
            selected_ids = [item for item in supporting_ids if item in evidence_by_id]
            if not selected_ids:
                raise WorkflowExecutionError(
                    "Strong evidence assessment did not provide supporting chunk IDs."
                )
            selected_evidence = [evidence_by_id[chunk_id] for chunk_id in selected_ids]
            citations = [
                {
                    "chunk_id": item["chunk_id"],
                    "doc_id": item["doc_id"],
                    "section": item["section"],
                    "page": item.get("page"),
                }
                for item in selected_evidence
            ]
            claim_artifact = {
                "answer": claim,
                "citations": citations,
                "confidence": float(assessment.get("confidence", 0.0)),
                "refused": False,
                "refusal_reason": None,
                "evidence_chunk_ids": selected_ids,
                "model": "user_supplied_claim",
                "prompt_version": "claim_audit_workflow_v1",
                "config_version": "claim_audit_workflow_v1.0",
                "timestamp": utc_timestamp(),
                "cache_hit": False,
                "fallback_used": False,
                "fallback_reason": None,
                "api_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0.0,
            }
            state.answer = copy.deepcopy(claim_artifact)
            audit_result = self._invoke(
                state,
                trace,
                started,
                "audit_answer",
                answer=claim_artifact,
                evidence_bundle=bundle.to_dict(),
            )
            audit = audit_result.data.get("audit")
            audit = dict(audit) if isinstance(audit, Mapping) else None
            state.audit_result = copy.deepcopy(audit)
            output = ClaimAuditResult(
                claim=claim,
                evidence=tuple(selected_evidence),
                support_level=support_level,
                citations=tuple(citations),
                audit_result=audit,
                trace=tuple(trace),
            ).to_dict()
            status = "success" if audit_result.status == "success" else (
                "rejected" if audit_result.status == "rejected" else "failed"
            )
            return self._finish(
                status=status,
                message="Claim audit completed." if status == "success" else "Claim was not verified.",
                reason=None if status == "success" else (
                    "claim_not_grounded" if status == "rejected" else "tool_error"
                ),
                output=output,
                trace=trace,
                started_at=started_at,
                started=started,
            )
        except (ValueError, TypeError, WorkflowExecutionError, WorkflowLimitError) as exc:
            return self._failure(
                str(exc),
                claim,
                evidence,
                support_level,
                citations,
                trace,
                started_at,
                started,
            )

    def _failure(
        self,
        reason: str,
        claim: str,
        evidence: list[dict[str, Any]],
        support_level: str,
        citations: list[dict[str, Any]],
        trace: list[dict[str, Any]],
        started_at: str,
        started: float,
    ) -> WorkflowResult:
        return self._finish(
            status="failed",
            message="Claim audit workflow failed.",
            reason=reason,
            output=ClaimAuditResult(
                claim=claim,
                evidence=tuple(evidence),
                support_level=support_level,
                citations=tuple(citations),
                audit_result=None,
                trace=tuple(trace),
            ).to_dict(),
            trace=trace,
            started_at=started_at,
            started=started,
        )
