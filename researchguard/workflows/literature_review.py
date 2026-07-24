# C:\Users\18449\Desktop\researchguard_workspace\researchguard\workflows\literature_review.py
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Mapping

from researchguard.tools import ScholarPaperRecord
from researchguard.workflows.base import (
    ResearchWorkflow,
    WorkflowExecutionError,
    WorkflowLimitError,
    WorkflowResult,
    utc_timestamp,
)


@dataclass(frozen=True)
class LiteratureReviewResult:
    topic: str
    papers: tuple[Mapping[str, Any], ...]
    evidence: tuple[Mapping[str, Any], ...]
    summary: str | None
    citations: tuple[Mapping[str, Any], ...]
    audit_result: Mapping[str, Any] | None
    trace: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_type": "literature_review",
            "topic": self.topic,
            "papers": copy.deepcopy(list(self.papers)),
            "evidence": copy.deepcopy(list(self.evidence)),
            "summary": self.summary,
            "citations": copy.deepcopy(list(self.citations)),
            "audit_result": copy.deepcopy(dict(self.audit_result)) if self.audit_result else None,
            "trace": copy.deepcopy(list(self.trace)),
        }


class LiteratureReviewWorkflow(ResearchWorkflow):
    workflow_name = "literature_review"
    version = "1.0.0"
    description = "Discover candidate papers and produce a corpus-grounded audited review."
    required_tools = (
        "search_scholarly_sources",
        "retrieve_evidence",
        "assess_evidence",
        "generate_grounded_answer",
        "audit_answer",
    )
    input_schema = {
        "topic": "optional topic override; defaults to state.query",
        "sources": "optional scholarly provider list",
        "candidate_limit": "optional integer, default 5",
    }
    output_schema = {
        "type": "LiteratureReviewResult",
        "fields": ["topic", "papers", "evidence", "summary", "citations", "audit_result", "trace"],
    }

    def run(self, state: Any) -> WorkflowResult:
        started = time.perf_counter()
        started_at = utc_timestamp()
        trace: list[dict[str, Any]] = []
        workflow_input = state.workflow_input if isinstance(state.workflow_input, Mapping) else {}
        topic = " ".join(str(workflow_input.get("topic") or state.query).split()).strip()
        sources = workflow_input.get("sources")
        candidate_limit = 5
        papers: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        try:
            if isinstance(workflow_input.get("candidate_limit", 5), bool):
                raise ValueError("candidate_limit must be an integer.")
            candidate_limit = int(workflow_input.get("candidate_limit", 5))
            if not 1 <= candidate_limit <= 50:
                raise ValueError("candidate_limit must be between 1 and 50.")
            search_kwargs: dict[str, Any] = {"query": topic, "limit": candidate_limit}
            if sources is not None:
                search_kwargs["sources"] = sources
            search_result = self._invoke(
                state,
                trace,
                started,
                "search_scholarly_sources",
                **search_kwargs,
            )
            if search_result.status == "failed":
                return self._tool_failure(search_result, topic, papers, evidence, trace, started_at, started)
            raw_papers = search_result.data.get("candidate_papers", [])
            if not isinstance(raw_papers, list):
                raise WorkflowExecutionError("Scholarly Search returned invalid candidate_papers.")
            papers = [
                ScholarPaperRecord.from_dict(item).to_dict()
                for item in raw_papers
            ]
            state.candidate_papers = copy.deepcopy(papers)

            retrieval_result = self._invoke(
                state,
                trace,
                started,
                "retrieve_evidence",
                query=topic,
            )
            if retrieval_result.status == "failed":
                return self._tool_failure(
                    retrieval_result, topic, papers, evidence, trace, started_at, started
                )
            bundle = self._bundle_from_result(retrieval_result, query=topic)
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
            if assessment_result.status == "failed":
                return self._tool_failure(
                    assessment_result, topic, papers, evidence, trace, started_at, started
                )
            gate = self._gate_from_result(assessment_result, bundle=bundle)
            state.gate_decision = gate.to_dict()
            if (
                assessment_result.status == "rejected"
                or gate.status != "strong"
                or not gate.answerable
            ):
                return self._rejected(topic, papers, evidence, trace, started_at, started)

            answer_result = self._invoke(
                state,
                trace,
                started,
                "generate_grounded_answer",
                evidence_bundle=bundle.to_dict(),
                gate_decision=gate.to_dict(),
            )
            if answer_result.status != "success":
                if answer_result.status == "rejected":
                    return self._rejected(topic, papers, evidence, trace, started_at, started)
                return self._tool_failure(
                    answer_result, topic, papers, evidence, trace, started_at, started
                )
            answer = self._answer_from_result(answer_result, bundle=bundle)
            state.answer = copy.deepcopy(answer)

            audit_result = self._invoke(
                state,
                trace,
                started,
                "audit_answer",
                answer=answer,
                evidence_bundle=bundle.to_dict(),
            )
            audit = audit_result.data.get("audit")
            audit = dict(audit) if isinstance(audit, Mapping) else None
            state.audit_result = copy.deepcopy(audit)
            output = LiteratureReviewResult(
                topic=topic,
                papers=tuple(papers),
                evidence=tuple(evidence),
                summary=str(answer.get("answer", "")),
                citations=tuple(answer.get("citations", [])),
                audit_result=audit,
                trace=tuple(trace),
            ).to_dict()
            if audit_result.status == "success":
                return self._finish(
                    status="success",
                    message="Literature review completed with grounded evidence.",
                    reason=None,
                    output=output,
                    trace=trace,
                    started_at=started_at,
                    started=started,
                )
            status = "rejected" if audit_result.status == "rejected" else "failed"
            return self._finish(
                status=status,
                message="Literature review was not released.",
                reason="citation_audit_rejected" if status == "rejected" else "tool_error",
                output=output,
                trace=trace,
                started_at=started_at,
                started=started,
            )
        except (ValueError, TypeError, WorkflowExecutionError, WorkflowLimitError) as exc:
            return self._finish(
                status="failed",
                message="Literature review workflow failed.",
                reason=str(exc),
                output=LiteratureReviewResult(
                    topic=topic,
                    papers=tuple(papers),
                    evidence=tuple(evidence),
                    summary=None,
                    citations=(),
                    audit_result=None,
                    trace=tuple(trace),
                ).to_dict(),
                trace=trace,
                started_at=started_at,
                started=started,
            )

    def _rejected(
        self,
        topic: str,
        papers: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        trace: list[dict[str, Any]],
        started_at: str,
        started: float,
    ) -> WorkflowResult:
        return self._finish(
            status="rejected",
            message="Current corpus does not contain sufficient evidence for a review.",
            reason="insufficient_evidence",
            output=LiteratureReviewResult(
                topic=topic,
                papers=tuple(papers),
                evidence=tuple(evidence),
                summary=None,
                citations=(),
                audit_result=None,
                trace=tuple(trace),
            ).to_dict(),
            trace=trace,
            started_at=started_at,
            started=started,
        )

    def _tool_failure(
        self,
        result: Any,
        topic: str,
        papers: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        trace: list[dict[str, Any]],
        started_at: str,
        started: float,
    ) -> WorkflowResult:
        return self._finish(
            status="failed",
            message="Literature review tool execution failed.",
            reason=result.reason or "tool_error",
            output=LiteratureReviewResult(
                topic=topic,
                papers=tuple(papers),
                evidence=tuple(evidence),
                summary=None,
                citations=(),
                audit_result=None,
                trace=tuple(trace),
            ).to_dict(),
            trace=trace,
            started_at=started_at,
            started=started,
        )
