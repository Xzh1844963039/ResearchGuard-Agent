# C:\Users\18449\Desktop\researchguard_workspace\researchguard\workflows\paper_comparison.py
from __future__ import annotations

import copy
import re
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


COMPARE_PREFIX_RE = re.compile(r"^\s*(compare|contrast)\s+", re.IGNORECASE)
COMPARE_SPLIT_RE = re.compile(r"\s+(?:and|vs\.?|versus)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class ComparisonResult:
    papers: tuple[Mapping[str, Any], ...]
    comparison_dimensions: tuple[str, ...]
    evidence_table: tuple[Mapping[str, Any], ...]
    summary: str | None
    citations: tuple[Mapping[str, Any], ...]
    audit_result: Mapping[str, Any] | None
    trace: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_type": "paper_comparison",
            "papers": copy.deepcopy(list(self.papers)),
            "comparison_dimensions": list(self.comparison_dimensions),
            "evidence_table": copy.deepcopy(list(self.evidence_table)),
            "summary": self.summary,
            "citations": copy.deepcopy(list(self.citations)),
            "audit_result": copy.deepcopy(dict(self.audit_result)) if self.audit_result else None,
            "trace": copy.deepcopy(list(self.trace)),
        }


class PaperComparisonWorkflow(ResearchWorkflow):
    workflow_name = "paper_comparison"
    version = "1.0.0"
    description = "Compare papers using separated retrieval traces and a grounded audited answer."
    required_tools = (
        "search_scholarly_sources",
        "retrieve_evidence",
        "assess_evidence",
        "generate_grounded_answer",
        "audit_answer",
    )
    input_schema = {
        "papers": "optional two-item list of names or {name, doc_id} mappings",
        "comparison_dimensions": "optional list; defaults to method/dataset/metric/limitation",
    }
    output_schema = {
        "type": "ComparisonResult",
        "fields": ["papers", "comparison_dimensions", "evidence_table", "summary", "citations"],
    }

    def run(self, state: Any) -> WorkflowResult:
        started = time.perf_counter()
        started_at = utc_timestamp()
        trace: list[dict[str, Any]] = []
        workflow_input = state.workflow_input if isinstance(state.workflow_input, Mapping) else {}
        dimensions = ("method", "dataset", "metric", "limitation")
        paper_specs: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        evidence_table: list[dict[str, Any]] = []
        combined_evidence: list[dict[str, Any]] = []
        try:
            dimensions = self._dimensions(workflow_input.get("comparison_dimensions"))
            paper_specs = self._paper_specs(workflow_input.get("papers"), state.query)
            search_result = self._invoke(
                state,
                trace,
                started,
                "search_scholarly_sources",
                query=state.query,
                limit=5,
            )
            if search_result.status == "failed":
                return self._failure(
                    search_result.reason or "tool_error",
                    paper_specs,
                    dimensions,
                    evidence_table,
                    trace,
                    started_at,
                    started,
                )
            raw_candidates = search_result.data.get("candidate_papers", [])
            if not isinstance(raw_candidates, list):
                raise WorkflowExecutionError("Scholarly Search returned invalid candidate_papers.")
            candidates = [ScholarPaperRecord.from_dict(item).to_dict() for item in raw_candidates]
            state.candidate_papers = copy.deepcopy(candidates)
            if len(paper_specs) < 2:
                for candidate in candidates:
                    name = str(candidate.get("title", "")).strip()
                    if name and all(spec["name"].casefold() != name.casefold() for spec in paper_specs):
                        paper_specs.append({"name": name, "doc_id": None})
                    if len(paper_specs) == 2:
                        break
            if len(paper_specs) != 2:
                raise WorkflowExecutionError("Paper comparison requires exactly two identifiable papers.")

            for spec in paper_specs:
                kwargs: dict[str, Any] = {
                    "query": f"{spec['name']}. {state.query}",
                }
                if spec.get("doc_id"):
                    kwargs["filters"] = {"doc_ids": [spec["doc_id"]]}
                retrieval_result = self._invoke(
                    state,
                    trace,
                    started,
                    "retrieve_evidence",
                    **kwargs,
                )
                if retrieval_result.status == "failed":
                    return self._failure(
                        retrieval_result.reason or "tool_error",
                        paper_specs,
                        dimensions,
                        evidence_table,
                        trace,
                        started_at,
                        started,
                    )
                paper_evidence = self._evidence_from_result(retrieval_result)
                evidence_table.append(
                    {
                        "paper": spec["name"],
                        "doc_id_filter": spec.get("doc_id"),
                        "evidence": paper_evidence,
                    }
                )
                combined_evidence.extend(paper_evidence)

            combined_evidence = self._deduplicate_evidence(combined_evidence)
            state.evidence = copy.deepcopy(combined_evidence)
            assessment_result = self._invoke(
                state,
                trace,
                started,
                "assess_evidence",
                query=state.query,
                evidence=combined_evidence,
            )
            if assessment_result.status == "failed":
                return self._failure(
                    assessment_result.reason or "tool_error",
                    paper_specs,
                    dimensions,
                    evidence_table,
                    trace,
                    started_at,
                    started,
                )
            assessment = assessment_result.data.get("assessment")
            assessment = dict(assessment) if isinstance(assessment, Mapping) else {}
            if (
                assessment_result.status == "rejected"
                or str(assessment.get("support_level", "")).casefold() != "strong"
                or not bool(assessment.get("answerable", False))
            ):
                return self._rejected(
                    paper_specs, dimensions, evidence_table, trace, started_at, started
                )

            generation_query = (
                f"{state.query} Compare only the evidence-supported dimensions: "
                + ", ".join(dimensions)
                + "."
            )
            answer_result = self._invoke(
                state,
                trace,
                started,
                "generate_grounded_answer",
                query=generation_query,
            )
            if answer_result.status != "success":
                if answer_result.status == "rejected":
                    return self._rejected(
                        paper_specs, dimensions, evidence_table, trace, started_at, started
                    )
                return self._failure(
                    answer_result.reason or "tool_error",
                    paper_specs,
                    dimensions,
                    evidence_table,
                    trace,
                    started_at,
                    started,
                )
            answer, grounded_evidence, pipeline_audit = self._guarded_artifacts(answer_result)
            state.answer = copy.deepcopy(answer)
            state.evidence = copy.deepcopy(grounded_evidence)
            state.audit_result = copy.deepcopy(pipeline_audit)
            audit_result = self._invoke(
                state,
                trace,
                started,
                "audit_answer",
                answer=answer,
                evidence=grounded_evidence,
            )
            audit = audit_result.data.get("audit")
            audit = dict(audit) if isinstance(audit, Mapping) else pipeline_audit
            state.audit_result = copy.deepcopy(audit)
            output = ComparisonResult(
                papers=tuple(
                    {
                        **spec,
                        "candidate_matches": [
                            candidate
                            for candidate in candidates
                            if spec["name"].casefold() in str(candidate.get("title", "")).casefold()
                            or str(candidate.get("title", "")).casefold() in spec["name"].casefold()
                        ],
                    }
                    for spec in paper_specs
                ),
                comparison_dimensions=dimensions,
                evidence_table=tuple(evidence_table),
                summary=str(answer.get("answer", "")),
                citations=tuple(answer.get("citations", [])),
                audit_result=audit,
                trace=tuple(trace),
            ).to_dict()
            status = "success" if audit_result.status == "success" else (
                "rejected" if audit_result.status == "rejected" else "failed"
            )
            return self._finish(
                status=status,
                message="Paper comparison completed." if status == "success" else "Comparison not released.",
                reason=None if status == "success" else (
                    "citation_audit_rejected" if status == "rejected" else "tool_error"
                ),
                output=output,
                trace=trace,
                started_at=started_at,
                started=started,
            )
        except (ValueError, TypeError, WorkflowExecutionError, WorkflowLimitError) as exc:
            return self._failure(
                str(exc),
                paper_specs,
                dimensions,
                evidence_table,
                trace,
                started_at,
                started,
            )

    @staticmethod
    def _dimensions(value: Any) -> tuple[str, ...]:
        if value is None:
            return ("method", "dataset", "metric", "limitation")
        if not isinstance(value, (list, tuple)):
            raise ValueError("comparison_dimensions must be a list.")
        dimensions = tuple(" ".join(str(item).split()).strip() for item in value if str(item).strip())
        if not dimensions or len(dimensions) > 8:
            raise ValueError("comparison_dimensions must contain between 1 and 8 items.")
        return dimensions

    @staticmethod
    def _paper_specs(value: Any, query: str) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        if value is not None:
            if not isinstance(value, (list, tuple)):
                raise ValueError("papers must be a list.")
            for item in value[:2]:
                if isinstance(item, Mapping):
                    name = " ".join(str(item.get("name", "")).split()).strip()
                    doc_id = " ".join(str(item.get("doc_id", "")).split()).strip() or None
                else:
                    name = " ".join(str(item).split()).strip()
                    doc_id = None
                if name:
                    specs.append({"name": name, "doc_id": doc_id})
            return specs
        remainder = COMPARE_PREFIX_RE.sub("", query).strip().rstrip("?.")
        parts = [part.strip(" ,") for part in COMPARE_SPLIT_RE.split(remainder) if part.strip(" ,")]
        return [{"name": part, "doc_id": None} for part in parts[:2]]

    @staticmethod
    def _deduplicate_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in evidence:
            chunk_id = str(item.get("chunk_id", ""))
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                result.append(item)
        return result

    def _rejected(
        self,
        papers: list[dict[str, Any]],
        dimensions: tuple[str, ...],
        evidence_table: list[dict[str, Any]],
        trace: list[dict[str, Any]],
        started_at: str,
        started: float,
    ) -> WorkflowResult:
        return self._finish(
            status="rejected",
            message="Current corpus does not contain sufficient comparison evidence.",
            reason="insufficient_evidence",
            output=ComparisonResult(
                papers=tuple(papers),
                comparison_dimensions=dimensions,
                evidence_table=tuple(evidence_table),
                summary=None,
                citations=(),
                audit_result=None,
                trace=tuple(trace),
            ).to_dict(),
            trace=trace,
            started_at=started_at,
            started=started,
        )

    def _failure(
        self,
        reason: str,
        papers: list[dict[str, Any]],
        dimensions: tuple[str, ...],
        evidence_table: list[dict[str, Any]],
        trace: list[dict[str, Any]],
        started_at: str,
        started: float,
    ) -> WorkflowResult:
        return self._finish(
            status="failed",
            message="Paper comparison workflow failed.",
            reason=reason,
            output=ComparisonResult(
                papers=tuple(papers),
                comparison_dimensions=dimensions,
                evidence_table=tuple(evidence_table),
                summary=None,
                citations=(),
                audit_result=None,
                trace=tuple(trace),
            ).to_dict(),
            trace=trace,
            started_at=started_at,
            started=started,
        )
