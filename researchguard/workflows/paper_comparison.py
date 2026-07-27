# C:\Users\18449\Desktop\researchguard_workspace\researchguard\workflows\paper_comparison.py
from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping

from researchguard.tools import EvidenceBundle, ScholarPaperRecord
from researchguard.workflows.base import (
    ResearchWorkflow,
    WorkflowExecutionError,
    WorkflowLimitError,
    WorkflowResult,
    utc_timestamp,
)


COMPARE_PREFIX_RE = re.compile(
    r"^\s*(?:(?:compare|contrast)\s+|"
    r"(?:what\s+is\s+)?the\s+difference\s+between\s+)",
    re.IGNORECASE,
)
COMPARE_SPLIT_RE = re.compile(
    r"\s+(?:and|with|vs\.?|versus)\s+",
    re.IGNORECASE,
)
COMPARISON_TAIL_RE = re.compile(
    r"\s+(?:for|in|on)\s+"
    r"(?:how\b|terms?\s+of\b|reducing\b|detecting\b|their\b|the\b)",
    re.IGNORECASE,
)
PAPER_PHRASE_RE = re.compile(
    r"\b(?:the\s+)?([A-Za-z][A-Za-z0-9_-]*\s+paper)\b",
    re.IGNORECASE,
)
TECHNICAL_NAME_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+|"
    r"[A-Z]{2,}[A-Za-z0-9]*|"
    r"[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]+)+)\b"
)
DIMENSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _/().-]{0,63}$")


def normalize_comparison_dimensions(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("method", "dataset", "metric", "limitation")
    if not isinstance(value, (list, tuple)):
        raise ValueError("comparison_dimensions must be a list.")
    dimensions: list[str] = []
    for item in value:
        dimension = " ".join(str(item).split()).strip()
        if not dimension:
            continue
        if not DIMENSION_RE.fullmatch(dimension):
            raise ValueError(
                "comparison_dimensions contains an invalid dimension label."
            )
        if dimension.casefold() not in {item.casefold() for item in dimensions}:
            dimensions.append(dimension)
    if not dimensions or len(dimensions) > 8:
        raise ValueError(
            "comparison_dimensions must contain between 1 and 8 unique items."
        )
    return tuple(dimensions)


def extract_paper_specs(value: Any, query: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if value is not None:
        if not isinstance(value, (list, tuple)):
            raise ValueError("papers must be a list.")
        for item in value[:2]:
            if isinstance(item, Mapping):
                name = " ".join(str(item.get("name", "")).split()).strip()
                doc_id = " ".join(
                    str(item.get("doc_id") or "").split()
                ).strip() or None
            else:
                name = " ".join(str(item).split()).strip()
                doc_id = None
            if name:
                specs.append(
                    {
                        "identity_id": f"paper-{len(specs) + 1}",
                        "name": name,
                        "doc_id": doc_id,
                    }
                )
        return specs

    remainder = COMPARE_PREFIX_RE.sub("", query).strip().rstrip("?.")
    parts = [
        part.strip(" ,")
        for part in COMPARE_SPLIT_RE.split(remainder)
        if part.strip(" ,")
    ]
    if len(parts) > 1:
        parts[-1] = COMPARISON_TAIL_RE.split(
            parts[-1],
            maxsplit=1,
        )[0].strip()
    for part in parts[:2]:
        paper_phrases = PAPER_PHRASE_RE.findall(part)
        technical_names = TECHNICAL_NAME_RE.findall(part)
        short_phrase = len(part.split()) <= 4
        name = (
            part
            if short_phrase
            else paper_phrases[-1]
            if paper_phrases
            else technical_names[-1]
            if technical_names
            else part
        )
        specs.append(
            {
                "identity_id": f"paper-{len(specs) + 1}",
                "name": " ".join(name.split()).strip(),
                "doc_id": None,
            }
        )
    return specs


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
    skill_template = (
        "search_papers",
        "retrieve_evidence",
        "compare_evidence",
        "generate_report",
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
            dimensions = normalize_comparison_dimensions(
                workflow_input.get(
                    "comparison_dimensions",
                    workflow_input.get("dimensions"),
                )
            )
            paper_specs = extract_paper_specs(
                workflow_input.get("papers"),
                state.query,
            )
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

            for spec_index, spec in enumerate(paper_specs):
                if not spec.get("doc_id"):
                    probe_result = self._invoke(
                        state,
                        trace,
                        started,
                        "retrieve_evidence",
                        query=spec["name"],
                        top_k=10,
                        rewrite=False,
                        multi_query=False,
                    )
                    if probe_result.status == "failed":
                        return self._failure(
                            probe_result.reason or "paper_identity_resolution_failed",
                            paper_specs,
                            dimensions,
                            evidence_table,
                            trace,
                            started_at,
                            started,
                        )
                    probe_bundle = self._bundle_from_result(
                        probe_result,
                        query=spec["name"],
                    )
                    spec["doc_id"] = self._resolve_doc_id(
                        spec["name"],
                        probe_bundle,
                        excluded_doc_ids=tuple(
                            str(item.get("doc_id", ""))
                            for item in paper_specs[:spec_index]
                            if item.get("doc_id")
                            and not self._identity_tokens(item["name"]).intersection(
                                self._identity_tokens(spec["name"])
                            )
                        ),
                    )
                    if not spec["doc_id"]:
                        raise WorkflowExecutionError(
                            "Paper identity could not be resolved to a corpus document."
                        )
                kwargs: dict[str, Any] = {
                    "query": f"{spec['name']}. {state.query}",
                    "filters": {"doc_ids": [spec["doc_id"]]},
                }
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
                paper_bundle = self._bundle_from_result(
                    retrieval_result,
                    query=str(kwargs["query"]),
                )
                paper_evidence = [
                    record.to_dict() for record in paper_bundle.evidence_records
                ]
                if spec.get("doc_id") and any(
                    item["doc_id"] != spec["doc_id"] for item in paper_evidence
                ):
                    raise WorkflowExecutionError(
                        "Retrieval returned evidence outside the paper identity filter."
                    )
                evidence_table.append(
                    {
                        "identity_id": spec["identity_id"],
                        "paper": spec["name"],
                        "doc_id_filter": spec.get("doc_id"),
                        "evidence": paper_evidence,
                    }
                )
                combined_evidence.extend(paper_evidence)

            chunk_owners: dict[str, str] = {}
            for row in evidence_table:
                separated: list[dict[str, Any]] = []
                for item in row["evidence"]:
                    chunk_id = str(item["chunk_id"])
                    if chunk_id in chunk_owners:
                        continue
                    chunk_owners[chunk_id] = str(row["identity_id"])
                    separated.append(item)
                row["evidence"] = separated
                if not separated:
                    raise WorkflowExecutionError(
                        "A paper identity has no independent comparison evidence."
                    )
            combined_evidence = [
                item
                for row in evidence_table
                for item in row["evidence"]
            ]
            combined_evidence = self._deduplicate_evidence(combined_evidence)
            generation_query = (
                f"{state.query} Compare only the evidence-supported dimensions: "
                + ", ".join(dimensions)
                + "."
            )
            bundle = EvidenceBundle.create(
                query=generation_query,
                evidence=combined_evidence,
                provenance={
                    "source": "paper_comparison_workflow",
                    "document_queries": [
                        row["paper"] for row in evidence_table
                    ],
                },
            )
            state.evidence = copy.deepcopy(combined_evidence)
            state.evidence_bundle = bundle.to_dict()
            assessment_result = self._invoke(
                state,
                trace,
                started,
                "assess_evidence",
                evidence_bundle=bundle.to_dict(),
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
            gate = self._gate_from_result(assessment_result, bundle=bundle)
            state.gate_decision = gate.to_dict()
            if (
                assessment_result.status == "rejected"
                or gate.status != "strong"
                or not gate.answerable
            ):
                return self._rejected(
                    paper_specs, dimensions, evidence_table, trace, started_at, started
                )

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
        return normalize_comparison_dimensions(value)

    @staticmethod
    def _paper_specs(value: Any, query: str) -> list[dict[str, Any]]:
        return extract_paper_specs(value, query)

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

    @staticmethod
    def _resolve_doc_id(
        name: str,
        bundle: EvidenceBundle,
        *,
        excluded_doc_ids: tuple[str, ...] = (),
    ) -> str | None:
        name_tokens = PaperComparisonWorkflow._identity_tokens(name)
        scores: dict[str, float] = {}
        for index, record in enumerate(bundle.evidence_records, start=1):
            haystack = " ".join(
                (
                    record.doc_id,
                    record.source,
                    str(record.provenance.get("title", "")),
                )
            )
            tokens = {
                token.casefold()
                for token in re.findall(r"[A-Za-z0-9]+", haystack)
                if len(token) > 1
            }
            scores[record.doc_id] = scores.get(record.doc_id, 0.0) + (
                1.0 / index + 2.0 * len(name_tokens.intersection(tokens))
            )
        if not scores:
            return None
        eligible = {
            doc_id: score
            for doc_id, score in scores.items()
            if doc_id not in set(excluded_doc_ids)
        } or scores
        return min(eligible, key=lambda item: (-eligible[item], item))

    @staticmethod
    def _identity_tokens(value: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9]+", value)
            if len(token) > 1
            and token.casefold()
            not in {"paper", "method", "model", "original", "standard"}
        }

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
