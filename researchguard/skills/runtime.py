# C:\Users\18449\Desktop\researchguard_workspace\researchguard\skills\runtime.py
from __future__ import annotations

import copy
import re
import time
from collections import defaultdict
from typing import Any, Mapping

from researchguard.skills.base import (
    ResearchSkill,
    SkillContext,
    SkillResult,
    utc_timestamp,
)
from researchguard.tools import (
    EvidenceBundle,
    EvidenceRecord,
    GateDecision,
    ScholarPaperRecord,
    ToolResult,
)


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _object_schema(
    title: str,
    *,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "type": "object",
        "title": title,
        "required": list(required),
        "additionalProperties": True,
    }


def _tool_failure(
    skill: ResearchSkill,
    started: float,
    result: ToolResult,
) -> SkillResult:
    return skill._result(
        started,
        status="failed" if result.status == "failed" else "rejected",
        reason=result.reason or f"{result.tool_name}_{result.status}",
        output={
            "tool_name": result.tool_name,
            "tool_status": result.status,
            "tool_trace_id": result.trace_id,
            "retryable": bool(result.error and result.error.retryable),
            "error_type": result.error.category if result.error else None,
            "error_code": result.error.code if result.error else None,
        },
    )


def _bundle_from_state(context: SkillContext) -> EvidenceBundle:
    bundle = getattr(context.state, "evidence_bundle", None)
    if not isinstance(bundle, Mapping):
        raise ValueError("Skill requires a canonical EvidenceBundle.")
    return EvidenceBundle.from_mapping(bundle)


def _gate_from_tool(
    result: ToolResult,
    bundle: EvidenceBundle,
) -> GateDecision:
    raw_gate = result.data.get("gate_decision")
    if isinstance(raw_gate, Mapping):
        gate = GateDecision.from_mapping(raw_gate)
    else:
        assessment = result.data.get("assessment")
        if not isinstance(assessment, Mapping):
            raise ValueError("Evidence Tool returned no GateDecision.")
        gate = GateDecision.from_assessment(
            evidence_bundle_id=bundle.bundle_id,
            assessment=assessment,
        )
    if gate.evidence_bundle_id != bundle.bundle_id:
        raise ValueError("GateDecision belongs to another EvidenceBundle.")
    return gate


def _ensure_gate(
    context: SkillContext,
    bundle: EvidenceBundle,
) -> tuple[GateDecision, ToolResult | None]:
    raw_gate = getattr(context.state, "gate_decision", None)
    if isinstance(raw_gate, Mapping):
        gate = GateDecision.from_mapping(raw_gate)
        if gate.evidence_bundle_id == bundle.bundle_id:
            return gate, None
    result = context.invoke_tool(
        "assess_evidence",
        {"evidence_bundle": bundle.to_dict()},
    )
    if not isinstance(result, ToolResult):
        raise TypeError("Skill tool adapter must return ToolResult.")
    return _gate_from_tool(result, bundle), result


class SearchPapersSkill(ResearchSkill):
    version = "1.0.0"
    description = "Discover metadata-only scholarly candidate papers."
    input_schema = _object_schema("SearchPapersInput", required=("query",))
    output_schema = _object_schema(
        "CandidatePaperSet",
        required=("candidate_papers",),
    )
    required_tools = ("search_scholarly_sources",)
    risk_level = "low"

    def __init__(self, name: str = "search_papers"):
        self.name = name

    def run(self, context: SkillContext) -> SkillResult:
        started = time.perf_counter()
        tool_input = {
            "query": str(context.inputs.get("query") or context.query),
            "limit": int(context.inputs.get("limit", 5)),
        }
        if context.inputs.get("sources") is not None:
            tool_input["sources"] = context.inputs["sources"]
        result = context.invoke_tool("search_scholarly_sources", tool_input)
        if result.status != "success":
            return _tool_failure(self, started, result)
        raw = result.data.get("candidate_papers", [])
        if not isinstance(raw, list):
            raise TypeError("Scholarly Search returned invalid candidate_papers.")
        papers = [
            ScholarPaperRecord.from_dict(item).to_dict()
            for item in raw
        ]
        return self._result(
            started,
            status="success",
            output={
                "candidate_papers": papers,
                "metadata_only": True,
                "evidence_eligible": False,
            },
        )


class RetrieveEvidenceSkill(ResearchSkill):
    name = "retrieve_evidence"
    version = "1.0.0"
    description = "Retrieve canonical local-corpus evidence for the current task."
    input_schema = _object_schema("RetrieveEvidenceInput", required=("query",))
    output_schema = _object_schema(
        "EvidenceBundle",
        required=("evidence", "evidence_count"),
    )
    required_tools = ("retrieve_evidence",)
    risk_level = "low"

    def run(self, context: SkillContext) -> SkillResult:
        started = time.perf_counter()
        papers = context.inputs.get("papers")
        paper_specs = (
            [item for item in papers if isinstance(item, Mapping)][:2]
            if isinstance(papers, (list, tuple))
            else []
        )
        calls: list[dict[str, Any]] = []
        if context.task_type == "paper_comparison" and paper_specs:
            for paper in paper_specs:
                name = " ".join(str(paper.get("name", "")).split()).strip()
                doc_id = " ".join(str(paper.get("doc_id", "")).split()).strip()
                tool_input = self._tool_input(context)
                tool_input["query"] = (
                    f"{name}: {context.query}" if name else context.query
                )
                if doc_id:
                    tool_input["filters"] = {"doc_ids": [doc_id]}
                calls.append(tool_input)
        else:
            calls.append(self._tool_input(context))

        evidence_by_id: dict[str, dict[str, Any]] = {}
        retrieval_rows: list[dict[str, Any]] = []
        source_bundles: list[EvidenceBundle] = []
        last_failure: ToolResult | None = None
        for tool_input in calls:
            result = context.invoke_tool("retrieve_evidence", tool_input)
            if result.status == "failed":
                last_failure = result
                continue
            raw = result.data.get("evidence", [])
            if not isinstance(raw, list):
                raise TypeError("Retrieval Tool returned invalid evidence.")
            for item in raw:
                record = EvidenceRecord.from_mapping(item)
                evidence_by_id.setdefault(record.chunk_id, record.to_dict())
            raw_bundle = result.data.get("evidence_bundle")
            if isinstance(raw_bundle, Mapping):
                source_bundles.append(EvidenceBundle.from_mapping(raw_bundle))
            retrieval_rows.append(copy.deepcopy(dict(result.data)))

        evidence = tuple(evidence_by_id.values())
        if not evidence:
            if last_failure is not None:
                return _tool_failure(self, started, last_failure)
            return self._result(
                started,
                status="failed",
                reason="no_evidence",
                output={
                    "evidence": [],
                    "evidence_count": 0,
                    "coverage": 0.0,
                    "retryable": True,
                },
            )

        bundle = (
            source_bundles[0]
            if len(calls) == 1 and len(source_bundles) == 1
            else EvidenceBundle.create(
                query=context.query,
                evidence=evidence,
                retrieval_metadata={
                    "skill": self.name,
                    "retrieval_count": len(calls),
                    "source_bundle_ids": [
                        item.bundle_id for item in source_bundles
                    ],
                },
                provenance={
                    "source": "dynamic_skill_retrieval",
                    "tool_result_count": len(retrieval_rows),
                },
            )
        )
        coverage_values = [
            row.get("retrieval", {}).get("coverage")
            for row in retrieval_rows
            if isinstance(row.get("retrieval"), Mapping)
            and isinstance(row.get("retrieval", {}).get("coverage"), (int, float))
        ]
        coverage = min(coverage_values) if coverage_values else None
        return self._result(
            started,
            status="success",
            output={
                "evidence": list(evidence),
                "evidence_bundle": bundle.to_dict(),
                "evidence_count": len(evidence),
                "coverage": coverage,
                "retrieval_count": len(calls),
            },
            evidence=evidence,
        )

    @staticmethod
    def _tool_input(context: SkillContext) -> dict[str, Any]:
        result = {"query": str(context.inputs.get("query") or context.query)}
        for key in (
            "top_k",
            "candidate_k",
            "read_cache",
            "rewrite",
            "multi_query",
        ):
            if key in context.inputs:
                result[key] = context.inputs[key]
        return result


class AssessEvidenceSkill(ResearchSkill):
    name = "assess_evidence"
    version = "1.0.0"
    description = "Apply the existing Evidence Sufficiency gate to a canonical bundle."
    input_schema = _object_schema(
        "AssessEvidenceInput",
        required=("evidence_bundle",),
    )
    output_schema = _object_schema(
        "GateDecision",
        required=("gate_decision", "support_level"),
    )
    required_tools = ("assess_evidence",)
    risk_level = "medium"

    def run(self, context: SkillContext) -> SkillResult:
        started = time.perf_counter()
        bundle = _bundle_from_state(context)
        result = context.invoke_tool(
            "assess_evidence",
            {"evidence_bundle": bundle.to_dict()},
        )
        gate = _gate_from_tool(result, bundle)
        status = (
            "success"
            if result.status == "success" and gate.status == "strong"
            else "rejected"
        )
        return self._result(
            started,
            status=status,
            reason=None if status == "success" else "insufficient_evidence",
            output={
                "gate_decision": gate.to_dict(),
                "support_level": gate.status,
                "evidence_bundle_id": bundle.bundle_id,
            },
            evidence=tuple(record.to_dict() for record in bundle.evidence_records),
        )


class AnalyzeClaimsSkill(ResearchSkill):
    name = "analyze_claims"
    version = "1.0.0"
    description = "Extract bounded provenance-bearing claim candidates from evidence."
    input_schema = _object_schema(
        "AnalyzeClaimsInput",
        required=("evidence_bundle",),
    )
    output_schema = _object_schema(
        "StructuredClaims",
        required=("claims",),
    )
    required_tools = ()
    risk_level = "medium"

    def run(self, context: SkillContext) -> SkillResult:
        started = time.perf_counter()
        bundle = _bundle_from_state(context)
        claims: list[dict[str, Any]] = []
        for index, record in enumerate(bundle.evidence_records[:12], start=1):
            sentences = [
                item.strip()
                for item in SENTENCE_RE.split(record.content)
                if item.strip()
            ]
            text = (sentences[0] if sentences else record.content.strip())[:600]
            if not text:
                continue
            claims.append(
                {
                    "claim_id": f"evidence-claim-{index}",
                    "text": text,
                    "chunk_id": record.chunk_id,
                    "doc_id": record.doc_id,
                    "section": record.section,
                    "page": record.page,
                    "citations": [
                        {
                            "chunk_id": record.chunk_id,
                            "doc_id": record.doc_id,
                            "section": record.section,
                            "page": record.page,
                        }
                    ],
                }
            )
        return self._result(
            started,
            status="success",
            output={
                "claims": claims,
                "claim_count": len(claims),
                "evidence_bundle_id": bundle.bundle_id,
            },
            evidence=tuple(record.to_dict() for record in bundle.evidence_records),
        )


class CompareEvidenceSkill(ResearchSkill):
    name = "compare_evidence"
    version = "1.0.0"
    description = "Group and compare canonical evidence without generating conclusions."
    input_schema = _object_schema(
        "CompareEvidenceInput",
        required=("evidence_bundle",),
    )
    output_schema = _object_schema(
        "EvidenceComparison",
        required=("evidence_groups",),
    )
    required_tools = ()
    risk_level = "medium"

    def run(self, context: SkillContext) -> SkillResult:
        started = time.perf_counter()
        bundle = _bundle_from_state(context)
        groups: dict[str, list[EvidenceRecord]] = defaultdict(list)
        for record in bundle.evidence_records:
            groups[record.doc_id].append(record)
        if len(groups) < 2:
            return self._result(
                started,
                status="rejected",
                reason="insufficient_comparison_groups",
                output={
                    "evidence_groups": [],
                    "group_count": len(groups),
                    "evidence_bundle_id": bundle.bundle_id,
                },
                evidence=tuple(record.to_dict() for record in bundle.evidence_records),
            )
        rows = [
            {
                "doc_id": doc_id,
                "evidence_count": len(records),
                "chunk_ids": [record.chunk_id for record in records],
                "sections": list(
                    dict.fromkeys(record.section for record in records)
                ),
            }
            for doc_id, records in sorted(groups.items())
        ]
        return self._result(
            started,
            status="success",
            output={
                "evidence_groups": rows,
                "group_count": len(rows),
                "evidence_bundle_id": bundle.bundle_id,
            },
            evidence=tuple(record.to_dict() for record in bundle.evidence_records),
        )


class AuditClaimsSkill(ResearchSkill):
    name = "audit_claims"
    version = "1.0.0"
    description = "Audit supplied or extracted claims against canonical evidence."
    input_schema = _object_schema(
        "AuditClaimsInput",
        required=("evidence_bundle",),
    )
    output_schema = _object_schema(
        "CitationAuditResult",
        required=("audit", "gate_decision"),
    )
    required_tools = ("assess_evidence", "audit_answer")
    risk_level = "high"

    def run(self, context: SkillContext) -> SkillResult:
        started = time.perf_counter()
        bundle = _bundle_from_state(context)
        gate, gate_result = _ensure_gate(context, bundle)
        if (
            gate.status != "strong"
            or not gate.answerable
            or (gate_result is not None and gate_result.status != "success")
        ):
            return self._result(
                started,
                status="rejected",
                reason="insufficient_evidence",
                output={
                    "audit": None,
                    "gate_decision": gate.to_dict(),
                    "support_level": gate.status,
                },
                evidence=tuple(record.to_dict() for record in bundle.evidence_records),
            )
        answer = getattr(context.state, "answer", None)
        if not isinstance(answer, Mapping):
            answer = self._claim_artifact(context, bundle, gate)
        result = context.invoke_tool(
            "audit_answer",
            {
                "answer": copy.deepcopy(dict(answer)),
                "evidence_bundle": bundle.to_dict(),
            },
        )
        audit = result.data.get("audit")
        if not isinstance(audit, Mapping):
            return _tool_failure(self, started, result)
        status = "success" if result.status == "success" else (
            "rejected" if result.status == "rejected" else "failed"
        )
        return self._result(
            started,
            status=status,
            reason=None if status == "success" else (
                result.reason or "claim_audit_failed"
            ),
            output={
                "audit": copy.deepcopy(dict(audit)),
                "answer_artifact": copy.deepcopy(dict(answer)),
                "gate_decision": gate.to_dict(),
                "support_level": gate.status,
            },
            evidence=tuple(record.to_dict() for record in bundle.evidence_records),
        )

    @staticmethod
    def _claim_artifact(
        context: SkillContext,
        bundle: EvidenceBundle,
        gate: GateDecision,
    ) -> dict[str, Any]:
        analyze_output = getattr(context.state, "skill_outputs", {}).get(
            "analyze_claims", {}
        )
        raw_claims = (
            analyze_output.get("claims", [])
            if isinstance(analyze_output, Mapping)
            else []
        )
        claim_texts = [
            str(item.get("text", "")).strip()
            for item in raw_claims
            if isinstance(item, Mapping) and str(item.get("text", "")).strip()
        ]
        text = "\n".join(claim_texts[:5]) or context.query
        evidence_by_id = {
            record.chunk_id: record for record in bundle.evidence_records
        }
        supporting = [
            evidence_by_id[chunk_id]
            for chunk_id in gate.supporting_chunk_ids
            if chunk_id in evidence_by_id
        ]
        citations = [
            {
                "chunk_id": record.chunk_id,
                "doc_id": record.doc_id,
                "section": record.section,
                "page": record.page,
            }
            for record in supporting
        ]
        return {
            "answer": text,
            "citations": citations,
            "confidence": gate.confidence,
            "refused": False,
            "refusal_reason": None,
            "evidence_chunk_ids": [record.chunk_id for record in supporting],
            "model": "skill_claim_artifact",
            "prompt_version": "dynamic_skill_claim_audit_v1",
            "config_version": "dynamic_skill_composition_v1",
            "timestamp": gate.assessment.get("timestamp") or utc_timestamp(),
            "cache_hit": False,
            "fallback_used": False,
            "fallback_reason": None,
            "api_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.0,
        }


class GenerateReportSkill(ResearchSkill):
    name = "generate_report"
    version = "1.0.0"
    description = "Generate and audit a grounded report through protected tools."
    input_schema = _object_schema(
        "GenerateReportInput",
        required=("evidence_bundle",),
    )
    output_schema = _object_schema(
        "GroundedReport",
        required=("answer", "audit", "gate_decision"),
    )
    required_tools = (
        "assess_evidence",
        "generate_grounded_answer",
        "audit_answer",
    )
    risk_level = "high"

    def run(self, context: SkillContext) -> SkillResult:
        started = time.perf_counter()
        bundle = _bundle_from_state(context)
        gate, gate_result = _ensure_gate(context, bundle)
        if (
            gate.status != "strong"
            or not gate.answerable
            or (gate_result is not None and gate_result.status != "success")
        ):
            return self._result(
                started,
                status="rejected",
                reason="insufficient_evidence",
                output={
                    "answer": None,
                    "audit": None,
                    "gate_decision": gate.to_dict(),
                    "support_level": gate.status,
                },
                evidence=tuple(record.to_dict() for record in bundle.evidence_records),
            )
        answer_result = context.invoke_tool(
            "generate_grounded_answer",
            {
                "evidence_bundle": bundle.to_dict(),
                "gate_decision": gate.to_dict(),
            },
        )
        if answer_result.status != "success":
            return _tool_failure(self, started, answer_result)
        answer = answer_result.data.get(
            "answer",
            answer_result.data.get("answer_artifact"),
        )
        if not isinstance(answer, Mapping):
            raise TypeError("Answer Tool returned no answer artifact.")
        audit_result = context.invoke_tool(
            "audit_answer",
            {
                "answer": copy.deepcopy(dict(answer)),
                "evidence_bundle": bundle.to_dict(),
            },
        )
        audit = audit_result.data.get("audit")
        if not isinstance(audit, Mapping):
            return _tool_failure(self, started, audit_result)
        status = "success" if audit_result.status == "success" else (
            "rejected" if audit_result.status == "rejected" else "failed"
        )
        return self._result(
            started,
            status=status,
            reason=None if status == "success" else (
                audit_result.reason or "citation_audit_failed"
            ),
            output={
                "answer": copy.deepcopy(dict(answer)),
                "audit": copy.deepcopy(dict(audit)),
                "gate_decision": gate.to_dict(),
                "evidence_bundle_id": bundle.bundle_id,
                "support_level": gate.status,
            },
            evidence=tuple(record.to_dict() for record in bundle.evidence_records),
        )
