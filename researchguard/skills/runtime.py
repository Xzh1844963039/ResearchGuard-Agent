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
from researchguard.workflows.claim_audit import (
    extract_atomic_claims,
    extract_claim_entities,
)
from researchguard.workflows.paper_comparison import (
    extract_paper_specs,
    normalize_comparison_dimensions,
)


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
        paper_specs = extract_paper_specs(papers, context.query)
        dimensions = normalize_comparison_dimensions(
            context.inputs.get("comparison_dimensions")
        )
        calls: list[dict[str, Any]] = []
        identities: list[dict[str, Any]] = []
        if context.task_type == "paper_comparison":
            if len(paper_specs) != 2:
                return self._result(
                    started,
                    status="rejected",
                    reason="paper_identity_extraction_failed",
                    output={
                        "evidence": [],
                        "evidence_count": 0,
                        "paper_identities": paper_specs,
                        "comparison_dimensions": list(dimensions),
                    },
                )
            for paper in paper_specs:
                related_to_existing = any(
                    self._identity_tokens(str(item["name"])).intersection(
                        self._identity_tokens(str(paper["name"]))
                    )
                    for item in identities
                )
                identity = self._resolve_identity(
                    context,
                    paper,
                    excluded_doc_ids=(
                        ()
                        if related_to_existing
                        else tuple(item["doc_id"] for item in identities)
                    ),
                )
                if identity is None:
                    return self._result(
                        started,
                        status="rejected",
                        reason="paper_identity_resolution_failed",
                        output={
                            "evidence": [],
                            "evidence_count": 0,
                            "paper_identities": identities,
                            "comparison_dimensions": list(dimensions),
                        },
                    )
                identities.append(identity)
            resolved_doc_ids = [item["doc_id"] for item in identities]
            resolved_names = [item["name"].casefold() for item in identities]
            if (
                len(set(resolved_doc_ids)) != len(resolved_doc_ids)
                and len(set(resolved_names)) != len(resolved_names)
            ):
                return self._result(
                    started,
                    status="rejected",
                    reason="paper_identity_collision",
                    output={
                        "evidence": [],
                        "evidence_count": 0,
                        "paper_identities": identities,
                        "comparison_dimensions": list(dimensions),
                    },
                )
            for paper in identities:
                name = paper["name"]
                doc_id = paper["doc_id"]
                tool_input = self._tool_input(context)
                tool_input["query"] = (
                    f"{name}: {context.query}" if name else context.query
                )
                tool_input["filters"] = {"doc_ids": [doc_id]}
                calls.append(tool_input)
        elif context.task_type == "claim_audit":
            entities = extract_claim_entities(context.query)
            if len(entities) > 1:
                for entity in entities:
                    tool_input = self._tool_input(context)
                    tool_input["query"] = f"{entity}: {context.query}"
                    calls.append(tool_input)
            else:
                tool_input = self._tool_input(context)
                if re.search(
                    r"\b(?:all|every)\s+(?:indexed\s+)?papers?\b",
                    context.query,
                    re.IGNORECASE,
                ):
                    tool_input.update(
                        {
                            "top_k": max(
                                50,
                                int(tool_input.get("top_k", 50)),
                            ),
                            "candidate_k": max(
                                160,
                                int(tool_input.get("candidate_k", 160)),
                            ),
                            "multi_query": True,
                        }
                    )
                calls.append(tool_input)
        else:
            calls.append(self._tool_input(context))

        evidence_by_id: dict[str, dict[str, Any]] = {}
        retrieval_rows: list[dict[str, Any]] = []
        source_bundles: list[EvidenceBundle] = []
        last_failure: ToolResult | None = None
        identity_rows: list[dict[str, Any]] = []
        for call_index, tool_input in enumerate(calls):
            result = context.invoke_tool("retrieve_evidence", tool_input)
            if result.status == "failed":
                last_failure = result
                continue
            raw = result.data.get("evidence", [])
            if not isinstance(raw, list):
                raise TypeError("Retrieval Tool returned invalid evidence.")
            identity = (
                identities[call_index]
                if context.task_type == "paper_comparison"
                else None
            )
            identity_chunk_ids: list[str] = []
            for item in raw:
                record = EvidenceRecord.from_mapping(item)
                if identity is not None and record.doc_id != identity["doc_id"]:
                    return self._result(
                        started,
                        status="failed",
                        reason="comparison_evidence_identity_mismatch",
                        output={
                            "evidence": [],
                            "evidence_count": 0,
                            "paper_identities": identities,
                            "comparison_dimensions": list(dimensions),
                        },
                    )
                serialized = record.to_dict()
                if identity is not None:
                    provenance = dict(serialized.get("provenance", {}))
                    provenance["comparison_identity"] = {
                        "identity_id": identity["identity_id"],
                        "name": identity["name"],
                        "doc_id": identity["doc_id"],
                    }
                    serialized["provenance"] = provenance
                existing = evidence_by_id.get(record.chunk_id)
                if (
                    existing is not None
                    and identity is not None
                    and existing.get("provenance", {}).get(
                        "comparison_identity", {}
                    ).get("identity_id") != identity["identity_id"]
                ):
                    continue
                evidence_by_id.setdefault(record.chunk_id, serialized)
                identity_chunk_ids.append(record.chunk_id)
            raw_bundle = result.data.get("evidence_bundle")
            if isinstance(raw_bundle, Mapping):
                source_bundles.append(EvidenceBundle.from_mapping(raw_bundle))
            retrieval_rows.append(copy.deepcopy(dict(result.data)))
            if identity is not None:
                if not identity_chunk_ids:
                    return self._result(
                        started,
                        status="rejected",
                        reason="incomplete_comparison_evidence",
                        output={
                            "evidence": list(evidence_by_id.values()),
                            "evidence_count": len(evidence_by_id),
                            "paper_identities": identities,
                            "comparison_dimensions": list(dimensions),
                        },
                    )
                identity_rows.append(
                    {
                        **identity,
                        "chunk_ids": list(dict.fromkeys(identity_chunk_ids)),
                    }
                )

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
                    "paper_identities": identity_rows,
                    "comparison_dimensions": list(dimensions),
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
                "paper_identities": identity_rows,
                "comparison_dimensions": list(dimensions),
            },
            evidence=evidence,
        )

    def _resolve_identity(
        self,
        context: SkillContext,
        paper: Mapping[str, Any],
        *,
        excluded_doc_ids: tuple[str, ...] = (),
    ) -> dict[str, Any] | None:
        name = " ".join(str(paper.get("name", "")).split()).strip()
        identity_id = str(paper.get("identity_id", "")).strip()
        explicit_doc_id = " ".join(
            str(paper.get("doc_id") or "").split()
        ).strip()
        if explicit_doc_id:
            return {
                "identity_id": identity_id,
                "name": name,
                "doc_id": explicit_doc_id,
                "resolution": "explicit",
            }
        probe_input = self._tool_input(context)
        probe_input.update(
            {
                "query": name,
                "top_k": max(10, int(probe_input.get("top_k", 10))),
                "rewrite": False,
                "multi_query": False,
            }
        )
        result = context.invoke_tool("retrieve_evidence", probe_input)
        if result.status == "failed":
            return None
        raw = result.data.get("evidence", [])
        if not isinstance(raw, list) or not raw:
            return None
        identity_tokens = self._identity_tokens(name)
        scores: dict[str, float] = defaultdict(float)
        for index, item in enumerate(raw, start=1):
            record = EvidenceRecord.from_mapping(item)
            haystack = " ".join(
                (
                    record.doc_id,
                    record.source,
                    str(record.provenance.get("title", "")),
                )
            )
            overlap = len(identity_tokens.intersection(
                self._identity_tokens(haystack)
            ))
            scores[record.doc_id] += 1.0 / index + 2.0 * overlap
        if not scores:
            return None
        eligible = {
            doc_id: score
            for doc_id, score in scores.items()
            if doc_id not in set(excluded_doc_ids)
        } or scores
        doc_id = min(
            eligible,
            key=lambda item: (-eligible[item], item),
        )
        return {
            "identity_id": identity_id,
            "name": name,
            "doc_id": doc_id,
            "resolution": "retrieval_probe",
        }

    @staticmethod
    def _identity_tokens(value: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9]+", value)
            if len(token) > 1
            and token.casefold()
            not in {
                "paper",
                "method",
                "model",
                "original",
                "standard",
                "retrieval",
                "evaluation",
                "claim",
            }
        }

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
    description = "Decompose the user-supplied statement into bounded atomic claims."
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
        claim_text = str(
            context.inputs.get("claim")
            or getattr(context.state, "workflow_input", {}).get("claim")
            or context.query
        )
        claims = list(extract_atomic_claims(claim_text))
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
        dimensions = normalize_comparison_dimensions(
            context.inputs.get(
                "comparison_dimensions",
                bundle.provenance.get("comparison_dimensions"),
            )
        )
        expected_identities = bundle.provenance.get("paper_identities", [])
        if not isinstance(expected_identities, list) or len(expected_identities) != 2:
            return self._result(
                started,
                status="rejected",
                reason="missing_paper_identity_context",
                output={
                    "evidence_groups": [],
                    "group_count": 0,
                    "comparison_dimensions": list(dimensions),
                    "evidence_bundle_id": bundle.bundle_id,
                },
                evidence=tuple(
                    record.to_dict() for record in bundle.evidence_records
                ),
            )
        groups: dict[str, list[EvidenceRecord]] = defaultdict(list)
        for record in bundle.evidence_records:
            identity = record.provenance.get("comparison_identity", {})
            identity_id = (
                str(identity.get("identity_id", ""))
                if isinstance(identity, Mapping)
                else ""
            )
            if not identity_id:
                return self._result(
                    started,
                    status="failed",
                    reason="unscoped_comparison_evidence",
                    output={
                        "evidence_groups": [],
                        "group_count": len(groups),
                        "comparison_dimensions": list(dimensions),
                        "evidence_bundle_id": bundle.bundle_id,
                    },
                    evidence=tuple(
                        item.to_dict() for item in bundle.evidence_records
                    ),
                )
            groups[identity_id].append(record)
        expected_ids = {
            str(item.get("identity_id", ""))
            for item in expected_identities
            if isinstance(item, Mapping)
        }
        if len(groups) != 2 or set(groups) != expected_ids:
            return self._result(
                started,
                status="rejected",
                reason="insufficient_comparison_groups",
                output={
                    "evidence_groups": [],
                    "group_count": len(groups),
                    "comparison_dimensions": list(dimensions),
                    "evidence_bundle_id": bundle.bundle_id,
                },
                evidence=tuple(record.to_dict() for record in bundle.evidence_records),
            )
        rows: list[dict[str, Any]] = []
        for identity in expected_identities:
            identity_id = str(identity["identity_id"])
            records = groups[identity_id]
            expected_doc_id = str(identity["doc_id"])
            if any(record.doc_id != expected_doc_id for record in records):
                return self._result(
                    started,
                    status="failed",
                    reason="comparison_group_doc_mismatch",
                    output={
                        "evidence_groups": rows,
                        "group_count": len(rows),
                        "comparison_dimensions": list(dimensions),
                        "evidence_bundle_id": bundle.bundle_id,
                    },
                    evidence=tuple(
                        item.to_dict() for item in bundle.evidence_records
                    ),
                )
            rows.append(
                {
                    "identity_id": identity_id,
                    "paper_name": identity["name"],
                    "doc_id": expected_doc_id,
                    "evidence_count": len(records),
                    "chunk_ids": [record.chunk_id for record in records],
                    "sections": list(
                        dict.fromkeys(record.section for record in records)
                    ),
                    "dimensions": list(dimensions),
                }
            )
        return self._result(
            started,
            status="success",
            output={
                "evidence_groups": rows,
                "group_count": len(rows),
                "comparison_dimensions": list(dimensions),
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
                "claims": copy.deepcopy(
                    list(context.inputs.get("claims", []))
                ),
                "gate_decision": gate.to_dict(),
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
                "gate_decision": gate.to_dict(),
                "support_level": gate.status,
                "claims": copy.deepcopy(
                    list(context.inputs.get("claims", []))
                ),
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
        text = "\n".join(claim_texts[:8]) or context.query
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
        answer_text = str(answer.get("answer", "")).strip()
        try:
            explicit_claims = list(
                extract_atomic_claims(answer_text, max_claims=20)
            )
        except ValueError:
            explicit_claims = []
        audit_input: dict[str, Any] = {
            "answer": copy.deepcopy(dict(answer)),
            "evidence_bundle": bundle.to_dict(),
            "gate_decision": gate.to_dict(),
        }
        if explicit_claims:
            audit_input["claims"] = explicit_claims
        audit_result = context.invoke_tool(
            "audit_answer",
            audit_input,
        )
        audit = audit_result.data.get("audit")
        if not isinstance(audit, Mapping):
            return _tool_failure(self, started, audit_result)
        status = "success" if audit_result.status == "success" else (
            "rejected" if audit_result.status == "rejected" else "failed"
        )
        released_answer = (
            copy.deepcopy(dict(answer))
            if status == "success"
            else None
        )
        return self._result(
            started,
            status=status,
            reason=None if status == "success" else (
                audit_result.reason or "citation_audit_failed"
            ),
            output={
                "answer": released_answer,
                "audit": copy.deepcopy(dict(audit)),
                "gate_decision": gate.to_dict(),
                "evidence_bundle_id": bundle.bundle_id,
                "support_level": gate.status,
            },
            evidence=tuple(record.to_dict() for record in bundle.evidence_records),
        )
