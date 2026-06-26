# C:\Users\18449\Desktop\researchguard_workspace\researchguard\schemas.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SourceRecord:
    source_id: str
    case_id: str
    source_type: str
    source_name: str
    source_path: str
    total_pages: int | None
    loaded_at: str
    parser_version: str
    status: str
    error_message: str | None = None
    discipline: str | None = None
    topic: str | None = None
    parser_used: str | None = None
    text_length: int = 0


@dataclass
class EvidenceRecord:
    evidence_id: str
    case_id: str
    source_id: str
    source_type: str
    source_name: str
    page: int | None
    section: str | None
    paragraph_index: int | None
    table_id: str | None
    row_index: int | None
    location_text: str
    content: str
    content_summary: str
    created_by_tool: str
    created_at: str


@dataclass
class ClaimRecord:
    claim_id: str
    case_id: str
    original_text: str
    normalized_text: str
    claim_type: str
    generated_by: str
    source_report_section: str
    created_at: str


@dataclass
class ReviewRecord:
    review_id: str
    case_id: str
    claim_id: str
    claim_text: str
    status: str
    color: str
    evidence_ids: list[str]
    counter_evidence_ids: list[str]
    citation_ids: list[str] = field(default_factory=list)
    explanation: str = ""
    confidence: float | None = None
    checked_by_tools: list[str] = field(default_factory=list)
    created_at: str = ""


@dataclass
class CitationRecord:
    citation_id: str
    raw_citation_text: str
    title: str | None
    authors: str | None
    year: str | None
    exists_in_input_references: bool
    matched_reference_text: str | None
    supports_claim: bool | None
    status: str


@dataclass
class ToolTraceRecord:
    trace_id: str
    case_id: str
    tool_name: str
    input_summary: str
    output_summary: str
    status: str
    error_message: str | None
    started_at: str
    finished_at: str
    duration_ms: int


@dataclass
class SearchQueryRecord:
    query_id: str
    case_id: str
    query: str
    discipline: str | None
    generated_from_evidence_ids: list[str]
    generated_by_skill: str
    created_at: str


@dataclass
class LiteratureReferenceRecord:
    ref_id: str
    case_id: str
    query_id: str
    title: str
    authors: list[str]
    year: str | None
    venue: str | None
    doi: str | None
    pmid: str | None
    arxiv_id: str | None
    url: str | None
    abstract: str | None
    source_api: str
    raw_result_path: str | None
    metadata_confidence: str
    verified_existence: bool | str
    matched_sources: list[str]
    created_at: str
    source_mode: str = "unknown"
    is_demo_reference: bool = False


@dataclass
class HypothesisRecord:
    hypothesis_id: str
    case_id: str
    hypothesis_text: str
    rationale: str
    supporting_evidence_ids: list[str]
    supporting_ref_ids: list[str]
    novelty_claim: str
    expected_validation_method: str
    status: str
    color: str
    generated_by_skill: str
    created_at: str


@dataclass
class HypothesisSupportClaimRecord:
    support_claim_id: str
    case_id: str
    hypothesis_id: str
    claim_text: str
    source_type: str
    source_ids: list[str]
    created_at: str


@dataclass
class ReferenceAuditRecord:
    audit_id: str
    case_id: str
    hypothesis_id: str
    support_claim_id: str
    ref_id: str
    citation_text: str
    existence_status: str
    metadata_match_status: str
    support_status: str
    final_status: str
    color: str
    evidence_ids: list[str]
    explanation: str
    checked_by_skills: list[str]
    created_at: str


@dataclass
class SkillTraceRecord:
    trace_id: str
    case_id: str
    skill_name: str
    input_summary: str
    output_summary: str
    status: str
    error_message: str | None
    started_at: str
    finished_at: str
    duration_ms: int
    cached: bool = False


@dataclass
class MemorySnapshot:
    case_id: str
    source_memory_count: int
    evidence_memory_count: int
    literature_memory_count: int
    hypothesis_memory_count: int
    review_memory_count: int
    tool_trace_count: int
    skill_trace_count: int
    failure_memory_count: int
    updated_files: list[str]
    created_at: str


def to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported record type: {type(obj)!r}")
