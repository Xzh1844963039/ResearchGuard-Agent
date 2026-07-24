# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tools\contracts.py
from __future__ import annotations

import json
import hashlib
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping

from researchguard.retrieval.models import RetrievalHit


TOOL_RESULT_SCHEMA_VERSION = "researchguard.tool_result.v1"
EVIDENCE_RECORD_SCHEMA_VERSION = "researchguard.evidence_record.v1"
TOOL_ERROR_SCHEMA_VERSION = "researchguard.tool_error.v1"
TOOL_SPEC_SCHEMA_VERSION = "researchguard.tool_spec.v1"
EVIDENCE_BUNDLE_SCHEMA_VERSION = "researchguard.evidence_bundle.v1"
GATE_DECISION_SCHEMA_VERSION = "researchguard.gate_decision.v1"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_trace_id(tool_name: str) -> str:
    return f"{tool_name}-{uuid.uuid4().hex}"


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_value(value.to_dict())
    if is_dataclass(value):
        return _json_value(asdict(value))
    return str(value)


@dataclass(frozen=True)
class ToolError:
    code: str
    category: str
    message: str
    retryable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = TOOL_ERROR_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "code": self.code,
            "category": self.category,
            "message": self.message,
            "retryable": self.retryable,
            "details": _json_value(self.details),
        }


@dataclass(frozen=True)
class EvidenceRecord:
    chunk_id: str
    doc_id: str
    section: str
    page: int | None
    content: str
    source: str
    score: float | None
    provenance: Mapping[str, Any]
    rank: int | None = None
    page_end: int | None = None
    schema_version: str = EVIDENCE_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("EvidenceRecord.chunk_id must not be empty.")
        if not self.doc_id.strip():
            raise ValueError("EvidenceRecord.doc_id must not be empty.")
        if not self.content.strip():
            raise ValueError("EvidenceRecord.content must not be empty.")

    @classmethod
    def from_retrieval_hit(cls, hit: RetrievalHit) -> "EvidenceRecord":
        row = hit.to_dict(include_text=True)
        score = next(
            (
                float(row[key])
                for key in (
                    "rerank_score",
                    "multi_query_fusion_score",
                    "fusion_score",
                    "dense_score",
                    "sparse_score",
                )
                if row.get(key) is not None
            ),
            None,
        )
        provenance_keys = (
            "title",
            "section_heading",
            "heading_path",
            "chunk_type",
            "source_block_ids",
            "overlap_source_block_ids",
            "content_types",
            "has_equation",
            "has_table",
            "has_caption",
            "dense_score",
            "sparse_score",
            "fusion_score",
            "rerank_score",
            "multi_query_fusion_score",
            "dense_rank",
            "sparse_rank",
            "rerank_rank",
            "retrieval_sources",
            "query_variant_hits",
            "query_variant_ids",
            "query_variant_types",
            "query_variant_ranks",
            "original_query_recalled",
            "rewrite_query_recalled",
            "expansion_query_recalled",
        )
        provenance = {key: row.get(key) for key in provenance_keys}
        provenance["canonical"] = {
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "section": row.get("section", ""),
            "page_start": row.get("page_start"),
            "page_end": row.get("page_end"),
        }
        return cls(
            chunk_id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            section=str(row.get("section", "")),
            page=row.get("page_start"),
            page_end=row.get("page_end"),
            content=str(row.get("text", "")),
            source=str(row.get("title") or row["doc_id"]),
            score=score,
            rank=row.get("rank"),
            provenance=provenance,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceRecord":
        provenance = dict(value.get("provenance", {}) or {})
        page = value.get("page", value.get("page_start"))
        page_end = value.get("page_end", page)
        for key in (
            "title",
            "section_heading",
            "heading_path",
            "chunk_type",
            "source_block_ids",
            "overlap_source_block_ids",
            "content_types",
            "has_equation",
            "has_table",
            "has_caption",
            "dense_score",
            "sparse_score",
            "fusion_score",
            "rerank_score",
            "multi_query_fusion_score",
            "dense_rank",
            "sparse_rank",
            "rerank_rank",
            "retrieval_sources",
            "query_variant_hits",
        ):
            if key in value and key not in provenance:
                provenance[key] = value[key]
        raw_score = value.get("score")
        if raw_score is None:
            raw_score = next(
                (
                    value[key]
                    for key in (
                        "rerank_score",
                        "multi_query_fusion_score",
                        "fusion_score",
                        "dense_score",
                        "sparse_score",
                    )
                    if value.get(key) is not None
                ),
                None,
            )
        return cls(
            chunk_id=str(value.get("chunk_id", "")),
            doc_id=str(value.get("doc_id", "")),
            section=str(value.get("section", "")),
            page=int(page) if page is not None else None,
            page_end=int(page_end) if page_end is not None else None,
            content=str(value.get("content", value.get("text", ""))),
            source=str(value.get("source", value.get("title", value.get("doc_id", "")))),
            score=float(raw_score) if raw_score is not None else None,
            rank=int(value["rank"]) if value.get("rank") is not None else None,
            provenance=provenance,
        )

    def to_retrieval_mapping(self) -> dict[str, Any]:
        provenance = dict(self.provenance)
        mapping = {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": provenance.get("title") or self.source,
            "section": self.section,
            "section_heading": provenance.get("section_heading", ""),
            "heading_path": provenance.get("heading_path", []),
            "chunk_type": provenance.get("chunk_type", "text"),
            "page_start": self.page,
            "page_end": self.page_end if self.page_end is not None else self.page,
            "source_block_ids": provenance.get("source_block_ids", []),
            "overlap_source_block_ids": provenance.get("overlap_source_block_ids", []),
            "content_types": provenance.get("content_types", []),
            "has_equation": bool(provenance.get("has_equation", False)),
            "has_table": bool(provenance.get("has_table", False)),
            "has_caption": bool(provenance.get("has_caption", False)),
            "text": self.content,
        }
        for key in (
            "dense_score",
            "sparse_score",
            "fusion_score",
            "rerank_score",
            "multi_query_fusion_score",
            "dense_rank",
            "sparse_rank",
            "rerank_rank",
            "retrieval_sources",
            "query_variant_hits",
            "query_variant_ids",
            "query_variant_types",
            "query_variant_ranks",
            "original_query_recalled",
            "rewrite_query_recalled",
            "expansion_query_recalled",
        ):
            if key in provenance:
                mapping[key] = provenance[key]
        return mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section": self.section,
            "page": self.page,
            "page_end": self.page_end,
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "rank": self.rank,
            "provenance": _json_value(self.provenance),
        }


def _evidence_bundle_id(
    query: str,
    records: tuple[EvidenceRecord, ...],
    retrieval_metadata: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    payload = {
        "query": query,
        "evidence_records": [record.to_dict() for record in records],
        "retrieval_metadata": _json_value(retrieval_metadata),
        "provenance": _json_value(provenance),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"evidence-{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class EvidenceBundle:
    query: str
    evidence_records: tuple[EvidenceRecord, ...]
    retrieval_metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    bundle_id: str = ""
    version: str = "1.0.0"
    schema_version: str = EVIDENCE_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        normalized_query = " ".join(str(self.query).split()).strip()
        if not normalized_query:
            raise ValueError("EvidenceBundle.query must not be empty.")
        if not self.evidence_records:
            raise ValueError("EvidenceBundle requires at least one evidence record.")
        chunk_ids = [record.chunk_id for record in self.evidence_records]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("EvidenceBundle contains duplicate chunk_id values.")
        object.__setattr__(self, "query", normalized_query)
        expected_id = _evidence_bundle_id(
            normalized_query,
            self.evidence_records,
            self.retrieval_metadata,
            self.provenance,
        )
        if self.bundle_id and self.bundle_id != expected_id:
            raise ValueError("EvidenceBundle.bundle_id does not match its canonical content.")
        object.__setattr__(self, "bundle_id", expected_id)

    @classmethod
    def create(
        cls,
        *,
        query: str,
        evidence: Iterable[EvidenceRecord | Mapping[str, Any]],
        retrieval_metadata: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "EvidenceBundle":
        records = tuple(
            item if isinstance(item, EvidenceRecord) else EvidenceRecord.from_mapping(item)
            for item in evidence
        )
        return cls(
            query=query,
            evidence_records=records,
            retrieval_metadata=dict(retrieval_metadata or {}),
            provenance=dict(provenance or {}),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceBundle":
        raw_records = value.get("evidence_records", value.get("evidence", ()))
        if not isinstance(raw_records, (list, tuple)):
            raise TypeError("EvidenceBundle.evidence_records must be a list.")
        return cls(
            query=str(value.get("query", "")),
            evidence_records=tuple(
                item if isinstance(item, EvidenceRecord) else EvidenceRecord.from_mapping(item)
                for item in raw_records
            ),
            retrieval_metadata=dict(value.get("retrieval_metadata", {}) or {}),
            provenance=dict(value.get("provenance", {}) or {}),
            bundle_id=str(value.get("bundle_id", "")),
            version=str(value.get("version", "1.0.0")),
            schema_version=str(
                value.get("schema_version", EVIDENCE_BUNDLE_SCHEMA_VERSION)
            ),
        )

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(record.chunk_id for record in self.evidence_records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "bundle_id": self.bundle_id,
            "query": self.query,
            "evidence_records": [record.to_dict() for record in self.evidence_records],
            "retrieval_metadata": _json_value(self.retrieval_metadata),
            "provenance": _json_value(self.provenance),
        }


@dataclass(frozen=True)
class GateDecision:
    status: str
    reason: str
    supporting_chunk_ids: tuple[str, ...]
    evidence_bundle_id: str
    confidence: float = 0.0
    answerable: bool = False
    assessment: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    schema_version: str = GATE_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        normalized_status = str(self.status).strip().casefold()
        if normalized_status not in {"strong", "partial", "unsupported"}:
            raise ValueError(f"Unsupported GateDecision status: {self.status}")
        if not str(self.reason).strip():
            raise ValueError("GateDecision.reason must not be empty.")
        if not self.evidence_bundle_id.strip():
            raise ValueError("GateDecision.evidence_bundle_id must not be empty.")
        if len(self.supporting_chunk_ids) != len(set(self.supporting_chunk_ids)):
            raise ValueError("GateDecision supporting_chunk_ids must be unique.")
        expected_answerable = normalized_status == "strong"
        if bool(self.answerable) != expected_answerable:
            raise ValueError("GateDecision answerable flag does not match status.")
        if normalized_status in {"strong", "partial"} and not self.supporting_chunk_ids:
            raise ValueError(f"{normalized_status} GateDecision requires supporting evidence.")
        if normalized_status == "unsupported" and self.supporting_chunk_ids:
            raise ValueError("Unsupported GateDecision cannot contain supporting evidence.")
        object.__setattr__(self, "status", normalized_status)

    @classmethod
    def from_assessment(
        cls,
        *,
        evidence_bundle_id: str,
        assessment: Mapping[str, Any],
    ) -> "GateDecision":
        status = str(assessment.get("support_level", "unsupported")).casefold()
        return cls(
            status=status,
            reason=str(assessment.get("reason") or f"evidence_{status}"),
            supporting_chunk_ids=tuple(
                dict.fromkeys(
                    str(item)
                    for item in assessment.get("supporting_chunk_ids", ())
                    if str(item).strip()
                )
            ),
            evidence_bundle_id=evidence_bundle_id,
            confidence=float(assessment.get("confidence", 0.0)),
            answerable=bool(assessment.get("answerable", status == "strong")),
            assessment=dict(assessment),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GateDecision":
        return cls(
            status=str(value.get("status", value.get("support_level", ""))),
            reason=str(value.get("reason", "")),
            supporting_chunk_ids=tuple(
                str(item) for item in value.get("supporting_chunk_ids", ())
            ),
            evidence_bundle_id=str(value.get("evidence_bundle_id", "")),
            confidence=float(value.get("confidence", 0.0)),
            answerable=bool(value.get("answerable", False)),
            assessment=dict(value.get("assessment", {}) or {}),
            version=str(value.get("version", "1.0.0")),
            schema_version=str(
                value.get("schema_version", GATE_DECISION_SCHEMA_VERSION)
            ),
        )

    def to_sufficiency_result(self) -> Any:
        from researchguard.retrieval.evidence_judge import EvidenceSufficiencyResult

        source = dict(self.assessment)
        return EvidenceSufficiencyResult(
            answerable=self.answerable,
            support_level=self.status,
            confidence=self.confidence,
            reason=self.reason,
            supporting_chunk_ids=self.supporting_chunk_ids,
            model=str(source.get("model", "agent_evidence_gate")),
            prompt_version=str(source.get("prompt_version", "agent_evidence_gate_v1")),
            config_version=str(source.get("config_version", "agent_evidence_gate_v1.0")),
            timestamp=str(source.get("timestamp", utc_timestamp())),
            cache_hit=bool(source.get("cache_hit", False)),
            fallback_used=bool(source.get("fallback_used", False)),
            fallback_reason=source.get("fallback_reason"),
            api_call_count=int(source.get("api_call_count", 0)),
            input_tokens=int(source.get("input_tokens", 0)),
            output_tokens=int(source.get("output_tokens", 0)),
            latency_ms=float(source.get("latency_ms", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "status": self.status,
            "reason": self.reason,
            "supporting_chunk_ids": list(self.supporting_chunk_ids),
            "evidence_bundle_id": self.evidence_bundle_id,
            "confidence": self.confidence,
            "answerable": self.answerable,
            "assessment": _json_value(self.assessment),
        }


@dataclass(frozen=True)
class ToolResult:
    status: str
    message: str
    reason: str | None
    timestamp: str
    latency_ms: float
    tool_name: str
    tool_version: str
    trace_id: str
    data: Mapping[str, Any] = field(default_factory=dict)
    error: ToolError | None = None
    schema_version: str = TOOL_RESULT_SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        return self.status == "success"

    @classmethod
    def create(
        cls,
        *,
        status: str,
        message: str,
        tool_name: str,
        tool_version: str,
        latency_ms: float,
        reason: str | None = None,
        data: Mapping[str, Any] | None = None,
        error: ToolError | None = None,
        trace_id: str | None = None,
    ) -> "ToolResult":
        if status not in {"success", "rejected", "failed"}:
            raise ValueError(f"Unsupported ToolResult status: {status}")
        return cls(
            status=status,
            message=message,
            reason=reason,
            timestamp=utc_timestamp(),
            latency_ms=max(0.0, float(latency_ms)),
            tool_name=tool_name,
            tool_version=tool_version,
            trace_id=trace_id or new_trace_id(tool_name),
            data=data or {},
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "message": self.message,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "trace_id": self.trace_id,
            "data": _json_value(self.data),
            "error": self.error.to_dict() if self.error else None,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: str = TOOL_RESULT_SCHEMA_VERSION
    schema_version: str = TOOL_SPEC_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "input_schema": _json_value(self.input_schema),
            "output_schema": self.output_schema,
        }
