# C:\Users\18449\Desktop\researchguard_workspace\researchguard\memory\schemas.py
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


RUN_RECORD_SCHEMA_VERSION = "researchguard.memory.run_record.v1"
EVIDENCE_REF_SCHEMA_VERSION = "researchguard.memory.evidence_ref.v1"
LEDGER_RECORD_SCHEMA_VERSION = "researchguard.memory.ledger_record.v1"
FAILURE_RECORD_SCHEMA_VERSION = "researchguard.memory.failure_record.v1"
MEMORY_VERSION = "1.0.0"
RUN_STATUSES = {"created", "planned", "running", "completed", "rejected", "failed"}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(content: str) -> str:
    digest = hashlib.sha256(str(content).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class EvidenceRef:
    chunk_id: str
    doc_id: str
    section: str
    page: int | None
    source: str
    hash: str
    schema_version: str = EVIDENCE_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.chunk_id.strip() or not self.doc_id.strip():
            raise ValueError("EvidenceRef requires canonical chunk_id and doc_id.")
        if not self.hash.startswith("sha256:"):
            raise ValueError("EvidenceRef.hash must be a SHA-256 content hash.")

    @classmethod
    def from_evidence(cls, value: Mapping[str, Any]) -> "EvidenceRef":
        page = value.get("page", value.get("page_start"))
        content = str(value.get("content", value.get("text", "")))
        return cls(
            chunk_id=str(value.get("chunk_id", "")).strip(),
            doc_id=str(value.get("doc_id", "")).strip(),
            section=str(value.get("section", "")).strip(),
            page=int(page) if page is not None else None,
            source=str(value.get("source", value.get("title", value.get("doc_id", "")))),
            hash=content_hash(content),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRef":
        if value.get("schema_version") != EVIDENCE_REF_SCHEMA_VERSION:
            raise ValueError("Unsupported EvidenceRef schema.")
        page = value.get("page")
        return cls(
            chunk_id=str(value.get("chunk_id", "")),
            doc_id=str(value.get("doc_id", "")),
            section=str(value.get("section", "")),
            page=int(page) if page is not None else None,
            source=str(value.get("source", "")),
            hash=str(value.get("hash", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section": self.section,
            "page": self.page,
            "source": self.source,
            "hash": self.hash,
        }


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    created_at: str
    updated_at: str
    query: str
    workflow_name: str | None
    status: str
    plan: tuple[Mapping[str, Any], ...]
    tool_trace: tuple[Mapping[str, Any], ...]
    papers: tuple[Mapping[str, Any], ...]
    evidence_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    answer_summary: str | None
    audit_result: Mapping[str, Any] | None
    latency_ms: float
    reason: str | None = None
    version: str = MEMORY_VERSION
    schema_version: str = RUN_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.query.strip():
            raise ValueError("RunRecord requires run_id and query.")
        if self.status not in RUN_STATUSES:
            raise ValueError(f"Unsupported RunRecord status: {self.status}")
        if self.latency_ms < 0:
            raise ValueError("RunRecord.latency_ms must not be negative.")

    @classmethod
    def from_state(
        cls,
        state: Any,
        *,
        latency_ms: float,
        claim_ids: tuple[str, ...] = (),
    ) -> "RunRecord":
        workflow_output = (
            state.workflow_result.get("output")
            if isinstance(state.workflow_result, Mapping)
            else None
        )
        answer_summary: str | None = None
        if isinstance(workflow_output, Mapping):
            answer_summary = str(
                workflow_output.get("summary") or workflow_output.get("claim") or ""
            ).strip() or None
        if answer_summary is None and isinstance(state.answer, Mapping):
            answer_summary = str(state.answer.get("answer", "")).strip() or None
        return cls(
            run_id=state.run_id,
            created_at=state.created_at,
            updated_at=state.updated_at,
            query=state.query,
            workflow_name=state.workflow_name,
            status=state.status,
            plan=tuple(copy.deepcopy(state.plan)),
            tool_trace=tuple(copy.deepcopy(state.tool_history)),
            papers=tuple(copy.deepcopy(state.candidate_papers)),
            evidence_ids=tuple(
                dict.fromkeys(
                    str(item.get("chunk_id"))
                    for item in state.evidence
                    if isinstance(item, Mapping) and item.get("chunk_id")
                )
            ),
            claim_ids=tuple(dict.fromkeys(claim_ids)),
            answer_summary=answer_summary,
            audit_result=copy.deepcopy(state.audit_result),
            latency_ms=max(0.0, float(latency_ms)),
            reason=state.reason,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunRecord":
        if value.get("schema_version") != RUN_RECORD_SCHEMA_VERSION:
            raise ValueError("Unsupported RunRecord schema.")
        return cls(
            run_id=str(value.get("run_id", "")),
            created_at=str(value.get("created_at", "")),
            updated_at=str(value.get("updated_at", "")),
            query=str(value.get("query", "")),
            workflow_name=(
                str(value["workflow_name"])
                if value.get("workflow_name") is not None
                else None
            ),
            status=str(value.get("status", "")),
            plan=tuple(copy.deepcopy(list(value.get("plan", [])))),
            tool_trace=tuple(copy.deepcopy(list(value.get("tool_trace", [])))),
            papers=tuple(copy.deepcopy(list(value.get("papers", [])))),
            evidence_ids=tuple(str(item) for item in value.get("evidence_ids", [])),
            claim_ids=tuple(str(item) for item in value.get("claim_ids", [])),
            answer_summary=(
                str(value["answer_summary"])
                if value.get("answer_summary") is not None
                else None
            ),
            audit_result=copy.deepcopy(value.get("audit_result")),
            latency_ms=float(value.get("latency_ms", 0.0)),
            reason=value.get("reason"),
            version=str(value.get("version", MEMORY_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "query": self.query,
            "workflow_name": self.workflow_name,
            "status": self.status,
            "reason": self.reason,
            "plan": copy.deepcopy(list(self.plan)),
            "tool_trace": copy.deepcopy(list(self.tool_trace)),
            "papers": copy.deepcopy(list(self.papers)),
            "evidence_ids": list(self.evidence_ids),
            "claim_ids": list(self.claim_ids),
            "answer_summary": self.answer_summary,
            "audit_result": copy.deepcopy(dict(self.audit_result)) if self.audit_result else None,
            "latency_ms": self.latency_ms,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class LedgerRecord:
    claim_id: str
    run_id: str
    claim_text: str
    evidence_refs: tuple[EvidenceRef, ...]
    source: str
    verification_status: str
    created_at: str = field(default_factory=utc_timestamp)
    version: str = MEMORY_VERSION
    schema_version: str = LEDGER_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.run_id.strip() or not self.claim_text.strip():
            raise ValueError("LedgerRecord requires claim_id, run_id, and claim_text.")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LedgerRecord":
        if value.get("schema_version") != LEDGER_RECORD_SCHEMA_VERSION:
            raise ValueError("Unsupported LedgerRecord schema.")
        return cls(
            claim_id=str(value.get("claim_id", "")),
            run_id=str(value.get("run_id", "")),
            claim_text=str(value.get("claim_text", "")),
            evidence_refs=tuple(
                EvidenceRef.from_dict(item) for item in value.get("evidence_refs", [])
            ),
            source=str(value.get("source", "")),
            verification_status=str(value.get("verification_status", "unknown")),
            created_at=str(value.get("created_at", utc_timestamp())),
            version=str(value.get("version", MEMORY_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "claim_id": self.claim_id,
            "run_id": self.run_id,
            "claim_text": self.claim_text,
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "source": self.source,
            "verification_status": self.verification_status,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class FailureRecord:
    failure_id: str
    run_id: str
    query: str
    workflow_name: str | None
    failure_type: str
    reason: str
    timestamp: str
    details: Mapping[str, Any] = field(default_factory=dict)
    version: str = MEMORY_VERSION
    schema_version: str = FAILURE_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.failure_id.strip(),
                self.run_id.strip(),
                self.query.strip(),
                self.failure_type.strip(),
                self.reason.strip(),
                self.timestamp.strip(),
            )
        ):
            raise ValueError("FailureRecord requires complete failure provenance.")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FailureRecord":
        if value.get("schema_version") != FAILURE_RECORD_SCHEMA_VERSION:
            raise ValueError("Unsupported FailureRecord schema.")
        return cls(
            failure_id=str(value.get("failure_id", "")),
            run_id=str(value.get("run_id", "")),
            query=str(value.get("query", "")),
            workflow_name=(
                str(value["workflow_name"])
                if value.get("workflow_name") is not None
                else None
            ),
            failure_type=str(value.get("failure_type", "")),
            reason=str(value.get("reason", "")),
            timestamp=str(value.get("timestamp", "")),
            details=copy.deepcopy(dict(value.get("details", {}))),
            version=str(value.get("version", MEMORY_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "failure_id": self.failure_id,
            "run_id": self.run_id,
            "query": self.query,
            "workflow_name": self.workflow_name,
            "failure_type": self.failure_type,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "details": copy.deepcopy(dict(self.details)),
        }
