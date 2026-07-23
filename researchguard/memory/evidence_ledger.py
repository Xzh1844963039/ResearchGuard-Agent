# C:\Users\18449\Desktop\researchguard_workspace\researchguard\memory\evidence_ledger.py
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from researchguard.memory.schemas import EvidenceRef, LedgerRecord
from researchguard.memory.storage import DEFAULT_MEMORY_ROOT, JsonlStorage


class EvidenceLedger:
    def __init__(self, root: str | Path = DEFAULT_MEMORY_ROOT):
        self.root = Path(root)
        self.storage = JsonlStorage(self.root / "evidence_ledger.jsonl")

    def add(self, record: LedgerRecord) -> LedgerRecord:
        existing = {item.claim_id for item in self.for_run(record.run_id)}
        if record.claim_id not in existing:
            self.storage.append(record.to_dict())
        return record

    def for_run(self, run_id: str) -> list[LedgerRecord]:
        records: list[LedgerRecord] = []
        for row in self.storage.read_all():
            if str(row.get("run_id")) != run_id:
                continue
            try:
                records.append(LedgerRecord.from_dict(row))
            except (TypeError, ValueError):
                continue
        return records

    def build_from_state(self, state: Any) -> list[LedgerRecord]:
        evidence_by_id = {
            str(item.get("chunk_id")): EvidenceRef.from_evidence(item)
            for item in state.evidence
            if isinstance(item, Mapping) and item.get("chunk_id")
        }
        claims = self._claims_from_state(state)
        source = f"workflow:{state.workflow_name}" if state.workflow_name else f"task:{state.task_type}"
        records: list[LedgerRecord] = []
        for index, claim in enumerate(claims, start=1):
            text = str(claim.get("text", "")).strip()
            if not text:
                continue
            citation_ids = [
                str(item.get("chunk_id"))
                for item in claim.get("citations", [])
                if isinstance(item, Mapping) and item.get("chunk_id")
            ]
            refs = tuple(
                evidence_by_id[chunk_id]
                for chunk_id in dict.fromkeys(citation_ids)
                if chunk_id in evidence_by_id
            )
            verification_status = str(
                claim.get("support_level", claim.get("verification_status", "unknown"))
            ).casefold()
            if verification_status in {"strong", "supported", "partial"} and not refs:
                continue
            raw_id = str(claim.get("id") or f"claim-{index}")
            digest = hashlib.sha256(
                f"{state.run_id}|{raw_id}|{text}".encode("utf-8")
            ).hexdigest()[:16]
            records.append(
                LedgerRecord(
                    claim_id=f"{state.run_id}:claim-{digest}",
                    run_id=state.run_id,
                    claim_text=text,
                    evidence_refs=refs,
                    source=source,
                    verification_status=verification_status,
                )
            )
        return records

    @staticmethod
    def _claims_from_state(state: Any) -> list[dict[str, Any]]:
        audit = state.audit_result if isinstance(state.audit_result, Mapping) else {}
        claims = audit.get("claims", [])
        if isinstance(claims, list) and claims:
            return [dict(item) for item in claims if isinstance(item, Mapping)]

        workflow_output = (
            state.workflow_result.get("output")
            if isinstance(state.workflow_result, Mapping)
            else None
        )
        if not isinstance(workflow_output, Mapping):
            return []
        nested_audit = workflow_output.get("audit_result")
        if isinstance(nested_audit, Mapping):
            nested_claims = nested_audit.get("claims", [])
            if isinstance(nested_claims, list) and nested_claims:
                return [dict(item) for item in nested_claims if isinstance(item, Mapping)]

        text = str(
            workflow_output.get("claim") or workflow_output.get("summary") or ""
        ).strip()
        citations = workflow_output.get("citations", [])
        if not text:
            return []
        return [
            {
                "id": "workflow-result",
                "text": text,
                "support_level": workflow_output.get("support_level", "supported"),
                "citations": citations if isinstance(citations, list) else [],
            }
        ]
