# C:\Users\18449\Desktop\researchguard_workspace\researchguard\memory\failure_store.py
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from researchguard.memory.schemas import FailureRecord
from researchguard.memory.storage import DEFAULT_MEMORY_ROOT, JsonlStorage


class FailureStore:
    def __init__(self, root: str | Path = DEFAULT_MEMORY_ROOT):
        self.root = Path(root)
        self.storage = JsonlStorage(self.root / "failures.jsonl")

    def save(self, record: FailureRecord) -> FailureRecord:
        existing = {item.failure_id for item in self.for_run(record.run_id)}
        if record.failure_id not in existing:
            self.storage.append(record.to_dict())
        return record

    def for_run(self, run_id: str) -> list[FailureRecord]:
        records: list[FailureRecord] = []
        for row in self.storage.read_all():
            if str(row.get("run_id")) != run_id:
                continue
            try:
                records.append(FailureRecord.from_dict(row))
            except (TypeError, ValueError):
                continue
        return records

    def from_state(self, state: Any) -> FailureRecord | None:
        if state.status not in {"rejected", "failed"}:
            return None
        reason = str(state.reason or state.status)
        failure_type = self._classify(state.status, reason, bool(state.evidence))
        digest = hashlib.sha256(
            f"{state.run_id}|{failure_type}|{reason}".encode("utf-8")
        ).hexdigest()[:16]
        return FailureRecord(
            failure_id=f"{state.run_id}:failure-{digest}",
            run_id=state.run_id,
            query=state.query,
            workflow_name=state.workflow_name,
            failure_type=failure_type,
            reason=reason,
            timestamp=state.updated_at,
            details={
                "status": state.status,
                "task_type": state.task_type,
                "tool_call_count": len(state.tool_history),
                "evidence_count": len(state.evidence),
            },
        )

    @staticmethod
    def _classify(status: str, reason: str, has_evidence: bool) -> str:
        normalized = reason.casefold()
        if "insufficient" in normalized:
            return "insufficient_evidence"
        if "tool" in normalized or "timeout" in normalized:
            return "tool_failure"
        if status == "rejected":
            return "rejected_answer" if has_evidence else "no_evidence"
        return "execution_failure"
