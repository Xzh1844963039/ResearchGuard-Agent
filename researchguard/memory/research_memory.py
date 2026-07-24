# C:\Users\18449\Desktop\researchguard_workspace\researchguard\memory\research_memory.py
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from researchguard.memory.evidence_ledger import EvidenceLedger
from researchguard.memory.failure_store import FailureStore
from researchguard.memory.run_store import ResearchRunStore
from researchguard.memory.schemas import RunRecord
from researchguard.memory.storage import DEFAULT_MEMORY_ROOT


class ResearchMemory:
    """Persistent research-process memory; it stores no chat history or user profile."""

    def __init__(self, root: str | Path = DEFAULT_MEMORY_ROOT):
        self.root = Path(root)
        self.runs = ResearchRunStore(self.root)
        self.ledger = EvidenceLedger(self.root)
        self.failures = FailureStore(self.root)
        self._started: dict[str, float] = {}

    def start_run(self, state: Any) -> RunRecord:
        self._started[state.run_id] = time.perf_counter()
        return self.runs.save(RunRecord.from_state(state, latency_ms=0.0))

    def complete_run(self, state: Any) -> dict[str, Any]:
        started = self._started.pop(state.run_id, None)
        latency_ms = (
            (time.perf_counter() - started) * 1000.0 if started is not None else 0.0
        )
        errors: list[str] = []
        ledger_records = []
        if state.status == "completed":
            try:
                for record in self.ledger.build_from_state(state):
                    self.ledger.add(record)
                    ledger_records.append(record)
            except Exception as exc:
                errors.append(f"evidence_ledger: {type(exc).__name__}: {exc}")

        failure_record = None
        failure_recorded = False
        try:
            failure_record = self.failures.from_state(state)
            if failure_record is not None:
                self.failures.save(failure_record)
                failure_recorded = True
        except Exception as exc:
            errors.append(f"failure_store: {type(exc).__name__}: {exc}")

        run_record = RunRecord.from_state(
            state,
            latency_ms=latency_ms,
            claim_ids=tuple(item.claim_id for item in ledger_records),
        )
        self.runs.save(run_record)
        return {
            "run_id": state.run_id,
            "persisted": True,
            "ledger_record_count": len(ledger_records),
            "failure_recorded": failure_recorded,
            "errors": errors,
        }

    def show(self, run_id: str) -> dict[str, Any] | None:
        run = self.runs.get(run_id)
        if run is None:
            return None
        return {
            "run": run.to_dict(),
            "evidence_ledger": [item.to_dict() for item in self.ledger.for_run(run_id)],
            "failures": [item.to_dict() for item in self.failures.for_run(run_id)],
        }

    def find_previous_runs(
        self,
        query: str,
        *,
        workflow: str | None = None,
        since: str | None = None,
        limit: int = 20,
    ) -> list[RunRecord]:
        return self.runs.find_previous_runs(
            query,
            workflow=workflow,
            since=since,
            limit=limit,
        )

    def search_context(
        self,
        query: str,
        *,
        workflow: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        records = self.find_previous_runs(
            query,
            workflow=workflow,
            limit=limit,
        )
        papers: list[dict[str, Any]] = []
        seen_papers: set[str] = set()
        failures: list[dict[str, Any]] = []
        for record in records:
            for paper in record.papers:
                identity = str(
                    paper.get("paper_id")
                    or paper.get("doi")
                    or paper.get("title")
                    or ""
                ).strip()
                if not identity or identity in seen_papers:
                    continue
                seen_papers.add(identity)
                papers.append(
                    {
                        "paper_id": paper.get("paper_id"),
                        "title": paper.get("title"),
                        "doi": paper.get("doi"),
                        "source": paper.get("source"),
                    }
                )
            for failure in self.failures.for_run(record.run_id):
                failures.append(
                    {
                        "run_id": failure.run_id,
                        "failure_type": failure.failure_type,
                        "reason": failure.reason,
                        "workflow_name": failure.workflow_name,
                        "timestamp": failure.timestamp,
                    }
                )
        return {
            "schema_version": "researchguard.memory_context.v1",
            "version": "1.0.0",
            "query": str(query),
            "matched_run_ids": [record.run_id for record in records],
            "matched_runs": [
                {
                    "run_id": record.run_id,
                    "query": record.query,
                    "workflow_name": record.workflow_name,
                    "status": record.status,
                    "updated_at": record.updated_at,
                    "evidence_ids": list(record.evidence_ids),
                }
                for record in records
            ],
            "previous_workflows": list(
                dict.fromkeys(
                    record.workflow_name
                    for record in records
                    if record.workflow_name
                )
            ),
            "previous_papers": papers[:10],
            "previous_failures": failures[:limit],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
