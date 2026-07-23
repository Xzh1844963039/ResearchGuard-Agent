# C:\Users\18449\Desktop\researchguard_workspace\researchguard\memory\research_memory.py
from __future__ import annotations

import time
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
