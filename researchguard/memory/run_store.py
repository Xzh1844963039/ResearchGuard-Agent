# C:\Users\18449\Desktop\researchguard_workspace\researchguard\memory\run_store.py
from __future__ import annotations

import re
from pathlib import Path

from researchguard.memory.schemas import RunRecord
from researchguard.memory.storage import DEFAULT_MEMORY_ROOT, JsonlStorage


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class ResearchRunStore:
    def __init__(self, root: str | Path = DEFAULT_MEMORY_ROOT):
        self.root = Path(root)
        self.storage = JsonlStorage(self.root / "runs.jsonl")

    def save(self, record: RunRecord) -> RunRecord:
        self.storage.append(record.to_dict())
        return record

    def get(self, run_id: str) -> RunRecord | None:
        matches: list[RunRecord] = []
        for row in self.storage.read_all():
            if str(row.get("run_id")) != run_id:
                continue
            try:
                matches.append(RunRecord.from_dict(row))
            except (TypeError, ValueError):
                continue
        return matches[-1] if matches else None

    def list_runs(
        self,
        *,
        limit: int = 20,
        workflow: str | None = None,
        status: str | None = None,
    ) -> list[RunRecord]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        records = list(self._latest_by_run().values())
        if workflow is not None:
            records = [item for item in records if item.workflow_name == workflow]
        if status is not None:
            records = [item for item in records if item.status == status]
        records.sort(key=lambda item: item.updated_at, reverse=True)
        return records[:limit]

    def find_previous_runs(
        self,
        query: str,
        *,
        workflow: str | None = None,
        since: str | None = None,
        limit: int = 20,
    ) -> list[RunRecord]:
        tokens = set(TOKEN_RE.findall(str(query).casefold()))
        if not tokens:
            return []
        ranked: list[tuple[int, RunRecord]] = []
        for record in self._latest_by_run().values():
            if workflow is not None and record.workflow_name != workflow:
                continue
            if since is not None and record.updated_at < since:
                continue
            haystack = f"{record.query} {record.answer_summary or ''}".casefold()
            score = sum(token in haystack for token in tokens)
            if score:
                ranked.append((score, record))
        ranked.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return [record for _, record in ranked[:limit]]

    def _latest_by_run(self) -> dict[str, RunRecord]:
        records: dict[str, RunRecord] = {}
        for row in self.storage.read_all():
            try:
                record = RunRecord.from_dict(row)
            except (TypeError, ValueError):
                continue
            records[record.run_id] = record
        return records
