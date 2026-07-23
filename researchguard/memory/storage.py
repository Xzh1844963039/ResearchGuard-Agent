# C:\Users\18449\Desktop\researchguard_workspace\researchguard\memory\storage.py
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_ROOT = PROJECT_ROOT / "data" / "memory" / "research_runs"
_JSONL_LOCK = threading.RLock()


class JsonlStorage:
    """Small append-only JSONL store that tolerates interrupted trailing writes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(record)
        line = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with _JSONL_LOCK:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return row

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with _JSONL_LOCK:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows
