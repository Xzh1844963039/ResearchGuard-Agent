# C:\Users\18449\Desktop\researchguard_workspace\researchguard\indexing\embedding_cache.py
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class EmbeddingCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_path = cache_dir / "embedding_cache.jsonl"
        self.entries: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.load()

    @staticmethod
    def make_key(*, provider: str, model: str, content_hash: str) -> str:
        return f"{provider}:{model}:{content_hash}"

    def load(self) -> None:
        self.entries = {}
        if not self.cache_path.exists():
            return
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                key = str(row.get("cache_key", ""))
                if key:
                    self.entries[key] = row

    def get(self, *, provider: str, model: str, content_hash: str) -> list[float] | None:
        key = self.make_key(provider=provider, model=model, content_hash=content_hash)
        entry = self.entries.get(key)
        if not entry:
            self.misses += 1
            return None
        vector = entry.get("embedding")
        if not isinstance(vector, list):
            self.misses += 1
            return None
        self.hits += 1
        return [float(value) for value in vector]

    def set(
        self,
        *,
        provider: str,
        model: str,
        content_hash: str,
        embedding: list[float],
        dimensions: int,
    ) -> None:
        key = self.make_key(provider=provider, model=model, content_hash=content_hash)
        self.entries[key] = {
            "cache_key": key,
            "provider": provider,
            "model": model,
            "content_hash": content_hash,
            "dimensions": dimensions,
            "embedding": [float(value) for value in embedding],
        }

    def save(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="embedding_cache_", suffix=".jsonl.tmp", dir=str(self.cache_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                for key in sorted(self.entries):
                    handle.write(json.dumps(self.entries[key], ensure_ascii=False, sort_keys=True) + "\n")
            os.replace(temp_name, self.cache_path)
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
        }
