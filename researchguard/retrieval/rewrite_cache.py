# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\rewrite_cache.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from researchguard.indexing.corpus_loader import stable_json_hash
from researchguard.retrieval.query_rewriter import QueryRewriteSettings, normalize_query_text


class QueryRewriteCache:
    def __init__(self, directory: Path, *, enabled: bool = True):
        self.directory = directory
        self.enabled = enabled

    @staticmethod
    def make_key(*, original_query: str, settings: QueryRewriteSettings) -> str:
        return stable_json_hash(
            {
                "original_query": normalize_query_text(original_query),
                "rewrite_config": settings.cache_identity,
            }
        )

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("cache_key") != key or not isinstance(payload.get("result"), dict):
                return None
            return dict(payload["result"])
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def put(self, key: str, result: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cache_key": key, "result": result}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
