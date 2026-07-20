# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\rerank_cache.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from researchguard.indexing.corpus_loader import stable_json_hash
from researchguard.retrieval.reranker import RerankerSettings


class RerankCache:
    def __init__(self, directory: Path, *, enabled: bool = True):
        self.directory = directory
        self.enabled = enabled

    @staticmethod
    def make_key(
        *,
        query: str,
        content_hash: str,
        metadata_hash: str = "",
        settings: RerankerSettings,
    ) -> str:
        return stable_json_hash(
            {
                "query": str(query).strip(),
                "content_hash": str(content_hash),
                "metadata_hash": str(metadata_hash),
                "reranker_model": settings.model_identity,
                "reranker_config_version": settings.config_version,
                "input_template_version": settings.input_template_version,
            }
        )

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> float | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            score = float(payload["score"])
            if payload.get("cache_key") != key or not math.isfinite(score):
                return None
            return score
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def put(self, key: str, score: float, *, metadata: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cache_key": key, "score": float(score), **metadata}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
