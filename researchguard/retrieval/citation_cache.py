# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\citation_cache.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from researchguard.indexing.corpus_loader import stable_json_hash
from researchguard.retrieval.claim_extractor import CitationAuditSettings


class CitationAuditCache:
    def __init__(self, directory: Path, *, enabled: bool = True):
        self.directory = directory
        self.enabled = enabled

    @staticmethod
    def make_key(
        *,
        answer_hash: str,
        evidence_chunk_ids: list[str],
        input_hash: str,
        settings: CitationAuditSettings,
    ) -> str:
        return stable_json_hash(
            {
                "answer_hash": answer_hash,
                "extraction_prompt_version": settings.extraction_prompt_version,
                "verification_prompt_version": settings.verification_prompt_version,
                "evidence_chunk_ids": evidence_chunk_ids,
                "model": settings.model,
                "config_version": settings.config_version,
                "audit_config": settings.cache_identity,
                "input_hash": input_hash,
            }
        )

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str, *, input_hash: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("cache_key") != key
                or payload.get("input_hash") != input_hash
                or not isinstance(payload.get("output"), dict)
            ):
                return None
            return dict(payload["output"])
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def put(
        self,
        key: str,
        *,
        input_hash: str,
        output: dict[str, Any],
        timestamp: str,
    ) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_key": key,
            "input_hash": input_hash,
            "output": output,
            "timestamp": timestamp,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
