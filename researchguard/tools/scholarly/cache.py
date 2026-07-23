# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tools\scholarly\cache.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from researchguard.tools.scholarly.base import ScholarPaperRecord, utc_timestamp


class ScholarlySearchCache:
    def __init__(self, directory: str | Path, *, enabled: bool = True):
        self.directory = Path(directory)
        self.enabled = enabled

    @staticmethod
    def make_key(
        *,
        query: str,
        provider: str,
        config_version: str,
        limit: int,
    ) -> str:
        payload = {
            "query": query,
            "provider": provider,
            "config_version": config_version,
            "limit": limit,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(
        self,
        key: str,
        *,
        request: Mapping[str, Any],
    ) -> list[ScholarPaperRecord] | None:
        if not self.enabled:
            return None
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("request") != dict(request):
                return None
            response = payload.get("response")
            if not isinstance(response, list):
                return None
            return [ScholarPaperRecord.from_dict(item) for item in response]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def set(
        self,
        key: str,
        *,
        request: Mapping[str, Any],
        records: list[ScholarPaperRecord],
    ) -> None:
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        temp_path = self.directory / f".{key}.tmp"
        payload = {
            "request": dict(request),
            "response": [record.to_dict() for record in records],
            "timestamp": utc_timestamp(),
        }
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
