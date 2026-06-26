# C:\Users\18449\Desktop\researchguard_workspace\researchguard\utils_ids.py
from __future__ import annotations


def make_id(prefix: str, index: int, width: int = 3) -> str:
    return f"{prefix}{index:0{width}d}"


def next_id(prefix: str, records: list[dict], key: str, width: int = 3) -> str:
    max_seen = 0
    for record in records:
        value = str(record.get(key, ""))
        if value.startswith(prefix):
            suffix = value[len(prefix) :]
            if suffix.isdigit():
                max_seen = max(max_seen, int(suffix))
    return make_id(prefix, max_seen + 1, width)

