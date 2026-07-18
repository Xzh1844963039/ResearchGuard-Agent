# C:\Users\18449\Desktop\researchguard_workspace\researchguard\indexing\chroma_metadata.py
from __future__ import annotations

import json
from typing import Any


COLLECTION_SCHEMA_VERSION = "chroma_collection_v1"
LIST_FIELDS = (
    "heading_path",
    "source_block_ids",
    "heading_block_ids",
    "overlap_source_block_ids",
    "content_types",
)
SCALAR_FIELDS = (
    "chunk_id",
    "doc_id",
    "title",
    "section",
    "section_heading",
    "chunk_type",
    "page_start",
    "page_end",
    "has_equation",
    "has_table",
    "has_caption",
    "short_chunk",
    "char_count",
    "word_count",
    "content_hash",
    "metadata_hash",
    "schema_version",
)


class ChromaMetadataError(ValueError):
    pass


def stable_json_list(value: Any) -> str:
    items = value if isinstance(value, list) else ([] if value is None else [value])
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def parse_json_list(value: Any, *, field: str) -> list[Any]:
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ChromaMetadataError(f"Invalid JSON list in {field}: {value!r}") from exc
    if not isinstance(parsed, list):
        raise ChromaMetadataError(f"Expected a JSON list in {field}.")
    return parsed


def encode_record_metadata(document: dict[str, Any], *, corpus_fingerprint: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field in SCALAR_FIELDS:
        value = document.get(field)
        if value is None:
            value = "" if field == "section_heading" else 0
        if not isinstance(value, (str, int, float, bool)):
            raise ChromaMetadataError(f"Unsupported scalar metadata value for {field}: {type(value).__name__}")
        metadata[field] = value
    for field in LIST_FIELDS:
        metadata[f"{field}_json"] = stable_json_list(document.get(field, []))
    metadata["corpus_fingerprint"] = str(corpus_fingerprint)
    return metadata


def decode_record_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(metadata or {})
    decoded = {field: raw.get(field) for field in SCALAR_FIELDS if field in raw}
    if decoded.get("section_heading") == "":
        decoded["section_heading"] = None
    for field in LIST_FIELDS:
        decoded[field] = parse_json_list(raw.get(f"{field}_json"), field=f"{field}_json")
    decoded["corpus_fingerprint"] = raw.get("corpus_fingerprint")
    return decoded


def build_collection_metadata(index_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "corpus_fingerprint": str(index_manifest.get("corpus_fingerprint", "")),
        "embedding_provider": str(index_manifest.get("embedding_provider", "")),
        "embedding_model": str(index_manifest.get("embedding_model", "")),
        "embedding_dimensions": int(index_manifest.get("embedding_dimensions", 0)),
        "distance_metric": str(index_manifest.get("dense_metric", "cosine")),
        "source_index_backend": str(index_manifest.get("dense_backend", "numpy")),
        "build_timestamp": str(index_manifest.get("build_timestamp", "")),
    }


def validate_collection_metadata(
    actual: dict[str, Any] | None,
    expected: dict[str, Any],
    *,
    check_fingerprint: bool,
) -> list[str]:
    actual = actual or {}
    fields = [
        "schema_version",
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "distance_metric",
        "source_index_backend",
    ]
    if check_fingerprint:
        fields.append("corpus_fingerprint")
    return [field for field in fields if actual.get(field) != expected.get(field)]
