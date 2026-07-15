# C:\Users\18449\Desktop\researchguard_workspace\researchguard\indexing\corpus_loader.py
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "corpus_manifest_v1"
REQUIRED_FIELDS = (
    "chunk_id",
    "doc_id",
    "section",
    "page_start",
    "page_end",
    "source_block_ids",
    "text",
)
METADATA_HASH_FIELDS = (
    "chunk_id",
    "doc_id",
    "title",
    "section",
    "section_heading",
    "heading_path",
    "chunk_type",
    "page_start",
    "page_end",
    "source_block_ids",
    "heading_block_ids",
    "overlap_source_block_ids",
    "content_types",
    "has_equation",
    "has_table",
    "has_caption",
    "short_chunk",
    "char_count",
    "word_count",
)
MANIFEST_FIELDS = (
    "chunk_id",
    "doc_id",
    "title",
    "section",
    "section_heading",
    "heading_path",
    "chunk_type",
    "page_start",
    "page_end",
    "source_block_ids",
    "heading_block_ids",
    "overlap_source_block_ids",
    "content_types",
    "has_equation",
    "has_table",
    "has_caption",
    "short_chunk",
    "text",
    "char_count",
    "word_count",
)


class CorpusValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CorpusBuildResult:
    documents: list[dict[str, Any]]
    summary: dict[str, Any]
    validation_errors: list[dict[str, Any]]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise CorpusValidationError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_json_hash(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_bool(value: Any) -> bool:
    return bool(value)


def normalize_text(text: Any) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def source_path_for_paper(input_root: Path, paper: str, chunk_filename: str) -> Path:
    return input_root / paper / chunk_filename


def required_metadata_missing(chunk: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in REQUIRED_FIELDS:
        value = chunk.get(field)
        if field == "text":
            if not normalize_text(value):
                missing.append(field)
        elif field == "source_block_ids":
            if not normalize_list(value):
                missing.append(field)
        elif field in {"page_start", "page_end"}:
            if value is None:
                missing.append(field)
        elif value in (None, ""):
            missing.append(field)
    return missing


def manifest_document(chunk: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    text = normalize_text(chunk.get("text"))
    doc = {
        "chunk_id": str(chunk.get("chunk_id", "")),
        "doc_id": str(chunk.get("doc_id", "")),
        "title": str(chunk.get("title", "")),
        "section": str(chunk.get("section", "")),
        "section_heading": chunk.get("section_heading"),
        "heading_path": [str(item) for item in normalize_list(chunk.get("heading_path"))],
        "chunk_type": str(chunk.get("chunk_type", "")),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "source_block_ids": [str(item) for item in normalize_list(chunk.get("source_block_ids")) if str(item).strip()],
        "heading_block_ids": [str(item) for item in normalize_list(chunk.get("heading_block_ids")) if str(item).strip()],
        "overlap_source_block_ids": [
            str(item) for item in normalize_list(chunk.get("overlap_source_block_ids")) if str(item).strip()
        ],
        "content_types": [str(item) for item in normalize_list(chunk.get("content_types"))],
        "has_equation": normalize_bool(chunk.get("has_equation")),
        "has_table": normalize_bool(chunk.get("has_table")),
        "has_caption": normalize_bool(chunk.get("has_caption")),
        "short_chunk": normalize_bool(chunk.get("short_chunk")),
        "text": text,
        "char_count": int(chunk.get("char_count", len(text)) or 0),
        "word_count": int(chunk.get("word_count", len(text.split())) or 0),
    }
    content_hash = stable_json_hash({"text": text})
    metadata_hash = stable_json_hash({field: doc.get(field) for field in METADATA_HASH_FIELDS})
    doc.update(
        {
            "content_hash": content_hash,
            "metadata_hash": metadata_hash,
            "source_path": str(source_path),
            "schema_version": SCHEMA_VERSION,
        }
    )
    return doc


def validate_manifest_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for index, doc in enumerate(documents, start=1):
        missing = required_metadata_missing(doc)
        if missing:
            errors.append({"type": "required_metadata_missing", "chunk_id": doc.get("chunk_id"), "fields": missing})
        chunk_id = str(doc.get("chunk_id", ""))
        if chunk_id in seen:
            errors.append({"type": "duplicate_chunk_id", "chunk_id": chunk_id, "first_position": seen[chunk_id], "position": index})
        else:
            seen[chunk_id] = index
        if int(doc.get("char_count", 0)) != len(str(doc.get("text", ""))):
            errors.append({"type": "char_count_mismatch", "chunk_id": chunk_id})
        if int(doc.get("word_count", 0)) != len(str(doc.get("text", "")).split()):
            errors.append({"type": "word_count_mismatch", "chunk_id": chunk_id})
        try:
            if int(doc.get("page_start")) > int(doc.get("page_end")):
                errors.append({"type": "page_range_invalid", "chunk_id": chunk_id})
        except Exception:
            errors.append({"type": "page_range_invalid", "chunk_id": chunk_id})
        if set(doc.get("overlap_source_block_ids", [])) & set(doc.get("source_block_ids", [])):
            pass
    return errors


def summarize_documents(
    documents: list[dict[str, Any]],
    *,
    papers: list[str],
    source_file_hashes: dict[str, str],
    validation_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    section_counts = Counter(str(doc.get("section", "unknown")) for doc in documents)
    chunk_type_counts = Counter(str(doc.get("chunk_type", "unknown")) for doc in documents)
    per_paper = Counter(str(doc.get("doc_id", "unknown")) for doc in documents)
    content_hash_counts = Counter(str(doc.get("content_hash", "")) for doc in documents)
    duplicate_content_hashes = {
        key: count for key, count in content_hash_counts.items() if key and count > 1
    }
    missing_counts = Counter(error["type"] for error in validation_errors)
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_count": len(papers),
        "papers": papers,
        "chunk_count": len(documents),
        "chunks_per_paper": dict(sorted(per_paper.items())),
        "section_distribution": dict(sorted(section_counts.items())),
        "chunk_type_distribution": dict(sorted(chunk_type_counts.items())),
        "special_chunk_counts": {
            "has_equation": sum(1 for doc in documents if doc.get("has_equation")),
            "has_table": sum(1 for doc in documents if doc.get("has_table")),
            "has_caption": sum(1 for doc in documents if doc.get("has_caption")),
            "mixed": sum(1 for doc in documents if doc.get("chunk_type") == "mixed"),
            "references": sum(1 for doc in documents if doc.get("section") == "references"),
        },
        "short_chunk_count": sum(1 for doc in documents if doc.get("short_chunk") or int(doc.get("char_count", 0)) < 150),
        "metadata_missing_count": missing_counts.get("required_metadata_missing", 0),
        "duplicate_chunk_id_count": missing_counts.get("duplicate_chunk_id", 0),
        "content_hash_duplicate_count": len(duplicate_content_hashes),
        "content_hash_duplicate_examples": dict(list(duplicate_content_hashes.items())[:20]),
        "total_char_count": sum(int(doc.get("char_count", 0)) for doc in documents),
        "total_word_count": sum(int(doc.get("word_count", 0)) for doc in documents),
        "source_file_hashes": source_file_hashes,
        "corpus_fingerprint": corpus_fingerprint(documents),
        "validation_error_counts": dict(sorted(missing_counts.items())),
    }


def corpus_fingerprint(documents: list[dict[str, Any]]) -> str:
    payload = [
        {
            "chunk_id": doc.get("chunk_id"),
            "content_hash": doc.get("content_hash"),
            "metadata_hash": doc.get("metadata_hash"),
        }
        for doc in sorted(documents, key=lambda item: str(item.get("chunk_id", "")))
    ]
    return stable_json_hash(payload)


def build_corpus_manifest(config: dict[str, Any], *, input_root_override: Path | None = None) -> CorpusBuildResult:
    corpus = config.get("corpus", {}) or {}
    input_root = Path(input_root_override or corpus.get("input_root", "data/parsed/chunk_eval_v1"))
    chunk_filename = str(corpus.get("chunk_filename", "chunks.jsonl"))
    papers = [str(paper) for paper in corpus.get("papers", [])]
    if not papers:
        papers = sorted(path.name for path in input_root.iterdir() if path.is_dir())

    documents: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for paper in papers:
        chunk_path = source_path_for_paper(input_root, paper, chunk_filename)
        rows = read_jsonl(chunk_path)
        source_hashes[str(chunk_path)] = file_hash(chunk_path)
        for row in rows:
            documents.append(manifest_document(row, source_path=chunk_path))

    documents.sort(key=lambda item: (str(item.get("doc_id", "")), str(item.get("chunk_id", ""))))
    errors = validate_manifest_documents(documents)
    summary = summarize_documents(
        documents,
        papers=papers,
        source_file_hashes=source_hashes,
        validation_errors=errors,
    )
    return CorpusBuildResult(documents=documents, summary=summary, validation_errors=errors)


def write_corpus_outputs(result: CorpusBuildResult, output_dir: Path) -> None:
    write_jsonl(output_dir / "corpus_manifest.jsonl", result.documents)
    write_json(output_dir / "corpus_summary.json", result.summary)
