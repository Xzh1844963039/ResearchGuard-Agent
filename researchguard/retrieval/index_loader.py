# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\index_loader.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from researchguard.indexing.corpus_loader import corpus_fingerprint, load_yaml, read_jsonl
from researchguard.indexing.dense_index import DenseNumpyIndex
from researchguard.indexing.sparse_index import LocalBM25Index
from researchguard.retrieval.models import RetrievalError


@dataclass
class RetrievalIndexBundle:
    config: dict[str, Any]
    indexing_config: dict[str, Any]
    index_dir: Path
    manifest: dict[str, Any]
    dense_manifest: dict[str, Any]
    sparse_payload: dict[str, Any]
    documents: list[dict[str, Any]]
    document_by_id: dict[str, dict[str, Any]]
    dense_index: DenseNumpyIndex
    sparse_index: LocalBM25Index
    hard_checks: dict[str, int]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RetrievalError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_retrieval_config(path: Path) -> dict[str, Any]:
    return load_yaml(path)


def load_index_bundle(config_path: Path, *, strict: bool = True) -> RetrievalIndexBundle:
    config = load_retrieval_config(config_path)
    index_cfg = config.get("index", {}) or {}
    index_dir = Path(index_cfg.get("output_dir", "data/indexes/index_v1"))
    indexing_config_path = Path(index_cfg.get("indexing_config_path", "configs/indexing_v1.yaml"))
    indexing_config = load_yaml(indexing_config_path)

    manifest = read_json(index_dir / "index_manifest.json")
    dense_manifest = read_json(index_dir / "dense" / "dense_manifest.json")
    sparse_payload = read_json(index_dir / "sparse" / "bm25_index.json")
    documents = read_jsonl(index_dir / "corpus_manifest.jsonl")
    dense_index = DenseNumpyIndex.load(index_dir / "dense")
    sparse_index = LocalBM25Index.load(index_dir / "sparse")
    document_by_id = {str(doc.get("chunk_id")): doc for doc in documents}

    hard_checks = validate_loaded_index(
        manifest=manifest,
        dense_manifest=dense_manifest,
        sparse_payload=sparse_payload,
        documents=documents,
        dense_index=dense_index,
        sparse_index=sparse_index,
    )
    if strict:
        failures = {key: value for key, value in hard_checks.items() if value}
        if failures:
            raise RetrievalError(f"Index hard checks failed: {failures}")

    return RetrievalIndexBundle(
        config=config,
        indexing_config=indexing_config,
        index_dir=index_dir,
        manifest=manifest,
        dense_manifest=dense_manifest,
        sparse_payload=sparse_payload,
        documents=documents,
        document_by_id=document_by_id,
        dense_index=dense_index,
        sparse_index=sparse_index,
        hard_checks=hard_checks,
    )


def validate_loaded_index(
    *,
    manifest: dict[str, Any],
    dense_manifest: dict[str, Any],
    sparse_payload: dict[str, Any],
    documents: list[dict[str, Any]],
    dense_index: DenseNumpyIndex,
    sparse_index: LocalBM25Index,
) -> dict[str, int]:
    checks = {
        "index_load_failure": 0,
        "fingerprint_mismatch": 0,
        "chunk_id_mapping_mismatch": 0,
        "metadata_missing": 0,
        "dense_dimension_mismatch": 0,
        "duplicate_chunk_id": 0,
        "schema_error": 0,
    }

    if manifest.get("build_status") != "complete":
        checks["schema_error"] += 1
    if manifest.get("schema_version") != "index_manifest_v1":
        checks["schema_error"] += 1
    if dense_manifest.get("schema_version") != "dense_numpy_v1":
        checks["schema_error"] += 1
    if sparse_payload.get("schema_version") != "local_bm25_v1":
        checks["schema_error"] += 1

    chunk_ids = [str(doc.get("chunk_id", "")) for doc in documents]
    if len(set(chunk_ids)) != len(chunk_ids):
        checks["duplicate_chunk_id"] += len(chunk_ids) - len(set(chunk_ids))
    if len(chunk_ids) != int(manifest.get("chunk_count", -1)):
        checks["chunk_id_mapping_mismatch"] += 1
    if chunk_ids != list(dense_index.chunk_ids):
        checks["chunk_id_mapping_mismatch"] += 1
    if chunk_ids != list(sparse_index.chunk_ids):
        checks["chunk_id_mapping_mismatch"] += 1

    required = ("chunk_id", "doc_id", "section", "chunk_type", "page_start", "page_end", "text", "source_block_ids")
    for doc in documents:
        for field in required:
            value = doc.get(field)
            if value in (None, "", []):
                checks["metadata_missing"] += 1

    if dense_index.dimension != int(manifest.get("embedding_dimensions", -1)):
        checks["dense_dimension_mismatch"] += 1
    if dense_index.dimension != int(dense_manifest.get("dimension", -1)):
        checks["dense_dimension_mismatch"] += 1

    actual_fingerprint = corpus_fingerprint(documents)
    expected = str(manifest.get("corpus_fingerprint", ""))
    if actual_fingerprint != expected:
        checks["fingerprint_mismatch"] += 1
    if str(dense_manifest.get("corpus_fingerprint", "")) != expected:
        checks["fingerprint_mismatch"] += 1
    sparse_manifest = sparse_payload.get("manifest", {}) or {}
    if str(sparse_manifest.get("corpus_fingerprint", "")) != expected:
        checks["fingerprint_mismatch"] += 1

    return checks
