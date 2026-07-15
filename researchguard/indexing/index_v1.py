# C:\Users\18449\Desktop\researchguard_workspace\researchguard\indexing\index_v1.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from researchguard.indexing.corpus_loader import (
    CorpusBuildResult,
    build_corpus_manifest,
    corpus_fingerprint,
    load_yaml,
    read_jsonl,
    write_corpus_outputs,
    write_json,
)
from researchguard.indexing.dense_index import DenseNumpyIndex
from researchguard.indexing.embedding_cache import EmbeddingCache
from researchguard.indexing.embedding_provider import OpenAIEmbeddingProvider, parse_embedding_config
from researchguard.indexing.sparse_index import LocalBM25Index


INDEX_SCHEMA_VERSION = "index_manifest_v1"


class IndexBuildError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_output_dir(config: dict[str, Any], output_dir_override: Path | None = None) -> Path:
    build = config.get("build", {}) or {}
    return Path(output_dir_override or build.get("output_dir", "data/indexes/index_v1"))


def resolve_dense_dir(config: dict[str, Any], output_dir: Path) -> Path:
    dense = config.get("dense_index", {}) or {}
    return Path(dense.get("output_dir") or output_dir / "dense")


def resolve_sparse_dir(config: dict[str, Any], output_dir: Path) -> Path:
    sparse = config.get("sparse_index", {}) or {}
    return Path(sparse.get("output_dir") or output_dir / "sparse")


def load_existing_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def plan_incremental(current: list[dict[str, Any]], previous: list[dict[str, Any]]) -> dict[str, Any]:
    current_by_id = {str(row.get("chunk_id")): row for row in current}
    previous_by_id = {str(row.get("chunk_id")): row for row in previous}
    added: list[str] = []
    updated: list[str] = []
    metadata_changed: list[str] = []
    reused: list[str] = []
    removed: list[str] = []

    for chunk_id, row in current_by_id.items():
        old = previous_by_id.get(chunk_id)
        if old is None:
            added.append(chunk_id)
        elif old.get("content_hash") != row.get("content_hash"):
            updated.append(chunk_id)
        elif old.get("metadata_hash") != row.get("metadata_hash"):
            metadata_changed.append(chunk_id)
            reused.append(chunk_id)
        else:
            reused.append(chunk_id)

    for chunk_id in previous_by_id:
        if chunk_id not in current_by_id:
            removed.append(chunk_id)

    return {
        "added": len(added),
        "updated": len(updated),
        "metadata_changed": len(metadata_changed),
        "reused": len(reused),
        "removed": len(removed),
        "added_chunk_ids": added[:50],
        "updated_chunk_ids": updated[:50],
        "metadata_changed_chunk_ids": metadata_changed[:50],
        "removed_chunk_ids": removed[:50],
    }


def prepare_build(
    config_path: Path,
    *,
    input_root_override: Path | None = None,
    output_dir_override: Path | None = None,
) -> tuple[dict[str, Any], Path, CorpusBuildResult, dict[str, Any]]:
    config = load_yaml(config_path)
    output_dir = resolve_output_dir(config, output_dir_override)
    result = build_corpus_manifest(config, input_root_override=input_root_override)
    previous = load_existing_manifest(output_dir / "corpus_manifest.jsonl")
    plan = plan_incremental(result.documents, previous)
    return config, output_dir, result, plan


def dry_run(
    config_path: Path,
    *,
    input_root_override: Path | None = None,
    output_dir_override: Path | None = None,
) -> dict[str, Any]:
    config, output_dir, result, plan = prepare_build(
        config_path,
        input_root_override=input_root_override,
        output_dir_override=output_dir_override,
    )
    embedding = parse_embedding_config(config)
    return {
        "status": "dry_run",
        "output_dir": str(output_dir),
        "paper_count": result.summary["paper_count"],
        "chunk_count": result.summary["chunk_count"],
        "corpus_fingerprint": result.summary["corpus_fingerprint"],
        "validation_error_count": len(result.validation_errors),
        "validation_error_examples": result.validation_errors[:20],
        "embedding_provider": embedding.provider,
        "embedding_model": embedding.model,
        "embedding_dimensions": embedding.dimensions,
        **plan,
    }


def write_index_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    write_json(output_dir / "index_manifest.json", payload)


def base_index_manifest(
    *,
    config_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    corpus_result: CorpusBuildResult,
    build_status: str,
) -> dict[str, Any]:
    embedding = parse_embedding_config(config)
    dense = config.get("dense_index", {}) or {}
    sparse = config.get("sparse_index", {}) or {}
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "corpus_schema_version": corpus_result.summary["schema_version"],
        "corpus_fingerprint": corpus_result.summary["corpus_fingerprint"],
        "embedding_provider": embedding.provider,
        "embedding_model": embedding.model,
        "embedding_dimensions": embedding.dimensions,
        "normalize": embedding.normalize,
        "dense_backend": str(dense.get("backend", "numpy")),
        "dense_metric": str(dense.get("metric", "cosine")),
        "dense_index_path": str(resolve_dense_dir(config, output_dir)),
        "sparse_enabled": bool(sparse.get("enabled", True)),
        "sparse_backend": str(sparse.get("backend", "local_bm25")),
        "sparse_index_path": str(resolve_sparse_dir(config, output_dir)),
        "paper_count": corpus_result.summary["paper_count"],
        "chunk_count": corpus_result.summary["chunk_count"],
        "build_timestamp": utc_now(),
        "config_path": str(config_path),
        "manifest_path": str(output_dir / "corpus_manifest.jsonl"),
        "cache_hits": 0,
        "cache_misses": 0,
        "added": 0,
        "updated": 0,
        "reused": 0,
        "removed": 0,
        "embedded": 0,
        "build_status": build_status,
    }


def embed_documents_with_cache(
    *,
    documents: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    embedding_config = parse_embedding_config(config)
    cache_enabled = bool((config.get("embedding", {}) or {}).get("cache_enabled", True))
    cache = EmbeddingCache(output_dir / "embedding_cache")
    provider = OpenAIEmbeddingProvider(embedding_config)

    vectors_by_cache_key: dict[str, list[float]] = {}
    missing_by_cache_key: dict[str, dict[str, Any]] = {}
    keys_by_document: list[str] = []

    for doc in documents:
        key = EmbeddingCache.make_key(
            provider=embedding_config.provider,
            model=embedding_config.model,
            content_hash=str(doc["content_hash"]),
        )
        keys_by_document.append(key)
        cached = cache.get(
            provider=embedding_config.provider,
            model=embedding_config.model,
            content_hash=str(doc["content_hash"]),
        ) if cache_enabled else None
        if cached is not None:
            vectors_by_cache_key[key] = cached
        elif key not in missing_by_cache_key:
            missing_by_cache_key[key] = doc

    missing_docs = list(missing_by_cache_key.values())
    embedded_count = 0
    if missing_docs:
        texts = [str(doc["text"]) for doc in missing_docs]
        new_vectors = provider.embed_documents(texts)
        embedded_count = len(new_vectors)
        for doc, vector in zip(missing_docs, new_vectors):
            key = EmbeddingCache.make_key(
                provider=embedding_config.provider,
                model=embedding_config.model,
                content_hash=str(doc["content_hash"]),
            )
            vectors_by_cache_key[key] = vector
            if cache_enabled:
                cache.set(
                    provider=embedding_config.provider,
                    model=embedding_config.model,
                    content_hash=str(doc["content_hash"]),
                    embedding=vector,
                    dimensions=embedding_config.dimensions,
                )

    if cache_enabled:
        cache.save()

    vectors = [vectors_by_cache_key[key] for key in keys_by_document]
    array = np.asarray(vectors, dtype="float32")
    stats = cache.stats()
    stats.update({"embedded": embedded_count, "cache_enabled": cache_enabled})
    return array, stats


def build_index(
    config_path: Path,
    *,
    input_root_override: Path | None = None,
    output_dir_override: Path | None = None,
    force_rebuild: bool | None = None,
    incremental: bool | None = None,
) -> dict[str, Any]:
    config, output_dir, corpus_result, plan = prepare_build(
        config_path,
        input_root_override=input_root_override,
        output_dir_override=output_dir_override,
    )
    build_config = config.get("build", {}) or {}
    if force_rebuild is not None:
        build_config["force_rebuild"] = force_rebuild
    if incremental is not None:
        build_config["incremental"] = incremental

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = base_index_manifest(
        config_path=config_path,
        output_dir=output_dir,
        config=config,
        corpus_result=corpus_result,
        build_status="building",
    )
    manifest.update(plan)
    write_index_manifest(output_dir, manifest)

    if corpus_result.validation_errors and (
        build_config.get("fail_on_invalid_metadata", True)
        or build_config.get("fail_on_duplicate_chunk_id", True)
    ):
        manifest["build_status"] = "failed"
        manifest["failure_reason"] = "corpus validation failed"
        manifest["validation_errors"] = corpus_result.validation_errors[:50]
        write_index_manifest(output_dir, manifest)
        raise IndexBuildError("Corpus validation failed; refusing to build index.")

    write_corpus_outputs(corpus_result, output_dir)

    vectors, cache_stats = embed_documents_with_cache(
        documents=corpus_result.documents,
        config=config,
        output_dir=output_dir,
    )
    dense = config.get("dense_index", {}) or {}
    dense_index = DenseNumpyIndex(
        chunk_ids=[str(doc["chunk_id"]) for doc in corpus_result.documents],
        vectors=vectors,
        metadata=[
            {
                key: doc.get(key)
                for key in (
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
                    "content_hash",
                    "metadata_hash",
                )
            }
            for doc in corpus_result.documents
        ],
        metric=str(dense.get("metric", "cosine")),
    )
    dense_index.save(
        resolve_dense_dir(config, output_dir),
        manifest={
            "embedding_model": parse_embedding_config(config).model,
            "corpus_fingerprint": corpus_result.summary["corpus_fingerprint"],
        },
    )

    sparse = config.get("sparse_index", {}) or {}
    if bool(sparse.get("enabled", True)):
        sparse_index = LocalBM25Index.build(corpus_result.documents)
        sparse_index.save(
            resolve_sparse_dir(config, output_dir),
            manifest={"corpus_fingerprint": corpus_result.summary["corpus_fingerprint"]},
        )

    complete_manifest = base_index_manifest(
        config_path=config_path,
        output_dir=output_dir,
        config=config,
        corpus_result=corpus_result,
        build_status="complete",
    )
    complete_manifest.update(plan)
    complete_manifest.update(
        {
            "cache_hits": int(cache_stats.get("hits", 0)),
            "cache_misses": int(cache_stats.get("misses", 0)),
            "cache_entries": int(cache_stats.get("entries", 0)),
            "embedded": int(cache_stats.get("embedded", 0)),
            "force_rebuild": bool(build_config.get("force_rebuild", False)),
            "incremental": bool(build_config.get("incremental", True)),
        }
    )
    write_index_manifest(output_dir, complete_manifest)
    write_json(
        output_dir / "build_report.json",
        {
            "status": "complete",
            "summary": {
                "paper_count": complete_manifest["paper_count"],
                "chunk_count": complete_manifest["chunk_count"],
                "embedding_model": complete_manifest["embedding_model"],
                "embedding_dimensions": complete_manifest["embedding_dimensions"],
                "dense_backend": complete_manifest["dense_backend"],
                "sparse_backend": complete_manifest["sparse_backend"],
                "cache_hits": complete_manifest["cache_hits"],
                "cache_misses": complete_manifest["cache_misses"],
                "added": complete_manifest["added"],
                "updated": complete_manifest["updated"],
                "removed": complete_manifest["removed"],
                "output_dir": str(output_dir),
            },
            "corpus_summary": corpus_result.summary,
        },
    )
    return complete_manifest


def validate_existing_index(config_path: Path, *, output_dir_override: Path | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    output_dir = resolve_output_dir(config, output_dir_override)
    manifest_path = output_dir / "index_manifest.json"
    if not manifest_path.exists():
        raise IndexBuildError(f"Index manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dense_index = DenseNumpyIndex.load(resolve_dense_dir(config, output_dir))
    sparse = config.get("sparse_index", {}) or {}
    sparse_loaded = False
    if bool(sparse.get("enabled", True)):
        LocalBM25Index.load(resolve_sparse_dir(config, output_dir))
        sparse_loaded = True
    current_manifest = load_existing_manifest(output_dir / "corpus_manifest.jsonl")
    return {
        "status": "validate_only",
        "build_status": manifest.get("build_status"),
        "chunk_count": len(current_manifest),
        "dense_vector_count": len(dense_index.chunk_ids),
        "dense_dimension": dense_index.dimension,
        "sparse_loaded": sparse_loaded,
        "corpus_fingerprint": corpus_fingerprint(current_manifest),
        "manifest_corpus_fingerprint": manifest.get("corpus_fingerprint"),
    }
