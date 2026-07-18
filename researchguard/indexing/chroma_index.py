# C:\Users\18449\Desktop\researchguard_workspace\researchguard\indexing\chroma_index.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import chromadb
import numpy as np

from researchguard.indexing.chroma_metadata import (
    build_collection_metadata,
    encode_record_metadata,
    validate_collection_metadata,
)
from researchguard.indexing.corpus_loader import corpus_fingerprint, load_yaml, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ChromaIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChromaSettings:
    persist_directory: Path
    collection_name: str
    distance_metric: str
    batch_size: int
    allow_reset: bool
    source_index_directory: Path
    corpus_manifest_path: Path
    vectors_path: Path
    ids_path: Path
    dense_manifest_path: Path
    index_manifest_path: Path
    delete_stale: bool
    large_delete_ratio: float
    validation_output_directory: Path


@dataclass
class ChromaSourceIndex:
    documents: list[dict[str, Any]]
    chunk_ids: list[str]
    vectors: np.ndarray
    index_manifest: dict[str, Any]
    dense_manifest: dict[str, Any]
    hard_checks: dict[str, int]

    @property
    def corpus_fingerprint(self) -> str:
        return str(self.index_manifest.get("corpus_fingerprint", ""))


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_chroma_settings(config_path: str | Path) -> tuple[dict[str, Any], ChromaSettings]:
    config = load_yaml(Path(config_path))
    chroma_cfg = config.get("chroma", {}) or {}
    source_cfg = config.get("source", {}) or {}
    sync_cfg = config.get("sync", {}) or {}
    validation_cfg = config.get("validation", {}) or {}
    source_dir = resolve_project_path(source_cfg.get("index_directory", "data/indexes/index_v1"))
    settings = ChromaSettings(
        persist_directory=resolve_project_path(chroma_cfg.get("persist_directory", "data/indexes/chroma_v1")),
        collection_name=str(chroma_cfg.get("collection_name", "researchguard_papers_v1")),
        distance_metric=str(chroma_cfg.get("distance_metric", "cosine")),
        batch_size=max(1, int(chroma_cfg.get("batch_size", 100))),
        allow_reset=bool(chroma_cfg.get("allow_reset", False)),
        source_index_directory=source_dir,
        corpus_manifest_path=source_dir / str(source_cfg.get("corpus_manifest", "corpus_manifest.jsonl")),
        vectors_path=source_dir / str(source_cfg.get("vectors_file", "dense/vectors.npy")),
        ids_path=source_dir / str(source_cfg.get("ids_file", "dense/ids.json")),
        dense_manifest_path=source_dir / str(source_cfg.get("dense_manifest", "dense/dense_manifest.json")),
        index_manifest_path=source_dir / str(source_cfg.get("index_manifest", "index_manifest.json")),
        delete_stale=bool(sync_cfg.get("delete_stale", True)),
        large_delete_ratio=float(sync_cfg.get("large_delete_ratio", 0.10)),
        validation_output_directory=resolve_project_path(
            validation_cfg.get("output_directory", "outputs/chroma_validation_v1")
        ),
    )
    if settings.distance_metric != "cosine":
        raise ChromaIndexError("Chroma v1 currently requires cosine distance to match the NumPy baseline.")
    if settings.allow_reset:
        raise ChromaIndexError("allow_reset must remain false for Chroma v1.")
    return config, settings


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise ChromaIndexError(f"Required source index file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_index(settings: ChromaSettings, *, strict: bool = True) -> ChromaSourceIndex:
    documents = read_jsonl(settings.corpus_manifest_path)
    chunk_ids = [str(item) for item in _read_json(settings.ids_path)]
    vectors = np.load(settings.vectors_path)
    index_manifest = dict(_read_json(settings.index_manifest_path))
    dense_manifest = dict(_read_json(settings.dense_manifest_path))
    document_ids = [str(doc.get("chunk_id", "")) for doc in documents]
    expected_count = int(index_manifest.get("chunk_count", -1))
    expected_dimension = int(index_manifest.get("embedding_dimensions", -1))
    actual_fingerprint = corpus_fingerprint(documents)
    expected_fingerprint = str(index_manifest.get("corpus_fingerprint", ""))
    norms = np.linalg.norm(vectors, axis=1) if vectors.ndim == 2 else np.asarray([])
    hard_checks = {
        "source_count_mismatch": int(
            len(documents) != expected_count or len(chunk_ids) != expected_count or len(vectors) != expected_count
        ),
        "duplicate_record_id": len(chunk_ids) - len(set(chunk_ids)),
        "chunk_id_mapping_mismatch": int(document_ids != chunk_ids),
        "embedding_dimension_mismatch": int(
            vectors.ndim != 2
            or (vectors.ndim == 2 and vectors.shape[1] != expected_dimension)
            or int(dense_manifest.get("dimension", -1)) != expected_dimension
        ),
        "invalid_embedding": int(
            vectors.ndim != 2
            or not np.isfinite(vectors).all()
            or (norms.size > 0 and bool(np.any(norms == 0)))
        ),
        "corpus_fingerprint_mismatch": int(
            actual_fingerprint != expected_fingerprint
            or str(dense_manifest.get("corpus_fingerprint", "")) != expected_fingerprint
        ),
        "source_manifest_incomplete": int(index_manifest.get("build_status") != "complete"),
    }
    source = ChromaSourceIndex(
        documents=documents,
        chunk_ids=chunk_ids,
        vectors=np.asarray(vectors, dtype="float32"),
        index_manifest=index_manifest,
        dense_manifest=dense_manifest,
        hard_checks=hard_checks,
    )
    if strict:
        failures = {key: value for key, value in hard_checks.items() if value}
        if failures:
            raise ChromaIndexError(f"Source index hard checks failed: {failures}")
    return source


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class ChromaIndexManager:
    def __init__(self, settings: ChromaSettings):
        self.settings = settings

    def client(self):
        self.settings.persist_directory.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self.settings.persist_directory))

    def collection_exists(self, client: Any) -> bool:
        return self.settings.collection_name in {item.name for item in client.list_collections()}

    def get_collection(self, *, strict_fingerprint: bool, source: ChromaSourceIndex):
        client = self.client()
        if not self.collection_exists(client):
            raise ChromaIndexError(f"Chroma collection not found: {self.settings.collection_name}")
        collection = client.get_collection(name=self.settings.collection_name, embedding_function=None)
        self.validate_collection(collection, source=source, strict_fingerprint=strict_fingerprint)
        return client, collection

    def validate_collection(self, collection: Any, *, source: ChromaSourceIndex, strict_fingerprint: bool) -> None:
        expected = build_collection_metadata(source.index_manifest)
        mismatches = validate_collection_metadata(
            collection.metadata,
            expected,
            check_fingerprint=strict_fingerprint,
        )
        configuration = collection.configuration_json or {}
        actual_space = ((configuration.get("hnsw") or {}).get("space"))
        if actual_space != self.settings.distance_metric:
            mismatches.append("configuration.hnsw.space")
        if mismatches:
            raise ChromaIndexError(f"Chroma collection metadata mismatch: {sorted(set(mismatches))}")

    def create_collection(self, client: Any, source: ChromaSourceIndex):
        metadata = build_collection_metadata(source.index_manifest)
        return client.create_collection(
            name=self.settings.collection_name,
            configuration={"hnsw": {"space": self.settings.distance_metric}},
            metadata=metadata,
            embedding_function=None,
        )

    def _all_record_metadata(self, collection: Any) -> dict[str, dict[str, Any]]:
        payload = collection.get(include=["metadatas"])
        ids = [str(item) for item in payload.get("ids", [])]
        metadatas = payload.get("metadatas") or [{} for _ in ids]
        return {chunk_id: dict(metadata or {}) for chunk_id, metadata in zip(ids, metadatas)}

    def _write_records(
        self,
        collection: Any,
        source: ChromaSourceIndex,
        chunk_ids: list[str],
        *,
        operation: str,
    ) -> None:
        position = {chunk_id: index for index, chunk_id in enumerate(source.chunk_ids)}
        for batch_ids in batched(chunk_ids, self.settings.batch_size):
            indexes = [position[chunk_id] for chunk_id in batch_ids]
            documents = [str(source.documents[index].get("text", "")) for index in indexes]
            metadatas = [
                encode_record_metadata(
                    source.documents[index],
                    corpus_fingerprint=source.corpus_fingerprint,
                )
                for index in indexes
            ]
            if operation == "add":
                collection.add(
                    ids=batch_ids,
                    embeddings=source.vectors[indexes].tolist(),
                    documents=documents,
                    metadatas=metadatas,
                )
            elif operation == "upsert":
                collection.upsert(
                    ids=batch_ids,
                    embeddings=source.vectors[indexes].tolist(),
                    documents=documents,
                    metadatas=metadatas,
                )
            elif operation == "metadata_update":
                collection.update(ids=batch_ids, metadatas=metadatas)
            else:
                raise ChromaIndexError(f"Unsupported Chroma write operation: {operation}")

    def sync(self, source: ChromaSourceIndex, *, allow_large_delete: bool = False) -> dict[str, Any]:
        client = self.client()
        created = not self.collection_exists(client)
        collection = self.create_collection(client, source) if created else client.get_collection(
            name=self.settings.collection_name,
            embedding_function=None,
        )
        if not created:
            self.validate_collection(collection, source=source, strict_fingerprint=False)

        existing = self._all_record_metadata(collection)
        current_ids = set(source.chunk_ids)
        existing_ids = set(existing)
        added = [chunk_id for chunk_id in source.chunk_ids if chunk_id not in existing_ids]
        updated: list[str] = []
        metadata_updated: list[str] = []
        reused: list[str] = []
        for document in source.documents:
            chunk_id = str(document["chunk_id"])
            previous = existing.get(chunk_id)
            if previous is None:
                continue
            if previous.get("content_hash") != document.get("content_hash"):
                updated.append(chunk_id)
            elif (
                previous.get("metadata_hash") != document.get("metadata_hash")
                or previous.get("corpus_fingerprint") != source.corpus_fingerprint
            ):
                metadata_updated.append(chunk_id)
            else:
                reused.append(chunk_id)
        stale = sorted(existing_ids - current_ids)
        stale_ratio = len(stale) / max(len(existing_ids), 1)
        if stale and not self.settings.delete_stale:
            raise ChromaIndexError(f"Stale Chroma records found while delete_stale=false: {stale[:20]}")
        if stale and stale_ratio > self.settings.large_delete_ratio and not allow_large_delete:
            raise ChromaIndexError(
                "Large stale-record deletion requires explicit confirmation: "
                f"{len(stale)}/{len(existing_ids)} ({stale_ratio:.2%}); ids={stale[:50]}"
            )

        self._write_records(collection, source, added, operation="add")
        self._write_records(collection, source, updated, operation="upsert")
        self._write_records(collection, source, metadata_updated, operation="metadata_update")
        for batch_ids in batched(stale, self.settings.batch_size):
            collection.delete(ids=batch_ids)

        expected_metadata = build_collection_metadata(source.index_manifest)
        if collection.metadata != expected_metadata:
            collection.modify(metadata=expected_metadata)
        final_count = collection.count()
        if final_count != len(source.chunk_ids):
            raise ChromaIndexError(
                f"Collection count mismatch after sync: expected {len(source.chunk_ids)}, got {final_count}."
            )
        return {
            "status": "complete",
            "created_collection": created,
            "collection_name": self.settings.collection_name,
            "persist_directory": str(self.settings.persist_directory),
            "source_chunks": len(source.documents),
            "source_vectors": len(source.vectors),
            "inserted": len(added),
            "added": len(added),
            "updated": len(updated),
            "metadata_updated": len(metadata_updated),
            "deleted": len(stale),
            "reused": len(reused),
            "collection_count": final_count,
            "added_chunk_ids": added,
            "updated_chunk_ids": updated,
            "metadata_updated_chunk_ids": metadata_updated,
            "deleted_chunk_ids": stale,
            "corpus_fingerprint": source.corpus_fingerprint,
            "embedding_model": source.index_manifest.get("embedding_model"),
            "embedding_dimensions": source.index_manifest.get("embedding_dimensions"),
            "distance_metric": self.settings.distance_metric,
            "source_hard_checks": source.hard_checks,
        }


def build_or_sync_chroma(
    config_path: str | Path,
    *,
    allow_large_delete: bool = False,
) -> tuple[dict[str, Any], ChromaSettings]:
    _, settings = load_chroma_settings(config_path)
    source = load_source_index(settings, strict=True)
    summary = ChromaIndexManager(settings).sync(source, allow_large_delete=allow_large_delete)
    return summary, settings
