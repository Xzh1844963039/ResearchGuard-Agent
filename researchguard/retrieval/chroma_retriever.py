# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\chroma_retriever.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from researchguard.indexing.chroma_index import ChromaIndexManager, load_chroma_settings, load_source_index
from researchguard.retrieval.filters import metadata_matches
from researchguard.retrieval.index_loader import RetrievalIndexBundle
from researchguard.retrieval.models import MetadataFilter, RetrievalError


def _condition(field: str, operator: str, value: Any) -> dict[str, Any]:
    return {field: {operator: value}}


def build_chroma_where(filters: MetadataFilter) -> dict[str, Any] | None:
    conditions: list[dict[str, Any]] = []
    if filters.doc_ids:
        conditions.append(_condition("doc_id", "$in", list(filters.doc_ids)))
    if filters.sections:
        conditions.append(_condition("section", "$in", list(filters.sections)))
    if filters.chunk_types:
        conditions.append(_condition("chunk_type", "$in", list(filters.chunk_types)))
    if filters.exclude_references:
        conditions.append(_condition("section", "$ne", "references"))
    if filters.page_start_min is not None:
        conditions.append(_condition("page_end", "$gte", int(filters.page_start_min)))
    if filters.page_end_max is not None:
        conditions.append(_condition("page_start", "$lte", int(filters.page_end_max)))
    if filters.has_equation is not None:
        conditions.append(_condition("has_equation", "$eq", bool(filters.has_equation)))
    if filters.has_table is not None:
        conditions.append(_condition("has_table", "$eq", bool(filters.has_table)))
    if filters.has_caption is not None:
        conditions.append(_condition("has_caption", "$eq", bool(filters.has_caption)))
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class ChromaDenseRetrieverBackend:
    name = "chroma"

    def __init__(self, bundle: RetrievalIndexBundle, *, chroma_config_path: str | Path):
        self.bundle = bundle
        _, self.settings = load_chroma_settings(chroma_config_path)
        source = load_source_index(self.settings, strict=True)
        if source.chunk_ids != bundle.dense_index.chunk_ids:
            raise RetrievalError("Chroma source IDs do not match the loaded Retrieval v1 NumPy baseline.")
        try:
            _, self.collection = ChromaIndexManager(self.settings).get_collection(
                strict_fingerprint=True,
                source=source,
            )
        except Exception as exc:
            raise RetrievalError(f"Unable to load validated Chroma backend: {exc}") from exc

    def search(
        self,
        query_vector: np.ndarray,
        *,
        candidate_k: int,
        filters: MetadataFilter,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        where = build_chroma_where(filters)
        n_results = min(candidate_k, self.collection.count())
        if n_results <= 0:
            return [], {
                "dense_backend": self.name,
                "native_filter": where is not None,
                "post_filter": True,
                "native_candidate_count": 0,
                "filtered_count": 0,
            }
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "query_embeddings": [np.asarray(query_vector, dtype="float32").tolist()],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        try:
            result = self.collection.query(**kwargs)
        except Exception as exc:
            raise RetrievalError(f"Chroma dense query failed: {exc}") from exc
        ids = [str(item) for item in (result.get("ids") or [[]])[0]]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        candidates: list[dict[str, Any]] = []
        for chunk_id, stored_text, stored_metadata, distance in zip(ids, documents, metadatas, distances):
            source_document = self.bundle.document_by_id.get(chunk_id)
            if source_document is None:
                raise RetrievalError(f"Chroma returned unknown chunk_id: {chunk_id}")
            if str(stored_text or "") != str(source_document.get("text", "")):
                raise RetrievalError(f"Chroma document mismatch for chunk_id: {chunk_id}")
            if str((stored_metadata or {}).get("content_hash", "")) != str(source_document.get("content_hash", "")):
                raise RetrievalError(f"Chroma content_hash mismatch for chunk_id: {chunk_id}")
            if not metadata_matches(source_document, filters):
                continue
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "document": source_document,
                    "dense_score": 1.0 - float(distance),
                    "dense_rank": None,
                    "retrieval_sources": ["dense"],
                }
            )
        candidates.sort(key=lambda item: (-float(item["dense_score"]), str(item["chunk_id"])))
        for rank, item in enumerate(candidates, start=1):
            item["dense_rank"] = rank
        return candidates[:candidate_k], {
            "dense_backend": self.name,
            "native_filter": where is not None,
            "native_where": where,
            "post_filter": True,
            "native_candidate_count": len(ids),
            "filtered_count": len(candidates),
            "backend_latency_ms": (time.perf_counter() - started) * 1000.0,
        }
