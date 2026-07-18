# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\dense_backend.py
from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

from researchguard.retrieval.filters import metadata_matches
from researchguard.retrieval.index_loader import RetrievalIndexBundle
from researchguard.retrieval.models import MetadataFilter, RetrievalError


class DenseRetrieverBackend(Protocol):
    name: str

    def search(
        self,
        query_vector: np.ndarray,
        *,
        candidate_k: int,
        filters: MetadataFilter,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...


class NumpyDenseRetrieverBackend:
    name = "numpy"

    def __init__(self, bundle: RetrievalIndexBundle):
        self.bundle = bundle

    def search(
        self,
        query_vector: np.ndarray,
        *,
        candidate_k: int,
        filters: MetadataFilter,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        q_norm = float(np.linalg.norm(query_vector))
        if q_norm == 0 or not math.isfinite(q_norm):
            raise RetrievalError("Query embedding has invalid norm.")
        scores = self.bundle.dense_index.vectors @ (query_vector / q_norm)
        candidates: list[dict[str, Any]] = []
        for index, score in enumerate(scores):
            document = self.bundle.documents[index]
            if not metadata_matches(document, filters):
                continue
            candidates.append(
                {
                    "chunk_id": str(document["chunk_id"]),
                    "document": document,
                    "dense_score": float(score),
                    "dense_rank": None,
                    "retrieval_sources": ["dense"],
                }
            )
        candidates.sort(key=lambda item: (-float(item["dense_score"]), str(item["chunk_id"])))
        for rank, item in enumerate(candidates, start=1):
            item["dense_rank"] = rank
        return candidates[:candidate_k], {
            "dense_backend": self.name,
            "native_filter": False,
            "post_filter": True,
            "scored_count": len(scores),
            "filtered_count": len(candidates),
        }
