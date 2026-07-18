# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\retrieval_v1.py
from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from researchguard.indexing.embedding_provider import OpenAIEmbeddingProvider, parse_embedding_config, validate_vector
from researchguard.indexing.sparse_index import tokenize
from researchguard.retrieval.chroma_retriever import ChromaDenseRetrieverBackend
from researchguard.retrieval.dense_backend import DenseRetrieverBackend, NumpyDenseRetrieverBackend
from researchguard.retrieval.filters import metadata_matches
from researchguard.retrieval.index_loader import RetrievalIndexBundle, load_index_bundle
from researchguard.retrieval.models import MetadataFilter, RetrievalError, RetrievalHit, RetrievalResponse


VALID_MODES = {"dense", "sparse", "hybrid"}


class RetrievalEngine:
    def __init__(self, bundle: RetrievalIndexBundle, *, dense_backend_override: str | None = None):
        self.bundle = bundle
        self.config = bundle.config
        embedding_config = parse_embedding_config(bundle.indexing_config)
        if embedding_config.dimensions != bundle.dense_index.dimension:
            raise RetrievalError(
                f"Embedding config dimension {embedding_config.dimensions} does not match dense index {bundle.dense_index.dimension}."
            )
        self.embedding_provider = OpenAIEmbeddingProvider(embedding_config)
        self._query_vector_cache: dict[str, np.ndarray] = {}
        dense_cfg = self.config.get("dense", {}) or {}
        backend_name = str(dense_backend_override or dense_cfg.get("backend", "numpy"))
        if backend_name == "numpy":
            self.dense_backend: DenseRetrieverBackend = NumpyDenseRetrieverBackend(bundle)
        elif backend_name == "chroma":
            chroma_config_path = dense_cfg.get("chroma_config_path", "configs/chroma_v1.yaml")
            self.dense_backend = ChromaDenseRetrieverBackend(bundle, chroma_config_path=chroma_config_path)
        else:
            raise RetrievalError(f"Unsupported dense backend: {backend_name}")
        self.dense_backend_name = backend_name
        self._last_dense_trace: dict[str, Any] = {}

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        dense_backend_override: str | None = None,
    ) -> "RetrievalEngine":
        return cls(
            load_index_bundle(Path(config_path), strict=True),
            dense_backend_override=dense_backend_override,
        )

    def retrieve(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        candidate_k: int | None = None,
        filters: MetadataFilter | None = None,
    ) -> RetrievalResponse:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise RetrievalError("Query must not be empty.")
        retrieval_cfg = self.config.get("retrieval", {}) or {}
        mode = str(mode or retrieval_cfg.get("default_mode", "hybrid"))
        if mode not in VALID_MODES:
            raise RetrievalError(f"Unsupported retrieval mode: {mode}")
        top_k = int(top_k or retrieval_cfg.get("default_top_k", 10))
        candidate_k = int(candidate_k or retrieval_cfg.get("default_candidate_k", max(top_k, 10)))
        max_top_k = int(retrieval_cfg.get("max_top_k", 50))
        if top_k <= 0 or top_k > max_top_k:
            raise RetrievalError(f"top_k must be between 1 and {max_top_k}.")
        if candidate_k < top_k:
            candidate_k = top_k
        filters = filters or MetadataFilter()

        started = time.perf_counter()
        trace: dict[str, Any] = {
            "index_dir": str(self.bundle.index_dir),
            "corpus_fingerprint": self.bundle.manifest.get("corpus_fingerprint"),
            "mode": mode,
            "candidate_k": candidate_k,
            "dense_backend": self.dense_backend_name,
        }

        if mode == "dense":
            ranked = self._dense_candidates(normalized_query, candidate_k, filters)
        elif mode == "sparse":
            ranked = self._sparse_candidates(normalized_query, candidate_k, filters)
        else:
            ranked = self._hybrid_candidates(normalized_query, candidate_k, filters)

        hits = [self._hit_from_candidate(rank, candidate) for rank, candidate in enumerate(ranked[:top_k], start=1)]
        latency_ms = (time.perf_counter() - started) * 1000.0
        trace["returned"] = len(hits)
        if mode in {"dense", "hybrid"}:
            trace["dense_backend_trace"] = self._last_dense_trace
        return RetrievalResponse(
            query=normalized_query,
            mode=mode,
            top_k=top_k,
            candidate_k=candidate_k,
            filters=filters,
            hits=hits,
            latency_ms=latency_ms,
            trace=trace,
        )

    def _embed_query(self, query: str) -> np.ndarray:
        if query in self._query_vector_cache:
            return self._query_vector_cache[query]
        vector = self.embedding_provider.embed_query(query)
        validate_vector(vector, dimensions=self.bundle.dense_index.dimension)
        query_vector = np.asarray(vector, dtype="float32")
        self._query_vector_cache[query] = query_vector
        return query_vector

    def _dense_candidates(self, query: str, candidate_k: int, filters: MetadataFilter) -> list[dict[str, Any]]:
        query_vector = self._embed_query(query)
        candidates, trace = self.dense_backend.search(query_vector, candidate_k=candidate_k, filters=filters)
        self._last_dense_trace = trace
        return candidates

    def _sparse_candidates(self, query: str, candidate_k: int, filters: MetadataFilter) -> list[dict[str, Any]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        query_terms = Counter(tokens)
        sparse = self.bundle.sparse_index
        n_docs = len(sparse.chunk_ids)
        candidates: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(sparse.chunk_ids):
            doc = self.bundle.documents[index]
            if not metadata_matches(doc, filters):
                continue
            score = 0.0
            tf_map = sparse.doc_term_freqs[index]
            length = sparse.doc_lengths[index]
            for term, query_count in query_terms.items():
                df = sparse.df.get(term, 0)
                if df == 0:
                    continue
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + sparse.k1 * (1 - sparse.b + sparse.b * length / max(sparse.avgdl, 1.0))
                score += query_count * idf * (tf * (sparse.k1 + 1) / denom)
            if score > 0.0:
                candidates.append(
                    {
                        "chunk_id": str(chunk_id),
                        "document": doc,
                        "sparse_score": float(score),
                        "sparse_rank": None,
                        "retrieval_sources": ["sparse"],
                    }
                )
        candidates.sort(key=lambda item: (-float(item["sparse_score"]), str(item["chunk_id"])))
        for rank, item in enumerate(candidates, start=1):
            item["sparse_rank"] = rank
        return candidates[:candidate_k]

    def _hybrid_candidates(self, query: str, candidate_k: int, filters: MetadataFilter) -> list[dict[str, Any]]:
        hybrid_cfg = self.config.get("hybrid", {}) or {}
        rrf_k = float(hybrid_cfg.get("rrf_k", 60))
        dense_weight = float(hybrid_cfg.get("dense_weight", 1.0))
        sparse_weight = float(hybrid_cfg.get("sparse_weight", 1.0))
        dense = self._dense_candidates(query, candidate_k, filters)
        sparse = self._sparse_candidates(query, candidate_k, filters)

        merged: dict[str, dict[str, Any]] = {}
        for item in dense:
            chunk_id = str(item["chunk_id"])
            score = dense_weight / (rrf_k + int(item["dense_rank"]))
            merged[chunk_id] = {
                "chunk_id": chunk_id,
                "document": item["document"],
                "dense_score": item.get("dense_score"),
                "sparse_score": None,
                "dense_rank": item.get("dense_rank"),
                "sparse_rank": None,
                "fusion_score": score,
                "retrieval_sources": ["dense"],
            }
        for item in sparse:
            chunk_id = str(item["chunk_id"])
            score = sparse_weight / (rrf_k + int(item["sparse_rank"]))
            if chunk_id not in merged:
                merged[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document": item["document"],
                    "dense_score": None,
                    "sparse_score": item.get("sparse_score"),
                    "dense_rank": None,
                    "sparse_rank": item.get("sparse_rank"),
                    "fusion_score": score,
                    "retrieval_sources": ["sparse"],
                }
            else:
                merged[chunk_id]["sparse_score"] = item.get("sparse_score")
                merged[chunk_id]["sparse_rank"] = item.get("sparse_rank")
                merged[chunk_id]["fusion_score"] = float(merged[chunk_id]["fusion_score"]) + score
                merged[chunk_id]["retrieval_sources"] = ["dense", "sparse"]

        candidates = list(merged.values())
        candidates.sort(key=lambda item: (-float(item.get("fusion_score") or 0.0), str(item["chunk_id"])))
        return candidates[:candidate_k]

    def _hit_from_candidate(self, rank: int, candidate: dict[str, Any]) -> RetrievalHit:
        doc = candidate["document"]
        return RetrievalHit(
            rank=rank,
            chunk_id=str(doc.get("chunk_id", "")),
            doc_id=str(doc.get("doc_id", "")),
            title=str(doc.get("title", "")),
            section=str(doc.get("section", "")),
            section_heading=doc.get("section_heading"),
            heading_path=[str(item) for item in doc.get("heading_path", [])],
            chunk_type=str(doc.get("chunk_type", "")),
            page_start=doc.get("page_start"),
            page_end=doc.get("page_end"),
            source_block_ids=[str(item) for item in doc.get("source_block_ids", [])],
            overlap_source_block_ids=[str(item) for item in doc.get("overlap_source_block_ids", [])],
            content_types=[str(item) for item in doc.get("content_types", [])],
            has_equation=bool(doc.get("has_equation")),
            has_table=bool(doc.get("has_table")),
            has_caption=bool(doc.get("has_caption")),
            text=str(doc.get("text", "")),
            dense_score=candidate.get("dense_score"),
            sparse_score=candidate.get("sparse_score"),
            fusion_score=candidate.get("fusion_score"),
            dense_rank=candidate.get("dense_rank"),
            sparse_rank=candidate.get("sparse_rank"),
            retrieval_sources=list(candidate.get("retrieval_sources", [])),
        )
