# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\rerank_pipeline.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from researchguard.retrieval.rerank_cache import RerankCache
from researchguard.retrieval.reranker import CrossEncoderReranker, RerankerBackend, RerankerSettings


@dataclass(frozen=True)
class RerankResult:
    candidates: list[dict[str, Any]]
    latency_ms: float
    inference_latency_ms: float
    cache_hits: int
    cache_misses: int


class RerankPipeline:
    def __init__(
        self,
        settings: RerankerSettings,
        *,
        backend: RerankerBackend | None = None,
        cache: RerankCache | None = None,
    ):
        self.settings = settings
        self.backend = backend or CrossEncoderReranker(settings)
        self.cache = cache or RerankCache(settings.cache_directory, enabled=settings.cache_enabled)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int,
        read_cache: bool = True,
    ) -> RerankResult:
        started = time.perf_counter()
        prepared: list[dict[str, Any]] = []
        misses: list[dict[str, Any]] = []
        miss_keys: list[str] = []
        cache_hits = 0
        for pre_rank, candidate in enumerate(candidates, start=1):
            row = dict(candidate)
            row["pre_rerank_rank"] = pre_rank
            content_hash = str(row["document"].get("content_hash", ""))
            metadata_hash = str(row["document"].get("metadata_hash", ""))
            key = self.cache.make_key(
                query=query,
                content_hash=content_hash,
                metadata_hash=metadata_hash,
                settings=self.settings,
            )
            score = self.cache.get(key) if read_cache else None
            if score is None:
                misses.append(row)
                miss_keys.append(key)
            else:
                row["rerank_score"] = score
                cache_hits += 1
            prepared.append(row)

        inference_started = time.perf_counter()
        if misses:
            miss_scores = self.backend.score(query, misses)
            if len(miss_scores) != len(misses):
                raise RuntimeError("Reranker score count does not match cache misses.")
            for row, key, score in zip(misses, miss_keys, miss_scores):
                row["rerank_score"] = float(score)
                self.cache.put(
                    key,
                    float(score),
                    metadata={
                        "chunk_id": str(row["chunk_id"]),
                        "content_hash": str(row["document"].get("content_hash", "")),
                        "metadata_hash": str(row["document"].get("metadata_hash", "")),
                        "reranker_model": self.settings.model_identity,
                        "reranker_config_version": self.settings.config_version,
                        "input_template_version": self.settings.input_template_version,
                    },
                )
        inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0
        prepared.sort(
            key=lambda item: (
                -float(item["rerank_score"]),
                int(item["pre_rerank_rank"]),
                str(item["chunk_id"]),
            )
        )
        for rerank_rank, row in enumerate(prepared, start=1):
            row["rerank_rank"] = rerank_rank
            row["reranker_backend"] = self.backend.backend_name
            row["reranker_model"] = self.backend.model_name
        return RerankResult(
            candidates=prepared[:top_k],
            latency_ms=(time.perf_counter() - started) * 1000.0,
            inference_latency_ms=inference_latency_ms,
            cache_hits=cache_hits,
            cache_misses=len(misses),
        )
