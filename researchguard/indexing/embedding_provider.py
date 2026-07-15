# C:\Users\18449\Desktop\researchguard_workspace\researchguard\indexing\embedding_provider.py
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


class EmbeddingError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    dimensions: int
    batch_size: int
    max_retries: int
    timeout: float
    normalize: bool


def parse_embedding_config(config: dict[str, Any]) -> EmbeddingConfig:
    embedding = config.get("embedding", {}) or {}
    return EmbeddingConfig(
        provider=str(embedding.get("provider", "openai")),
        model=str(embedding.get("model", "text-embedding-3-small")),
        dimensions=int(embedding.get("dimensions", 1536)),
        batch_size=int(embedding.get("batch_size", 64)),
        max_retries=int(embedding.get("max_retries", 4)),
        timeout=float(embedding.get("timeout", 60)),
        normalize=bool(embedding.get("normalize", True)),
    )


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0 or not math.isfinite(norm):
        raise EmbeddingError("Embedding vector has invalid norm.")
    return [value / norm for value in vector]


def validate_vector(vector: list[float], *, dimensions: int) -> None:
    if not vector:
        raise EmbeddingError("Embedding vector is empty.")
    if len(vector) != dimensions:
        raise EmbeddingError(f"Embedding dimension mismatch: expected {dimensions}, got {len(vector)}.")
    for value in vector:
        if not math.isfinite(float(value)):
            raise EmbeddingError("Embedding vector contains NaN or Inf.")
    if all(float(value) == 0.0 for value in vector):
        raise EmbeddingError("Embedding vector is all zeros.")


class OpenAIEmbeddingProvider:
    def __init__(self, config: EmbeddingConfig):
        if config.provider != "openai":
            raise EmbeddingError(f"Unsupported embedding provider: {config.provider}")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY is missing; refusing to generate fake embeddings.")
        self.config = config
        self.client = OpenAI(api_key=api_key, timeout=config.timeout, max_retries=0)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not str(text).strip() for text in texts):
            raise EmbeddingError("Cannot embed empty text.")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            vectors.extend(self._embed_batch(batch))

        if len(vectors) != len(texts):
            raise EmbeddingError(f"Embedding count mismatch: expected {len(texts)}, got {len(vectors)}.")
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.embeddings.create(model=self.config.model, input=texts)
                vectors = [[float(value) for value in item.embedding] for item in response.data]
                for vector in vectors:
                    validate_vector(vector, dimensions=self.config.dimensions)
                if self.config.normalize:
                    vectors = [l2_normalize(vector) for vector in vectors]
                return vectors
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(min(2**attempt, 16))
        raise EmbeddingError(f"Embedding API failed after retries: {type(last_error).__name__}: {last_error}") from last_error
