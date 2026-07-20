# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\reranker.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from researchguard.indexing.corpus_loader import load_yaml
from researchguard.retrieval.models import RetrievalError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RerankerSettings:
    enabled: bool
    backend: str
    model_name: str
    model_revision: str
    model_path: Path
    device: str
    candidate_k: int
    final_top_k: int
    batch_size: int
    max_length: int
    config_version: str
    input_template_version: str
    cache_enabled: bool
    cache_directory: Path

    @property
    def model_identity(self) -> str:
        return f"{self.model_name}@{self.model_revision}"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_reranker_settings(path: str | Path) -> tuple[dict[str, Any], RerankerSettings]:
    config = load_yaml(Path(path))
    reranker = config.get("reranker", {}) or {}
    cache = config.get("cache", {}) or {}
    settings = RerankerSettings(
        enabled=bool(reranker.get("enabled", False)),
        backend=str(reranker.get("backend", "cross_encoder")),
        model_name=str(reranker.get("model_name", "cross-encoder/ms-marco-MiniLM-L6-v2")),
        model_revision=str(reranker.get("model_revision", "")),
        model_path=resolve_project_path(reranker.get("model_path", "data/cache/reranker_models/ms-marco-MiniLM-L6-v2")),
        device=str(reranker.get("device", "cpu")),
        candidate_k=max(1, int(reranker.get("candidate_k", 20))),
        final_top_k=max(1, int(reranker.get("final_top_k", 10))),
        batch_size=max(1, int(reranker.get("batch_size", 8))),
        max_length=max(32, int(reranker.get("max_length", 512))),
        config_version=str(reranker.get("config_version", "reranker_v1")),
        input_template_version=str(
            reranker.get("input_template_version", "title_section_heading_content_v1")
        ),
        cache_enabled=bool(cache.get("enabled", True)),
        cache_directory=resolve_project_path(cache.get("directory", "data/cache/reranker_v1")),
    )
    if settings.backend != "cross_encoder":
        raise RetrievalError(f"Unsupported reranker backend: {settings.backend}")
    if settings.final_top_k > settings.candidate_k:
        raise RetrievalError("Reranker final_top_k must not exceed candidate_k.")
    return config, settings


def render_rerank_document(document: dict[str, Any]) -> str:
    title = " ".join(str(document.get("title", "")).split())
    section = " ".join(str(document.get("section", "")).split())
    section_heading = document.get("section_heading")
    heading_path = [str(item).strip() for item in document.get("heading_path", []) if str(item).strip()]
    heading = " ".join(str(section_heading or (heading_path[-1] if heading_path else "")).split())
    content = str(document.get("text", "")).strip()
    return f"Title: {title}\nSection: {section}\nHeading: {heading}\nContent: {content}"


class RerankerBackend(ABC):
    backend_name: str
    model_name: str

    @abstractmethod
    def score(self, query: str, candidates: list[dict[str, Any]]) -> list[float]:
        raise NotImplementedError

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        scores = self.score(query, candidates)
        if len(scores) != len(candidates):
            raise RetrievalError("Reranker returned a score count that does not match candidate count.")
        ranked: list[dict[str, Any]] = []
        for pre_rank, (candidate, score) in enumerate(zip(candidates, scores), start=1):
            row = dict(candidate)
            row["rerank_score"] = float(score)
            row["pre_rerank_rank"] = pre_rank
            row["reranker_backend"] = self.backend_name
            row["reranker_model"] = self.model_name
            ranked.append(row)
        ranked.sort(
            key=lambda item: (
                -float(item["rerank_score"]),
                int(item["pre_rerank_rank"]),
                str(item["chunk_id"]),
            )
        )
        for rerank_rank, row in enumerate(ranked, start=1):
            row["rerank_rank"] = rerank_rank
        return ranked[:top_k]


class CrossEncoderReranker(RerankerBackend):
    backend_name = "cross_encoder"

    def __init__(self, settings: RerankerSettings):
        self.settings = settings
        self.model_name = settings.model_identity
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.settings.model_path.exists():
            raise RetrievalError(
                f"Reranker model files not found at {self.settings.model_path}. Download is never implicit."
            )
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                str(self.settings.model_path),
                device=self.settings.device,
                max_length=self.settings.max_length,
                local_files_only=True,
            )
        except Exception as exc:
            raise RetrievalError(f"Unable to load Cross-Encoder reranker: {exc}") from exc
        return self._model

    def score(self, query: str, candidates: list[dict[str, Any]]) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, render_rerank_document(candidate["document"])) for candidate in candidates]
        try:
            scores = self._load_model().predict(
                pairs,
                batch_size=self.settings.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RetrievalError(f"Cross-Encoder inference failed: {exc}") from exc
        array = np.asarray(scores, dtype="float64").reshape(-1)
        if len(array) != len(candidates) or not np.isfinite(array).all():
            raise RetrievalError("Cross-Encoder returned invalid scores.")
        return [float(value) for value in array]
