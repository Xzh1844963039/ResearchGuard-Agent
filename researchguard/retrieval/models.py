# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class RetrievalError(RuntimeError):
    pass


@dataclass(frozen=True)
class MetadataFilter:
    doc_ids: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    chunk_types: tuple[str, ...] = ()
    page_start_min: int | None = None
    page_end_max: int | None = None
    has_equation: bool | None = None
    has_table: bool | None = None
    has_caption: bool | None = None
    exclude_references: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "MetadataFilter":
        data = data or {}
        return cls(
            doc_ids=tuple(str(item) for item in data.get("doc_ids", []) if str(item).strip()),
            sections=tuple(str(item) for item in data.get("sections", []) if str(item).strip()),
            chunk_types=tuple(str(item) for item in data.get("chunk_types", []) if str(item).strip()),
            page_start_min=data.get("page_start_min"),
            page_end_max=data.get("page_end_max"),
            has_equation=data.get("has_equation"),
            has_table=data.get("has_table"),
            has_caption=data.get("has_caption"),
            exclude_references=bool(data.get("exclude_references", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_ids": list(self.doc_ids),
            "sections": list(self.sections),
            "chunk_types": list(self.chunk_types),
            "page_start_min": self.page_start_min,
            "page_end_max": self.page_end_max,
            "has_equation": self.has_equation,
            "has_table": self.has_table,
            "has_caption": self.has_caption,
            "exclude_references": self.exclude_references,
        }


@dataclass
class RetrievalHit:
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    section: str
    section_heading: str | None
    heading_path: list[str]
    chunk_type: str
    page_start: int | None
    page_end: int | None
    source_block_ids: list[str]
    overlap_source_block_ids: list[str]
    content_types: list[str]
    has_equation: bool
    has_table: bool
    has_caption: bool
    text: str
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fusion_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    pre_rerank_rank: int | None = None
    reranker_backend: str | None = None
    reranker_model: str | None = None
    retrieval_sources: list[str] = field(default_factory=list)

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        row = {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "section_heading": self.section_heading,
            "heading_path": self.heading_path,
            "chunk_type": self.chunk_type,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "source_block_ids": self.source_block_ids,
            "overlap_source_block_ids": self.overlap_source_block_ids,
            "content_types": self.content_types,
            "has_equation": self.has_equation,
            "has_table": self.has_table,
            "has_caption": self.has_caption,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "fusion_score": self.fusion_score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "fusion_rank": self.fusion_rank,
            "rerank_score": self.rerank_score,
            "rerank_rank": self.rerank_rank,
            "pre_rerank_rank": self.pre_rerank_rank,
            "reranker_backend": self.reranker_backend,
            "reranker_model": self.reranker_model,
            "retrieval_sources": self.retrieval_sources,
        }
        if include_text:
            row["text"] = self.text
        return row


@dataclass
class RetrievalResponse:
    query: str
    mode: str
    top_k: int
    candidate_k: int
    filters: MetadataFilter
    hits: list[RetrievalHit]
    latency_ms: float
    trace: dict[str, Any]
    retrieval_latency_ms: float | None = None
    rerank_latency_ms: float = 0.0
    total_latency_ms: float | None = None

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "filters": self.filters.to_dict(),
            "latency_ms": self.latency_ms,
            "retrieval_latency_ms": self.retrieval_latency_ms if self.retrieval_latency_ms is not None else self.latency_ms,
            "rerank_latency_ms": self.rerank_latency_ms,
            "total_latency_ms": self.total_latency_ms if self.total_latency_ms is not None else self.latency_ms,
            "trace": self.trace,
            "hits": [hit.to_dict(include_text=include_text) for hit in self.hits],
        }
