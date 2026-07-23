# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tools\scholarly\base.py
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


SCHOLAR_PAPER_SCHEMA_VERSION = "researchguard.scholar_paper.v1"
ALLOWED_SOURCE_TYPES = {
    "arxiv",
    "journal",
    "conference",
    "repository",
    "metadata",
    "other",
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


class ScholarlyProviderError(RuntimeError):
    pass


class ScholarlyProviderConfigurationError(ScholarlyProviderError):
    pass


class ScholarlyProviderTimeout(ScholarlyProviderError):
    pass


class ScholarlyProviderAPIError(ScholarlyProviderError):
    pass


@dataclass(frozen=True)
class ScholarPaperRecord:
    title: str
    authors: tuple[str, ...]
    year: int | None
    venue: str
    doi: str | None
    url: str
    abstract: str
    source: str
    paper_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_type: str = "other"
    metadata_only: bool = True
    retrieved_at: str = field(default_factory=utc_timestamp)
    schema_version: str = SCHOLAR_PAPER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("ScholarPaperRecord.title must not be empty.")
        if not self.paper_id.strip():
            raise ValueError("ScholarPaperRecord.paper_id must not be empty.")
        if not self.source.strip():
            raise ValueError("ScholarPaperRecord.source must not be empty.")
        if self.source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"Unsupported scholarly source_type: {self.source_type}")
        if self.metadata_only is not True:
            raise ValueError("ScholarPaperRecord must remain metadata_only=true.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "url": self.url,
            "abstract": self.abstract,
            "source": self.source,
            "paper_id": self.paper_id,
            "metadata": copy.deepcopy(dict(self.metadata)),
            "source_type": self.source_type,
            "metadata_only": True,
            "retrieved_at": self.retrieved_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScholarPaperRecord":
        if str(value.get("schema_version", "")) != SCHOLAR_PAPER_SCHEMA_VERSION:
            raise ValueError("Unsupported or missing ScholarPaperRecord schema_version.")
        year = value.get("year")
        return cls(
            title=normalize_text(value.get("title")),
            authors=tuple(
                normalize_text(author)
                for author in value.get("authors", [])
                if normalize_text(author)
            ),
            year=int(year) if year is not None else None,
            venue=normalize_text(value.get("venue")),
            doi=normalize_text(value.get("doi")) or None,
            url=normalize_text(value.get("url")),
            abstract=normalize_text(value.get("abstract")),
            source=normalize_text(value.get("source")),
            paper_id=normalize_text(value.get("paper_id")),
            metadata=copy.deepcopy(dict(value.get("metadata", {}) or {})),
            source_type=normalize_text(value.get("source_type")) or "other",
            metadata_only=bool(value.get("metadata_only", False)),
            retrieved_at=normalize_text(value.get("retrieved_at")) or utc_timestamp(),
        )


class ScholarlyProvider(ABC):
    name: str
    version: str

    @abstractmethod
    def search(self, query: str, *, limit: int) -> list[ScholarPaperRecord]:
        raise NotImplementedError
