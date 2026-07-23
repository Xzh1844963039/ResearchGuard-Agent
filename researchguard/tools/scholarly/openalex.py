# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tools\scholarly\openalex.py
from __future__ import annotations

import os
from typing import Any, Mapping

import httpx

from researchguard.tools.scholarly.base import (
    ScholarPaperRecord,
    ScholarlyProvider,
    ScholarlyProviderAPIError,
    ScholarlyProviderConfigurationError,
    ScholarlyProviderTimeout,
    normalize_text,
)


class OpenAlexProvider(ScholarlyProvider):
    name = "openalex"
    version = "1.0.0"
    endpoint = "https://api.openalex.org/works"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        timeout: float = 20.0,
    ):
        self.api_key = normalize_text(api_key or os.getenv("OPENALEX_API_KEY"))
        self.timeout = timeout
        self.client = client or httpx.Client(follow_redirects=True)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, *, limit: int) -> list[ScholarPaperRecord]:
        normalized_query = normalize_text(query)
        if not normalized_query:
            raise ValueError("OpenAlex query must not be empty.")
        if limit < 1 or limit > 50:
            raise ValueError("OpenAlex limit must be between 1 and 50.")
        if not self.api_key:
            raise ScholarlyProviderConfigurationError(
                "OPENALEX_API_KEY is required for the OpenAlex provider."
            )
        params = {
            "search": normalized_query,
            "per_page": limit,
            "sort": "-relevance_score",
            "api_key": self.api_key,
        }
        try:
            response = self.client.get(self.endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise ScholarlyProviderTimeout("OpenAlex request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise ScholarlyProviderAPIError(
                f"OpenAlex request returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.HTTPError as exc:
            raise ScholarlyProviderAPIError(
                f"OpenAlex request failed with {type(exc).__name__}."
            ) from exc
        except ValueError as exc:
            raise ScholarlyProviderAPIError("OpenAlex returned invalid JSON.") from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("results"), list):
            raise ScholarlyProviderAPIError("OpenAlex response does not contain a results list.")
        return [record for item in payload["results"] if (record := self.parse_work(item)) is not None]

    @classmethod
    def parse_work(cls, item: Any) -> ScholarPaperRecord | None:
        if not isinstance(item, Mapping):
            return None
        title = normalize_text(item.get("display_name") or item.get("title"))
        openalex_url = normalize_text(item.get("id"))
        openalex_id = openalex_url.rstrip("/").split("/")[-1]
        if not title or not openalex_id:
            return None
        authors = tuple(
            name
            for authorship in item.get("authorships", []) or []
            if isinstance(authorship, Mapping)
            and isinstance(authorship.get("author"), Mapping)
            and (name := normalize_text(authorship["author"].get("display_name")))
        )
        location = item.get("primary_location")
        location = location if isinstance(location, Mapping) else {}
        source = location.get("source")
        source = source if isinstance(source, Mapping) else {}
        source_type = cls._source_type(source.get("type"))
        doi = normalize_text(item.get("doi"))
        if doi.lower().startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/") :]
        url = (
            normalize_text(location.get("landing_page_url"))
            or (f"https://doi.org/{doi}" if doi else "")
            or openalex_url
        )
        year = item.get("publication_year")
        return ScholarPaperRecord(
            title=title,
            authors=authors,
            year=int(year) if year is not None else None,
            venue=normalize_text(source.get("display_name")),
            doi=doi or None,
            url=url,
            abstract=cls._abstract_from_inverted_index(item.get("abstract_inverted_index")),
            source=cls.name,
            paper_id=f"openalex:{openalex_id}",
            source_type=source_type,
            metadata={
                "openalex_id": openalex_id,
                "publication_date": item.get("publication_date"),
                "work_type": item.get("type"),
                "cited_by_count": item.get("cited_by_count"),
                "open_access": item.get("open_access"),
                "ids": item.get("ids"),
                "source_id": source.get("id"),
                "source_type": source.get("type"),
            },
        )

    @staticmethod
    def _abstract_from_inverted_index(value: Any) -> str:
        if not isinstance(value, Mapping):
            return ""
        positioned: list[tuple[int, str]] = []
        for word, positions in value.items():
            if not isinstance(positions, list):
                continue
            for position in positions:
                if isinstance(position, int):
                    positioned.append((position, str(word)))
        return normalize_text(" ".join(word for _, word in sorted(positioned)))

    @staticmethod
    def _source_type(value: Any) -> str:
        normalized = normalize_text(value).casefold()
        if normalized in {"journal", "conference", "repository"}:
            return normalized
        if normalized == "metadata":
            return "metadata"
        return "other"
