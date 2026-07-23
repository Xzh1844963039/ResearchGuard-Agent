# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tools\scholarly\arxiv.py
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from researchguard.tools.scholarly.base import (
    ScholarPaperRecord,
    ScholarlyProvider,
    ScholarlyProviderAPIError,
    ScholarlyProviderTimeout,
    normalize_text,
)


ATOM = "http://www.w3.org/2005/Atom"
ARXIV = "http://arxiv.org/schemas/atom"
VERSION_SUFFIX_RE = re.compile(r"v\d+$", re.IGNORECASE)


class ArxivProvider(ScholarlyProvider):
    name = "arxiv"
    version = "1.0.0"
    endpoint = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        *,
        client: Any | None = None,
        timeout: float = 20.0,
        user_agent: str = "ResearchGuard-Agent/2.0 (+https://github.com/Xzh1844963039/ResearchGuard-Agent)",
    ):
        self.timeout = timeout
        self.client = client or httpx.Client(
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )

    def search(self, query: str, *, limit: int) -> list[ScholarPaperRecord]:
        normalized_query = normalize_text(query)
        if not normalized_query:
            raise ValueError("arXiv query must not be empty.")
        if limit < 1 or limit > 50:
            raise ValueError("arXiv limit must be between 1 and 50.")
        escaped_query = normalized_query.replace('"', '\\"')
        params = {
            "search_query": f'all:"{escaped_query}"',
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            response = self.client.get(self.endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ScholarlyProviderTimeout("arXiv request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise ScholarlyProviderAPIError(
                f"arXiv request returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.HTTPError as exc:
            raise ScholarlyProviderAPIError(
                f"arXiv request failed with {type(exc).__name__}."
            ) from exc
        return self.parse_feed(response.text)

    @classmethod
    def parse_feed(cls, xml_text: str) -> list[ScholarPaperRecord]:
        if "<!DOCTYPE" in xml_text.upper():
            raise ScholarlyProviderAPIError("arXiv response contains a disallowed DOCTYPE.")
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise ScholarlyProviderAPIError(f"Invalid arXiv Atom response: {exc}") from exc

        records: list[ScholarPaperRecord] = []
        namespaces = {"atom": ATOM, "arxiv": ARXIV}
        for entry in root.findall("atom:entry", namespaces):
            title = normalize_text(entry.findtext("atom:title", default="", namespaces=namespaces))
            entry_url = normalize_text(entry.findtext("atom:id", default="", namespaces=namespaces))
            raw_id = entry_url.rstrip("/").split("/")[-1]
            stable_id = VERSION_SUFFIX_RE.sub("", raw_id)
            if not title or not stable_id:
                continue
            authors = tuple(
                name
                for author in entry.findall("atom:author", namespaces)
                if (name := normalize_text(author.findtext("atom:name", default="", namespaces=namespaces)))
            )
            published = normalize_text(
                entry.findtext("atom:published", default="", namespaces=namespaces)
            )
            year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
            doi = normalize_text(entry.findtext("arxiv:doi", default="", namespaces=namespaces)) or None
            journal_ref = normalize_text(
                entry.findtext("arxiv:journal_ref", default="", namespaces=namespaces)
            )
            links = [
                {
                    "href": normalize_text(link.attrib.get("href")),
                    "rel": normalize_text(link.attrib.get("rel")),
                    "type": normalize_text(link.attrib.get("type")),
                }
                for link in entry.findall("atom:link", namespaces)
            ]
            alternate_url = next(
                (link["href"] for link in links if link["rel"] == "alternate" and link["href"]),
                entry_url,
            )
            categories = [
                normalize_text(category.attrib.get("term"))
                for category in entry.findall("atom:category", namespaces)
                if normalize_text(category.attrib.get("term"))
            ]
            primary_category = entry.find("arxiv:primary_category", namespaces)
            primary_term = (
                normalize_text(primary_category.attrib.get("term"))
                if primary_category is not None
                else ""
            )
            records.append(
                ScholarPaperRecord(
                    title=title,
                    authors=authors,
                    year=year,
                    venue=journal_ref or "arXiv",
                    doi=doi,
                    url=alternate_url,
                    abstract=normalize_text(
                        entry.findtext("atom:summary", default="", namespaces=namespaces)
                    ),
                    source=cls.name,
                    paper_id=f"arxiv:{stable_id}",
                    source_type="arxiv",
                    metadata={
                        "arxiv_id": stable_id,
                        "versioned_arxiv_id": raw_id,
                        "published": published,
                        "updated": normalize_text(
                            entry.findtext("atom:updated", default="", namespaces=namespaces)
                        ),
                        "journal_reference": journal_ref or None,
                        "comment": normalize_text(
                            entry.findtext("arxiv:comment", default="", namespaces=namespaces)
                        )
                        or None,
                        "primary_category": primary_term or None,
                        "categories": categories,
                        "links": links,
                    },
                )
            )
        return records
