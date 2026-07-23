# C:\Users\18449\Desktop\researchguard_workspace\tests\scholarly\test_search_tool.py
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from researchguard.tools.scholarly import (
    ScholarPaperRecord,
    ScholarlyProvider,
    ScholarlyProviderAPIError,
    ScholarlyProviderTimeout,
    ScholarlySearchCache,
)
from researchguard.tools.scholarly_search_tool import ScholarlySearchTool


def _paper(*, source: str = "arxiv", doi: str | None = "10.1000/shared") -> ScholarPaperRecord:
    return ScholarPaperRecord(
        title="Corrective Retrieval Augmented Generation",
        authors=("Author A",),
        year=2024,
        venue="arXiv" if source == "arxiv" else "Example Journal",
        doi=doi,
        url=f"https://example.org/{source}",
        abstract="A candidate metadata abstract.",
        source=source,
        paper_id=f"{source}:paper-1",
        source_type="arxiv" if source == "arxiv" else "journal",
        metadata={"provider": source},
    )


class FakeProvider(ScholarlyProvider):
    version = "test"

    def __init__(
        self,
        name: str,
        *,
        records: list[ScholarPaperRecord] | None = None,
        error: Exception | None = None,
    ):
        self.name = name
        self.records = records or []
        self.error = error
        self.call_count = 0

    def search(self, query: str, *, limit: int) -> list[ScholarPaperRecord]:
        del query, limit
        self.call_count += 1
        if self.error:
            raise self.error
        return list(self.records)


class ScholarlySearchToolTests(unittest.TestCase):
    def test_result_is_cached_by_query_provider_config_and_limit(self) -> None:
        provider = FakeProvider("arxiv", records=[_paper()])
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ScholarlySearchCache(Path(temp_dir))
            tool = ScholarlySearchTool(providers={"arxiv": provider}, cache=cache)

            first = tool.search_scholarly_sources("CRAG", limit=5)
            second = tool.search_scholarly_sources("CRAG", limit=5)

        self.assertEqual(first.status, "success")
        self.assertEqual(second.status, "success")
        self.assertEqual(provider.call_count, 1)
        self.assertFalse(first.data["cache_hits"]["arxiv"])
        self.assertTrue(second.data["cache_hits"]["arxiv"])

    def test_empty_provider_result_is_successful_discovery(self) -> None:
        provider = FakeProvider("arxiv", records=[])
        tool = ScholarlySearchTool(
            providers={"arxiv": provider},
            cache_enabled=False,
        )

        result = tool.search_scholarly_sources("nothing", limit=3)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.data["candidate_count"], 0)
        self.assertFalse(result.data["evidence_eligible"])

    def test_empty_source_preference_uses_default_arxiv(self) -> None:
        provider = FakeProvider("arxiv", records=[_paper()])
        tool = ScholarlySearchTool(
            providers={"arxiv": provider},
            cache_enabled=False,
        )

        result = tool.search_scholarly_sources("CRAG", sources=[], limit=3)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.data["providers_requested"], ["arxiv"])
        self.assertEqual(provider.call_count, 1)

    def test_timeout_and_api_errors_are_structured(self) -> None:
        cases = (
            (
                ScholarlyProviderTimeout("timeout"),
                "timeout",
                "scholarly_provider_timeout",
            ),
            (
                ScholarlyProviderAPIError("api error"),
                "api_failure",
                "scholarly_provider_failure",
            ),
        )
        for error, category, reason in cases:
            with self.subTest(category=category):
                provider = FakeProvider("arxiv", error=error)
                tool = ScholarlySearchTool(
                    providers={"arxiv": provider},
                    cache_enabled=False,
                )

                result = tool.search_scholarly_sources("query", limit=3)

                self.assertEqual(result.status, "failed")
                self.assertEqual(result.error.category, category)
                self.assertEqual(result.reason, reason)

    def test_cross_provider_deduplication_uses_doi(self) -> None:
        providers = {
            "arxiv": FakeProvider("arxiv", records=[_paper(source="arxiv")]),
            "openalex": FakeProvider("openalex", records=[_paper(source="openalex")]),
        }
        tool = ScholarlySearchTool(providers=providers, cache_enabled=False)

        result = tool.search_scholarly_sources(
            "CRAG",
            sources=["arxiv", "openalex"],
            limit=10,
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.data["candidate_count"], 1)
        self.assertEqual(result.data["candidate_papers"][0]["source"], "arxiv")


if __name__ == "__main__":
    unittest.main()
