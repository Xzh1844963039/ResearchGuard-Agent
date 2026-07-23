# C:\Users\18449\Desktop\researchguard_workspace\tests\scholarly\test_providers.py
from __future__ import annotations

import unittest

import httpx

from researchguard.tools.scholarly import (
    ArxivProvider,
    OpenAlexProvider,
    ScholarlyProviderAPIError,
    ScholarlyProviderConfigurationError,
    ScholarlyProviderTimeout,
)


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>https://arxiv.org/abs/2401.12345v2</id>
    <updated>2024-02-02T00:00:00Z</updated>
    <published>2024-01-20T00:00:00Z</published>
    <title>Corrective Retrieval Augmented Generation</title>
    <summary>A retrieval evaluator triggers corrective actions.</summary>
    <author><name>Author One</name></author>
    <author><name>Author Two</name></author>
    <link href="https://arxiv.org/abs/2401.12345v2" rel="alternate" type="text/html"/>
    <link href="https://arxiv.org/pdf/2401.12345v2" rel="related" type="application/pdf"/>
    <category term="cs.CL"/>
    <arxiv:primary_category term="cs.CL"/>
    <arxiv:doi>10.1000/example</arxiv:doi>
    <arxiv:journal_ref>Example Conference 2024</arxiv:journal_ref>
  </entry>
</feed>
"""

EMPTY_ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
"""


class FakeClient:
    def __init__(self, response: httpx.Response | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        if self.error:
            raise self.error
        assert self.response is not None
        return self.response


def _response(status: int, *, text: str | None = None, payload: object | None = None) -> httpx.Response:
    request = httpx.Request("GET", "https://example.test")
    if payload is not None:
        return httpx.Response(status, request=request, json=payload)
    return httpx.Response(status, request=request, text=text or "")


class ArxivProviderTests(unittest.TestCase):
    def test_atom_response_is_parsed_to_metadata_record(self) -> None:
        client = FakeClient(_response(200, text=ARXIV_FEED))
        provider = ArxivProvider(client=client)

        records = provider.search("corrective retrieval", limit=3)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.paper_id, "arxiv:2401.12345")
        self.assertEqual(record.authors, ("Author One", "Author Two"))
        self.assertEqual(record.year, 2024)
        self.assertEqual(record.doi, "10.1000/example")
        self.assertEqual(record.source_type, "arxiv")
        self.assertTrue(record.metadata_only)
        self.assertEqual(client.calls[0]["params"]["max_results"], 3)

    def test_empty_atom_feed_returns_empty_list(self) -> None:
        provider = ArxivProvider(client=FakeClient(_response(200, text=EMPTY_ARXIV_FEED)))

        self.assertEqual(provider.search("no result", limit=2), [])

    def test_timeout_is_normalized(self) -> None:
        request = httpx.Request("GET", "https://export.arxiv.org/api/query")
        provider = ArxivProvider(
            client=FakeClient(error=httpx.ReadTimeout("timeout", request=request))
        )

        with self.assertRaises(ScholarlyProviderTimeout):
            provider.search("query", limit=2)

    def test_http_error_is_normalized(self) -> None:
        provider = ArxivProvider(client=FakeClient(_response(503, text="unavailable")))

        with self.assertRaises(ScholarlyProviderAPIError):
            provider.search("query", limit=2)


class OpenAlexProviderTests(unittest.TestCase):
    def test_json_response_and_inverted_abstract_are_parsed(self) -> None:
        payload = {
            "meta": {"count": 1},
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "display_name": "A Grounded Retrieval Paper",
                    "publication_year": 2025,
                    "publication_date": "2025-01-10",
                    "doi": "https://doi.org/10.2000/test",
                    "type": "article",
                    "cited_by_count": 12,
                    "authorships": [
                        {"author": {"display_name": "Researcher A"}},
                    ],
                    "primary_location": {
                        "landing_page_url": "https://example.org/paper",
                        "source": {
                            "id": "https://openalex.org/S1",
                            "display_name": "Example Journal",
                            "type": "journal",
                        },
                    },
                    "abstract_inverted_index": {
                        "Evidence": [0],
                        "is": [1],
                        "grounded": [2],
                    },
                    "open_access": {"is_oa": True},
                    "ids": {"openalex": "https://openalex.org/W123"},
                }
            ],
        }
        client = FakeClient(_response(200, payload=payload))
        provider = OpenAlexProvider(api_key="test-key", client=client)

        records = provider.search("grounded retrieval", limit=5)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].paper_id, "openalex:W123")
        self.assertEqual(records[0].abstract, "Evidence is grounded")
        self.assertEqual(records[0].source_type, "journal")
        self.assertEqual(records[0].doi, "10.2000/test")
        self.assertEqual(client.calls[0]["params"]["api_key"], "test-key")

    def test_missing_api_key_fails_before_network(self) -> None:
        client = FakeClient(_response(200, payload={"results": []}))
        provider = OpenAlexProvider(api_key="", client=client)

        with self.assertRaises(ScholarlyProviderConfigurationError):
            provider.search("query", limit=2)
        self.assertEqual(client.calls, [])

    def test_invalid_json_schema_is_api_error(self) -> None:
        provider = OpenAlexProvider(
            api_key="test-key",
            client=FakeClient(_response(200, payload={"unexpected": []})),
        )

        with self.assertRaises(ScholarlyProviderAPIError):
            provider.search("query", limit=2)

    def test_http_error_does_not_expose_api_key(self) -> None:
        provider = OpenAlexProvider(
            api_key="sensitive-unit-test-key",
            client=FakeClient(_response(429, payload={"error": "rate limited"})),
        )

        with self.assertRaises(ScholarlyProviderAPIError) as raised:
            provider.search("query", limit=2)

        self.assertNotIn("sensitive-unit-test-key", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
