# C:\Users\18449\Desktop\researchguard_workspace\tests\tools\test_contracts.py
from __future__ import annotations

import json
import unittest

from researchguard.tools.contracts import EvidenceRecord, ToolError, ToolResult


class ContractTests(unittest.TestCase):
    def test_tool_result_is_stable_json(self) -> None:
        evidence = EvidenceRecord(
            chunk_id="doc-a::chunk-1",
            doc_id="doc-a",
            section="method",
            page=4,
            page_end=4,
            content="The method retrieves corrective evidence.",
            source="paper_a.pdf",
            score=0.91,
            rank=1,
            provenance={"source_block_ids": ["b-1"], "content_types": ["paragraph"]},
        )
        result = ToolResult.create(
            status="success",
            message="ok",
            tool_name="test_tool",
            tool_version="1.0.0",
            latency_ms=3.5,
            data={"evidence": [evidence]},
        )

        payload = json.loads(result.to_json())

        self.assertEqual(payload["schema_version"], "researchguard.tool_result.v1")
        self.assertEqual(payload["data"]["evidence"][0]["chunk_id"], "doc-a::chunk-1")
        self.assertEqual(payload["data"]["evidence"][0]["page"], 4)
        self.assertEqual(payload["data"]["evidence"][0]["section"], "method")
        self.assertTrue(payload["trace_id"].startswith("test_tool-"))

    def test_tool_error_is_json_serializable(self) -> None:
        error = ToolError(
            code="retrieval_timeout",
            category="timeout",
            message="timed out",
            retryable=True,
            details={"attempt": 2},
        )
        result = ToolResult.create(
            status="failed",
            message="failed",
            reason=error.code,
            tool_name="retrieve_evidence",
            tool_version="1.0.0",
            latency_ms=1000,
            error=error,
        )

        payload = json.loads(result.to_json())

        self.assertEqual(payload["error"]["schema_version"], "researchguard.tool_error.v1")
        self.assertEqual(payload["error"]["category"], "timeout")
        self.assertTrue(payload["error"]["retryable"])

    def test_invalid_evidence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EvidenceRecord(
                chunk_id="",
                doc_id="doc-a",
                section="method",
                page=1,
                content="content",
                source="paper.pdf",
                score=None,
                provenance={},
            )


if __name__ == "__main__":
    unittest.main()
