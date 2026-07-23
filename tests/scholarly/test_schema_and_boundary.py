# C:\Users\18449\Desktop\researchguard_workspace\tests\scholarly\test_schema_and_boundary.py
from __future__ import annotations

import json
import unittest

from researchguard.tools import EvidenceRecord, ScholarPaperRecord, ToolResult
from researchguard.tools.answer_tool import GuardedAnswerTool


def _paper() -> ScholarPaperRecord:
    return ScholarPaperRecord(
        title="Candidate Paper",
        authors=("Author A",),
        year=2025,
        venue="arXiv",
        doi=None,
        url="https://arxiv.org/abs/2501.00001",
        abstract="Metadata abstract only.",
        source="arxiv",
        paper_id="arxiv:2501.00001",
        source_type="arxiv",
        metadata={"categories": ["cs.CL"]},
    )


class ScholarlySchemaTests(unittest.TestCase):
    def test_record_and_tool_result_are_json_serializable(self) -> None:
        paper = _paper()
        result = ToolResult.create(
            status="success",
            message="found",
            tool_name="search_scholarly_sources",
            tool_version="1.0.0",
            latency_ms=1,
            data={
                "candidate_papers": [paper],
                "metadata_only": True,
                "evidence_eligible": False,
            },
        )

        payload = json.loads(result.to_json())

        self.assertEqual(
            payload["data"]["candidate_papers"][0]["schema_version"],
            "researchguard.scholar_paper.v1",
        )
        self.assertTrue(payload["data"]["candidate_papers"][0]["metadata_only"])
        self.assertFalse(payload["data"]["evidence_eligible"])

    def test_metadata_only_flag_cannot_be_disabled(self) -> None:
        payload = _paper().to_dict()
        payload["metadata_only"] = False

        with self.assertRaises(ValueError):
            ScholarPaperRecord.from_dict(payload)

    def test_candidate_metadata_is_not_an_evidence_record(self) -> None:
        paper = _paper()

        self.assertNotIsInstance(paper, EvidenceRecord)
        with self.assertRaises(ValueError):
            EvidenceRecord.from_mapping(paper.to_dict())

    def test_answer_tool_does_not_accept_candidate_papers(self) -> None:
        tool = GuardedAnswerTool(pipeline=object())

        with self.assertRaises(TypeError):
            tool.generate_grounded_answer(
                "Question",
                candidate_papers=[_paper().to_dict()],
            )


if __name__ == "__main__":
    unittest.main()
