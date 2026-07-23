# C:\Users\18449\Desktop\researchguard_workspace\tests\scholarly\test_agent_integration.py
from __future__ import annotations

import unittest

from researchguard.agent import BoundedResearchAgentController
from researchguard.tools import ToolRegistry, ToolResult, ToolSpec


CANDIDATE = {
    "schema_version": "researchguard.scholar_paper.v1",
    "title": "Corrective Retrieval Augmented Generation",
    "authors": ["Author A"],
    "year": 2024,
    "venue": "arXiv",
    "doi": None,
    "url": "https://arxiv.org/abs/2401.00001",
    "abstract": "Candidate metadata.",
    "source": "arxiv",
    "paper_id": "arxiv:2401.00001",
    "metadata": {},
    "source_type": "arxiv",
    "metadata_only": True,
    "retrieved_at": "2026-01-01T00:00:00+00:00",
}


class FakeScholarlyTool:
    name = "search_scholarly_sources"
    version = "test"

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            version=self.version,
            description="Synthetic scholarly search.",
            input_schema={"query": "string"},
        )

    def invoke(self, **kwargs: object) -> ToolResult:
        self.call_count += 1
        return ToolResult.create(
            status="success",
            message="found",
            tool_name=self.name,
            tool_version=self.version,
            latency_ms=1,
            data={
                "query": kwargs["query"],
                "candidate_papers": [CANDIDATE],
                "metadata_only": True,
                "evidence_eligible": False,
            },
        )


class InvalidScholarlyTool(FakeScholarlyTool):
    def invoke(self, **kwargs: object) -> ToolResult:
        del kwargs
        return ToolResult.create(
            status="success",
            message="invalid",
            tool_name=self.name,
            tool_version=self.version,
            latency_ms=1,
            data={
                "candidate_papers": [
                    {
                        **CANDIDATE,
                        "metadata_only": False,
                    }
                ]
            },
        )


class ScholarlyAgentIntegrationTests(unittest.TestCase):
    def test_literature_search_stops_after_candidate_discovery(self) -> None:
        registry = ToolRegistry()
        tool = FakeScholarlyTool()
        registry.register(tool)
        controller = BoundedResearchAgentController(
            registry=registry,
            memory_enabled=False,
        )

        state = controller.run("Find papers about CRAG")

        self.assertEqual(state.task_type, "literature_search")
        self.assertEqual(state.status, "completed")
        self.assertEqual(
            [entry["tool_name"] for entry in state.tool_history],
            ["search_scholarly_sources"],
        )
        self.assertEqual(state.candidate_papers[0]["paper_id"], "arxiv:2401.00001")
        self.assertEqual(state.evidence, [])
        self.assertIsNone(state.answer)
        self.assertIsNone(state.audit_result)
        self.assertEqual(tool.call_count, 1)

    def test_invalid_candidate_schema_fails_closed(self) -> None:
        registry = ToolRegistry()
        registry.register(InvalidScholarlyTool())
        controller = BoundedResearchAgentController(
            registry=registry,
            memory_enabled=False,
        )

        state = controller.run("Find papers about CRAG")

        self.assertEqual(state.status, "failed")
        self.assertEqual(state.reason, "invalid_tool_output")
        self.assertEqual(state.candidate_papers, [])
        self.assertEqual(
            state.observations[0]["output_validation_error"]["exception_type"],
            "ValueError",
        )


if __name__ == "__main__":
    unittest.main()
