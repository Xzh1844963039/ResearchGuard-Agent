# C:\Users\18449\Desktop\researchguard_workspace\tests\tools\test_gate_bypass.py
from __future__ import annotations

import unittest
from pathlib import Path

from researchguard.pipeline import PipelineSettings, ResearchGuardPipeline
from researchguard.retrieval.evidence_judge import EvidenceSufficiencyResult
from researchguard.retrieval.models import MetadataFilter, RetrievalHit, RetrievalResponse
from researchguard.tools.answer_tool import GuardedAnswerTool


def _hit() -> RetrievalHit:
    return RetrievalHit(
        rank=1,
        chunk_id="doc-a::chunk-1",
        doc_id="doc-a",
        title="Paper A",
        section="method",
        section_heading="Method",
        heading_path=["Method"],
        chunk_type="text",
        page_start=3,
        page_end=3,
        source_block_ids=["p3-b1"],
        overlap_source_block_ids=[],
        content_types=["paragraph"],
        has_equation=False,
        has_table=False,
        has_caption=False,
        text="This passage is related but does not fully answer the question.",
    )


class FakeRetrievalEngine:
    embedding_provider = None

    def retrieve(self, query: str, **_: object) -> RetrievalResponse:
        return RetrievalResponse(
            query=query,
            mode="hybrid",
            top_k=10,
            candidate_k=80,
            filters=MetadataFilter(),
            hits=[_hit()],
            latency_ms=1.0,
            retrieval_latency_ms=1.0,
            total_latency_ms=1.0,
            trace={},
        )


class FakeEvidencePipeline:
    def __init__(self, support_level: str):
        self.support_level = support_level

    def assess(self, query: str, hits: object, *, read_cache: bool) -> EvidenceSufficiencyResult:
        del query, hits, read_cache
        return EvidenceSufficiencyResult(
            answerable=self.support_level == "strong",
            support_level=self.support_level,
            confidence=0.9,
            reason=f"synthetic_{self.support_level}",
            supporting_chunk_ids=("doc-a::chunk-1",),
            model="synthetic",
            prompt_version="test",
            config_version="test",
            timestamp="2026-01-01T00:00:00+00:00",
            cache_hit=False,
            fallback_used=False,
            fallback_reason=None,
            api_call_count=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.1,
        )


class ForbiddenAnswerPipeline:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, *_: object, **__: object) -> object:
        self.call_count += 1
        raise AssertionError("Answer generation must not run after a failed evidence gate.")


class ForbiddenAuditPipeline:
    def __init__(self) -> None:
        self.call_count = 0

    def audit(self, *_: object, **__: object) -> object:
        self.call_count += 1
        raise AssertionError("Citation audit must not run when no answer was generated.")


def _settings() -> PipelineSettings:
    placeholder = Path("unused.yaml")
    return PipelineSettings(
        schema_version="researchguard_pipeline_v1",
        config_version="test",
        read_cache=False,
        include_retrieval_text=True,
        rewrite_enabled=False,
        multi_query_enabled=False,
        retrieval_enabled=True,
        retrieval_config_path=placeholder,
        retrieval_mode="hybrid",
        retrieval_top_k=10,
        retrieval_candidate_k=80,
        reranker_enabled=False,
        reranker_candidate_k=20,
        evidence_check_enabled=True,
        evidence_config_path=placeholder,
        answer_generation_enabled=True,
        answer_config_path=placeholder,
        citation_audit_enabled=True,
        citation_audit_config_path=placeholder,
    )


class GateBypassTests(unittest.TestCase):
    def test_unsupported_and_partial_cannot_reach_answer_generator(self) -> None:
        for support_level in ("unsupported", "partial"):
            with self.subTest(support_level=support_level):
                answer = ForbiddenAnswerPipeline()
                audit = ForbiddenAuditPipeline()
                pipeline = ResearchGuardPipeline(
                    _settings(),
                    retrieval_engine=FakeRetrievalEngine(),
                    evidence_pipeline=FakeEvidencePipeline(support_level),
                    answer_pipeline=answer,
                    citation_audit_pipeline=audit,
                )
                tool = GuardedAnswerTool(pipeline=pipeline)

                result = tool.generate_grounded_answer("What is supported?")

                self.assertEqual(result.status, "rejected")
                self.assertEqual(
                    result.data["pipeline_result"]["final_status"],
                    "rejected",
                )
                self.assertEqual(answer.call_count, 0)
                self.assertEqual(audit.call_count, 0)

    def test_tool_has_no_raw_generation_method(self) -> None:
        tool = GuardedAnswerTool(pipeline=object())

        self.assertFalse(hasattr(tool, "generate_answer"))
        self.assertTrue(hasattr(tool, "generate_grounded_answer"))


if __name__ == "__main__":
    unittest.main()
