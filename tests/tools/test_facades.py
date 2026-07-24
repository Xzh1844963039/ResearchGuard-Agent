# C:\Users\18449\Desktop\researchguard_workspace\tests\tools\test_facades.py
from __future__ import annotations

import unittest
from types import SimpleNamespace

from researchguard.pipeline import load_pipeline_settings
from researchguard.retrieval.answer_generator import AnswerCitation, AnswerGenerationResult
from researchguard.retrieval.evidence_judge import EvidenceSufficiencyResult
from researchguard.retrieval.models import MetadataFilter, RetrievalHit, RetrievalResponse
from researchguard.tools.audit_tool import CitationAuditTool
from researchguard.tools.contracts import EvidenceBundle
from researchguard.tools.evidence_tool import EvidenceTool
from researchguard.tools.retrieval_tool import RetrievalTool


def _hit() -> RetrievalHit:
    return RetrievalHit(
        rank=1,
        chunk_id="paper-a::chunk-2",
        doc_id="paper-a",
        title="Paper A",
        section="results",
        section_heading="4 Results",
        heading_path=["4 Results"],
        chunk_type="mixed",
        page_start=7,
        page_end=7,
        source_block_ids=["p7-b2"],
        overlap_source_block_ids=[],
        content_types=["paragraph", "table"],
        has_equation=False,
        has_table=True,
        has_caption=False,
        text="The evaluated method improves retrieval quality.",
        fusion_score=0.08,
        rerank_score=0.95,
        rerank_rank=1,
        retrieval_sources=["dense", "sparse"],
    )


class FakeRetrievalEngine:
    def __init__(self) -> None:
        self.call_kwargs: dict[str, object] | None = None

    def retrieve(self, query: str, **kwargs: object) -> RetrievalResponse:
        self.call_kwargs = dict(kwargs)
        return RetrievalResponse(
            query=query,
            mode="hybrid",
            top_k=int(kwargs["top_k"]),
            candidate_k=int(kwargs["candidate_k"]),
            filters=kwargs.get("filters") or MetadataFilter(),
            hits=[_hit()],
            latency_ms=2.0,
            retrieval_latency_ms=1.0,
            rerank_latency_ms=1.0,
            total_latency_ms=2.0,
            trace={"reranker": {"candidate_count": 1}},
        )


class FakeEvidencePipeline:
    def __init__(self) -> None:
        self.received_hits: list[dict[str, object]] = []

    def assess(
        self,
        query: str,
        hits: list[dict[str, object]],
        *,
        read_cache: bool,
    ) -> EvidenceSufficiencyResult:
        del query, read_cache
        self.received_hits = hits
        return EvidenceSufficiencyResult(
            answerable=True,
            support_level="strong",
            confidence=0.96,
            reason="direct_support",
            supporting_chunk_ids=(str(hits[0]["chunk_id"]),),
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


class FakeCitationAuditPipeline:
    def __init__(self) -> None:
        self.received_hits: list[dict[str, object]] = []

    def audit(
        self,
        answer: AnswerGenerationResult,
        hits: list[dict[str, object]],
        *,
        read_cache: bool,
    ) -> SimpleNamespace:
        del answer, read_cache
        self.received_hits = hits
        payload = {
            "audit_completed": True,
            "overall_grounded": True,
            "fallback_used": False,
            "fallback_reason": None,
            "audit_reason": None,
        }
        return SimpleNamespace(**payload, to_dict=lambda: payload)


class FacadeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.settings = load_pipeline_settings()

    def test_retrieval_tool_wraps_engine_and_emits_canonical_evidence(self) -> None:
        engine = FakeRetrievalEngine()
        tool = RetrievalTool(engine=engine, settings=self.settings)

        result = tool.retrieve_evidence("What improved?", top_k=5, candidate_k=20)

        self.assertEqual(result.status, "success")
        evidence = result.data["evidence"][0]
        self.assertEqual(evidence["chunk_id"], "paper-a::chunk-2")
        self.assertEqual(evidence["page"], 7)
        self.assertEqual(evidence["section"], "results")
        self.assertEqual(evidence["score"], 0.95)
        self.assertEqual(engine.call_kwargs["top_k"], 5)
        self.assertTrue(engine.call_kwargs["rerank"])

    def test_evidence_tool_passes_provenance_to_existing_pipeline(self) -> None:
        pipeline = FakeEvidencePipeline()
        tool = EvidenceTool(pipeline=pipeline, settings=self.settings)
        evidence = {
            "chunk_id": "paper-a::chunk-2",
            "doc_id": "paper-a",
            "section": "results",
            "page": 7,
            "content": "The evaluated method improves retrieval quality.",
            "source": "Paper A",
            "provenance": {"source_block_ids": ["p7-b2"]},
        }

        bundle = EvidenceBundle.create(query="What improved?", evidence=[evidence])
        result = tool.assess_evidence(bundle)

        self.assertEqual(result.status, "success")
        self.assertEqual(pipeline.received_hits[0]["chunk_id"], "paper-a::chunk-2")
        self.assertEqual(pipeline.received_hits[0]["page_start"], 7)
        self.assertEqual(pipeline.received_hits[0]["section"], "results")
        self.assertEqual(pipeline.received_hits[0]["source_block_ids"], ["p7-b2"])
        self.assertEqual(
            result.data["gate_decision"]["evidence_bundle_id"],
            result.data["evidence_bundle_id"],
        )

    def test_audit_tool_wraps_existing_auditor_with_complete_artifact(self) -> None:
        pipeline = FakeCitationAuditPipeline()
        tool = CitationAuditTool(pipeline=pipeline, settings=self.settings)
        answer = AnswerGenerationResult(
            answer="The method improves retrieval quality.",
            citations=(
                AnswerCitation(
                    chunk_id="paper-a::chunk-2",
                    doc_id="paper-a",
                    section="results",
                    page=7,
                ),
            ),
            confidence=0.9,
            refused=False,
            refusal_reason=None,
            evidence_chunk_ids=("paper-a::chunk-2",),
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
        evidence = {
            "chunk_id": "paper-a::chunk-2",
            "doc_id": "paper-a",
            "section": "results",
            "page": 7,
            "content": "The evaluated method improves retrieval quality.",
            "source": "Paper A",
        }

        bundle = EvidenceBundle.create(query="What improved?", evidence=[evidence])
        result = tool.audit_answer(answer, bundle)

        self.assertEqual(result.status, "success")
        self.assertEqual(pipeline.received_hits[0]["chunk_id"], "paper-a::chunk-2")
        self.assertEqual(result.data["evidence_bundle_id"], bundle.bundle_id)


if __name__ == "__main__":
    unittest.main()
