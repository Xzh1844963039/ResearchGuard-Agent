# C:\Users\18449\Desktop\researchguard_workspace\tests\tools\test_gate_bypass.py
from __future__ import annotations

import unittest

from researchguard.retrieval.answer_generator import (
    AnswerCitation,
    AnswerGenerationResult,
)
from researchguard.tools import EvidenceBundle, GateDecision
from researchguard.tools.answer_tool import GuardedAnswerTool


EVIDENCE = {
    "chunk_id": "doc-a::chunk-1",
    "doc_id": "doc-a",
    "section": "method",
    "page": 3,
    "content": "CRAG uses a retrieval evaluator before generation.",
    "source": "Paper A",
    "provenance": {"source_block_ids": ["p3-b1"]},
}


class RecordingAnswerPipeline:
    def __init__(self) -> None:
        self.call_count = 0
        self.received_hits: list[dict[str, object]] = []

    def generate(
        self,
        query: str,
        hits: list[dict[str, object]],
        sufficiency: object,
        *,
        read_cache: bool,
    ) -> AnswerGenerationResult:
        del query, sufficiency, read_cache
        self.call_count += 1
        self.received_hits = hits
        return AnswerGenerationResult(
            answer="CRAG evaluates retrieval quality.",
            citations=(
                AnswerCitation(
                    chunk_id=EVIDENCE["chunk_id"],
                    doc_id=EVIDENCE["doc_id"],
                    section=EVIDENCE["section"],
                    page=EVIDENCE["page"],
                ),
            ),
            confidence=0.9,
            refused=False,
            refusal_reason=None,
            evidence_chunk_ids=(EVIDENCE["chunk_id"],),
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


def bundle() -> EvidenceBundle:
    return EvidenceBundle.create(query="How does CRAG work?", evidence=[EVIDENCE])


def gate(value: EvidenceBundle, status: str) -> GateDecision:
    supporting = (EVIDENCE["chunk_id"],) if status in {"strong", "partial"} else ()
    return GateDecision(
        status=status,
        reason=f"synthetic_{status}",
        supporting_chunk_ids=supporting,
        evidence_bundle_id=value.bundle_id,
        confidence=0.9,
        answerable=status == "strong",
    )


class GateBypassTests(unittest.TestCase):
    def test_unsupported_and_partial_cannot_reach_answer_generator(self) -> None:
        for support_level in ("unsupported", "partial"):
            with self.subTest(support_level=support_level):
                answer_pipeline = RecordingAnswerPipeline()
                evidence_bundle = bundle()
                tool = GuardedAnswerTool(answer_pipeline=answer_pipeline)

                result = tool.generate_grounded_answer(
                    evidence_bundle,
                    gate(evidence_bundle, support_level),
                )

                self.assertEqual(result.status, "rejected")
                self.assertEqual(answer_pipeline.call_count, 0)

    def test_strong_gate_reuses_bundle_without_retrieval_or_rejudging(self) -> None:
        answer_pipeline = RecordingAnswerPipeline()
        evidence_bundle = bundle()
        tool = GuardedAnswerTool(answer_pipeline=answer_pipeline)

        result = tool.generate_grounded_answer(
            evidence_bundle,
            gate(evidence_bundle, "strong"),
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(answer_pipeline.call_count, 1)
        self.assertEqual(
            answer_pipeline.received_hits[0]["chunk_id"],
            EVIDENCE["chunk_id"],
        )
        self.assertEqual(
            result.data["evidence_bundle_id"],
            evidence_bundle.bundle_id,
        )
        self.assertNotIn("pipeline_result", result.data)
        self.assertFalse(hasattr(tool, "_guarded_pipeline"))

    def test_gate_for_another_bundle_is_rejected_before_generation(self) -> None:
        answer_pipeline = RecordingAnswerPipeline()
        evidence_bundle = bundle()
        other = EvidenceBundle.create(
            query="Different question",
            evidence=[EVIDENCE],
        )
        tool = GuardedAnswerTool(answer_pipeline=answer_pipeline)

        result = tool.generate_grounded_answer(
            evidence_bundle,
            gate(other, "strong"),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "invalid_answer_input")
        self.assertEqual(answer_pipeline.call_count, 0)


if __name__ == "__main__":
    unittest.main()
