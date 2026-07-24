# C:\Users\18449\Desktop\researchguard_workspace\tests\tools\test_provenance.py
from __future__ import annotations

import unittest

from researchguard.retrieval.models import RetrievalHit
from researchguard.tools.contracts import EvidenceBundle, EvidenceRecord, GateDecision


class EvidenceProvenanceTests(unittest.TestCase):
    def test_canonical_fields_survive_round_trip(self) -> None:
        hit = RetrievalHit(
            rank=2,
            chunk_id="paper-crag::chunk-17",
            doc_id="paper-crag",
            title="Corrective Retrieval Augmented Generation",
            section="method",
            section_heading="3 Method",
            heading_path=["3 Method", "3.2 Retrieval Evaluator"],
            chunk_type="mixed",
            page_start=5,
            page_end=6,
            source_block_ids=["p5-b8", "p6-b1"],
            overlap_source_block_ids=["p5-b7"],
            content_types=["paragraph", "equation"],
            has_equation=True,
            has_table=False,
            has_caption=False,
            text="The retrieval evaluator assigns a confidence score.",
            dense_score=0.78,
            sparse_score=4.2,
            fusion_score=0.054,
            rerank_score=0.93,
            rerank_rank=2,
            retrieval_sources=["dense", "sparse"],
        )

        record = EvidenceRecord.from_retrieval_hit(hit)
        restored = record.to_retrieval_mapping()

        self.assertEqual(record.chunk_id, hit.chunk_id)
        self.assertEqual(record.page, hit.page_start)
        self.assertEqual(record.page_end, hit.page_end)
        self.assertEqual(record.section, hit.section)
        self.assertEqual(record.score, hit.rerank_score)
        self.assertEqual(restored["chunk_id"], hit.chunk_id)
        self.assertEqual(restored["page_start"], hit.page_start)
        self.assertEqual(restored["page_end"], hit.page_end)
        self.assertEqual(restored["section"], hit.section)
        self.assertEqual(restored["source_block_ids"], hit.source_block_ids)
        self.assertEqual(restored["overlap_source_block_ids"], hit.overlap_source_block_ids)
        self.assertEqual(restored["text"], hit.text)

    def test_mapping_preserves_required_provenance(self) -> None:
        source = {
            "chunk_id": "doc-2::chunk-3",
            "doc_id": "doc-2",
            "section": "results",
            "page_start": 8,
            "page_end": 8,
            "text": "Table 3 reports the main result.",
            "title": "Paper Two",
            "rank": 3,
            "score": 0.8,
            "source_block_ids": ["p8-b4"],
            "content_types": ["table", "caption"],
            "has_table": True,
            "has_caption": True,
        }

        record = EvidenceRecord.from_mapping(source)
        payload = record.to_dict()

        self.assertEqual(payload["chunk_id"], source["chunk_id"])
        self.assertEqual(payload["page"], 8)
        self.assertEqual(payload["section"], "results")
        self.assertEqual(payload["provenance"]["source_block_ids"], ["p8-b4"])
        self.assertTrue(payload["provenance"]["has_table"])

    def test_bundle_and_gate_preserve_canonical_provenance(self) -> None:
        record = EvidenceRecord.from_mapping(
            {
                "chunk_id": "doc-2::chunk-3",
                "doc_id": "doc-2",
                "section": "results",
                "page": 8,
                "content": "Table 3 reports the main result.",
                "source": "Paper Two",
                "provenance": {"source_block_ids": ["p8-b4"]},
            }
        )
        bundle = EvidenceBundle.create(
            query="What does Table 3 report?",
            evidence=[record],
            retrieval_metadata={"mode": "hybrid"},
        )
        restored = EvidenceBundle.from_mapping(bundle.to_dict())
        gate = GateDecision(
            status="strong",
            reason="direct support",
            supporting_chunk_ids=(record.chunk_id,),
            evidence_bundle_id=restored.bundle_id,
            confidence=0.95,
            answerable=True,
        )

        self.assertEqual(restored.bundle_id, bundle.bundle_id)
        self.assertEqual(restored.evidence_records[0].page, 8)
        self.assertEqual(
            restored.evidence_records[0].provenance["source_block_ids"],
            ["p8-b4"],
        )
        self.assertEqual(gate.evidence_bundle_id, bundle.bundle_id)


if __name__ == "__main__":
    unittest.main()
