# C:\Users\18449\Desktop\researchguard_workspace\tests\evaluation\test_holdout.py
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from researchguard.evaluation.holdout import (
    build_holdout_report,
    evaluate_holdout_run,
    load_holdout_questions,
)


ROOT = Path(__file__).resolve().parents[2]


class HoldoutEvaluationTests(unittest.TestCase):
    def test_repository_split_is_blind_and_sized(self) -> None:
        path = ROOT / "data" / "eval" / "holdout" / "blind_questions.jsonl"
        questions = load_holdout_questions(path)
        self.assertEqual(25, len(questions))
        raw = path.read_text(encoding="utf-8")
        self.assertNotIn("relevant_chunk_ids", raw)
        self.assertNotIn('"answer"', raw)

    def test_evaluation_uses_document_and_provenance_labels(self) -> None:
        question = load_holdout_questions(
            ROOT / "data" / "eval" / "holdout" / "blind_questions.jsonl"
        )[0]
        state = SimpleNamespace(
            task_type="qa",
            plan=[
                {"skill": "retrieve_evidence"},
                {"skill": "generate_report"},
            ],
            evidence=[
                {
                    "chunk_id": "hyde-c1",
                    "doc_id": "hyde",
                    "section": "method",
                    "page": 2,
                    "content": "Grounded evidence.",
                }
            ],
            status="completed",
            reason=None,
            answer={"answer": "Grounded answer."},
            audit_result={"claims": [{"status": "supported"}]},
            gate_decision={"status": "strong"},
            tool_history=[{}],
            skill_history=[{}, {}],
        )
        row = evaluate_holdout_run(state, question)
        self.assertTrue(row["passed"])
        self.assertEqual(1.0, row["metrics"]["document_hit_at_10"])
        self.assertEqual(1.0, row["metrics"]["provenance_validity"])

    def test_unbuilt_corpus_is_reported_not_run(self) -> None:
        report = build_holdout_report(
            [],
            corpus_status={
                "ready": False,
                "paper_count": 0,
                "reason": "not built",
            },
        )
        self.assertEqual("NOT_RUN", report["conclusion"])


if __name__ == "__main__":
    unittest.main()
