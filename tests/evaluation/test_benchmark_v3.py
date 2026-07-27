# C:\Users\18449\Desktop\researchguard_workspace\tests\evaluation\test_benchmark_v3.py
from __future__ import annotations

import json
import unittest
from pathlib import Path

from researchguard.agent import ResearchAgentState
from researchguard.evaluation import (
    AgentBenchmarkCaseV3,
    build_benchmark_v3_report,
    evaluate_benchmark_v3_run,
    load_benchmark_v3_cases,
    render_benchmark_v3_markdown,
)


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = {
    "chunk_id": "paper_rag_chunk_00001",
    "doc_id": "paper_rag",
    "section": "method",
    "page": 3,
    "content": "RAG retrieves external passages before generation.",
    "source": "paper_rag.pdf",
    "score": 0.9,
    "provenance": {"source_block_ids": ["p3-b1"]},
}


class AgentBenchmarkV3Tests(unittest.TestCase):
    def test_repository_benchmark_has_required_real_task_coverage(self) -> None:
        cases = load_benchmark_v3_cases(
            ROOT / "data" / "eval" / "agent_benchmark_v3" / "cases.jsonl"
        )

        self.assertEqual(len(cases), 50)
        self.assertEqual(
            sum(case.category == "literature_research" for case in cases),
            20,
        )
        self.assertEqual(
            sum(case.category == "paper_comparison" for case in cases),
            15,
        )
        self.assertEqual(
            sum(case.category == "claim_audit" for case in cases),
            15,
        )
        serialized = json.dumps(
            [case.to_dict() for case in cases],
            ensure_ascii=False,
        )
        self.assertNotIn("expected_answer", serialized)
        self.assertNotIn("relevant_chunk_ids", serialized)

    def test_v3_metrics_use_skill_plan_and_canonical_provenance(self) -> None:
        case = AgentBenchmarkCaseV3(
            case_id="metric-case",
            category="literature_research",
            query="Explain RAG.",
            expected_task_type="qa",
            expected_skill_pattern=("retrieve_evidence", "generate_report"),
            expected_behavior="answer_or_refuse",
            difficulty="easy",
            expected_doc_ids=("paper_rag",),
        )
        state = ResearchAgentState(
            query=case.query,
            task_type="qa",
            planner_plan={
                "skills": [
                    {"skill": "retrieve_evidence"},
                    {"skill": "generate_report"},
                ]
            },
            planner_metadata={
                "validation": {"valid": True},
                "mode": "deterministic",
                "input_tokens": 10,
                "output_tokens": 5,
            },
            skill_history=[
                {"skill_name": "retrieve_evidence"},
                {"skill_name": "generate_report"},
            ],
            tool_history=[
                {"tool_name": "retrieve_evidence"},
                {"tool_name": "assess_evidence"},
                {
                    "tool_name": "generate_grounded_answer",
                    "input_summary": {"gate_status": "strong"},
                },
                {
                    "tool_name": "audit_answer",
                    "input_summary": {"gate_status": "strong"},
                },
            ],
            evidence=[EVIDENCE],
            gate_decision={"status": "strong"},
            answer={"answer": "Grounded.", "citations": []},
            audit_result={"audit_completed": True, "claims": []},
            status="completed",
        )

        result = evaluate_benchmark_v3_run(
            state,
            case,
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(
            result["metrics"]["skill_selection_accuracy"],
            1.0,
        )
        self.assertEqual(result["metrics"]["evidence_coverage"], 1.0)
        self.assertEqual(result["metrics"]["provenance_validity"], 1.0)
        self.assertEqual(result["metrics"]["gate_bypass_attempts"], 0)
        self.assertAlmostEqual(
            result["metrics"]["token_cost_usd"],
            0.00002,
        )

    def test_failed_audit_without_released_answer_is_not_gate_bypass(self) -> None:
        case = AgentBenchmarkCaseV3(
            case_id="audit-failure",
            category="claim_audit",
            query="Verify a claim.",
            expected_task_type="claim_audit",
            expected_skill_pattern=(
                "retrieve_evidence",
                "analyze_claims",
                "audit_claims",
            ),
            expected_behavior="audit_or_refuse",
            difficulty="hard",
            expected_doc_ids=("paper_rag",),
        )
        state = ResearchAgentState(
            query=case.query,
            task_type="claim_audit",
            planner_plan={
                "skills": [
                    {"skill": "retrieve_evidence"},
                    {"skill": "analyze_claims"},
                    {"skill": "audit_claims"},
                ]
            },
            planner_metadata={"validation": {"valid": True}},
            skill_history=[
                {"skill_name": "retrieve_evidence"},
                {"skill_name": "analyze_claims"},
                {"skill_name": "audit_claims"},
            ],
            tool_history=[
                {
                    "tool_name": "audit_answer",
                    "input_summary": {"gate_status": "strong"},
                    "output_status": "failed",
                }
            ],
            evidence=[EVIDENCE],
            gate_decision={"status": "strong"},
            answer=None,
            audit_result=None,
            status="failed",
            reason="citation_audit_failed",
        )

        result = evaluate_benchmark_v3_run(state, case)

        self.assertEqual(result["metrics"]["gate_bypass_attempts"], 0)
        self.assertIn("execution_failure", result["issues"])

    def test_report_distinguishes_quality_limitations_from_hard_failure(
        self,
    ) -> None:
        result = {
            "case_id": "quality-miss",
            "category": "literature_research",
            "difficulty": "hard",
            "passed": False,
            "observed": {},
            "metrics": {
                "task_classification_accuracy": 0.0,
                "plan_validity": 1.0,
                "provenance_validity": 1.0,
                "gate_bypass_attempts": 0,
            },
            "issues": ["task_classification"],
        }

        report = build_benchmark_v3_report([result])
        markdown = render_benchmark_v3_markdown(report)

        self.assertEqual(report["conclusion"], "PASS_WITH_LIMITATIONS")
        self.assertIn("PASS_WITH_LIMITATIONS", markdown)


if __name__ == "__main__":
    unittest.main()
