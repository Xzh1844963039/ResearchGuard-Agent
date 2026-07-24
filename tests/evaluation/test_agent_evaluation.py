# C:\Users\18449\Desktop\researchguard_workspace\tests\evaluation\test_agent_evaluation.py
from __future__ import annotations

import json
import tempfile
import unittest

from researchguard.agent import ResearchAgentState
from researchguard.evaluation import (
    AgentEvaluationCase,
    AgentEvaluator,
    write_evaluation_report,
)


TOOLS = (
    "search_scholarly_sources",
    "retrieve_evidence",
    "assess_evidence",
    "generate_grounded_answer",
    "audit_answer",
)
EVIDENCE = {
    "chunk_id": "paper-crag::chunk-7",
    "doc_id": "paper-crag",
    "section": "method",
    "page": 5,
    "content": "CRAG uses a retrieval evaluator.",
    "source": "paper_crag.pdf",
    "score": 0.94,
    "provenance": {"source_block_ids": ["p5-b2"]},
}


def tool_history(names: tuple[str, ...], *, rejected_last: bool = False) -> list[dict[str, object]]:
    return [
        {
            "tool_name": name,
            "output_status": "rejected" if rejected_last and index == len(names) else "success",
            "latency_ms": 1.0,
            "trace_id": f"trace-{index}",
        }
        for index, name in enumerate(names, start=1)
    ]


def snapshot(run_id: str, *, completed: bool) -> dict[str, object]:
    return {
        "run": {"run_id": run_id},
        "evidence_ledger": [{"claim_id": f"{run_id}:claim-1"}] if completed else [],
        "failures": [] if completed else [{"failure_id": f"{run_id}:failure-1"}],
    }


class AgentEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = AgentEvaluator(TOOLS)

    def test_synthetic_research_workflows_and_refusal_pass(self) -> None:
        review_tools = (
            "search_scholarly_sources",
            "retrieve_evidence",
            "assess_evidence",
            "generate_grounded_answer",
            "audit_answer",
        )
        comparison_tools = (
            "search_scholarly_sources",
            "retrieve_evidence",
            "retrieve_evidence",
            "assess_evidence",
            "generate_grounded_answer",
            "audit_answer",
        )
        audit_tools = ("retrieve_evidence", "assess_evidence")
        specifications = (
            ("literature-review", "literature_review", review_tools, "completed"),
            ("paper-comparison", "paper_comparison", comparison_tools, "completed"),
            ("claim-audit-refusal", "claim_audit", audit_tools, "rejected"),
        )
        runs = []
        for case_id, workflow, tools, status in specifications:
            state = ResearchAgentState(
                query=f"Synthetic {workflow} query",
                task_type=workflow,
                workflow_name=workflow,
                workflow_steps=[
                    {"step": index, "tool_name": name, "status": "success"}
                    for index, name in enumerate(tools, start=1)
                ],
                tool_history=tool_history(
                    tools,
                    rejected_last=status == "rejected",
                ),
                evidence=[EVIDENCE],
                audit_result=(
                    {
                        "claims": [
                            {
                                "text": "CRAG uses a retrieval evaluator.",
                                "support_level": "supported",
                            }
                        ]
                    }
                    if status == "completed"
                    else None
                ),
                memory_status={
                    "enabled": True,
                    "persisted": True,
                    "errors": [],
                },
                current_step=len(tools),
                status=status,
                reason="insufficient_evidence" if status == "rejected" else None,
            )
            case = AgentEvaluationCase(
                case_id=case_id,
                query=state.query,
                expected_task_type=workflow,
                expected_workflow=workflow,
                expected_tools=tools,
                relevant_evidence_ids=(EVIDENCE["chunk_id"],),
                expected_status=status,
            )
            runs.append((state, case, snapshot(state.run_id, completed=status == "completed")))

        report = self.evaluator.evaluate_many(runs)

        self.assertTrue(report.passed)
        self.assertEqual(report.case_count, 3)
        self.assertEqual(report.aggregate_metrics["task_classification_accuracy"], 1.0)
        self.assertEqual(report.aggregate_metrics["workflow_selection_accuracy"], 1.0)
        self.assertEqual(report.aggregate_metrics["invalid_tool_rate"], 0.0)
        self.assertEqual(report.aggregate_metrics["provenance_validity"], 1.0)
        self.assertEqual(report.aggregate_metrics["memory_persistence_success"], 1.0)

    def test_detects_wrong_plan_invalid_tool_and_bad_provenance(self) -> None:
        state = ResearchAgentState(
            query="Compare two papers.",
            task_type="qa",
            tool_history=[
                {
                    "tool_name": "unknown_tool",
                    "output_status": "success",
                    "trace_id": "trace-invalid",
                }
            ],
            evidence=[{**EVIDENCE, "page": None}],
            status="completed",
        )
        case = AgentEvaluationCase(
            case_id="bad-case",
            query=state.query,
            expected_task_type="paper_comparison",
            expected_workflow="paper_comparison",
            forbidden_tools=("unknown_tool",),
            relevant_evidence_ids=(EVIDENCE["chunk_id"],),
        )

        result = self.evaluator.evaluate(state, case)

        self.assertFalse(result.passed)
        self.assertFalse(result.metrics["task_classification_accuracy"].passed)
        self.assertFalse(result.metrics["invalid_tool_rate"].passed)
        self.assertFalse(result.metrics["provenance_validity"].passed)

    def test_report_is_json_serializable_and_written(self) -> None:
        state = ResearchAgentState(
            query="Review literature.",
            task_type="literature_review",
            workflow_name="literature_review",
            memory_status={"enabled": False},
            status="completed",
        )
        case = AgentEvaluationCase(
            case_id="serialization",
            query=state.query,
            expected_task_type="literature_review",
            expected_workflow="literature_review",
        )
        report = self.evaluator.evaluate_many([(state, case, None)])

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, markdown_path = write_evaluation_report(report, temp_dir)
            payload = json.loads(json_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["case_count"], 1)
            self.assertTrue(markdown_path.exists())

    def test_runtime_evaluation_marks_accuracy_as_unavailable(self) -> None:
        state = ResearchAgentState(query="Runtime query", status="completed")

        result = self.evaluator.evaluate_runtime(state)

        self.assertIsNone(result.metrics["task_classification_accuracy"].value)
        self.assertIsNone(result.metrics["workflow_selection_accuracy"].passed)
        self.assertIsNone(result.metrics["unsupported_claim_rate"].value)


if __name__ == "__main__":
    unittest.main()
