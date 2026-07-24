# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_agent_evaluation_v1.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.agent import ResearchAgentState
from researchguard.evaluation import (
    AgentEvaluationCase,
    AgentEvaluator,
    write_evaluation_report,
)


REGISTERED_TOOLS = (
    "search_scholarly_sources",
    "retrieve_evidence",
    "assess_evidence",
    "generate_grounded_answer",
    "audit_answer",
)
EVIDENCE = {
    "chunk_id": "synthetic-paper::chunk-1",
    "doc_id": "synthetic-paper",
    "section": "method",
    "page": 3,
    "content": "The method uses evidence-aware retrieval and verification.",
    "source": "synthetic_paper.pdf",
    "score": 0.95,
    "provenance": {"source_block_ids": ["p3-b1"]},
}


def _tool_history(
    tool_names: tuple[str, ...],
    *,
    rejected_last: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "tool_name": name,
            "output_status": (
                "rejected"
                if rejected_last and index == len(tool_names)
                else "success"
            ),
            "latency_ms": float(index),
            "trace_id": f"synthetic-trace-{index}",
        }
        for index, name in enumerate(tool_names, start=1)
    ]


def build_synthetic_runs() -> list[
    tuple[ResearchAgentState, AgentEvaluationCase, dict[str, Any]]
]:
    specifications = (
        (
            "literature-review",
            "literature_review",
            (
                "search_scholarly_sources",
                "retrieve_evidence",
                "assess_evidence",
                "generate_grounded_answer",
                "audit_answer",
            ),
            "completed",
        ),
        (
            "paper-comparison",
            "paper_comparison",
            (
                "search_scholarly_sources",
                "retrieve_evidence",
                "retrieve_evidence",
                "assess_evidence",
                "generate_grounded_answer",
                "audit_answer",
            ),
            "completed",
        ),
        (
            "claim-audit-refusal",
            "claim_audit",
            ("retrieve_evidence", "assess_evidence"),
            "rejected",
        ),
    )
    runs = []
    for case_id, workflow, tool_names, status in specifications:
        completed = status == "completed"
        state = ResearchAgentState(
            query=f"Synthetic benchmark for {workflow}",
            task_type=workflow,
            workflow_name=workflow,
            workflow_steps=[
                {
                    "step": index,
                    "tool_name": name,
                    "status": "success",
                    "trace_id": f"synthetic-trace-{index}",
                }
                for index, name in enumerate(tool_names, start=1)
            ],
            tool_history=_tool_history(
                tool_names,
                rejected_last=not completed,
            ),
            evidence=[EVIDENCE],
            audit_result=(
                {
                    "claims": [
                        {
                            "text": "The method uses evidence-aware retrieval.",
                            "support_level": "supported",
                        }
                    ]
                }
                if completed
                else None
            ),
            memory_status={
                "enabled": True,
                "persisted": True,
                "errors": [],
            },
            current_step=len(tool_names),
            status=status,
            reason=None if completed else "insufficient_evidence",
        )
        case = AgentEvaluationCase(
            case_id=case_id,
            query=state.query,
            expected_task_type=workflow,
            expected_workflow=workflow,
            expected_tools=tool_names,
            relevant_evidence_ids=(EVIDENCE["chunk_id"],),
            expected_status=status,
        )
        snapshot = {
            "run": {"run_id": state.run_id},
            "evidence_ledger": (
                [{"claim_id": f"{state.run_id}:claim-1"}] if completed else []
            ),
            "failures": (
                [] if completed else [{"failure_id": f"{state.run_id}:failure-1"}]
            ),
        }
        runs.append((state, case, snapshot))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ResearchGuard Agent Evaluation v1 synthetic benchmark."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/agent_evaluation_v1",
    )
    args = parser.parse_args()

    report = AgentEvaluator(REGISTERED_TOOLS).evaluate_many(build_synthetic_runs())
    json_path, markdown_path = write_evaluation_report(report, args.output_dir)
    print(report.to_json(indent=2))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
