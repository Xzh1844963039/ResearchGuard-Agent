# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_holdout_v1.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.agent import BoundedResearchAgentController  # noqa: E402
from researchguard.evaluation import (  # noqa: E402
    build_holdout_report,
    evaluate_holdout_run,
    load_holdout_questions,
    render_holdout_markdown,
)
from scripts.validate_agent_benchmark_v3 import (  # noqa: E402
    build_real_evidence_registry,
)


DEFAULT_QUESTIONS = (
    PROJECT_ROOT / "data" / "eval" / "holdout" / "blind_questions.jsonl"
)
DEFAULT_PIPELINE = PROJECT_ROOT / "configs" / "holdout_pipeline_v1.yaml"
DEFAULT_PLANNER = PROJECT_ROOT / "configs" / "planner_deterministic.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "holdout_validation_v1"
DEFAULT_INDEX = PROJECT_ROOT / "data" / "indexes" / "holdout_v1"


def corpus_status(index_root: Path) -> dict[str, Any]:
    manifest_path = index_root / "index_manifest.json"
    corpus_path = index_root / "corpus_manifest.jsonl"
    if not manifest_path.exists() or not corpus_path.exists():
        return {
            "ready": False,
            "paper_count": 0,
            "reason": "Dedicated holdout index has not been built.",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = {
        json.loads(line).get("doc_id")
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    ready = manifest.get("build_status") == "complete" and len(documents) >= 10
    return {
        "ready": ready,
        "paper_count": len(documents),
        "chunk_count": int(manifest.get("chunk_count", 0)),
        "corpus_fingerprint": manifest.get("corpus_fingerprint"),
        "reason": (
            None
            if ready
            else "Holdout index is incomplete or contains fewer than 10 papers."
        ),
    }


def workflow_input(question: Any) -> dict[str, Any]:
    if question.expected_task_type != "paper_comparison":
        return {}
    return {"candidate_limit": 3}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the blind, corpus-isolated holdout evaluation."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--planner-config", type=Path, default=DEFAULT_PLANNER)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    status = corpus_status(args.index_root)
    questions = load_holdout_questions(args.questions)
    if args.limit is not None:
        questions = questions[: max(0, args.limit)]
    results = []
    if status["ready"]:
        controller = BoundedResearchAgentController(
            registry=build_real_evidence_registry(
                args.pipeline_config,
                online_discovery=False,
            ),
            config_path=args.pipeline_config,
            planner_config_path=args.planner_config,
            memory_enabled=False,
        )
        for index, question in enumerate(questions, start=1):
            try:
                state = controller.run(
                    question.query,
                    workflow_input=workflow_input(question),
                )
                row = evaluate_holdout_run(state, question)
            except Exception as exc:
                row = {
                    "question_id": question.question_id,
                    "passed": False,
                    "expected": question.to_dict(),
                    "observed": {},
                    "metrics": {},
                    "limitations": ["execution_exception"],
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            results.append(row)
            print(
                f"[{index:02d}/{len(questions)}] {question.question_id}: "
                f"{'PASS' if row['passed'] else 'CHECK'}",
                flush=True,
            )

    report = build_holdout_report(results, corpus_status=status)
    markdown = render_holdout_markdown(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "holdout_evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "holdout_evaluation_report.md").write_text(
        markdown,
        encoding="utf-8",
        newline="\n",
    )
    print(markdown)
    if report["conclusion"] in {"FAIL", "NOT_RUN"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
