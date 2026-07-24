# C:\Users\18449\Desktop\researchguard_workspace\researchguard\cli.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from researchguard.audit.answer_auditor import AnswerAuditor
from researchguard.reporting.audit_report import render_audit_markdown


DEFAULT_PIPELINE_CONFIG = "configs/pipeline_v1.yaml"
DEFAULT_RESEARCH_MEMORY_ROOT = "data/memory/research_runs"


def cmd_status(args: argparse.Namespace) -> None:
    root = Path.cwd()

    print("ResearchGuard workspace status")
    print(f"Current directory: {root}")
    print(f"researchguard package: {(root / 'researchguard').exists()}")
    print(f"rag_agent_harness source: {(root / 'rag_agent_harness').exists()}")
    print(f"EvidenceClaw source: {(root / 'EvidenceClaw').exists()}")
    print(f"configs directory: {(root / 'configs').exists()}")
    print(f"data directory: {(root / 'data').exists()}")
    print(f"outputs directory: {(root / 'outputs').exists()}")


def cmd_check_imports(args: argparse.Namespace) -> None:
    modules = [
        "researchguard",
        "researchguard.agent.legacy_agentic_rag",
        "researchguard.indexing.index_builder",
        "researchguard.audit.paper_claim_extraction_skill",
        "researchguard.audit.evidence_verdict_validator",
        "researchguard.memory.memory_store",
        "researchguard.reporting.markdown_renderer",
        "researchguard.audit.answer_auditor",
        "researchguard.reporting.audit_report",
    ]

    ok = 0
    failed = 0

    for module_name in modules:
        try:
            __import__(module_name)
            print(f"[OK] {module_name}")
            ok += 1
        except Exception as exc:
            print(f"[FAIL] {module_name}")
            print(f"       {type(exc).__name__}: {exc}")
            failed += 1

    print(f"\nImport check finished: ok={ok}, failed={failed}")

    if failed > 0:
        raise SystemExit(1)


def cmd_smoke_audit(args: argparse.Namespace) -> None:
    question = "What is the main result of the student-oriented CoT optimization study?"

    answer = (
        "The study proposes a Teacher-Student-Controller framework for improving chain-of-thought data. "
        "It reports that repaired CoT improves Qwen2.5-Math-1.5B from 70.0 to 73.8 on math500 strict. "
        "It also proves that the method works for all reasoning tasks without failure."
    )

    evidence_nodes = [
        {
            "evidence_id": "E001",
            "text": (
                "The Teacher-Student-Controller framework uses student feedback to locate difficult "
                "reasoning steps and repairs missing transitions, compressed derivations, and unclear expressions."
            ),
        },
        {
            "evidence_id": "E002",
            "text": (
                "On math500 strict evaluation, repaired CoT improves Qwen2.5-Math-1.5B from 70.0 to 73.8 "
                "and improves Qwen2.5-Math-7B from 76.8 to 78.8."
            ),
        },
    ]

    auditor = AnswerAuditor()
    audit_result = auditor.audit(answer=answer, evidence_nodes=evidence_nodes)

    output_dir = Path("outputs") / "smoke_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "audit_result.json"
    report_path = output_dir / "audit_report.md"

    result_path.write_text(json.dumps(audit_result, ensure_ascii=False, indent=2), encoding="utf-8")

    report = render_audit_markdown(question=question, answer=answer, audit_result=audit_result)
    report_path.write_text(report, encoding="utf-8", newline="\n")

    print("Smoke audit finished.")
    print(f"Audit JSON: {result_path}")
    print(f"Audit report: {report_path}")
    print("")
    print(json.dumps(audit_result.get("summary", {}), ensure_ascii=False, indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    from researchguard.pipeline import run_pipeline

    result = run_pipeline(args.query, config_path=args.config)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Pipeline result: {output_path}")
    else:
        print(rendered)


def _load_json_file(path: str | None) -> object | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_agent_run(args: argparse.Namespace) -> None:
    from researchguard.agent import AgentPolicy, BoundedResearchAgentController
    from researchguard.evaluation import AgentEvaluator
    from researchguard.tools import build_default_registry
    from researchguard.tracing import TraceCollector

    answer_artifact = _load_json_file(args.answer_json)
    if answer_artifact is not None and not isinstance(answer_artifact, dict):
        raise ValueError("--answer-json must contain a JSON object.")

    evidence_value = _load_json_file(args.evidence_json)
    if isinstance(evidence_value, dict):
        evidence_value = evidence_value.get("evidence")
    if evidence_value is not None and not isinstance(evidence_value, list):
        raise ValueError("--evidence-json must contain a JSON list or an object with an evidence list.")
    workflow_input = _load_json_file(args.workflow_input_json)
    if workflow_input is not None and not isinstance(workflow_input, dict):
        raise ValueError("--workflow-input-json must contain a JSON object.")

    policy = AgentPolicy(
        max_steps=args.max_steps,
        max_tool_calls=args.max_tool_calls,
        max_retry=args.max_retry,
        max_plan_revisions=args.max_plan_revisions,
        timeout=args.timeout,
    )
    registry = build_default_registry(args.config)
    controller = BoundedResearchAgentController(
        registry=registry,
        policy=policy,
        config_path=args.config,
    )
    state = controller.run(
        args.query,
        task_type=args.task_type,
        answer_artifact=answer_artifact,
        evidence=evidence_value,
        workflow_input=workflow_input,
    )
    memory_snapshot = (
        controller.memory.show(state.run_id)
        if controller.memory is not None
        and (args.show_memory or args.show_trace or args.show_evaluation)
        else None
    )
    report = {
        "Agent Plan": state.plan,
        "Workflow Selected": state.workflow_name,
        "Workflow Steps": state.workflow_steps,
        "Workflow Result": state.workflow_result,
        "Research Memory": state.memory_status,
        "Tool Calls": state.tool_history,
        "Observations": state.observations,
        "Candidate Papers": state.candidate_papers,
        "Final Status": {
            "status": state.status,
            "reason": state.reason,
            "current_step": state.current_step,
            "run_id": state.run_id,
        },
        "Answer": state.answer,
        "Citation": state.audit_result,
    }
    if args.show_trace:
        report["Agent Trace"] = TraceCollector().collect(
            state,
            memory_snapshot=memory_snapshot,
        ).to_dict()
    if args.show_memory:
        report["Memory Context"] = state.memory_context
        report["Memory Snapshot"] = memory_snapshot
    if args.show_evaluation:
        report["Agent Evaluation"] = AgentEvaluator(
            registry.names
        ).evaluate_runtime(
            state,
            memory_snapshot=memory_snapshot,
        ).to_dict()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Agent result: {output_path}")
    else:
        print(rendered)
    if args.state_output:
        state_path = state.save(args.state_output)
        print(f"Agent state: {state_path}")
    if state.status == "failed":
        raise SystemExit(1)


def cmd_memory_list(args: argparse.Namespace) -> None:
    from researchguard.memory import ResearchRunStore

    store = ResearchRunStore(args.memory_root)
    records = store.list_runs(
        limit=args.limit,
        workflow=args.workflow,
        status=args.status,
    )
    print(
        json.dumps(
            {"count": len(records), "runs": [record.to_dict() for record in records]},
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_memory_show(args: argparse.Namespace) -> None:
    from researchguard.memory import ResearchMemory

    result = ResearchMemory(args.memory_root).show(args.run_id)
    if result is None:
        print(json.dumps({"error": "run_not_found", "run_id": args.run_id}, indent=2))
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_memory_search(args: argparse.Namespace) -> None:
    from researchguard.memory import ResearchRunStore

    records = ResearchRunStore(args.memory_root).find_previous_runs(
        args.query,
        workflow=args.workflow,
        since=args.since,
        limit=args.limit,
    )
    print(
        json.dumps(
            {"count": len(records), "runs": [record.to_dict() for record in records]},
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="researchguard",
        description="ResearchGuard-Agent: Agentic RAG with evidence auditing.",
    )

    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser("status", help="Show project status.")
    status_parser.set_defaults(func=cmd_status)

    import_parser = subparsers.add_parser("check-imports", help="Check core module imports.")
    import_parser.set_defaults(func=cmd_check_imports)

    smoke_parser = subparsers.add_parser("smoke-audit", help="Run a local claim-level audit smoke test.")
    smoke_parser.set_defaults(func=cmd_smoke_audit)

    run_parser = subparsers.add_parser("run", help="Run the unified ResearchGuard pipeline.")
    run_parser.add_argument("--query", required=True, help="Question to process.")
    run_parser.add_argument("--config", default=DEFAULT_PIPELINE_CONFIG, help="Pipeline YAML config path.")
    run_parser.add_argument("--output", help="Optional JSON output path.")
    run_parser.set_defaults(func=cmd_run)

    agent_parser = subparsers.add_parser(
        "agent-run",
        help="Run the bounded ResearchGuard single-agent controller.",
    )
    agent_parser.add_argument("--query", required=True, help="Research question or audit instruction.")
    agent_parser.add_argument(
        "--task-type",
        choices=(
            "qa",
            "comparison",
            "audit",
            "literature_search",
            "literature_review",
            "paper_comparison",
            "claim_audit",
        ),
        help="Optional task type override; otherwise inferred deterministically.",
    )
    agent_parser.add_argument(
        "--config",
        default=DEFAULT_PIPELINE_CONFIG,
        help="Pipeline YAML config used by the registered tools.",
    )
    agent_parser.add_argument(
        "--answer-json",
        help="Provenance-bearing answer artifact required for an explicit audit task.",
    )
    agent_parser.add_argument(
        "--evidence-json",
        help="Optional canonical evidence JSON for an audit task.",
    )
    agent_parser.add_argument(
        "--workflow-input-json",
        help="Optional workflow parameters such as papers, doc_ids, or comparison dimensions.",
    )
    agent_parser.add_argument("--output", help="Optional agent report JSON path.")
    agent_parser.add_argument("--state-output", help="Optional resumable agent state JSON path.")
    agent_parser.add_argument("--max-steps", type=int, default=6)
    agent_parser.add_argument("--max-tool-calls", type=int, default=10)
    agent_parser.add_argument("--max-retry", type=int, default=2)
    agent_parser.add_argument("--max-plan-revisions", type=int, default=2)
    agent_parser.add_argument("--timeout", type=float, default=120.0)
    agent_parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Include the complete JSON-serializable execution trace.",
    )
    agent_parser.add_argument(
        "--show-memory",
        action="store_true",
        help="Include advisory memory context and the persisted run snapshot.",
    )
    agent_parser.add_argument(
        "--show-evaluation",
        action="store_true",
        help="Include runtime agent health metrics without benchmark accuracy claims.",
    )
    agent_parser.set_defaults(func=cmd_agent_run)

    memory_list_parser = subparsers.add_parser(
        "memory-list",
        help="List persisted research runs.",
    )
    memory_list_parser.add_argument("--limit", type=int, default=20)
    memory_list_parser.add_argument("--workflow")
    memory_list_parser.add_argument(
        "--status",
        choices=("created", "planned", "running", "completed", "rejected", "failed"),
    )
    memory_list_parser.add_argument(
        "--memory-root",
        default=DEFAULT_RESEARCH_MEMORY_ROOT,
    )
    memory_list_parser.set_defaults(func=cmd_memory_list)

    memory_show_parser = subparsers.add_parser(
        "memory-show",
        help="Show one research run with evidence ledger and failures.",
    )
    memory_show_parser.add_argument("--run-id", required=True)
    memory_show_parser.add_argument(
        "--memory-root",
        default=DEFAULT_RESEARCH_MEMORY_ROOT,
    )
    memory_show_parser.set_defaults(func=cmd_memory_show)

    memory_search_parser = subparsers.add_parser(
        "memory-search",
        help="Find previous research runs by keyword.",
    )
    memory_search_parser.add_argument("--query", required=True)
    memory_search_parser.add_argument("--workflow")
    memory_search_parser.add_argument("--since")
    memory_search_parser.add_argument("--limit", type=int, default=20)
    memory_search_parser.add_argument(
        "--memory-root",
        default=DEFAULT_RESEARCH_MEMORY_ROOT,
    )
    memory_search_parser.set_defaults(func=cmd_memory_search)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
