# C:\Users\18449\Desktop\researchguard_workspace\researchguard\cli.py
#C:\Users\18449\Desktop\researchguard_workspace\cli.py
import argparse
from pathlib import Path


def cmd_status(args):
    root = Path.cwd()
    print("ResearchGuard workspace status")
    print(f"Current directory: {root}")
    print(f"researchguard package: {(root / 'researchguard').exists()}")
    print(f"rag_agent_harness source: {(root / 'rag_agent_harness').exists()}")
    print(f"EvidenceClaw source: {(root / 'EvidenceClaw').exists()}")
    print(f"configs directory: {(root / 'configs').exists()}")
    print(f"data directory: {(root / 'data').exists()}")
    print(f"outputs directory: {(root / 'outputs').exists()}")


def cmd_check_imports(args):
    modules = [
        "researchguard",
        "researchguard.agent.legacy_agentic_rag",
        "researchguard.indexing.index_builder",
        "researchguard.audit.paper_claim_extraction_skill",
        "researchguard.audit.evidence_verdict_validator",
        "researchguard.memory.memory_store",
        "researchguard.reporting.markdown_renderer",
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


def main():
    parser = argparse.ArgumentParser(
        prog="researchguard",
        description="ResearchGuard-Agent: Agentic RAG with evidence auditing."
    )

    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser("status", help="Show project status.")
    status_parser.set_defaults(func=cmd_status)

    import_parser = subparsers.add_parser("check-imports", help="Check core module imports.")
    import_parser.set_defaults(func=cmd_check_imports)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()

