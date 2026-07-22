# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_demo_v1.py
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.utils import (
    answer_view,
    audit_claims,
    evidence_sufficiency_view,
    retrieval_hits,
    sanitize_for_display,
    stage_rows,
    validate_pipeline_result,
)
from researchguard.pipeline import ResearchGuardPipeline, load_pipeline_settings


def check_streamlit_startup(app_path: Path) -> dict[str, Any]:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    exceptions = [str(item.value) for item in app.exception]
    return {
        "passed": not exceptions and len(app.title) == 1 and app.title[0].value == "ResearchGuard",
        "exceptions": exceptions,
        "title": app.title[0].value if app.title else None,
        "button_labels": [item.label for item in app.button],
    }


def result_checks(name: str, result: dict[str, Any]) -> dict[str, Any]:
    validated = validate_pipeline_result(result)
    evidence = evidence_sufficiency_view(validated)
    answer = answer_view(validated)
    claims = audit_claims(validated)
    statuses = {item["key"]: item["status"] for item in stage_rows(validated)}
    hits = retrieval_hits(validated)
    if name == "strong":
        passed = (
            validated["final_status"] == "grounded"
            and evidence["support_level"] == "strong"
            and bool(answer["answer"])
            and not answer["refused"]
            and bool(answer["citations"])
            and bool(claims)
            and statuses["answer_generation"] == "completed"
            and statuses["citation_audit"] == "completed"
        )
    else:
        passed = (
            validated["final_status"] == "rejected"
            and evidence["support_level"] == "unsupported"
            and answer["refused"]
            and statuses["answer_generation"] == "skipped"
            and statuses["citation_audit"] == "skipped"
            and not claims
        )
    return {
        "passed": passed,
        "final_status": validated["final_status"],
        "support_level": evidence["support_level"],
        "answer_refused": answer["refused"],
        "answer_length": len(answer["answer"]),
        "citation_count": len(answer["citations"]),
        "claim_count": len(claims),
        "evidence_count": len(hits),
        "stage_statuses": statuses,
        "latency_ms": float(validated["pipeline"].get("latency_ms") or 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Streamlit Demo v1.")
    parser.add_argument("--config", default="configs/pipeline_v1.yaml")
    parser.add_argument("--output", default="outputs/demo_validation_v1/demo_validation_summary.json")
    args = parser.parse_args()

    startup = check_streamlit_startup(PROJECT_ROOT / "demo" / "app.py")
    _, settings = load_pipeline_settings(args.config)
    pipeline = ResearchGuardPipeline(settings)
    queries = {
        "strong": "What is the difference between RAG-Sequence and RAG-Token?",
        "unsupported": "Does any indexed paper describe quantum error correction for superconducting qubits?",
    }
    results = {name: pipeline.run(query) for name, query in queries.items()}
    cases = {name: result_checks(name, result) for name, result in results.items()}
    sanitized = sanitize_for_display(results)
    serialized = json.dumps(sanitized, ensure_ascii=False)
    display_security = {
        "passed": "C:\\Users\\" not in serialized
        and "OPENAI_API_KEY" not in serialized
        and not any(key in serialized for key in ("index_dir", "cache_directory", "model_path")),
    }
    hard_checks = {
        "streamlit_startup_failure": 0 if startup["passed"] else 1,
        "strong_render_contract_failure": 0 if cases["strong"]["passed"] else 1,
        "unsupported_render_contract_failure": 0 if cases["unsupported"]["passed"] else 1,
        "display_sanitization_failure": 0 if display_security["passed"] else 1,
    }
    status = "PASS" if sum(hard_checks.values()) == 0 else "FAIL"
    summary = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "startup": startup,
        "cases": cases,
        "display_security": display_security,
        "hard_checks": hard_checks,
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
