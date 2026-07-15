# C:\Users\18449\Desktop\researchguard_workspace\scripts\run_functional_validation.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"C:\Users\18449\Desktop\researchguard_workspace")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from typing import Any

from researchguard.audit.answer_auditor import AnswerAuditor
from researchguard.audit.paper_claim_extraction_skill import PaperClaimExtractionSkill
from researchguard.memory.memory_store import MemoryStore
from researchguard.reporting.audit_report import render_audit_markdown


ROOT = Path(r"C:\Users\18449\Desktop\researchguard_workspace")
OUTPUT_DIR = ROOT / "outputs" / "functional_validation"
MEMORY_DIR = OUTPUT_DIR / "memory"


def instantiate_paper_claim_skill() -> PaperClaimExtractionSkill:
    """Instantiate migrated EvidenceClaw skill with tolerance for constructor differences."""
    try:
        return PaperClaimExtractionSkill()
    except TypeError:
        memory = MemoryStore(memory_dir=MEMORY_DIR, case_output_dir=OUTPUT_DIR)
        return PaperClaimExtractionSkill(memory)


def run_paper_claim_extraction() -> dict[str, Any]:
    """Run migrated EvidenceClaw paper claim extraction on a small simulated paper."""
    parsed_pages = [
        {
            "page": 1,
            "text": (
                "We propose a Teacher-Student-Controller framework for improving chain-of-thought data. "
                "The framework uses student feedback to locate difficult reasoning steps and repairs missing transitions. "
                "Our method improves Qwen2.5-Math-1.5B from 70.0 to 73.8 on math500 strict evaluation."
            ),
        },
        {
            "page": 2,
            "text": (
                "The repaired CoT also improves Qwen2.5-Math-7B from 76.8 to 78.8. "
                "However, the current experiment does not prove that the method works for all reasoning tasks."
            ),
        },
    ]

    source_evidence = [
        {
            "evidence_id": "E001",
            "clean_content": (
                "The Teacher-Student-Controller framework uses student feedback to locate difficult reasoning steps "
                "and repairs missing transitions."
            ),
            "section_guess": "method",
            "quality_label": "high",
            "evidence_role": "method",
        },
        {
            "evidence_id": "E002",
            "clean_content": (
                "On math500 strict evaluation, repaired CoT improves Qwen2.5-Math-1.5B from 70.0 to 73.8 "
                "and Qwen2.5-Math-7B from 76.8 to 78.8."
            ),
            "section_guess": "results",
            "quality_label": "high",
            "evidence_role": "result",
        },
    ]

    payload = {
        "paper_title": "Student-Oriented CoT Optimization",
        "topic": "chain-of-thought optimization for student models",
        "parsed_pages": parsed_pages,
        "source_evidence": source_evidence,
    }

    skill = instantiate_paper_claim_skill()
    result = skill.run(case_id="functional_validation_case", payload=payload)

    return {
        "payload": payload,
        "paper_claim_extraction_result": result,
    }


def run_answer_audit(source_evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Run ResearchGuard answer audit on a generated answer."""
    question = "What does the paper show about repaired CoT?"

    answer = (
        "The paper proposes a Teacher-Student-Controller framework for repairing chain-of-thought data. "
        "It reports that repaired CoT improves Qwen2.5-Math-1.5B from 70.0 to 73.8 on math500 strict. "
        "It also proves that the method works for all reasoning tasks."
    )

    auditor = AnswerAuditor()
    audit_result = auditor.audit(answer=answer, evidence_nodes=source_evidence)

    report = render_audit_markdown(
        question=question,
        answer=answer,
        audit_result=audit_result,
    )

    return {
        "question": question,
        "answer": answer,
        "audit_result": audit_result,
        "audit_report_markdown": report,
    }


def run_memory_validation(audit_result: dict[str, Any]) -> dict[str, Any]:
    """Validate migrated EvidenceClaw MemoryStore by writing real JSON/JSONL memory files."""
    store = MemoryStore(memory_dir=MEMORY_DIR, case_output_dir=OUTPUT_DIR)

    store.append_source(
        {
            "case_id": "functional_validation_case",
            "source_id": "S001",
            "title": "Student-Oriented CoT Optimization",
            "source_type": "simulated_paper",
        }
    )

    for idx, record in enumerate(audit_result.get("records", []), start=1):
        store.append_evidence(
            {
                "case_id": "functional_validation_case",
                "evidence_id": f"AE{idx:03d}",
                "content": record.get("claim", ""),
                "verdict": record.get("verdict", "unknown"),
                "risk_type": record.get("risk_type", "unknown"),
            }
        )

    store.append_failure(
        case_id="functional_validation_case",
        failure_type="demo_failure_record",
        message="This is a validation record showing that failure memory can be written.",
        details={"purpose": "functional validation"},
    )

    with store.trace_skill(
        case_id="functional_validation_case",
        skill_name="answer_auditor",
        input_summary="Run claim-level audit on generated answer.",
    ) as trace:
        trace.success("Claim-level audit finished successfully.")

    snapshot = store.snapshot("functional_validation_case")

    return {
        "memory_snapshot": snapshot,
        "memory_dir": str(MEMORY_DIR),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    claim_result = run_paper_claim_extraction()
    source_evidence = claim_result["payload"]["source_evidence"]

    audit_bundle = run_answer_audit(source_evidence=source_evidence)
    memory_result = run_memory_validation(audit_bundle["audit_result"])

    outputs = {
        "paper_claim_extraction": claim_result["paper_claim_extraction_result"],
        "answer_audit": audit_bundle["audit_result"],
        "memory_validation": memory_result,
    }

    (OUTPUT_DIR / "functional_validation_result.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "functional_audit_report.md").write_text(
        audit_bundle["audit_report_markdown"],
        encoding="utf-8",
        newline="\n",
    )

    print("Functional validation finished.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Result JSON: {OUTPUT_DIR / 'functional_validation_result.json'}")
    print(f"Audit report: {OUTPUT_DIR / 'functional_audit_report.md'}")
    print(f"Memory directory: {MEMORY_DIR}")
    print("")
    print("Paper claims extracted:", len(outputs["paper_claim_extraction"].get("paper_claims", [])))
    print("Audit summary:")
    print(json.dumps(outputs["answer_audit"].get("summary", {}), ensure_ascii=False, indent=2))
    print("")
    print("Memory snapshot:")
    print(json.dumps(outputs["memory_validation"].get("memory_snapshot", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()