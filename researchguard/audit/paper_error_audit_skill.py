# C:\Users\18449\Desktop\researchguard_workspace\researchguard\audit\paper_error_audit_skill.py
from __future__ import annotations

from collections import Counter

from researchguard.audit.base_skill import BaseSkill
from researchguard.audit.citation_in_paper_check_skill import check_citation_in_paper
from researchguard.audit.evidence_verdict_validator import detect_overclaim
from researchguard.audit.internal_consistency_check_skill import check_internal_consistency
from researchguard.audit.numerical_claim_check_skill import check_numerical_claim
from researchguard.text_utils_v2 import tokenize


RED_TYPES = {
    "unsupported_claim",
    "overclaim",
    "mis_citation",
    "invalid_citation",
    "contradiction",
    "numerical_error",
}
YELLOW_TYPES = {"partial_support", "needs_human_review"}


class PaperErrorAuditSkill(BaseSkill):
    name = "paper_error_audit_skill"
    description = "Audit input-paper claims for unsupported claims, overclaims, bad citations, contradictions, and numeric errors."
    input_schema = {"paper_claims": "list", "source_evidence": "list", "references": "list"}
    output_schema = {"paper_error_audit": "dict"}

    def run(self, case_id: str, payload: dict) -> dict:
        claims = payload.get("paper_claims", [])
        evidence = payload.get("source_evidence", [])
        references = payload.get("references", [])
        records = [
            audit_paper_claim(claim, evidence, references)
            for claim in claims
        ]
        stats = summarize_records(records)
        return {
            "paper_error_audit": {
                "records": records,
                "summary": stats,
                "key_error_points": select_key_error_points(records),
            }
        }


def audit_paper_claim(claim: dict, evidence: list[dict], references: list[dict]) -> dict:
    support = internal_support_status(claim, evidence)
    consistency = check_internal_consistency(claim, evidence)
    numeric = check_numerical_claim(claim, evidence)
    citation = check_citation_in_paper(claim, references)
    overclaim = detect_overclaim(claim.get("exact_text", ""))

    error_type = "supported"
    explanation_parts = []
    evidence_ids = list(dict.fromkeys(claim.get("candidate_evidence_ids", []) + support.get("evidence_ids", [])))

    if numeric.get("status") == "numerical_error":
        error_type = "numerical_error"
        explanation_parts.append(numeric["explanation"])
    elif consistency.get("status") == "contradiction":
        error_type = "contradiction"
        explanation_parts.append(consistency["explanation"])
    elif citation.get("status") == "invalid_citation":
        error_type = "invalid_citation"
        explanation_parts.append(citation["explanation"])
    elif citation.get("status") == "mis_citation":
        error_type = "mis_citation"
        explanation_parts.append(citation["explanation"])
    elif overclaim.get("severity") == "severe" and support.get("status") != "direct_support":
        error_type = "overclaim"
        explanation_parts.append(
            "The claim uses absolute or over-strong wording, while available evidence is not direct enough."
        )
    elif support.get("status") == "unsupported":
        error_type = "unsupported_claim"
        explanation_parts.append(support["explanation"])
    elif support.get("status") == "partial_support" or numeric.get("status") == "uncertain" or consistency.get("status") == "uncertain":
        error_type = "needs_human_review"
        explanation_parts.append("Evidence is partial or insufficient for a confident automatic decision.")
    else:
        explanation_parts.append("The claim has plausible internal support and no obvious numeric/citation conflict.")

    color = color_for_error_type(error_type)
    status = status_for_error_type(error_type)
    ref_ids = [
        row.get("matched_ref_id")
        for row in citation.get("citation_rows", [])
        if row.get("matched_ref_id")
    ]
    record = {
        "paper_claim_id": claim.get("paper_claim_id"),
        "exact_text": claim.get("exact_text", ""),
        "section": claim.get("section", "unknown"),
        "claim_type": claim.get("claim_type", "paper_claim"),
        "source_location": claim.get("source_location", ""),
        "status": status,
        "color": color,
        "error_type": error_type,
        "evidence_ids": list(dict.fromkeys(evidence_ids)),
        "ref_ids": list(dict.fromkeys(ref_ids)),
        "cited_refs": claim.get("cited_refs", []),
        "explanation": " ".join(dict.fromkeys(part for part in explanation_parts if part)),
        "checks": {
            "internal_support": support,
            "internal_consistency": consistency,
            "numerical": numeric,
            "citation": citation,
            "overclaim": overclaim,
        },
    }
    if color == "red" and record["status"] == "supported":
        record["status"] = "unsupported_claim"
    return record


def internal_support_status(claim: dict, evidence: list[dict]) -> dict:
    candidate_ids = claim.get("candidate_evidence_ids", [])
    if not candidate_ids:
        return {
            "status": "unsupported",
            "color": "red",
            "evidence_ids": [],
            "explanation": "No high-quality internal E-id evidence was matched to this paper claim.",
        }
    claim_tokens = tokenize(claim.get("exact_text", ""))
    evidence_by_id = {row.get("evidence_id"): row for row in evidence}
    best_overlap = 0.0
    usable_ids = []
    for eid in candidate_ids:
        row = evidence_by_id.get(eid)
        if not row:
            continue
        if row.get("evidence_role") in {"appendix", "reference", "noise"}:
            continue
        if row.get("quality_label", "medium") not in {"high", "medium"}:
            continue
        usable_ids.append(eid)
        ev_tokens = tokenize(row.get("clean_content") or row.get("content") or "")
        best_overlap = max(best_overlap, len(claim_tokens & ev_tokens) / max(1, len(claim_tokens)))
    if best_overlap >= 0.28:
        return {
            "status": "direct_support",
            "color": "green",
            "evidence_ids": usable_ids[:3],
            "overlap": round(best_overlap, 3),
            "explanation": "Matched internal E-id evidence directly overlaps the claim anchors.",
        }
    if usable_ids:
        return {
            "status": "partial_support",
            "color": "yellow",
            "evidence_ids": usable_ids[:3],
            "overlap": round(best_overlap, 3),
            "explanation": "Matched internal E-id evidence is related but not direct enough for green.",
        }
    return {
        "status": "unsupported",
        "color": "red",
        "evidence_ids": [],
        "overlap": round(best_overlap, 3),
        "explanation": "Matched evidence is noisy, appendix-only, or otherwise unsuitable for support.",
    }


def status_for_error_type(error_type: str) -> str:
    if error_type == "supported":
        return "supported"
    if error_type == "partial_support":
        return "partial_support"
    if error_type in YELLOW_TYPES:
        return error_type
    return error_type if error_type in RED_TYPES else "needs_human_review"


def color_for_error_type(error_type: str) -> str:
    if error_type == "supported":
        return "green"
    if error_type in YELLOW_TYPES:
        return "yellow"
    return "red"


def summarize_records(records: list[dict]) -> dict:
    colors = Counter(row.get("color") for row in records)
    errors = Counter(row.get("error_type") for row in records)
    return {
        "green_count": colors["green"],
        "yellow_count": colors["yellow"],
        "red_count": colors["red"],
        "supported_count": errors["supported"],
        "unsupported_claim_count": errors["unsupported_claim"],
        "overclaim_count": errors["overclaim"],
        "mis_citation_count": errors["mis_citation"],
        "invalid_citation_count": errors["invalid_citation"],
        "contradiction_count": errors["contradiction"],
        "numerical_error_count": errors["numerical_error"],
        "needs_human_review_count": errors["needs_human_review"],
    }


def select_key_error_points(records: list[dict]) -> list[dict]:
    priority = {
        "red": 0,
        "yellow": 1,
        "green": 2,
    }
    rows = sorted(
        [row for row in records if row.get("color") in {"red", "yellow"}],
        key=lambda row: (priority.get(row.get("color"), 9), row.get("paper_claim_id", "")),
    )
    return rows[:5]
