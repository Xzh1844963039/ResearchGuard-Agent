# C:\Users\18449\Desktop\researchguard_workspace\researchguard\audit\citation_in_paper_check_skill.py
from __future__ import annotations

from researchguard.audit.base_skill import BaseSkill
from researchguard.text_utils_v2 import tokenize


class CitationInPaperCheckSkill(BaseSkill):
    name = "citation_in_paper_check_skill"
    description = "Check whether citations attached to input-paper claims have plausible retrieved support."
    input_schema = {"paper_claim": "dict", "references": "list"}
    output_schema = {"status": "supported|mis_citation|invalid_citation|needs_human_review"}

    def run(self, case_id: str, payload: dict) -> dict:
        return check_citation_in_paper(payload.get("paper_claim", {}), payload.get("references", []))


def check_citation_in_paper(claim: dict, references: list[dict]) -> dict:
    citations = claim.get("cited_refs") or []
    if not citations:
        return {
            "status": "not_applicable",
            "color": "green",
            "explanation": "Claim does not carry an explicit in-paper citation.",
            "citation_rows": [],
        }
    if not references:
        return {
            "status": "invalid_citation",
            "color": "red",
            "explanation": "Claim cites references, but no reference metadata was retrieved for this run.",
            "citation_rows": [
                {"cited_ref": ref, "exists": False, "supports_claim": False, "status": "invalid_citation"}
                for ref in citations
            ],
        }

    claim_tokens = tokenize(claim.get("exact_text", ""))
    best_rows = []
    support_found = False
    for citation in citations:
        best_ref = None
        best_score = 0.0
        for ref in references:
            ref_text = " ".join([str(ref.get("title") or ""), str(ref.get("abstract") or "")])
            ref_tokens = tokenize(ref_text)
            score = len(claim_tokens & ref_tokens) / max(1, len(claim_tokens))
            if best_ref is None or score > best_score:
                best_score = score
                best_ref = ref
        exists = bool(best_ref and best_ref.get("title"))
        supports = exists and best_score >= 0.18 and not best_ref.get("is_demo_reference")
        support_found = support_found or supports
        best_rows.append(
            {
                "cited_ref": citation,
                "matched_ref_id": best_ref.get("ref_id") if best_ref else None,
                "exists": exists,
                "supports_claim": supports,
                "status": "supported" if supports else "mis_citation" if exists else "invalid_citation",
                "overlap": round(best_score, 3),
                "explanation": (
                    "Matched retrieved reference appears to support the claim."
                    if supports
                    else "Matched retrieved reference is weak or only topically related."
                    if exists
                    else "No retrieved reference metadata matched this citation."
                ),
            }
        )
    if support_found:
        return {
            "status": "supported",
            "color": "green",
            "explanation": "At least one cited reference has plausible retrieved support.",
            "citation_rows": best_rows,
        }
    if any(row["exists"] for row in best_rows):
        return {
            "status": "mis_citation",
            "color": "red",
            "explanation": "Citation metadata exists, but retrieved content does not support the specific claim.",
            "citation_rows": best_rows,
        }
    return {
        "status": "invalid_citation",
        "color": "red",
        "explanation": "No cited reference could be matched to retrieved metadata.",
        "citation_rows": best_rows,
    }
