# C:\Users\18449\Desktop\researchguard_workspace\researchguard\audit\internal_consistency_check_skill.py
from __future__ import annotations

import re

from researchguard.audit.base_skill import BaseSkill
from researchguard.text_utils_v2 import tokenize


NEGATION_TERMS = {"not", "no", "never", "without", "fails", "failed", "cannot", "unable"}


class InternalConsistencyCheckSkill(BaseSkill):
    name = "internal_consistency_check_skill"
    description = "Check whether an input-paper claim conflicts with internal PDF evidence."
    input_schema = {"paper_claim": "dict", "source_evidence": "list"}
    output_schema = {"status": "consistent|uncertain|contradiction", "explanation": "str"}

    def run(self, case_id: str, payload: dict) -> dict:
        claim = payload.get("paper_claim", {})
        evidence = payload.get("source_evidence", [])
        return check_internal_consistency(claim, evidence)


def check_internal_consistency(claim: dict, evidence: list[dict]) -> dict:
    text = claim.get("exact_text", "")
    claim_tokens = tokenize(text)
    claim_has_negation = bool(NEGATION_TERMS & claim_tokens)
    related = []
    for row in evidence:
        if row.get("evidence_role") in {"appendix", "reference", "noise"}:
            continue
        ev_text = row.get("clean_content") or row.get("content") or ""
        ev_tokens = tokenize(ev_text)
        overlap = len(claim_tokens & ev_tokens) / max(1, len(claim_tokens))
        if overlap < 0.14:
            continue
        ev_has_negation = bool(NEGATION_TERMS & ev_tokens)
        related.append(row.get("evidence_id"))
        if claim_has_negation != ev_has_negation and overlap >= 0.35 and has_direct_contradiction_cue(text, ev_text):
            return {
                "status": "contradiction",
                "color": "red",
                "explanation": "Related internal evidence appears to reverse the claim polarity.",
                "evidence_ids": [row.get("evidence_id")],
            }
    if related:
        return {
            "status": "consistent",
            "color": "green",
            "explanation": "No obvious contradiction was found in related internal evidence.",
            "evidence_ids": related[:3],
        }
    return {
        "status": "uncertain",
        "color": "yellow",
        "explanation": "No sufficiently related internal evidence was found for a consistency check.",
        "evidence_ids": [],
    }


def has_direct_contradiction_cue(claim_text: str, evidence_text: str) -> bool:
    claim = claim_text.lower()
    evidence = evidence_text.lower()
    cue_pairs = [
        (("with recurrence", "recurrent"), ("without recurrence", "no recurrence")),
        (("improves", "outperforms", "better", "higher"), ("worse", "lower", "decreases", "underperforms")),
        (("supports", "confirms", "enables"), ("does not support", "fails", "cannot", "unable")),
        (("increases", "higher", "more"), ("decreases", "lower", "less", "fewer")),
    ]
    for left, right in cue_pairs:
        claim_left = any(term in claim for term in left)
        claim_right = any(term in claim for term in right)
        ev_left = any(term in evidence for term in left)
        ev_right = any(term in evidence for term in right)
        if (claim_left and ev_right) or (claim_right and ev_left):
            return True
    return False
