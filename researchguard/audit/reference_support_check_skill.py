# C:\Users\18449\Desktop\researchguard_workspace\researchguard\audit\reference_support_check_skill.py
from __future__ import annotations

from researchguard.audit.base_skill import BaseSkill
from researchguard.text_utils_v2 import tokenize


SUPPORT_NOISE = {
    "abstract",
    "context",
    "discusses",
    "evidence",
    "findings",
    "input",
    "literature",
    "paper",
    "provides",
    "reference",
    "related",
    "reports",
    "retrieved",
    "study",
    "support",
}


class ReferenceSupportCheckSkill(BaseSkill):
    name = "reference_support_check_skill"
    description = "Check whether a LiteratureReferenceRecord supports a hypothesis support claim."
    input_schema = {"support_claim": "dict", "reference": "LiteratureReferenceRecord"}
    output_schema = {"support_status": "direct_support|partial_support|insufficient|contradicted|irrelevant"}

    def run(self, case_id: str, payload: dict) -> dict:
        claim = payload.get("support_claim", {})
        ref = payload.get("reference", {})
        abstract = ref.get("abstract") or ""
        if ref.get("is_demo_reference"):
            if not abstract:
                return {"support_status": "insufficient", "color": "yellow", "explanation": "Local demo reference has no abstract; it cannot be treated as direct support."}
        if not abstract:
            return {"support_status": "insufficient", "color": "yellow", "explanation": "Reference metadata is available but abstract is missing."}
        if float(ref.get("relevance_score", 1.0) or 0) < 0.25:
            return {"support_status": "irrelevant", "color": "red", "explanation": "Reference relevance score is below the topic threshold; it cannot support this claim."}
        claim_tokens = meaningful_tokens(claim.get("claim_text", ""))
        ref_tokens = meaningful_tokens(ref.get("title", "") + " " + abstract)
        common = claim_tokens & ref_tokens
        overlap = len(common) / max(1, len(claim_tokens))
        if any(term in abstract.lower() for term in ["contradict", "does not support", "unrelated"]):
            return {"support_status": "contradicted", "color": "red", "explanation": "Abstract appears to contradict or reject the support claim."}
        if ref.get("is_demo_reference") and len(common) >= 1:
            return {"support_status": "partial_support", "color": "yellow", "explanation": "Local demo reference is thematically related but is not real online evidence."}
        if ref.get("relevance_tier") == "weak_related":
            return {"support_status": "partial_support", "color": "yellow", "explanation": "Reference is weakly related to the topic and can provide background context only."}
        if len(common) >= 2 and overlap >= 0.25:
            return {"support_status": "direct_support", "color": "green", "explanation": "Abstract/title contain multiple substantive concepts from the support claim."}
        if len(common) >= 1:
            return {"support_status": "partial_support", "color": "yellow", "explanation": "Reference is thematically related but only indirectly supports the claim."}
        return {"support_status": "irrelevant", "color": "red", "explanation": "Reference appears unrelated to this support claim."}


def meaningful_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) >= 4 and token not in SUPPORT_NOISE}
