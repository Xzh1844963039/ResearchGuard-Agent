# C:\Users\18449\Desktop\researchguard_workspace\researchguard\audit\numerical_claim_check_skill.py
from __future__ import annotations

import re

from researchguard.audit.base_skill import BaseSkill
from researchguard.text_utils_v2 import tokenize


NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?\b")
NON_CLAIM_NUMBER_PREFIX = re.compile(
    r"\b(?:fig(?:ure)?|table|section|sec|appendix|eq|equation|page|chapter|ref(?:erence)?)\.?\s*$",
    re.I,
)


class NumericalClaimCheckSkill(BaseSkill):
    name = "numerical_claim_check_skill"
    description = "Check whether numbers in input-paper claims appear in related PDF evidence."
    input_schema = {"paper_claim": "dict", "source_evidence": "list"}
    output_schema = {"status": "consistent|uncertain|numerical_error", "explanation": "str"}

    def run(self, case_id: str, payload: dict) -> dict:
        return check_numerical_claim(payload.get("paper_claim", {}), payload.get("source_evidence", []))


def check_numerical_claim(claim: dict, evidence: list[dict]) -> dict:
    text = claim.get("exact_text", "")
    claim_numbers = extract_claim_numbers(text)
    if not claim_numbers:
        return {
            "status": "not_applicable",
            "color": "green",
            "explanation": "Claim does not contain an explicit numeric value.",
            "claim_numbers": [],
            "evidence_numbers": [],
        }
    claim_tokens = tokenize(text)
    related_numbers: list[str] = []
    related_ids: list[str] = []
    for row in evidence:
        if row.get("evidence_role") in {"appendix", "reference", "noise"}:
            continue
        ev_text = row.get("clean_content") or row.get("content") or ""
        ev_tokens = tokenize(ev_text)
        if len(claim_tokens & ev_tokens) / max(1, len(claim_tokens)) < 0.20:
            continue
        nums = extract_claim_numbers(ev_text)
        if nums:
            related_numbers.extend(nums)
            related_ids.append(row.get("evidence_id"))
    evidence_numbers = list(dict.fromkeys(related_numbers))
    if any(num in evidence_numbers for num in claim_numbers):
        return {
            "status": "consistent",
            "color": "green",
            "explanation": "At least one claim number appears in related internal evidence.",
            "claim_numbers": claim_numbers,
            "evidence_numbers": evidence_numbers,
            "evidence_ids": related_ids[:3],
        }
    if evidence_numbers:
        return {
            "status": "numerical_error",
            "color": "red",
            "explanation": "Related evidence contains numbers, but they do not match the claim numbers.",
            "claim_numbers": claim_numbers,
            "evidence_numbers": evidence_numbers[:8],
            "evidence_ids": related_ids[:3],
        }
    return {
        "status": "uncertain",
        "color": "yellow",
        "explanation": "Claim contains numbers, but related internal numeric evidence was not found.",
        "claim_numbers": claim_numbers,
        "evidence_numbers": [],
        "evidence_ids": [],
    }

def extract_claim_numbers(text: str) -> list[str]:
    cleaned = strip_citation_like_numbers(text)
    values: list[str] = []
    for match in NUMBER_PATTERN.finditer(cleaned):
        value = match.group(0)
        prefix = cleaned[max(0, match.start() - 24): match.start()]
        suffix = cleaned[match.end(): match.end() + 12]
        if NON_CLAIM_NUMBER_PREFIX.search(prefix):
            continue
        if re.match(r"\s*(?:[:.)]|-|–|—)", suffix) and re.search(r"\b(?:figure|fig|table|section|sec)\b", prefix, re.I):
            continue
        values.append(value)
    return list(dict.fromkeys(values))


def strip_citation_like_numbers(text: str) -> str:
    text = text or ""
    text = re.sub(r"\[\s*\d+(?:\s*[,;\-–—]\s*\d+)*\s*\]", " ", text)
    text = re.sub(r"\(\s*\d+\s*(?:[,;\-–—]\s*\d+\s*)+\)", " ", text)
    return text
    return re.sub(r"\(\s*\d+\s*(?:[,;\-–]\s*\d+\s*)+\)", " ", text or "")
