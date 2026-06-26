# C:\Users\18449\Desktop\researchguard_workspace\researchguard\audit\paper_claim_extraction_skill.py
from __future__ import annotations

import re

from researchguard.parsers.reference_parser import citation_patterns
from researchguard.audit.base_skill import BaseSkill
from researchguard.text_utils_v2 import compact_text, tokenize
from researchguard.utils_ids import make_id


CLAIM_SECTIONS = {"abstract", "introduction", "method", "results", "experiments", "discussion", "conclusion"}
WEAK_SECTIONS = {"appendix", "references"}
STRONG_CLAIM_TERMS = re.compile(
    r"\b("
    r"prove|proves|demonstrate|demonstrates|show|shows|outperform|outperforms|"
    r"improve|improves|achieve|achieves|reduce|reduces|enable|enables|"
    r"always|completely|fully|state-of-the-art|all|best"
    r")\b",
    re.I,
)
NUMERIC_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?\b")
NOISY_CLAIM_PATTERNS = [
    re.compile(r"\b(prompt|rubric|reasoning chain|chain start|chain end|instructions|query start|query end|cot start|cot end|answer start|answer end)\b", re.I),
    re.compile(r"^\s*(step|example)\s*\d+[:.)-]", re.I),
    re.compile(r"^\s*(table|figure)\s+\d+", re.I),
    re.compile(r"\barxiv\s+preprint\b|\bdoi:\s*10\.|\bproceedings of\b|\btransactions on\b", re.I),
    re.compile(r"\bbest results are marked\b|\bsecond-best results are underlined\b", re.I),
]


class PaperClaimExtractionSkill(BaseSkill):
    name = "paper_claim_extraction_skill"
    description = "Extract original paper claims as PC001 records for input-paper error audit."
    input_schema = {"parsed_pages": "list", "source_evidence": "list"}
    output_schema = {"paper_claims": "list"}

    def run(self, case_id: str, payload: dict) -> dict:
        pages = payload.get("parsed_pages", [])
        evidence = payload.get("source_evidence", [])
        topic_tokens = significant_topic_tokens(f"{payload.get('paper_title', '')} {payload.get('topic', '')}")
        evidence_by_id = {row.get("evidence_id"): row for row in evidence}
        candidates: list[dict] = []
        for page in pages:
            page_no = page.get("page")
            text = page.get("text", "")
            for sentence in split_sentences(text):
                section = infer_sentence_section(sentence, evidence)
                if is_noisy_claim(sentence, section, topic_tokens):
                    continue
                score = claim_score(sentence, section)
                if score <= 0:
                    continue
                candidates.append(
                    {
                        "text": sentence.strip(),
                        "section": section,
                        "page": page_no,
                        "score": score,
                        "citations": citation_patterns(sentence),
                    }
                )

        candidates.sort(key=lambda row: (-row["score"], row.get("page") or 0))
        claims: list[dict] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = normalize_for_dedupe(candidate["text"])
            if normalized in seen:
                continue
            seen.add(normalized)
            pc_id = make_id("PC", len(claims) + 1)
            candidate_eids = match_candidate_evidence(candidate["text"], evidence_by_id)
            claims.append(
                {
                    "paper_claim_id": pc_id,
                    "exact_text": candidate["text"],
                    "section": candidate["section"],
                    "claim_type": classify_claim(candidate["text"], candidate["citations"]),
                    "source_location": f"page {candidate.get('page') or '?'}",
                    "cited_refs": candidate["citations"],
                    "candidate_evidence_ids": candidate_eids,
                }
            )
            if len(claims) >= 15:
                break

        return {"paper_claims": claims}


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", cleaned)
    return [part.strip() for part in parts if 45 <= len(part.strip()) <= 550]


def infer_sentence_section(sentence: str, evidence: list[dict]) -> str:
    sent_tokens = tokenize(sentence)
    best_section = "unknown"
    best_overlap = 0
    for row in evidence:
        row_tokens = tokenize(row.get("clean_content") or row.get("content") or "")
        overlap = len(sent_tokens & row_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_section = row.get("section_guess") or row.get("section") or "unknown"
    return best_section


def is_noisy_claim(sentence: str, section: str, topic_tokens: set[str] | None = None) -> bool:
    lower = sentence.lower()
    if section in WEAK_SECTIONS:
        return True
    if any(pattern.search(sentence) for pattern in NOISY_CLAIM_PATTERNS):
        return True
    if any(
        term in lower
        for term in [
            "downloaded from",
            "print issn",
            "online issn",
            "published weekly",
            "this copy is for your personal",
            "www.sciencemag.org",
            "permission",
            "copyright",
            "provided proper attribution",
            "reproduce the tables and figures",
            "equal contribution",
            "listing order is random",
            "work performed while at",
            "preprint arxiv",
            "arxiv preprint",
            "[query start]",
            "[cot start]",
            "[answer start]",
        ]
    ):
        return True
    if lower.startswith(("therefore, you need to", "you need to locate", "you must strictly", "ignore any", "do not output")):
        return True
    if "@" in lower and sum(1 for marker in ["google", "university", ".com", ".edu"] if marker in lower) >= 2:
        return True
    affiliation_numbers = len(re.findall(r"\b\d+[*,\u2020]?", sentence))
    if affiliation_numbers >= 3 and sentence.count(",") >= 2 and not STRONG_CLAIM_TERMS.search(sentence):
        return True
    if sentence.lstrip().startswith("(") and "fig" in lower:
        return True
    if re.match(r"^\s*\(?\s*[a-z]\s*\)\s*\(?\s*t\s*o\s*p", lower):
        return True
    if "[reasoning chain start]" in lower or "[reasoning chain end]" in lower:
        return True
    if lower.count("http://") + lower.count("https://") > 0 and len(sentence) < 220:
        return True
    if topic_tokens and NUMERIC_PATTERN.search(sentence):
        overlap = tokenize(sentence) & topic_tokens
        has_author_claim = any(term in lower for term in ["we show", "we propose", "we present", "we demonstrate", "our results"])
        if not overlap and not has_author_claim:
            return True
    return False


def significant_topic_tokens(text: str) -> set[str]:
    stop = {
        "paper",
        "uploaded",
        "research",
        "article",
        "hypothesis",
        "audit",
        "citation",
        "review",
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
    }
    return {token for token in tokenize(text) if len(token) >= 4 and token not in stop}


def claim_score(sentence: str, section: str) -> int:
    lower = sentence.lower()
    score = 0
    if section in CLAIM_SECTIONS:
        score += 20
    if section in {"abstract", "results", "experiments", "conclusion"}:
        score += 12
    if STRONG_CLAIM_TERMS.search(sentence):
        score += 18
    if citation_patterns(sentence):
        score += 15
    if NUMERIC_PATTERN.search(sentence):
        score += 12
    if any(term in lower for term in ["we propose", "we present", "our method", "our results", "we show"]):
        score += 12
    if len(sentence) < 60:
        score -= 10
    return score


def classify_claim(sentence: str, citations: list[str]) -> str:
    lower = sentence.lower()
    if NUMERIC_PATTERN.search(sentence):
        return "numerical_claim"
    if citations:
        return "citation_claim"
    if any(term in lower for term in ["outperform", "accuracy", "benchmark", "result", "performance"]):
        return "result_claim"
    if any(term in lower for term in ["method", "algorithm", "framework", "model"]):
        return "method_claim"
    if any(term in lower for term in ["all", "always", "completely", "generalize", "state-of-the-art"]):
        return "generalization_claim"
    return "paper_claim"


def match_candidate_evidence(claim: str, evidence_by_id: dict[str, dict]) -> list[str]:
    claim_tokens = tokenize(claim)
    scored: list[tuple[float, str]] = []
    for eid, row in evidence_by_id.items():
        if not eid:
            continue
        if row.get("evidence_role") in {"appendix", "reference", "noise"}:
            continue
        if row.get("quality_label", "medium") not in {"high", "medium"}:
            continue
        text = row.get("clean_content") or row.get("content") or row.get("content_summary") or ""
        tokens = tokenize(text)
        score = len(claim_tokens & tokens) / max(1, len(claim_tokens))
        if score >= 0.12:
            scored.append((score, eid))
    scored.sort(reverse=True)
    return [eid for _, eid in scored[:3]]


def normalize_for_dedupe(text: str) -> str:
    return compact_text(re.sub(r"\W+", " ", text.lower()), 180)
