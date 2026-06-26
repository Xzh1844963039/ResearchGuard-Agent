# C:\Users\18449\Desktop\researchguard_workspace\researchguard\audit\source_evidence_extract_skill.py
from __future__ import annotations

import re

from researchguard.schemas import EvidenceRecord
from researchguard.audit.base_skill import BaseSkill
from researchguard.text_utils_v2 import compact_text, detect_section, now_iso
from researchguard.utils_ids import make_id


class SourceEvidenceExtractSkill(BaseSkill):
    name = "source_evidence_extract_skill"
    description = "Extract E001 source evidence from parsed PDF/text pages."
    input_schema = {"source_record": "dict", "parsed_pages": "list"}
    output_schema = {"source_evidence": "list[SourceEvidenceRecord]"}

    def run(self, case_id: str, payload: dict) -> dict:
        source = payload["source_record"]
        pages = payload.get("parsed_pages", [])
        candidates = []
        for page in pages:
            paragraphs = split_page_text(page.get("text", ""))
            current_section = None
            for paragraph_index, paragraph in enumerate(paragraphs, start=1):
                clean = clean_text(paragraph)
                quality = evidence_quality(paragraph, clean)
                if quality["quality_label"] == "unusable":
                    continue
                current_section = detect_section(paragraph) or current_section
                role = evidence_role(clean, current_section)
                candidates.append(
                    {
                        "page": page.get("page"),
                        "paragraph_index": paragraph_index,
                        "section": current_section,
                        "paragraph": clean,
                        "raw_paragraph": paragraph,
                        "quality": quality,
                        "evidence_role": role,
                        "score": paragraph_score(clean, current_section, role) + quality["quality_score"],
                    }
                )
        candidates.sort(key=lambda row: (-row["score"], row["page"] or 0, row["paragraph_index"]))
        evidence = []
        for index, item in enumerate(candidates[:30], start=1):
            paragraph = item["paragraph"]
            current_section = item["section"]
            eid = make_id("E", index)
            location = f"page {item.get('page')}, paragraph {item.get('paragraph_index')}"
            if current_section:
                location += f", {current_section}"
            record = EvidenceRecord(
                evidence_id=eid,
                case_id=case_id,
                source_id=source["source_id"],
                source_type=source["source_type"],
                source_name=source["source_name"],
                page=item.get("page"),
                section=current_section,
                paragraph_index=item.get("paragraph_index"),
                table_id=None,
                row_index=None,
                location_text=location,
                content=paragraph,
                content_summary=compact_text(paragraph, 180),
                created_by_tool=self.name,
                created_at=now_iso(),
            )
            row = self.memory.append_evidence(record)
            row["used_for"] = []
            row["created_by_skill"] = self.name
            row["raw_content"] = item.get("raw_paragraph", paragraph)
            row["clean_content"] = paragraph
            row["quality_score"] = item["quality"]["quality_score"]
            row["quality_label"] = item["quality"]["quality_label"]
            row["noise_reasons"] = item["quality"]["noise_reasons"]
            row["section_guess"] = current_section
            row["evidence_role"] = item["evidence_role"]
            row["used_for_support"] = item["quality"]["quality_label"] in {"high", "medium"} and item["evidence_role"] not in {"appendix", "reference", "noise"}
            evidence.append(row)
        if not evidence:
            self.memory.append_failure(case_id, "no_source_evidence", "No source evidence extracted.", {"source": source})
        return {"source_evidence": evidence}


def split_page_text(text: str) -> list[str]:
    raw = [p.strip() for p in re.split(r"\n\s*\n|\r?\n", text or "") if p.strip()]
    merged: list[str] = []
    buffer = ""
    for part in raw:
        if len(part) < 45 and buffer:
            buffer += " " + part
        else:
            if buffer:
                merged.append(buffer)
            buffer = part
    if buffer:
        merged.append(buffer)
    return [p for p in merged if len(p) >= 30]


def is_noise(paragraph: str) -> bool:
    lower = paragraph.lower()
    noise_terms = [
        "this copy is for your personal",
        "downloaded from",
        "permission to republish",
        "american association for the advancement of science",
        "www.sciencemag.org",
        "high-quality copies",
        "registered trademark",
        "print issn",
        "online issn",
        "clicking here",
        "bibliography",
        "arxiv:",
        "proceedings of",
        "provided proper attribution",
        "reproduce the tables and figures",
        "equal contribution",
        "listing order is random",
        "work performed while at",
    ]
    if any(term in lower for term in noise_terms):
        return True
    if lower.count("http://") + lower.count("https://") >= 1 and len(lower) < 220:
        return True
    if lower.startswith("table ") and len(lower) < 350:
        return True
    if sum(1 for marker in [" et al.", " proceedings ", " conference ", "journal "] if marker in lower) >= 2 and len(lower) < 700:
        return True
    if "@" in lower and sum(1 for marker in ["google", "university", ".com", ".edu"] if marker in lower) >= 2:
        return True
    return False


def clean_text(text: str) -> str:
    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", text or "")
    cleaned = cleaned.replace("�", " ")
    cleaned = re.sub(r"\b([A-Za-z])\s+([A-Za-z]{2,})\b", r"\1\2", cleaned)
    cleaned = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    replacements = {
        "C as9": "Cas9",
        "C as 9": "Cas9",
        "T ransformer": "Transformer",
        "g enome": "genome",
        "v alues": "values",
        "Intr oduction": "Introduction",
        "f actor": "factor",
        "netw orks": "networks",
        "netw ork": "network",
        "g ated": "gated",
        "o verall": "overall",
        "se quence": "sequence",
        "f or": "for",
        "ha v e": "have",
        "ha ve": "have",
        "w orks": "works",
        "w ork": "work",
        "inf ormation": "information",
        "dif ferent": "different",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def evidence_quality(raw: str, clean: str) -> dict:
    lower = clean.lower()
    reasons = []
    if is_noise(raw) or is_noise(clean):
        reasons.append("known_noise_pattern")
    if len(clean) < 60:
        reasons.append("too_short")
    if lower.startswith("table ") or lower.startswith("figure ") or lower.startswith("attention visualizations figure"):
        reasons.append("table_or_figure_fragment")
    if "attention visualizations" in lower and ("<pad>" in lower or "<eos>" in lower):
        reasons.append("table_or_figure_fragment")
    if "depicted in figure" in lower or "shown in figure" in lower:
        reasons.append("table_or_figure_fragment")
    if any(term in lower for term in ["prompt appendix", "prompt template", "scoring prompt", "fitness prompt", "[reasoning chain start]", "[reasoning chain end]"]):
        reasons.append("prompt_or_reasoning_example")
    if re.match(r"^\s*(step|example)\s*\d+[:.)-]", lower) and ("reasoning" in lower or "prompt" in lower):
        reasons.append("prompt_or_reasoning_example")
    if re.search(r"\bcontents\b\s+\d+\s+\w+", lower) and len(clean) < 260:
        reasons.append("table_of_contents_fragment")
    if lower.count("figure ") >= 2 and len(clean) < 450:
        reasons.append("table_or_figure_fragment")
    if any(term in lower for term in ["copyright", "permission", "downloaded from", "references", "bibliography", "provided proper attribution", "equal contribution", "listing order is random", "work performed while"]):
        reasons.append("copyright_or_reference_fragment")
    alnum = sum(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in clean)
    symbol_ratio = 1 - (alnum / max(1, len(clean)))
    if symbol_ratio > 0.42:
        reasons.append("too_many_symbols")
    words = re.findall(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}", clean)
    if len(words) < 8:
        reasons.append("not_enough_readable_words")
    readable_bonus = 18 if len(words) >= 18 and any(p in clean for p in [".", "。", ";", "；"]) else 0
    domain_bonus = sum(
        6
        for term in [
            "abstract",
            "introduction",
            "conclusion",
            "cas9",
            "crispr",
            "guide rna",
            "genome editing",
            "target dna",
            "transformer",
            "self-attention",
            "multi-head",
            "sequence modeling",
        ]
        if term in lower
    )
    penalty = len(reasons) * 16
    score = max(0, min(100, 45 + readable_bonus + domain_bonus - penalty))
    if any(reason in reasons for reason in ["known_noise_pattern", "too_many_symbols", "table_or_figure_fragment", "copyright_or_reference_fragment", "prompt_or_reasoning_example", "table_of_contents_fragment"]) or len(clean) < 40:
        label = "unusable"
    elif score >= 70:
        label = "high"
    elif score >= 48:
        label = "medium"
    elif score >= 30:
        label = "low"
    else:
        label = "unusable"
    return {"quality_score": score, "quality_label": label, "noise_reasons": reasons}


def paragraph_score(paragraph: str, section: str | None, role: str | None = None) -> int:
    lower = paragraph.lower()
    priority_terms = [
        "abstract",
        "introduction",
        "conclusion",
        "discussion",
        "limitation",
        "reference",
        "result",
        "method",
        "crispr",
        "cas9",
        "guide",
        "dual-rna",
        "tracrrna",
        "crrna",
        "genome",
        "editing",
        "endonuclease",
        "target dna",
        "cleavage",
        "off-target",
        "delivery",
        "therapeutic",
        "transformer",
        "attention",
        "self-attention",
        "multi-head",
        "sequence",
        "translation",
        "encoder",
        "decoder",
        "recurrence",
        "convolution",
        "bleu",
        "long-range",
    ]
    score = sum(4 for term in priority_terms if term in lower)
    score += 8 if section in {"abstract", "introduction", "method", "experiments", "limitations"} else 0
    score += {"mechanism": 18, "contribution": 16, "limitation": 10, "result": 6, "table": -18, "appendix": -20, "reference": -24, "noise": -30}.get(role or "", 0)
    score += 3 if 90 <= len(paragraph) <= 1600 else 0
    score -= 4 if len(paragraph) < 50 else 0
    return score


def evidence_role(paragraph: str, section: str | None) -> str:
    lower = paragraph.lower()
    number_count = len(re.findall(r"\d+(?:\.\d+)?%?", lower))
    word_count = len(re.findall(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}", lower))
    if section == "references" or lower.startswith("references"):
        return "reference"
    if any(term in lower for term in ["appendix", "prompt template", "scoring rubric", "fitness prompt", "[reasoning chain start]", "[reasoning chain end]"]):
        return "appendix"
    if "contents" in lower and re.search(r"\b\d+\s+[a-z]", lower):
        return "noise"
    if lower.startswith("table ") or number_count >= max(4, word_count // 3):
        return "table"
    if any(term in lower for term in ["limitation", "however", "remain", "future work", "off-target", "failure"]):
        return "limitation"
    if any(term in lower for term in ["outperform", "accuracy", "benchmark", "bleu", "result", "performance"]):
        return "result"
    if any(term in lower for term in ["we propose", "our contribution", "contribution", "introduce", "present"]):
        return "contribution"
    if section in {"abstract", "introduction", "method"} or any(
        term in lower
        for term in [
            "mechanism",
            "algorithm",
            "chain-of-thought",
            "selection",
            "recombination",
            "mutation",
            "self-attention",
            "guide rna",
        ]
    ):
        return "mechanism"
    return "noise" if word_count < 8 else "contribution"
