# C:\Users\18449\Desktop\researchguard_workspace\researchguard\ingestion\heading_classifier.py
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


VALID_SECTIONS = {
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiment",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "references",
    "appendix",
    "main_text",
}

SECTION_ALIASES: dict[str, list[str]] = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "overview"],
    "related_work": ["related work", "background", "preliminaries", "literature review"],
    "method": [
        "method",
        "methods",
        "methodology",
        "approach",
        "framework",
        "model architecture",
        "architecture",
        "system design",
    ],
    "experiment": [
        "experiment",
        "experiments",
        "experimental setup",
        "experimental details",
        "evaluation setup",
        "implementation details",
        "datasets",
        "datasets and metrics",
        "task setup",
        "benchmark",
        "benchmarks",
    ],
    "results": [
        "results",
        "main results",
        "additional results",
        "analysis",
        "ablation",
        "case study",
        "error analysis",
        "automatic evaluation",
        "human evaluation",
        "evaluation",
    ],
    "discussion": ["discussion"],
    "limitations": ["limitation", "limitations", "threats to validity"],
    "conclusion": ["conclusion", "conclusions", "future work", "conclusion limitation", "conclusion and limitation"],
    "references": ["references", "bibliography"],
    "appendix": ["appendix", "appendices", "supplementary material", "supplementary"],
}

EXACT_SECTION_HEADINGS = {
    alias
    for aliases in SECTION_ALIASES.values()
    for alias in aliases
}


@dataclass
class HeadingPrediction:
    is_heading: bool
    section: str | None
    score: float
    confidence: float
    reasons: list[str]
    normalized: str


def normalize_heading(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*(\d{1,2}|[IVX]+)(\.\d{1,2})*\s*[\.\)]?\s+", "", text, flags=re.I)
    text = re.sub(r"^\s*[A-Z]\s*[\.\)]\s+", "", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def is_numbered_heading(text: str) -> bool:
    text = text.strip()
    arabic = re.match(r"^(\d{1,4})(\.\d{1,2})*\s+[A-Z][A-Za-z0-9][A-Za-z0-9 ,:\-/()&]+$", text)
    if arabic:
        return 1 <= int(arabic.group(1)) <= 99

    patterns = [
        r"^[IVX]+\.\s+[A-Z][A-Za-z0-9][A-Za-z0-9 ,:\-/()&]+$",
        r"^[A-Z]\.\s+[A-Z][A-Za-z0-9][A-Za-z0-9 ,:\-/()&]+$",
    ]
    return any(re.match(pattern, text) for pattern in patterns)


def is_all_caps_heading(text: str) -> bool:
    stripped = text.strip()

    if len(stripped) < 4:
        return False

    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return False

    upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return upper_ratio >= 0.8 and len(stripped.split()) <= 9


def is_reference_entry(text: str) -> bool:
    stripped = text.strip()
    return bool(re.match(r"^\[\d+\]\s+", stripped)) or bool(re.match(r"^\d+\.\s+[A-Z][A-Za-z\-]+,", stripped))


def is_bibliographic_or_venue_text(text: str) -> bool:
    lowered = text.strip().lower()

    if re.match(r"^(19|20)\d{2}\s+conference\s+on\b", lowered):
        return True

    venue_markers = [
        "proceedings of",
        "conference on empirical methods",
        "association for computational linguistics",
        "transactions of the association",
        "arxiv preprint",
        "international conference on",
        "advances in neural information processing systems",
    ]
    return any(marker in lowered for marker in venue_markers)


def is_caption(text: str) -> bool:
    return bool(re.match(r"^(figure|fig\.|table)\s+\d+", text.strip(), flags=re.I))


def is_table_or_chart_noise(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) < 4:
        return False

    numeric_lines = sum(1 for line in lines if re.search(r"\d", line))
    short_lines = sum(1 for line in lines if len(line.split()) <= 4)
    table_words = {"method", "score", "acc", "auc", "precision", "recall", "f1", "em", "pcc", "scc"}
    normalized_lines = {normalize_heading(line) for line in lines}

    return numeric_lines >= 3 and short_lines / len(lines) >= 0.55 and bool(table_words & normalized_lines)


def is_sentence_like(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    words = lowered.split()

    if len(words) >= 7 and stripped.endswith("."):
        return True

    if len(words) >= 8 and any(lowered.startswith(prefix) for prefix in ["we ", "our ", "this ", "these ", "the "]):
        return True

    if lowered.endswith("-"):
        return True

    if lowered.endswith(",") or lowered.endswith(";"):
        return True

    if re.search(r"\([A-Z][A-Za-z]+,?\s+\d{4}\)", stripped):
        return True

    return False


def map_heading_to_section(text: str) -> str | None:
    normalized = normalize_heading(text)

    if not normalized:
        return None

    if is_bibliographic_or_venue_text(text):
        return None

    if normalized.startswith(("appendix ", "appendices ", "supplementary ")):
        return "appendix"

    exact: dict[str, str] = {}
    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            exact[alias] = section

    if normalized in exact:
        return exact[normalized]

    weak_false_positives = {
        "model",
        "models",
        "language models",
        "large language models",
        "general purpose architectures for nlp",
    }

    if normalized in weak_false_positives:
        return None

    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            if normalized.startswith(alias + " "):
                return section

            if len(alias.split()) == 1:
                continue

            if alias in normalized and len(normalized.split()) <= 7:
                return section

    return None


def classify_heading(block: dict[str, Any], body_font_size: float, page_height: float, page_width: float) -> HeadingPrediction:
    text = str(block.get("text", "")).strip()
    normalized = normalize_heading(text)

    if not text:
        return HeadingPrediction(False, None, 0.0, 0.0, ["empty"], normalized)

    words = text.split()
    word_count = len(words)
    char_count = len(text)

    reasons: list[str] = []
    score = 0.0

    section = map_heading_to_section(text)

    if char_count > 130 or word_count > 14:
        return HeadingPrediction(False, section, 0.0, 0.0, ["too long for heading"], normalized)

    if is_reference_entry(text):
        return HeadingPrediction(False, "references", 0.0, 0.0, ["reference entry, not heading"], normalized)

    if is_bibliographic_or_venue_text(text):
        return HeadingPrediction(False, "references", 0.0, 0.0, ["bibliographic venue text, not heading"], normalized)

    if is_caption(text):
        return HeadingPrediction(False, None, 0.0, 0.0, ["caption, not heading"], normalized)

    if is_table_or_chart_noise(text):
        return HeadingPrediction(False, None, 0.0, 0.0, ["table/chart numeric block, not heading"], normalized)

    if is_sentence_like(text) and not section:
        return HeadingPrediction(False, None, 0.0, 0.0, ["sentence-like text"], normalized)

    if "\n" in text and re.search(r"\n(?:we|our|this|the)\s+", text, flags=re.I):
        return HeadingPrediction(False, section, 0.0, 0.0, ["heading merged with paragraph sentence"], normalized)

    font_size = float(block.get("font_size", body_font_size))
    font_delta = font_size - body_font_size

    if int(block.get("page", 0)) == 1 and float(block.get("y0", 0.0)) < page_height * 0.22:
        if font_delta >= 2.0 and section not in {"abstract", "introduction"}:
            return HeadingPrediction(False, section, 0.0, 0.0, ["first-page title area, not section heading"], normalized)

    if font_delta >= 3.0:
        score += 3.0
        reasons.append("font much larger than body")
    elif font_delta >= 1.2:
        score += 2.2
        reasons.append("font larger than body")
    elif font_delta >= 0.4:
        score += 0.8
        reasons.append("font slightly larger than body")

    if bool(block.get("is_bold", False)):
        score += 1.8
        reasons.append("bold font")

    if is_numbered_heading(text):
        score += 3.0
        reasons.append("numbered heading")

    if is_all_caps_heading(text):
        score += 1.4
        reasons.append("all caps style")

    if section:
        score += 3.2
        reasons.append(f"mapped to section={section}")

        if normalized in EXACT_SECTION_HEADINGS:
            score += 0.9
            reasons.append("exact section heading")

    if 1 <= word_count <= 8:
        score += 1.0
        reasons.append("short heading-like length")

    y0 = float(block.get("y0", 0.0))
    x0 = float(block.get("x0", 0.0))

    if y0 < page_height * 0.35:
        score += 0.7
        reasons.append("near page top")

    if x0 < page_width * 0.24:
        score += 0.3
        reasons.append("left aligned")

    lowered = text.lower()

    false_start_words = ["we ", "our ", "this paper", "these results", "the results", "in this"]
    if any(lowered.startswith(prefix) for prefix in false_start_words):
        score -= 3.0
        reasons.append("looks like paragraph sentence")

    if text.endswith(".") and word_count > 4:
        score -= 2.0
        reasons.append("sentence period ending")

    if text.endswith("-"):
        score -= 2.5
        reasons.append("line hyphenation ending")

    if "(" in text and ")" not in text:
        score -= 1.0
        reasons.append("broken citation or broken sentence")

    threshold = 4.8
    is_heading = score >= threshold

    confidence = 1 / (1 + math.exp(-(score - 5.8) / 1.4))
    confidence = round(max(0.0, min(1.0, confidence)), 4)

    return HeadingPrediction(
        is_heading=is_heading,
        section=section,
        score=round(score, 3),
        confidence=confidence,
        reasons=reasons,
        normalized=normalized,
    )
