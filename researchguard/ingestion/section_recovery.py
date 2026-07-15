# C:\Users\18449\Desktop\researchguard_workspace\researchguard\ingestion\section_recovery.py
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from researchguard.ingestion.heading_classifier import classify_heading


SECTION_ORDER = [
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
]


def soft_transition_score(previous: str, new: str, confidence: float) -> float:
    if previous == new:
        return 0.2

    if previous in {"main_text", "empty"}:
        return 0.0

    if new == "appendix":
        return 0.0

    if previous == "references" and new != "appendix":
        if confidence >= 0.70:
            return -0.2
        return -10.0

    try:
        prev_idx = SECTION_ORDER.index(previous)
        new_idx = SECTION_ORDER.index(new)
    except ValueError:
        return 0.0

    if new_idx >= prev_idx:
        return 0.3

    if previous in {"experiment", "results"} and new in {"experiment", "results", "method"}:
        return -0.2

    if confidence >= 0.78:
        return -0.4

    return -1.5


def normalized_line_marker(text: str) -> str:
    text = re.sub(r"^\s*(?:\d{1,2}(?:\.\d{1,2})*|[A-Z])\s*[\.\)]?\s+", "", text.strip())
    text = text.replace("&", " and ")
    text = re.sub(r"[^A-Za-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def reference_like_ratio(page_text: str) -> float:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if not lines:
        return 0.0

    markers = 0
    for line in lines:
        lowered = line.lower()
        if re.search(r"\b(19|20)\d{2}[a-z]?\.", lowered):
            markers += 1
        elif "arxiv" in lowered or "coRR" in line:
            markers += 1
        elif "association for computational linguistics" in lowered:
            markers += 1
        elif re.search(r"\b(pages|volume|proceedings|conference)\b", lowered):
            markers += 1

    return markers / len(lines)


def detect_implicit_section_from_page_text(page_text: str, page_number: int, previous_section: str) -> tuple[str | None, float, str]:
    lowered = page_text.lower()
    first_lines = [normalized_line_marker(line) for line in page_text.splitlines()[:18]]

    if page_number <= 2 and "abstract" in first_lines:
        return "abstract", 0.9, "implicit abstract on first pages"

    if page_number <= 4 and "introduction" in first_lines:
        return "introduction", 0.75, "implicit introduction near beginning"

    if "references" in lowered[:800] or "bibliography" in lowered[:800]:
        return "references", 0.85, "implicit references near page top"

    if page_number >= 5 and previous_section in {"conclusion", "references"} and reference_like_ratio(page_text) >= 0.16:
        return "references", 0.78, "implicit references by citation density"

    if "conclusion" in lowered[:1000] or "future work" in lowered[:1000]:
        return "conclusion", 0.72, "implicit conclusion/future work"

    return None, 0.0, ""


def recover_sections(
    *,
    layout: dict[str, Any],
    blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    body_font_size = float(layout.get("body_font_size", 10.0))
    pages_by_no = {int(page["page"]): page for page in layout["pages"]}

    blocks_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        blocks_by_page[int(block["page"])].append(block)

    current_section = "main_text"
    enriched_blocks: list[dict[str, Any]] = []
    page_records: list[dict[str, Any]] = []

    for page_no in sorted(pages_by_no):
        page = pages_by_no[page_no]
        page_height = float(page["height"])
        page_width = float(page["width"])
        page_blocks = sorted(blocks_by_page.get(page_no, []), key=lambda item: (int(item.get("column", 0)), float(item["y0"]), float(item["x0"])))

        page_text = "\n\n".join(block["text"] for block in page_blocks if block.get("block_type") != "equation")
        implicit_section, implicit_confidence, implicit_reason = detect_implicit_section_from_page_text(
            page_text=page_text,
            page_number=page_no,
            previous_section=current_section,
        )

        page_heading = None
        page_heading_confidence = 0.0
        page_heading_score = 0.0
        page_heading_reasons: list[str] = []
        page_section_reason = "inherited previous section"

        if implicit_section and soft_transition_score(current_section, implicit_section, implicit_confidence) > -2.0:
            current_section = implicit_section
            page_section_reason = implicit_reason
            page_heading_confidence = implicit_confidence

        for block in page_blocks:
            prediction = classify_heading(
                block=block,
                body_font_size=body_font_size,
                page_height=page_height,
                page_width=page_width,
            )

            block["heading_prediction"] = {
                "is_heading": prediction.is_heading,
                "section": prediction.section,
                "score": prediction.score,
                "confidence": prediction.confidence,
                "reasons": prediction.reasons,
                "normalized": prediction.normalized,
            }

            if prediction.is_heading and prediction.section:
                transition_score = soft_transition_score(
                    previous=current_section,
                    new=prediction.section,
                    confidence=prediction.confidence,
                )

                final_score = prediction.score + transition_score

                if final_score >= 4.6:
                    current_section = prediction.section
                    page_heading = block["text"]
                    page_heading_confidence = prediction.confidence
                    page_heading_score = prediction.score
                    page_heading_reasons = prediction.reasons
                    page_section_reason = "heading classifier + soft transition"
                    block["block_type"] = "heading"
                else:
                    block["block_type"] = "paragraph"

            block["section"] = current_section
            block["section_confidence"] = max(page_heading_confidence, 0.55 if current_section != "main_text" else 0.45)
            enriched_blocks.append(block)

        if page_blocks:
            section_chars: Counter[str] = Counter()
            for block in page_blocks:
                if block.get("block_type") in {"caption", "equation", "table"}:
                    continue

                section = str(block.get("section", current_section))
                text_len = max(len(str(block.get("text", ""))), 1)
                if block.get("block_type") == "heading":
                    text_len = max(20, text_len // 3)
                section_chars[section] += text_len

            dominant_section = section_chars.most_common(1)[0][0] if section_chars else current_section
        else:
            dominant_section = "empty"

        page_records.append(
            {
                "doc_id": layout["doc_id"],
                "title": layout["title"],
                "page": page_no,
                "section": dominant_section,
                "section_heading": page_heading,
                "section_confidence": round(float(page_heading_confidence or 0.55), 4),
                "section_reason": page_section_reason,
                "heading_score": page_heading_score or None,
                "heading_reasons": page_heading_reasons,
                "text": page_text.strip(),
                "char_count": len(page_text.strip()),
                "word_count": len(page_text.split()),
                "blocks": [block["block_id"] for block in page_blocks],
            }
        )

    return page_records, enriched_blocks
