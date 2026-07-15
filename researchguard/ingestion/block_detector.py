# C:\Users\18449\Desktop\researchguard_workspace\researchguard\ingestion\block_detector.py
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

SECTION_MARKER_WORDS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experiment",
    "experimental setup",
    "evaluation",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "references",
    "bibliography",
    "appendix",
    "appendices",
    "supplementary material",
}


@dataclass
class TextBlock:
    block_id: str
    doc_id: str
    page: int
    block_type: str
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    font_name: str
    is_bold: bool
    line_count: int
    column: int
    char_count: int
    word_count: int


def clean_join_lines(lines: list[dict[str, Any]]) -> str:
    parts: list[str] = []

    for line in lines:
        text = str(line["text"]).strip()

        if parts and parts[-1].endswith("-"):
            parts[-1] = parts[-1][:-1] + text
        else:
            parts.append(text)

    text = "\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def is_noise_line(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return True

    if re.match(r"^\d+$", stripped):
        return True

    if stripped.lower().startswith("arxiv:"):
        return True

    if stripped.lower() in {"preprint", "under review"}:
        return True

    return False


def is_caption_text(text: str) -> bool:
    return bool(re.match(r"^(figure|fig\.|table)\s+\d+", text.strip(), flags=re.I))


def is_reference_entry(text: str) -> bool:
    stripped = text.strip()
    return bool(re.match(r"^\[\d+\]\s+", stripped)) or bool(re.match(r"^\d+\.\s+[A-Z][A-Za-z\-]+,", stripped))


def normalize_marker_text(text: str) -> str:
    text = re.sub(r"^\s*(?:\d{1,2}(?:\.\d{1,2})*|[A-Z])\s*[\.\)]?\s+", "", text.strip())
    text = text.replace("&", " and ")
    text = re.sub(r"[^A-Za-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def has_valid_section_number(text: str) -> bool:
    match = re.match(r"^\s*(\d{1,4})(?:\.\d{1,2})*\s+[A-Z]", text.strip())
    if not match:
        return False

    return 1 <= int(match.group(1)) <= 99


def is_section_marker_line(text: str) -> bool:
    normalized = normalize_marker_text(text)

    if normalized in SECTION_MARKER_WORDS:
        return True

    if normalized.startswith(("background and related work", "conclusion and limitation")):
        return True

    return bool(has_valid_section_number(text) and 1 <= len(normalized.split()) <= 8)


def is_equation_like(text: str) -> bool:
    stripped = text.strip()

    if len(stripped) < 4:
        return False

    symbol_count = sum(1 for ch in stripped if not ch.isalnum() and not ch.isspace())
    symbol_ratio = symbol_count / max(len(stripped), 1)

    if symbol_ratio > 0.42 and len(stripped.split()) <= 10:
        return True

    if re.search(r"=\s*\\?sum|=\s*\\?frac|\\", stripped):
        return True

    return False


def is_bibliographic_venue_line(text: str) -> bool:
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
    ]
    return any(marker in lowered for marker in venue_markers)


def is_table_like_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) < 4:
        return False

    numeric_lines = sum(1 for line in lines if re.search(r"\d", line))
    short_lines = sum(1 for line in lines if len(line.split()) <= 4)
    table_words = {"method", "score", "acc", "auc", "precision", "recall", "f1", "em", "pcc", "scc"}
    table_word_hits = sum(1 for line in lines if normalize_marker_text(line) in table_words)

    return numeric_lines >= 3 and short_lines / len(lines) >= 0.55 and table_word_hits >= 1


def rough_line_is_standalone_heading(line: dict[str, Any], body_font_size: float) -> bool:
    text = str(line["text"]).strip()
    words = text.split()
    font_size = float(line.get("font_size", body_font_size))

    if is_section_marker_line(text) and font_size >= body_font_size - 1.2:
        return True

    if len(text) > 120 or len(words) > 14:
        return False

    if text.endswith("-"):
        return False

    if is_caption_text(text) or is_reference_entry(text) or is_equation_like(text):
        return False

    if is_bibliographic_venue_line(text):
        return False

    if is_table_like_text(text):
        return False

    if font_size < body_font_size - 1.4:
        return False

    numbered = has_valid_section_number(text)
    letters = [c for c in text if c.isalpha()]
    all_caps = bool(letters) and len(text) >= 4 and sum(1 for c in letters if c.isupper()) >= max(3, int(0.75 * len(letters)))

    is_bold = bool(line.get("is_bold", False))

    has_layout_signal = font_size >= body_font_size + 1.0 or is_bold or numbered or all_caps
    short_enough = 1 <= len(words) <= 10

    return bool(has_layout_signal and short_enough)


def detect_column(line: dict[str, Any], page_width: float, two_column: bool) -> int:
    if not two_column:
        return 0

    x0 = float(line.get("x0", 0.0))
    x1 = float(line.get("x1", 0.0))
    width = x1 - x0

    if width > page_width * 0.68:
        return 0

    return 0 if x0 < page_width * 0.52 else 1


def is_two_column_page(lines: list[dict[str, Any]], page_width: float, body_font_size: float) -> bool:
    body_lines = []

    for line in lines:
        text = str(line.get("text", "")).strip()

        if len(text) < 30:
            continue

        if abs(float(line.get("font_size", body_font_size)) - body_font_size) > 2.0:
            continue

        body_lines.append(line)

    if len(body_lines) < 10:
        return False

    left = sum(1 for line in body_lines if float(line.get("x0", 0.0)) < page_width * 0.45)
    right = sum(1 for line in body_lines if float(line.get("x0", 0.0)) > page_width * 0.45)

    return left >= 4 and right >= 4


def block_type_from_text(text: str) -> str:
    if is_caption_text(text):
        return "caption"

    if is_table_like_text(text):
        return "table"

    if is_reference_entry(text):
        return "reference_entry"

    if is_equation_like(text):
        return "equation"

    return "paragraph"


def make_block(
    *,
    doc_id: str,
    page: int,
    block_index: int,
    lines: list[dict[str, Any]],
    column: int,
    forced_type: str | None = None,
) -> TextBlock:
    text = clean_join_lines(lines)

    x0 = min(float(line["x0"]) for line in lines)
    y0 = min(float(line["y0"]) for line in lines)
    x1 = max(float(line["x1"]) for line in lines)
    y1 = max(float(line["y1"]) for line in lines)

    font_sizes = [float(line.get("font_size", 0.0)) for line in lines]
    font_names = [str(line.get("font_name", "")) for line in lines]

    font_size = float(median(font_sizes)) if font_sizes else 0.0
    font_name = max(set(font_names), key=font_names.count) if font_names else ""
    is_bold = any(bool(line.get("is_bold", False)) for line in lines)

    block_type = forced_type or block_type_from_text(text)

    return TextBlock(
        block_id=f"{doc_id}_p{page:04d}_b{block_index:04d}",
        doc_id=doc_id,
        page=page,
        block_type=block_type,
        text=text,
        x0=round(x0, 3),
        y0=round(y0, 3),
        x1=round(x1, 3),
        y1=round(y1, 3),
        font_size=round(font_size, 3),
        font_name=font_name,
        is_bold=is_bold,
        line_count=len(lines),
        column=column,
        char_count=len(text),
        word_count=len(text.split()),
    )


def detect_blocks(layout: dict[str, Any]) -> list[dict[str, Any]]:
    doc_id = layout["doc_id"]
    body_font_size = float(layout.get("body_font_size", 10.0))

    all_blocks: list[dict[str, Any]] = []

    for page in layout["pages"]:
        page_no = int(page["page"])
        page_width = float(page["width"])
        lines = [line for line in page.get("lines", []) if not is_noise_line(str(line.get("text", "")))]

        if not lines:
            continue

        two_column = is_two_column_page(lines, page_width=page_width, body_font_size=body_font_size)

        for line in lines:
            line["column"] = detect_column(line, page_width=page_width, two_column=two_column)

        if two_column:
            ordered = sorted(lines, key=lambda item: (int(item["column"]), float(item["y0"]), float(item["x0"])))
        else:
            ordered = sorted(lines, key=lambda item: (float(item["y0"]), float(item["x0"])))

        current: list[dict[str, Any]] = []
        current_column = 0
        block_index = 1

        for line in ordered:
            text = str(line.get("text", "")).strip()
            column = int(line.get("column", 0))

            if not text:
                continue

            if rough_line_is_standalone_heading(line, body_font_size):
                if current:
                    block = make_block(
                        doc_id=doc_id,
                        page=page_no,
                        block_index=block_index,
                        lines=current,
                        column=current_column,
                    )
                    all_blocks.append(asdict(block))
                    block_index += 1
                    current = []

                block = make_block(
                    doc_id=doc_id,
                    page=page_no,
                    block_index=block_index,
                    lines=[line],
                    column=column,
                    forced_type="heading_candidate",
                )
                all_blocks.append(asdict(block))
                block_index += 1
                current_column = column
                continue

            if current:
                previous = current[-1]
                y_gap = float(line["y0"]) - float(previous["y1"])
                same_column = column == current_column
                same_font = abs(float(line.get("font_size", body_font_size)) - float(previous.get("font_size", body_font_size))) <= 1.2

                should_split = False

                if not same_column:
                    should_split = True

                if y_gap > max(18.0, body_font_size * 2.2):
                    should_split = True

                if not same_font and y_gap > body_font_size * 1.4:
                    should_split = True

                if should_split:
                    block = make_block(
                        doc_id=doc_id,
                        page=page_no,
                        block_index=block_index,
                        lines=current,
                        column=current_column,
                    )
                    all_blocks.append(asdict(block))
                    block_index += 1
                    current = [line]
                    current_column = column
                else:
                    current.append(line)
            else:
                current = [line]
                current_column = column

        if current:
            block = make_block(
                doc_id=doc_id,
                page=page_no,
                block_index=block_index,
                lines=current,
                column=current_column,
            )
            all_blocks.append(asdict(block))

    return all_blocks
