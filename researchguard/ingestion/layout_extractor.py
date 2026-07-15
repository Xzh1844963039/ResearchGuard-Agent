# C:\Users\18449\Desktop\researchguard_workspace\researchguard\ingestion\layout_extractor.py
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import fitz


@dataclass
class LayoutLine:
    text: str
    page: int
    block_no: int
    line_no: int
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    font_name: str
    is_bold: bool
    is_italic: bool
    width: float
    height: float


def normalize_line_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_bold_font(font_name: str) -> bool:
    lowered = font_name.lower()
    return any(flag in lowered for flag in ["bold", "black", "heavy", "semibold", "demi"])


def is_italic_font(font_name: str) -> bool:
    lowered = font_name.lower()
    return "italic" in lowered or "oblique" in lowered


def extract_lines_from_page(page: fitz.Page, page_number: int) -> list[LayoutLine]:
    page_dict = page.get_text("dict")
    lines: list[LayoutLine] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        block_no = int(block.get("number", 0))

        for line_no, line in enumerate(block.get("lines", [])):
            spans = line.get("spans", [])
            if not spans:
                continue

            text_parts: list[str] = []
            font_sizes: list[float] = []
            font_names: list[str] = []
            bboxes: list[list[float]] = []

            for span in spans:
                span_text = span.get("text", "")
                if span_text:
                    text_parts.append(span_text)

                font_sizes.append(float(span.get("size", 0.0)))
                font_names.append(str(span.get("font", "")))
                bboxes.append(span.get("bbox", [0, 0, 0, 0]))

            text = normalize_line_text("".join(text_parts))
            if not text:
                continue

            x0 = min(float(b[0]) for b in bboxes)
            y0 = min(float(b[1]) for b in bboxes)
            x1 = max(float(b[2]) for b in bboxes)
            y1 = max(float(b[3]) for b in bboxes)

            font_size = float(median(font_sizes)) if font_sizes else 0.0
            font_name = Counter(font_names).most_common(1)[0][0] if font_names else ""

            lines.append(
                LayoutLine(
                    text=text,
                    page=page_number,
                    block_no=block_no,
                    line_no=line_no,
                    x0=round(x0, 3),
                    y0=round(y0, 3),
                    x1=round(x1, 3),
                    y1=round(y1, 3),
                    font_size=round(font_size, 3),
                    font_name=font_name,
                    is_bold=is_bold_font(font_name),
                    is_italic=is_italic_font(font_name),
                    width=round(x1 - x0, 3),
                    height=round(y1 - y0, 3),
                )
            )

    return sorted(lines, key=lambda item: (item.y0, item.x0))


def estimate_body_font_size(lines: list[LayoutLine]) -> float:
    candidates: list[float] = []

    for line in lines:
        text = line.text.strip()

        if len(text) < 35:
            continue

        if len(text.split()) < 6:
            continue

        rounded_size = round(line.font_size * 2) / 2
        if rounded_size > 0:
            candidates.append(rounded_size)

    if candidates:
        return float(Counter(candidates).most_common(1)[0][0])

    fallback = [round(line.font_size * 2) / 2 for line in lines if line.font_size > 0]
    if fallback:
        return float(Counter(fallback).most_common(1)[0][0])

    return 10.0


def extract_title(doc: fitz.Document, pdf_path: Path, first_page_lines: list[LayoutLine]) -> str:
    metadata_title = (doc.metadata or {}).get("title")

    if metadata_title and metadata_title.strip():
        title = metadata_title.strip()
        if 5 <= len(title) <= 200 and not title.lower().startswith("arxiv"):
            return title

    candidates = [
        line
        for line in first_page_lines[:30]
        if 10 <= len(line.text) <= 180
        and len(line.text.split()) >= 3
        and not line.text.lower().startswith("arxiv")
    ]

    if not candidates:
        return pdf_path.stem

    candidates = sorted(candidates, key=lambda item: (-item.font_size, item.y0, item.x0))
    return candidates[0].text.strip()


def extract_document_layout(pdf_path: Path) -> dict[str, Any]:
    doc = fitz.open(pdf_path)

    pages: list[dict[str, Any]] = []
    all_lines: list[LayoutLine] = []

    for page_index, page in enumerate(doc, start=1):
        lines = extract_lines_from_page(page=page, page_number=page_index)
        all_lines.extend(lines)

        pages.append(
            {
                "page": page_index,
                "width": float(page.rect.width),
                "height": float(page.rect.height),
                "lines": [asdict(line) for line in lines],
            }
        )

    body_font_size = estimate_body_font_size(all_lines)
    title = extract_title(doc, pdf_path, all_lines[:80])

    return {
        "doc_id": pdf_path.stem,
        "title": title,
        "pdf_path": str(pdf_path),
        "body_font_size": body_font_size,
        "pages": pages,
    }