# C:\Users\18449\Desktop\researchguard_workspace\researchguard\ingestion\parse_pdf.py
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import median
from typing import Any

import fitz


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
    "empty",
}

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
        "retrieval augmented generation",
        "rag model",
        "react",
    ],
    "experiment": [
        "experiment",
        "experiments",
        "experimental setup",
        "evaluation setup",
        "implementation details",
        "datasets",
        "tasks",
    ],
    "results": [
        "results",
        "main results",
        "additional results",
        "analysis",
        "ablation",
        "case study",
        "error analysis",
        "evaluation",
    ],
    "discussion": ["discussion"],
    "limitations": ["limitation", "limitations", "threats to validity"],
    "conclusion": ["conclusion", "conclusions", "future work"],
    "references": ["references", "bibliography"],
    "appendix": ["appendix", "appendices", "supplementary material", "supplementary"],
}


@dataclass
class TextLine:
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


@dataclass
class HeadingCandidate:
    text: str
    page: int
    section: str | None
    score: float
    confidence: float
    reasons: list[str]
    font_size: float
    body_font_size: float
    y0: float
    x0: float
    normalized: str


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-zA-Z])", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


def normalize_heading(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*(\d+|[IVX]+)(\.\d+)*\s*[\.\)]?\s+", "", text, flags=re.I)
    text = re.sub(r"^\s*[A-Z]\s*[\.\)]\s+", "", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def normalize_text_line(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_bold_font(font_name: str) -> bool:
    lowered = font_name.lower()
    return any(flag in lowered for flag in ["bold", "black", "heavy", "semibold", "demi"])


def is_italic_font(font_name: str) -> bool:
    return "italic" in font_name.lower() or "oblique" in font_name.lower()


def extract_text_lines(page: fitz.Page, page_number: int) -> list[TextLine]:
    page_dict = page.get_text("dict")
    lines: list[TextLine] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        block_no = int(block.get("number", 0))

        for line_no, line in enumerate(block.get("lines", [])):
            spans = line.get("spans", [])

            if not spans:
                continue

            text_parts = []
            font_sizes = []
            font_names = []
            bbox_values = []

            for span in spans:
                span_text = span.get("text", "")
                if span_text:
                    text_parts.append(span_text)

                font_sizes.append(float(span.get("size", 0.0)))
                font_names.append(str(span.get("font", "")))
                bbox_values.append(span.get("bbox", [0, 0, 0, 0]))

            text = normalize_text_line("".join(text_parts))

            if not text:
                continue

            x0 = min(float(b[0]) for b in bbox_values)
            y0 = min(float(b[1]) for b in bbox_values)
            x1 = max(float(b[2]) for b in bbox_values)
            y1 = max(float(b[3]) for b in bbox_values)

            size = median(font_sizes) if font_sizes else 0.0
            font_name_counter = Counter(font_names)
            dominant_font = font_name_counter.most_common(1)[0][0] if font_name_counter else ""

            lines.append(
                TextLine(
                    text=text,
                    page=page_number,
                    block_no=block_no,
                    line_no=line_no,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    font_size=float(size),
                    font_name=dominant_font,
                    is_bold=is_bold_font(dominant_font),
                    is_italic=is_italic_font(dominant_font),
                )
            )

    return sorted(lines, key=lambda item: (item.y0, item.x0))


def estimate_body_font_size(all_lines: list[TextLine]) -> float:
    sizes = []

    for line in all_lines:
        text = line.text.strip()

        if len(text) < 30:
            continue

        if len(text.split()) < 5:
            continue

        rounded = round(line.font_size * 2) / 2
        sizes.append(rounded)

    if not sizes:
        all_sizes = [round(line.font_size * 2) / 2 for line in all_lines if line.font_size > 0]
        if not all_sizes:
            return 10.0
        return Counter(all_sizes).most_common(1)[0][0]

    return Counter(sizes).most_common(1)[0][0]


def extract_title(doc: fitz.Document, pdf_path: Path, first_page_lines: list[TextLine]) -> str:
    metadata_title = (doc.metadata or {}).get("title")

    if metadata_title and metadata_title.strip():
        title = metadata_title.strip()
        if 5 <= len(title) <= 200:
            return title

    top_lines = [
        line
        for line in first_page_lines[:20]
        if 8 <= len(line.text) <= 180 and len(line.text.split()) >= 3
    ]

    if not top_lines:
        return pdf_path.stem

    top_lines = sorted(top_lines, key=lambda item: (-item.font_size, item.y0))
    return top_lines[0].text.strip()


def is_reference_line(text: str) -> bool:
    stripped = text.strip()
    return bool(re.match(r"^\[\d+\]\s+", stripped)) or bool(re.match(r"^\d+\.\s+[A-Z][A-Za-z\-]+,", stripped))


def is_equation_or_table_noise(text: str) -> bool:
    stripped = text.strip()

    if len(stripped) < 3:
        return True

    symbol_ratio = sum(1 for ch in stripped if not ch.isalnum() and not ch.isspace()) / max(len(stripped), 1)

    if symbol_ratio > 0.45 and len(stripped.split()) <= 8:
        return True

    if re.match(r"^(table|figure|fig\.)\s+\d+", stripped, flags=re.I):
        return True

    return False


def is_numbered_heading(text: str) -> bool:
    stripped = text.strip()
    patterns = [
        r"^\d+(\.\d+)*\s+[A-Z][A-Za-z0-9][A-Za-z0-9 ,:\-/()]+$",
        r"^[IVX]+\.\s+[A-Z][A-Za-z0-9][A-Za-z0-9 ,:\-/()]+$",
        r"^[A-Z]\.\s+[A-Z][A-Za-z0-9][A-Za-z0-9 ,:\-/()]+$",
    ]
    return any(re.match(pattern, stripped) for pattern in patterns)


def is_all_caps_heading(text: str) -> bool:
    stripped = text.strip()

    if len(stripped) < 4:
        return False

    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return False

    upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return upper_ratio > 0.8 and len(stripped.split()) <= 8


def map_heading_to_section(text: str) -> str | None:
    normalized = normalize_heading(text)

    if not normalized:
        return None

    exact_alias_map = {
        alias: section
        for section, aliases in SECTION_ALIASES.items()
        for alias in aliases
    }

    if normalized in exact_alias_map:
        return exact_alias_map[normalized]

    # Avoid common false positives.
    weak_method_titles = {
        "model",
        "models",
        "language models",
        "large language models",
        "general purpose architectures for nlp",
    }

    if normalized in weak_method_titles:
        return None

    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            if normalized.startswith(alias + " "):
                return section

            if alias in normalized and len(normalized.split()) <= 7:
                return section

    return None


def score_heading_candidate(
    line: TextLine,
    body_font_size: float,
    page_height: float,
    page_width: float,
) -> HeadingCandidate | None:
    text = line.text.strip()

    if not text:
        return None

    if is_equation_or_table_noise(text):
        return None

    word_count = len(text.split())
    char_count = len(text)

    if char_count > 130:
        return None

    if word_count > 14:
        return None

    normalized = normalize_heading(text)
    section = map_heading_to_section(text)

    score = 0.0
    reasons: list[str] = []

    font_delta = line.font_size - body_font_size

    if font_delta >= 3:
        score += 3.0
        reasons.append("font much larger than body")
    elif font_delta >= 1.2:
        score += 2.0
        reasons.append("font larger than body")
    elif font_delta >= 0.4:
        score += 0.8
        reasons.append("font slightly larger than body")

    if line.is_bold:
        score += 1.8
        reasons.append("bold font")

    if is_numbered_heading(text):
        score += 3.0
        reasons.append("numbered heading pattern")

    if is_all_caps_heading(text):
        score += 1.5
        reasons.append("all-caps heading style")

    if section:
        score += 3.0
        reasons.append(f"section keyword mapped to {section}")

    if 1 <= word_count <= 8:
        score += 1.0
        reasons.append("short heading-like length")

    if line.y0 < page_height * 0.32:
        score += 0.8
        reasons.append("near top of page")

    # Many paper headings are left-aligned with the text block.
    if line.x0 < page_width * 0.22:
        score += 0.4
        reasons.append("left aligned")

    if text.endswith(".") and word_count > 4:
        score -= 2.0
        reasons.append("sentence-like period ending")

    if text.endswith(",") or text.endswith(";"):
        score -= 1.5
        reasons.append("sentence-like punctuation")

    if is_reference_line(text):
        score -= 4.0
        reasons.append("reference entry pattern")

    if normalized in {"arxiv", "preprint", "conference paper"}:
        score -= 4.0
        reasons.append("document metadata noise")

    if score < 3.5:
        return None

    confidence = 1 / (1 + math.exp(-(score - 5.5) / 1.3))
    confidence = round(float(confidence), 4)

    return HeadingCandidate(
        text=text,
        page=line.page,
        section=section,
        score=round(score, 3),
        confidence=confidence,
        reasons=reasons,
        font_size=round(line.font_size, 2),
        body_font_size=round(body_font_size, 2),
        y0=round(line.y0, 2),
        x0=round(line.x0, 2),
        normalized=normalized,
    )


def select_best_heading(
    lines: list[TextLine],
    body_font_size: float,
    page_height: float,
    page_width: float,
) -> tuple[HeadingCandidate | None, list[HeadingCandidate]]:
    candidates = []

    # Usually the section heading appears in the first half of the page.
    for line in lines:
        if line.y0 > page_height * 0.62:
            continue

        candidate = score_heading_candidate(
            line=line,
            body_font_size=body_font_size,
            page_height=page_height,
            page_width=page_width,
        )

        if candidate:
            candidates.append(candidate)

    candidates = sorted(candidates, key=lambda item: (-item.score, item.y0))

    if not candidates:
        return None, []

    # Prefer candidates that map to a known section label.
    mapped = [item for item in candidates if item.section]

    if mapped:
        return mapped[0], candidates[:8]

    return candidates[0], candidates[:8]


def transition_allowed(previous: str, new: str) -> bool:
    if new == "empty":
        return True

    if previous in {"main_text", "empty"}:
        return True

    if previous == new:
        return True

    if new == "appendix":
        return True

    if previous == "references":
        return new == "appendix"

    if previous == "appendix":
        return True

    try:
        previous_index = SECTION_ORDER.index(previous)
        new_index = SECTION_ORDER.index(new)
    except ValueError:
        return True

    if new_index >= previous_index:
        return True

    # Experiment and result pages often alternate in appendices.
    if previous in {"experiment", "results"} and new in {"experiment", "results"}:
        return True

    return False


def infer_section_for_page(
    *,
    page_number: int,
    lines: list[TextLine],
    previous_section: str,
    body_font_size: float,
    page_height: float,
    page_width: float,
) -> tuple[str, HeadingCandidate | None, float, str, list[HeadingCandidate]]:
    if not lines:
        return "empty", None, 1.0, "empty page", []

    best_heading, candidates = select_best_heading(
        lines=lines,
        body_font_size=body_font_size,
        page_height=page_height,
        page_width=page_width,
    )

    # Strong special case: first pages often contain Abstract as a normal line.
    first_page_text = "\n".join(line.text for line in lines[:25])

    if page_number <= 2 and re.search(r"\bAbstract\b", first_page_text, flags=re.I):
        abstract_candidate = HeadingCandidate(
            text="Abstract",
            page=page_number,
            section="abstract",
            score=8.0,
            confidence=0.9,
            reasons=["Abstract found on first pages"],
            font_size=body_font_size,
            body_font_size=body_font_size,
            y0=0.0,
            x0=0.0,
            normalized="abstract",
        )
        return "abstract", abstract_candidate, 0.9, "first-page abstract heuristic", candidates

    if best_heading and best_heading.section:
        new_section = best_heading.section

        if transition_allowed(previous_section, new_section):
            return (
                new_section,
                best_heading,
                best_heading.confidence,
                "layout heading selected",
                candidates,
            )

        return (
            previous_section,
            None,
            0.55,
            f"blocked invalid transition {previous_section} -> {new_section}",
            candidates,
        )

    if best_heading and best_heading.score >= 7.0:
        # Strong heading style but no known section mapping. Keep previous section,
        # but record the heading for debugging.
        return (
            previous_section,
            best_heading,
            min(best_heading.confidence, 0.65),
            "strong heading without section mapping; kept previous section",
            candidates,
        )

    return (
        previous_section,
        None,
        0.45,
        "no reliable section heading; inherited previous section",
        candidates,
    )


def page_text_from_lines(lines: list[TextLine]) -> str:
    if not lines:
        return ""

    parts = []

    current_block = None

    for line in lines:
        if current_block is not None and line.block_no != current_block:
            parts.append("\n")

        parts.append(line.text)
        current_block = line.block_no

    return clean_text("\n".join(parts))


def build_markdown(rows: list[dict[str, Any]]) -> str:
    parts = []

    for row in rows:
        parts.append(
            f"\n\n<!-- doc_id={row['doc_id']} page={row['page']} "
            f"section={row['section']} confidence={row['section_confidence']} "
            f"heading={row.get('section_heading')} -->\n\n"
        )
        parts.append(row["text"])

    return "\n".join(parts).strip() + "\n"


def quality_status_and_warnings(rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    warnings = []

    total_pages = len(rows)
    total_chars = sum(row.get("char_count", 0) for row in rows)

    short_pages = [row["page"] for row in rows if row.get("char_count", 0) < 200]
    section_counts = Counter(row.get("section", "unknown") for row in rows)
    low_conf_pages = [
        row["page"]
        for row in rows
        if float(row.get("section_confidence", 0.0)) < 0.55
    ]

    if total_pages == 0:
        warnings.append("No pages were parsed.")

    if total_chars < 1000:
        warnings.append("Parsed text is very short. The PDF may be scanned or image-based.")

    if len(section_counts) <= 2 and total_pages >= 8:
        warnings.append("Very few sections detected. Section recovery may be weak.")

    if "references" not in section_counts and total_pages >= 8:
        warnings.append("References section was not detected.")

    if "introduction" not in section_counts and total_pages >= 8:
        warnings.append("Introduction section was not detected.")

    if len(short_pages) > max(2, total_pages // 3):
        warnings.append("Many pages are very short.")

    if len(low_conf_pages) > max(2, total_pages // 3):
        warnings.append("Many pages have low section confidence.")

    return ("ok" if not warnings else "warning"), warnings


def build_quality_report(
    *,
    rows: list[dict[str, Any]],
    pdf_path: Path,
    body_font_size: float,
    title: str,
) -> dict[str, Any]:
    section_counts = Counter(row.get("section", "unknown") for row in rows)
    status, warnings = quality_status_and_warnings(rows)

    detected_headings = []

    for row in rows:
        if row.get("section_heading"):
            detected_headings.append(
                {
                    "page": row["page"],
                    "section": row["section"],
                    "heading": row["section_heading"],
                    "confidence": row["section_confidence"],
                    "score": row.get("heading_score"),
                    "reason": row.get("section_reason"),
                }
            )

    return {
        "pdf_path": str(pdf_path),
        "doc_id": pdf_path.stem,
        "title": title,
        "parser": "layout_aware_rule_parser",
        "body_font_size": body_font_size,
        "parsed_pages": len(rows),
        "total_chars": sum(row.get("char_count", 0) for row in rows),
        "total_words": sum(row.get("word_count", 0) for row in rows),
        "avg_chars_per_page": round(
            sum(row.get("char_count", 0) for row in rows) / len(rows), 2
        )
        if rows
        else 0,
        "short_pages": [row["page"] for row in rows if row.get("char_count", 0) < 200],
        "section_counts": dict(section_counts),
        "detected_headings": detected_headings,
        "page_sections": [
            {
                "page": row["page"],
                "section": row["section"],
                "heading": row.get("section_heading"),
                "confidence": row.get("section_confidence"),
                "reason": row.get("section_reason"),
            }
            for row in rows
        ],
        "low_confidence_pages": [
            {
                "page": row["page"],
                "section": row["section"],
                "confidence": row["section_confidence"],
                "reason": row["section_reason"],
            }
            for row in rows
            if float(row.get("section_confidence", 0.0)) < 0.55
        ],
        "warnings": warnings,
        "status": status,
    }


def parse_pdf(pdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    doc = fitz.open(pdf_path)

    page_lines: dict[int, list[TextLine]] = {}
    all_lines: list[TextLine] = []

    for page_index, page in enumerate(doc, start=1):
        lines = extract_text_lines(page=page, page_number=page_index)
        page_lines[page_index] = lines
        all_lines.extend(lines)

    body_font_size = estimate_body_font_size(all_lines)
    first_page_lines = page_lines.get(1, [])
    title = extract_title(doc, pdf_path, first_page_lines)

    rows = []
    current_section = "main_text"

    for page_index, page in enumerate(doc, start=1):
        lines = page_lines.get(page_index, [])
        page_height = float(page.rect.height)
        page_width = float(page.rect.width)

        section, heading, confidence, reason, candidates = infer_section_for_page(
            page_number=page_index,
            lines=lines,
            previous_section=current_section,
            body_font_size=body_font_size,
            page_height=page_height,
            page_width=page_width,
        )

        if section != "empty":
            current_section = section

        text = page_text_from_lines(lines)

        rows.append(
            {
                "doc_id": pdf_path.stem,
                "title": title,
                "page": page_index,
                "section": section,
                "section_heading": heading.text if heading else None,
                "section_confidence": round(float(confidence), 4),
                "section_reason": reason,
                "heading_score": heading.score if heading else None,
                "heading_reasons": heading.reasons if heading else [],
                "text": text,
                "char_count": len(text),
                "word_count": len(text.split()),
                "heading_candidates": [asdict(candidate) for candidate in candidates],
                "debug_first_lines": [asdict(line) for line in lines[:12]],
            }
        )

    quality_report = build_quality_report(
        rows=rows,
        pdf_path=pdf_path,
        body_font_size=body_font_size,
        title=title,
    )

    return rows, quality_report


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse a PDF into layout-aware page-level JSONL for ResearchGuard."
    )

    parser.add_argument("--input", required=True, help="Input PDF path.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pdf_path = Path(args.input)
    out_dir = Path(args.out_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    rows, quality_report = parse_pdf(pdf_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(rows, out_dir / "parsed_pages.jsonl")
    (out_dir / "parsed.md").write_text(build_markdown(rows), encoding="utf-8", newline="\n")
    write_json(quality_report, out_dir / "parse_quality_report.json")

    debug_pages = [
        {
            "page": row["page"],
            "section": row["section"],
            "section_heading": row.get("section_heading"),
            "section_confidence": row.get("section_confidence"),
            "section_reason": row.get("section_reason"),
            "heading_score": row.get("heading_score"),
            "heading_reasons": row.get("heading_reasons"),
            "heading_candidates": row.get("heading_candidates"),
            "first_lines": row.get("debug_first_lines"),
        }
        for row in rows
    ]

    write_json(debug_pages, out_dir / "parse_debug_pages.json")

    print("PDF parsing finished.")
    print(f"Input PDF: {pdf_path}")
    print(f"Output directory: {out_dir}")
    print(f"Parsed pages JSONL: {out_dir / 'parsed_pages.jsonl'}")
    print(f"Parsed markdown: {out_dir / 'parsed.md'}")
    print(f"Quality report: {out_dir / 'parse_quality_report.json'}")
    print(f"Debug pages report: {out_dir / 'parse_debug_pages.json'}")
    print("")
    print(json.dumps(quality_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()