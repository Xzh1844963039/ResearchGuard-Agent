# C:\Users\18449\Desktop\researchguard_workspace\researchguard\ingestion\parse_pdf.py
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from researchguard.ingestion.block_detector import detect_blocks
from researchguard.ingestion.chunk_builder import build_chunks
from researchguard.ingestion.layout_extractor import extract_document_layout
from researchguard.ingestion.section_recovery import recover_sections


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_markdown(page_records: list[dict[str, Any]]) -> str:
    parts: list[str] = []

    for row in page_records:
        parts.append(
            f"\n\n<!-- doc_id={row['doc_id']} page={row['page']} "
            f"section={row['section']} confidence={row['section_confidence']} "
            f"heading={row.get('section_heading')} -->\n\n"
        )
        parts.append(row.get("text", ""))

    return "\n".join(parts).strip() + "\n"


def build_quality_report(
    *,
    layout: dict[str, Any],
    page_records: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    section_counts = Counter(row.get("section", "unknown") for row in page_records)
    block_type_counts = Counter(block.get("block_type", "unknown") for block in blocks)

    detected_headings = []

    for block in blocks:
        if block.get("block_type") == "heading":
            pred = block.get("heading_prediction", {})
            detected_headings.append(
                {
                    "page": block.get("page"),
                    "section": block.get("section"),
                    "heading": block.get("text"),
                    "confidence": pred.get("confidence"),
                    "score": pred.get("score"),
                    "reasons": pred.get("reasons"),
                }
            )

    low_confidence_pages = [
        {
            "page": row["page"],
            "section": row["section"],
            "confidence": row["section_confidence"],
            "reason": row["section_reason"],
        }
        for row in page_records
        if float(row.get("section_confidence", 0.0)) < 0.55
    ]

    warnings = []

    total_pages = len(page_records)
    total_chars = sum(row.get("char_count", 0) for row in page_records)
    short_pages = [row["page"] for row in page_records if row.get("char_count", 0) < 200]

    if total_pages == 0:
        warnings.append("No pages parsed.")

    if total_chars < 1000:
        warnings.append("Parsed text is very short. The PDF may be scanned or image-based.")

    if len(section_counts) <= 2 and total_pages >= 8:
        warnings.append("Very few sections detected.")

    if "abstract" not in section_counts and total_pages >= 5:
        warnings.append("Abstract section was not detected.")

    if "introduction" not in section_counts and total_pages >= 8:
        warnings.append("Introduction section was not detected.")

    if "references" not in section_counts and total_pages >= 8:
        warnings.append("References section was not detected.")

    if len(short_pages) > max(2, total_pages // 3):
        warnings.append("Many pages are very short.")

    if len(low_confidence_pages) > max(2, total_pages // 3):
        warnings.append("Many pages have low section confidence.")

    if not chunks:
        warnings.append("No chunks generated.")

    chunk_lengths = [chunk["char_count"] for chunk in chunks]

    return {
        "pdf_path": layout.get("pdf_path"),
        "doc_id": layout.get("doc_id"),
        "title": layout.get("title"),
        "parser": "layout_block_section_chunk_parser_v1",
        "body_font_size": layout.get("body_font_size"),
        "parsed_pages": total_pages,
        "total_chars": total_chars,
        "total_words": sum(row.get("word_count", 0) for row in page_records),
        "avg_chars_per_page": round(total_chars / total_pages, 2) if total_pages else 0,
        "section_counts": dict(section_counts),
        "block_type_counts": dict(block_type_counts),
        "detected_headings": detected_headings,
        "page_sections": [
            {
                "page": row["page"],
                "section": row["section"],
                "heading": row.get("section_heading"),
                "confidence": row.get("section_confidence"),
                "reason": row.get("section_reason"),
            }
            for row in page_records
        ],
        "short_pages": short_pages,
        "low_confidence_pages": low_confidence_pages,
        "chunk_count": len(chunks),
        "chunk_char_stats": {
            "min": min(chunk_lengths) if chunk_lengths else 0,
            "max": max(chunk_lengths) if chunk_lengths else 0,
            "avg": round(sum(chunk_lengths) / len(chunk_lengths), 2) if chunk_lengths else 0,
        },
        "warnings": warnings,
        "status": "ok" if not warnings else "warning",
    }


def parse_pdf_to_outputs(
    *,
    pdf_path: Path,
    out_dir: Path,
    max_chunk_chars: int,
    min_chunk_chars: int,
) -> dict[str, Any]:
    layout = extract_document_layout(pdf_path)
    blocks = detect_blocks(layout)
    page_records, enriched_blocks = recover_sections(layout=layout, blocks=blocks)

    chunks = build_chunks(
        doc_id=layout["doc_id"],
        title=layout["title"],
        blocks=enriched_blocks,
        max_chars=max_chunk_chars,
        min_chars=min_chunk_chars,
    )

    quality_report = build_quality_report(
        layout=layout,
        page_records=page_records,
        blocks=enriched_blocks,
        chunks=chunks,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(layout, out_dir / "layout.json")
    write_jsonl(enriched_blocks, out_dir / "blocks.jsonl")
    write_jsonl(page_records, out_dir / "parsed_pages.jsonl")
    write_jsonl(chunks, out_dir / "chunks.jsonl")
    write_json(quality_report, out_dir / "parse_quality_report.json")
    write_json(enriched_blocks[:200], out_dir / "parse_debug_blocks_sample.json")

    (out_dir / "parsed.md").write_text(
        build_markdown(page_records),
        encoding="utf-8",
        newline="\n",
    )

    return quality_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse PDF into layout-aware pages, blocks, and RAG chunks."
    )

    parser.add_argument("--input", required=True, help="Input PDF path.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument("--max_chunk_chars", type=int, default=1600)
    parser.add_argument("--min_chunk_chars", type=int, default=250)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pdf_path = Path(args.input)
    out_dir = Path(args.out_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    quality_report = parse_pdf_to_outputs(
        pdf_path=pdf_path,
        out_dir=out_dir,
        max_chunk_chars=args.max_chunk_chars,
        min_chunk_chars=args.min_chunk_chars,
    )

    print("PDF parsing finished.")
    print(f"Input PDF: {pdf_path}")
    print(f"Output directory: {out_dir}")
    print(f"Layout: {out_dir / 'layout.json'}")
    print(f"Blocks: {out_dir / 'blocks.jsonl'}")
    print(f"Parsed pages JSONL: {out_dir / 'parsed_pages.jsonl'}")
    print(f"Chunks JSONL: {out_dir / 'chunks.jsonl'}")
    print(f"Parsed markdown: {out_dir / 'parsed.md'}")
    print(f"Quality report: {out_dir / 'parse_quality_report.json'}")
    print("")
    print(json.dumps(quality_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()