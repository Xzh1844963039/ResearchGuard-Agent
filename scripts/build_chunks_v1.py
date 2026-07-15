# C:\Users\18449\Desktop\researchguard_workspace\scripts\build_chunks_v1.py
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from researchguard.ingestion.chunk_builder import build_chunks, summarize_chunks  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def infer_title(blocks_path: Path, fallback_doc_id: str) -> str:
    paper_dir = blocks_path.parent
    for name in ("parse_quality_report.json", "layout.json"):
        data = read_json(paper_dir / name)
        title = str(data.get("title", "")).strip()
        if title:
            return title
    return fallback_doc_id


def collect_block_ids(blocks: list[dict[str, Any]], block_type: str) -> set[str]:
    return {
        str(block.get("block_id"))
        for block in blocks
        if str(block.get("block_type")) == block_type and str(block.get("block_id", "")).strip()
    }


def coverage_summary(blocks: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = {
        str(block_id)
        for chunk in chunks
        for block_id in chunk.get("source_block_ids", chunk.get("block_ids", []))
        if str(block_id).strip()
    }
    heading_ids = {
        str(block_id)
        for chunk in chunks
        for block_id in chunk.get("heading_block_ids", [])
        if str(block_id).strip()
    }
    covered_ids = source_ids | heading_ids

    result: dict[str, Any] = {}
    for block_type in ("paragraph", "heading", "heading_candidate", "reference_entry", "equation", "caption", "table"):
        ids = collect_block_ids(blocks, block_type)
        result[block_type] = {
            "total": len(ids),
            "covered": len(ids & covered_ids) if block_type in {"heading", "heading_candidate"} else len(ids & source_ids),
            "lost": len(ids - covered_ids) if block_type in {"heading", "heading_candidate"} else len(ids - source_ids),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build section-aware chunks from parser v5 blocks.")
    parser.add_argument("--blocks", required=True, type=Path, help="Path to blocks.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Output chunks.jsonl path")
    parser.add_argument("--doc_id", required=True, help="Document id")
    parser.add_argument("--title", default=None, help="Document title; defaults to parser report/layout title")
    parser.add_argument("--max_chars", type=int, default=1600)
    parser.add_argument("--target_chars", type=int, default=1200)
    parser.add_argument("--min_chars", type=int, default=250)
    parser.add_argument("--overlap_sentences", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blocks_path = args.blocks.resolve()
    output_path = args.output.resolve()
    blocks = read_jsonl(blocks_path)
    title = args.title or infer_title(blocks_path, args.doc_id)

    chunks = build_chunks(
        doc_id=args.doc_id,
        title=title,
        blocks=blocks,
        max_chars=args.max_chars,
        target_chars=args.target_chars,
        min_chars=args.min_chars,
        overlap_sentences=args.overlap_sentences,
    )
    write_jsonl(output_path, chunks)

    summary = summarize_chunks(chunks)
    summary.update(
        {
            "doc_id": args.doc_id,
            "title": title,
            "input_blocks": len(blocks),
            "output": str(output_path),
            "block_type_counts": dict(Counter(str(block.get("block_type", "unknown")) for block in blocks)),
            "coverage": coverage_summary(blocks, chunks),
            "params": {
                "max_chars": args.max_chars,
                "target_chars": args.target_chars,
                "min_chars": args.min_chars,
                "overlap_sentences": args.overlap_sentences,
            },
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
