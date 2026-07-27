# scripts/prepare_holdout_v1.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.ingestion.parse_pdf import parse_pdf_to_outputs  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "eval" / "holdout" / "manifest.jsonl"
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw_docs" / "holdout_v1"
DEFAULT_PARSED = PROJECT_ROOT / "data" / "parsed" / "holdout_v1"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("Holdout manifest must contain JSON objects.")
    doc_ids = [str(row.get("doc_id", "")).strip() for row in rows]
    if not 10 <= len(rows) <= 15 or len(set(doc_ids)) != len(rows):
        raise ValueError("Holdout manifest requires 10-15 unique documents.")
    return [dict(row) for row in rows]


def download_pdf(row: Mapping[str, Any], raw_root: Path) -> dict[str, Any]:
    doc_id = str(row["doc_id"])
    destination = raw_root / f"{doc_id}.pdf"
    if destination.exists() and destination.stat().st_size > 4:
        with destination.open("rb") as handle:
            if handle.read(4) == b"%PDF":
                return {
                    "doc_id": doc_id,
                    "status": "reused",
                    "path": str(destination.relative_to(PROJECT_ROOT)),
                    "bytes": destination.stat().st_size,
                }
    response = requests.get(
        str(row["source_url"]),
        timeout=(10, 120),
        headers={"User-Agent": "ResearchGuard-Agent holdout preparation/1.0"},
    )
    response.raise_for_status()
    payload = response.content
    if not payload.startswith(b"%PDF"):
        raise ValueError(f"Downloaded content is not a PDF for {doc_id}.")
    raw_root.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "doc_id": doc_id,
        "status": "downloaded",
        "path": str(destination.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
    }


def parse_pdf(
    row: Mapping[str, Any],
    raw_root: Path,
    parsed_root: Path,
) -> dict[str, Any]:
    doc_id = str(row["doc_id"])
    pdf_path = raw_root / f"{doc_id}.pdf"
    if not pdf_path.exists():
        return {"doc_id": doc_id, "status": "missing_pdf"}
    report = parse_pdf_to_outputs(
        pdf_path=pdf_path,
        out_dir=parsed_root / doc_id,
        max_chunk_chars=1600,
        min_chunk_chars=250,
    )
    return {
        "doc_id": doc_id,
        "status": str(report.get("status", "unknown")),
        "pages": int(report.get("parsed_pages", 0)),
        "chunks": int(report.get("chunk_count", 0)),
        "warnings": list(report.get("warnings", ())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and parse the isolated ResearchGuard holdout corpus."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--parsed-root", type=Path, default=DEFAULT_PARSED)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--parse", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.download and not args.parse:
        parser.error("Select --download, --parse, or both.")

    rows = load_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    result: dict[str, Any] = {
        "manifest": str(args.manifest),
        "document_count": len(rows),
        "downloads": [],
        "parses": [],
    }
    for index, row in enumerate(rows, start=1):
        if args.download:
            item = download_pdf(row, args.raw_root)
            result["downloads"].append(item)
            print(
                f"[download {index:02d}/{len(rows)}] "
                f"{row['doc_id']}: {item['status']}"
            )
        if args.parse:
            item = parse_pdf(row, args.raw_root, args.parsed_root)
            result["parses"].append(item)
            print(
                f"[parse {index:02d}/{len(rows)}] "
                f"{row['doc_id']}: {item['status']}"
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
