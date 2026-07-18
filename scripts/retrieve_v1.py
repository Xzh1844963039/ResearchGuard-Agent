# C:\Users\18449\Desktop\researchguard_workspace\scripts\retrieve_v1.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(r"C:\Users\18449\Desktop\researchguard_workspace")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.retrieval import MetadataFilter, RetrievalEngine


DEFAULT_CONFIG = Path(r"C:\Users\18449\Desktop\researchguard_workspace\configs\retrieval_v1.yaml")


def comma_values(values: list[str] | None) -> tuple[str, ...]:
    items: list[str] = []
    for value in values or []:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(items)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ResearchGuard Retrieval v1 against index_v1.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to retrieval_v1.yaml.")
    parser.add_argument("--query", required=True, help="User query.")
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--dense-backend", choices=["numpy", "chroma"], default=None)
    parser.add_argument("--doc-id", action="append", default=[], help="Repeatable or comma-separated doc_id filter.")
    parser.add_argument("--section", action="append", default=[], help="Repeatable or comma-separated section filter.")
    parser.add_argument("--chunk-type", action="append", default=[], help="Repeatable or comma-separated chunk_type filter.")
    parser.add_argument("--exclude-references", action="store_true")
    parser.add_argument("--has-equation", action="store_true")
    parser.add_argument("--has-table", action="store_true")
    parser.add_argument("--has-caption", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--include-text", action="store_true", help="Include full chunk text in JSON output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    filters = MetadataFilter(
        doc_ids=comma_values(args.doc_id),
        sections=comma_values(args.section),
        chunk_types=comma_values(args.chunk_type),
        exclude_references=bool(args.exclude_references),
        has_equation=True if args.has_equation else None,
        has_table=True if args.has_table else None,
        has_caption=True if args.has_caption else None,
    )
    engine = RetrievalEngine.from_config(args.config, dense_backend_override=args.dense_backend)
    response = engine.retrieve(
        args.query,
        mode=args.mode,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        filters=filters,
    )
    payload: dict[str, Any] = response.to_dict(include_text=args.include_text)
    if not args.include_text:
        for hit in payload["hits"]:
            source_text = engine.bundle.document_by_id[hit["chunk_id"]].get("text", "")
            hit["text_preview"] = " ".join(str(source_text).split())[:240]

    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
