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

from researchguard.retrieval import (
    AnswerGenerationPipeline,
    EvidenceSufficiencyPipeline,
    MetadataFilter,
    RetrievalEngine,
)
from researchguard.retrieval.answer_generator import load_answer_generation_settings
from researchguard.retrieval.evidence_judge import load_evidence_judge_settings


DEFAULT_CONFIG = Path(r"C:\Users\18449\Desktop\researchguard_workspace\configs\retrieval_v1.yaml")
DEFAULT_EVIDENCE_CONFIG = Path(
    r"C:\Users\18449\Desktop\researchguard_workspace\configs\evidence_sufficiency_v1.yaml"
)
DEFAULT_ANSWER_CONFIG = Path(
    r"C:\Users\18449\Desktop\researchguard_workspace\configs\answer_generation_v1.yaml"
)


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
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument("--rerank", dest="rerank", action="store_true", help="Rerank hybrid candidates.")
    rerank_group.add_argument("--no-rerank", dest="rerank", action="store_false", help="Disable reranking.")
    parser.set_defaults(rerank=None)
    parser.add_argument("--rerank-candidate-k", type=int, default=None)
    rewrite_group = parser.add_mutually_exclusive_group()
    rewrite_group.add_argument("--rewrite", dest="rewrite", action="store_true", help="Use one normalized query.")
    rewrite_group.add_argument("--no-rewrite", dest="rewrite", action="store_false", help="Disable query rewrite.")
    parser.set_defaults(rewrite=None)
    parser.add_argument("--multi-query", action="store_true", help="Retrieve original, normalized, and expansion queries.")
    parser.add_argument("--evidence-check", action="store_true", help="Judge whether final Top-k evidence supports answering.")
    parser.add_argument(
        "--evidence-config",
        default=str(DEFAULT_EVIDENCE_CONFIG),
        help="Path to evidence_sufficiency_v1.yaml.",
    )
    parser.add_argument(
        "--generate-answer",
        action="store_true",
        help="Generate an evidence-grounded answer after the Evidence Sufficiency gate.",
    )
    parser.add_argument(
        "--answer-config",
        default=str(DEFAULT_ANSWER_CONFIG),
        help="Path to answer_generation_v1.yaml.",
    )
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args()
    if args.multi_query and args.rewrite is False:
        parser.error("--multi-query cannot be combined with --no-rewrite.")
    if args.generate_answer and not args.evidence_check:
        parser.error("--generate-answer requires --evidence-check.")
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
        rerank=args.rerank,
        rerank_candidate_k=args.rerank_candidate_k,
        rewrite=args.rewrite,
        multi_query=bool(args.multi_query),
    )
    payload: dict[str, Any] = response.to_dict(include_text=args.include_text)
    evidence_result = None
    if args.evidence_check:
        _, evidence_settings = load_evidence_judge_settings(args.evidence_config)
        evidence_result = EvidenceSufficiencyPipeline(evidence_settings).assess(args.query, response.hits)
        payload["evidence_sufficiency"] = evidence_result.to_dict()
        payload["evidence_check_latency_ms"] = evidence_result.latency_ms
    if args.generate_answer and evidence_result is not None:
        _, answer_settings = load_answer_generation_settings(args.answer_config)
        answer_result = AnswerGenerationPipeline(answer_settings).generate(
            args.query,
            response.hits,
            evidence_result,
        )
        payload["answer_generation"] = answer_result.to_dict()
        payload["answer_generation_latency_ms"] = answer_result.latency_ms
        payload["answer_pipeline_total_latency_ms"] = (
            float(response.total_latency_ms or response.latency_ms)
            + evidence_result.latency_ms
            + answer_result.latency_ms
        )
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
