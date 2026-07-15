# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_index_v1.py
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from researchguard.indexing.corpus_loader import (  # noqa: E402
    build_corpus_manifest,
    corpus_fingerprint,
    load_yaml,
    read_jsonl,
    stable_json_hash,
    write_json,
    write_jsonl,
)
from researchguard.indexing.dense_index import DenseNumpyIndex  # noqa: E402
from researchguard.indexing.embedding_cache import EmbeddingCache  # noqa: E402
from researchguard.indexing.embedding_provider import OpenAIEmbeddingProvider, parse_embedding_config  # noqa: E402
from researchguard.indexing.index_v1 import plan_incremental, resolve_dense_dir, resolve_output_dir, resolve_sparse_dir  # noqa: E402
from researchguard.indexing.sparse_index import LocalBM25Index, tokenize  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "index_validation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ResearchGuard persistent index v1.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "indexing_v1.yaml"))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def query_fragment(text: str, max_chars: int = 420) -> str:
    stripped = " ".join(str(text).split())
    if len(stripped) <= max_chars:
        return stripped
    start = min(120, max(0, len(stripped) // 4))
    return stripped[start : start + max_chars].strip()


def sample_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random("index-validation-v1")
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_many(candidates: list[dict[str, Any]], limit: int) -> None:
        pool = list(candidates)
        rng.shuffle(pool)
        for doc in pool[:limit]:
            chunk_id = str(doc.get("chunk_id"))
            if chunk_id not in selected_ids:
                selected.append(doc)
                selected_ids.add(chunk_id)

    by_paper: dict[str, list[dict[str, Any]]] = {}
    for doc in documents:
        by_paper.setdefault(str(doc.get("doc_id", "unknown")), []).append(doc)

    for paper, docs in sorted(by_paper.items()):
        ordinary = [
            doc
            for doc in docs
            if doc.get("section") != "references"
            and not doc.get("has_equation")
            and not doc.get("has_table")
            and not doc.get("has_caption")
            and int(doc.get("char_count", 0)) >= 150
        ]
        add_many(ordinary, 5)
        for section in ("method", "results", "experiment", "references"):
            add_many([doc for doc in docs if doc.get("section") == section], 1)
        add_many([doc for doc in docs if doc.get("has_equation") or doc.get("has_table") or doc.get("has_caption")], 2)
        add_many([doc for doc in docs if int(doc.get("char_count", 0)) < 150], 2)

    return selected


def validate_corpus(
    *,
    config: dict[str, Any],
    manifest_docs: list[dict[str, Any]],
    index_dir: Path,
) -> tuple[dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
    current = build_corpus_manifest(config)
    manifest_by_id = {str(doc.get("chunk_id")): doc for doc in manifest_docs}
    current_by_id = {str(doc.get("chunk_id")): doc for doc in current.documents}
    corpus_audit: list[dict[str, Any]] = []
    metadata_audit: list[dict[str, Any]] = []
    counts = Counter()

    if len(manifest_docs) != len(current.documents):
        counts["chunk_count_mismatch"] += 1
    if len(manifest_by_id) != len(manifest_docs):
        counts["duplicate_chunk_id"] += len(manifest_docs) - len(manifest_by_id)

    missing_from_manifest = sorted(set(current_by_id) - set(manifest_by_id))
    stale_in_manifest = sorted(set(manifest_by_id) - set(current_by_id))
    counts["missing_manifest_entries"] += len(missing_from_manifest)
    counts["stale_manifest_entries"] += len(stale_in_manifest)

    source_hash_mismatches = 0
    summary = load_json(index_dir / "corpus_summary.json")
    for path_text, expected_hash in summary.get("source_file_hashes", {}).items():
        path = Path(path_text)
        if not path.exists():
            source_hash_mismatches += 1
            continue
        actual_hash = current.summary.get("source_file_hashes", {}).get(path_text)
        if actual_hash != expected_hash:
            source_hash_mismatches += 1
    counts["source_file_hash_mismatch"] += source_hash_mismatches

    for chunk_id, doc in manifest_by_id.items():
        current_doc = current_by_id.get(chunk_id)
        text = str(doc.get("text", ""))
        missing_fields = []
        for field in ("chunk_id", "doc_id", "section", "page_start", "page_end", "source_block_ids", "text"):
            value = doc.get(field)
            if value in (None, "", []):
                missing_fields.append(field)
        content_hash_ok = bool(current_doc and current_doc.get("content_hash") == doc.get("content_hash"))
        metadata_hash_ok = bool(current_doc and current_doc.get("metadata_hash") == doc.get("metadata_hash"))
        if missing_fields:
            counts["required_metadata_missing"] += 1
        if not text.strip():
            counts["empty_text"] += 1
        if not doc.get("source_block_ids"):
            counts["source_block_ids_missing"] += 1
        if not content_hash_ok:
            counts["content_hash_mismatch"] += 1
        if not metadata_hash_ok:
            counts["metadata_hash_mismatch"] += 1
        row = {
            "chunk_id": chunk_id,
            "doc_id": doc.get("doc_id"),
            "section": doc.get("section"),
            "char_count": doc.get("char_count"),
            "missing_fields": missing_fields,
            "content_hash_ok": content_hash_ok,
            "metadata_hash_ok": metadata_hash_ok,
            "source_path": doc.get("source_path"),
        }
        corpus_audit.append(row)
        metadata_audit.append(row)

    counts["corpus_fingerprint_mismatch"] += int(corpus_fingerprint(manifest_docs) != summary.get("corpus_fingerprint"))
    return dict(counts), corpus_audit, metadata_audit


def validate_embeddings(
    *,
    dense_index: DenseNumpyIndex,
    manifest_docs: list[dict[str, Any]],
    index_manifest: dict[str, Any],
    index_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts = Counter()
    audit_rows: list[dict[str, Any]] = []
    vectors = dense_index.vectors
    expected_dim = int(index_manifest.get("embedding_dimensions", 0))
    if len(vectors) != len(manifest_docs):
        counts["embedding_count_mismatch"] += 1
    if vectors.ndim != 2 or vectors.shape[1] != expected_dim:
        counts["dimension_mismatch"] += 1
    if not np.isfinite(vectors).all():
        counts["invalid_embedding"] += 1
    norms = np.linalg.norm(vectors, axis=1)
    zero_count = int(np.sum(norms == 0))
    counts["zero_vector"] += zero_count
    if bool(index_manifest.get("normalize")):
        norm_bad = int(np.sum(np.abs(norms - 1.0) > 1e-3))
        counts["normalization_mismatch"] += norm_bad

    cache = EmbeddingCache(index_dir / "embedding_cache")
    embedding_config = parse_embedding_config(config)
    for doc, vector, norm in zip(manifest_docs, vectors, norms):
        cache_key = EmbeddingCache.make_key(
            provider=embedding_config.provider,
            model=embedding_config.model,
            content_hash=str(doc.get("content_hash")),
        )
        cache_entry_present = cache_key in cache.entries
        if not cache_entry_present:
            counts["cache_entry_missing"] += 1
        audit_rows.append(
            {
                "chunk_id": doc.get("chunk_id"),
                "doc_id": doc.get("doc_id"),
                "dimension": int(len(vector)),
                "norm": float(norm),
                "has_nan": bool(np.isnan(vector).any()),
                "has_inf": bool(np.isinf(vector).any()),
                "all_zero": bool(norm == 0),
                "cache_key": cache_key,
                "cache_entry_present": cache_entry_present,
                "model": embedding_config.model,
            }
        )
    return dict(counts), audit_rows


def validate_dense_sparse(
    *,
    dense_index: DenseNumpyIndex,
    sparse_index: LocalBM25Index | None,
    manifest_docs: list[dict[str, Any]],
    index_manifest: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    counts = Counter()
    manifest_ids = [str(doc.get("chunk_id")) for doc in manifest_docs]
    if len(dense_index.chunk_ids) != len(manifest_ids):
        counts["dense_index_count_mismatch"] += 1
    missing_mapping = sorted(set(manifest_ids) - set(dense_index.chunk_ids))
    stale_mapping = sorted(set(dense_index.chunk_ids) - set(manifest_ids))
    counts["missing_id_mapping"] += len(missing_mapping)
    counts["stale_entries"] += len(stale_mapping)
    counts["duplicate_index_id"] += len(dense_index.chunk_ids) - len(set(dense_index.chunk_ids))
    if str(index_manifest.get("dense_metric")) != dense_index.metric:
        counts["metric_mismatch"] += 1

    reload_audit = {
        "dense_loaded": True,
        "dense_vector_count": len(dense_index.chunk_ids),
        "dense_dimension": dense_index.dimension,
        "missing_mapping_examples": missing_mapping[:20],
        "stale_mapping_examples": stale_mapping[:20],
        "sparse_loaded": sparse_index is not None,
        "sparse_document_count": len(sparse_index.chunk_ids) if sparse_index else 0,
    }
    if sparse_index is not None:
        if len(sparse_index.chunk_ids) != len(manifest_ids):
            counts["sparse_index_count_mismatch"] += 1
        if sorted(sparse_index.chunk_ids) != sorted(manifest_ids):
            counts["sparse_id_mapping_mismatch"] += 1
        token_probe = "GPT-4o 2023 0.95 Qwen2.5-Math"
        if not tokenize(token_probe):
            counts["sparse_tokenization_failure"] += 1
    return dict(counts), reload_audit


def run_self_retrieval(
    *,
    dense_index: DenseNumpyIndex,
    sparse_index: LocalBM25Index | None,
    documents: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    provider = OpenAIEmbeddingProvider(parse_embedding_config(config))
    samples = sample_documents(documents)
    counts = Counter()
    rows: list[dict[str, Any]] = []
    for doc in samples:
        query = query_fragment(str(doc.get("text", "")))
        if not query:
            continue
        query_vector = provider.embed_query(query)
        dense_hits = dense_index.search_vector(query_vector, top_k=10)
        dense_ids = [hit["chunk_id"] for hit in dense_hits]
        sparse_hits = sparse_index.search(query, top_k=10) if sparse_index is not None else []
        sparse_ids = [hit["chunk_id"] for hit in sparse_hits]
        chunk_id = str(doc.get("chunk_id"))
        dense_ok = chunk_id in dense_ids
        sparse_ok = chunk_id in sparse_ids if sparse_index is not None else True
        if not dense_ok:
            counts["self_retrieval_catastrophic_mismatch"] += 1
        if sparse_index is not None and not sparse_ok:
            counts["bm25_self_mismatch"] += 1
        rows.append(
            {
                "query_chunk_id": chunk_id,
                "doc_id": doc.get("doc_id"),
                "section": doc.get("section"),
                "chunk_type": doc.get("chunk_type"),
                "query_preview": query[:180],
                "dense_ok": dense_ok,
                "dense_top_k": [
                    {
                        "chunk_id": hit["chunk_id"],
                        "score": hit["score"],
                        "doc_id": hit["metadata"].get("doc_id"),
                        "section": hit["metadata"].get("section"),
                    }
                    for hit in dense_hits
                ],
                "bm25_ok": sparse_ok,
                "bm25_top_k": [
                    {
                        "chunk_id": hit["chunk_id"],
                        "score": hit["score"],
                        "doc_id": hit["metadata"].get("doc_id"),
                        "section": hit["metadata"].get("section"),
                    }
                    for hit in sparse_hits
                ],
            }
        )
    return dict(counts), rows


def run_synthetic_incremental_tests() -> tuple[dict[str, int], dict[str, Any]]:
    base = [
        {"chunk_id": "a", "content_hash": "c1", "metadata_hash": "m1"},
        {"chunk_id": "b", "content_hash": "c2", "metadata_hash": "m2"},
    ]
    same = [dict(row) for row in base]
    changed = [
        {"chunk_id": "a", "content_hash": "c1", "metadata_hash": "m1"},
        {"chunk_id": "b", "content_hash": "c3", "metadata_hash": "m2"},
        {"chunk_id": "c", "content_hash": "c4", "metadata_hash": "m4"},
    ]
    removed = [{"chunk_id": "a", "content_hash": "c1", "metadata_hash": "m1"}]
    same_plan = plan_incremental(same, base)
    changed_plan = plan_incremental(changed, base)
    removed_plan = plan_incremental(removed, base)
    failures = []
    if any(same_plan[key] for key in ("added", "updated", "removed")):
        failures.append("unchanged plan reported changes")
    if changed_plan["added"] != 1 or changed_plan["updated"] != 1 or changed_plan["removed"] != 0:
        failures.append("changed plan did not isolate added/updated chunks")
    if removed_plan["removed"] != 1 or removed_plan["updated"] != 0:
        failures.append("removed plan did not isolate removed chunk")
    return (
        {"deterministic_incremental_synthetic_failure": len(failures)},
        {
            "failures": failures,
            "same_plan": same_plan,
            "changed_plan": changed_plan,
            "removed_plan": removed_plan,
            "same_hash": stable_json_hash(same_plan),
            "same_hash_rerun": stable_json_hash(plan_incremental(same, base)),
        },
    )


def conclusion_from_counts(counts: dict[str, int]) -> str:
    hard_keys = {
        "chunk_count_mismatch",
        "duplicate_chunk_id",
        "required_metadata_missing",
        "embedding_count_mismatch",
        "invalid_embedding",
        "dimension_mismatch",
        "dense_index_count_mismatch",
        "missing_id_mapping",
        "reload_failure",
        "stale_entries",
        "corpus_fingerprint_mismatch",
        "self_retrieval_catastrophic_mismatch",
        "deterministic_incremental_synthetic_failure",
    }
    if any(counts.get(key, 0) for key in hard_keys):
        return "FAIL"
    if any(value for key, value in counts.items() if value and key not in hard_keys):
        return "PASS_WITH_MINOR_ISSUES"
    return "PASS"


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Index Validation v1",
        "",
        f"Conclusion: **{summary['conclusion']}**",
        "",
        "## Counts",
        "",
    ]
    for key, value in sorted(summary["counts"].items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Index",
            "",
            f"- paper_count: `{summary['paper_count']}`",
            f"- chunk_count: `{summary['chunk_count']}`",
            f"- embedding_model: `{summary['embedding_model']}`",
            f"- embedding_dimensions: `{summary['embedding_dimensions']}`",
            f"- dense_backend: `{summary['dense_backend']}`",
            f"- sparse_backend: `{summary['sparse_backend']}`",
            f"- self_retrieval_samples: `{summary['self_retrieval_samples']}`",
        ]
    )
    (OUTPUT_DIR / "index_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)
    index_dir = Path(args.output_dir) if args.output_dir else resolve_output_dir(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    manifest_docs = read_jsonl(index_dir / "corpus_manifest.jsonl")
    index_manifest = load_json(index_dir / "index_manifest.json")
    dense_index = DenseNumpyIndex.load(resolve_dense_dir(config, index_dir))
    sparse_index = LocalBM25Index.load(resolve_sparse_dir(config, index_dir)) if index_manifest.get("sparse_enabled") else None

    corpus_counts, corpus_audit, metadata_audit = validate_corpus(
        config=config,
        manifest_docs=manifest_docs,
        index_dir=index_dir,
    )
    counts.update(corpus_counts)

    embedding_counts, embedding_audit = validate_embeddings(
        dense_index=dense_index,
        manifest_docs=manifest_docs,
        index_manifest=index_manifest,
        index_dir=index_dir,
        config=config,
    )
    counts.update(embedding_counts)

    index_counts, reload_audit = validate_dense_sparse(
        dense_index=dense_index,
        sparse_index=sparse_index,
        manifest_docs=manifest_docs,
        index_manifest=index_manifest,
    )
    counts.update(index_counts)

    self_counts, self_rows = run_self_retrieval(
        dense_index=dense_index,
        sparse_index=sparse_index,
        documents=manifest_docs,
        config=config,
    )
    counts.update(self_counts)

    synthetic_counts, synthetic_audit = run_synthetic_incremental_tests()
    counts.update(synthetic_counts)

    counts.setdefault("reload_failure", 0)
    conclusion = conclusion_from_counts(dict(counts))
    summary = {
        "conclusion": conclusion,
        "counts": dict(sorted(counts.items())),
        "paper_count": index_manifest.get("paper_count"),
        "chunk_count": index_manifest.get("chunk_count"),
        "embedding_model": index_manifest.get("embedding_model"),
        "embedding_dimensions": index_manifest.get("embedding_dimensions"),
        "dense_backend": index_manifest.get("dense_backend"),
        "sparse_backend": index_manifest.get("sparse_backend"),
        "cache_hits": index_manifest.get("cache_hits"),
        "cache_misses": index_manifest.get("cache_misses"),
        "self_retrieval_samples": len(self_rows),
        "synthetic_incremental": synthetic_audit,
        "output_dir": str(OUTPUT_DIR),
    }

    write_json(OUTPUT_DIR / "index_validation_summary.json", summary)
    write_jsonl(OUTPUT_DIR / "corpus_audit.jsonl", corpus_audit)
    write_jsonl(OUTPUT_DIR / "embedding_audit.jsonl", embedding_audit)
    write_json(OUTPUT_DIR / "index_reload_audit.json", reload_audit)
    write_jsonl(OUTPUT_DIR / "self_retrieval_audit.jsonl", self_rows)
    write_jsonl(OUTPUT_DIR / "metadata_audit.jsonl", metadata_audit)
    write_report(summary)
    print(json.dumps({"conclusion": conclusion, "output_dir": str(OUTPUT_DIR)}, ensure_ascii=False, indent=2))
    return 0 if conclusion != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
