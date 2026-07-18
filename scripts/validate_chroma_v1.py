# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_chroma_v1.py
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(r"C:\Users\18449\Desktop\researchguard_workspace")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchguard.indexing.chroma_index import (  # noqa: E402
    ChromaIndexManager,
    ChromaSettings,
    ChromaSourceIndex,
    load_chroma_settings,
    load_source_index,
)
from researchguard.indexing.chroma_metadata import (  # noqa: E402
    build_collection_metadata,
    decode_record_metadata,
    validate_collection_metadata,
)
from researchguard.indexing.corpus_loader import (  # noqa: E402
    corpus_fingerprint,
    read_jsonl,
    stable_json_hash,
    write_json,
    write_jsonl,
)
from researchguard.indexing.embedding_provider import validate_vector  # noqa: E402
from researchguard.retrieval.chroma_retriever import ChromaDenseRetrieverBackend  # noqa: E402
from researchguard.retrieval.dense_backend import NumpyDenseRetrieverBackend  # noqa: E402
from researchguard.retrieval.filters import metadata_matches  # noqa: E402
from researchguard.retrieval.models import MetadataFilter  # noqa: E402
from researchguard.retrieval.retrieval_v1 import RetrievalEngine  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "chroma_v1.yaml"
RETRIEVAL_CONFIG = PROJECT_ROOT / "configs" / "retrieval_v1.yaml"
LIST_FIELDS = (
    "heading_path",
    "source_block_ids",
    "heading_block_ids",
    "overlap_source_block_ids",
    "content_types",
)
HARD_CHECKS = (
    "source_count_mismatch",
    "collection_count_mismatch",
    "missing_record_id",
    "stale_record_id",
    "duplicate_record_id",
    "embedding_dimension_mismatch",
    "invalid_embedding",
    "document_mismatch",
    "content_hash_mismatch",
    "metadata_hash_mismatch",
    "corpus_fingerprint_mismatch",
    "persistence_reload_failure",
    "query_failure",
    "filter_correctness_failure",
    "incremental_sync_failure",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly validate Chroma backend v1.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def rank_correlation(left: list[str], right: list[str], k: int = 10) -> float:
    universe = sorted(set(left[:k]) | set(right[:k]))
    if len(universe) < 2:
        return 1.0
    missing_rank = k + 1
    left_rank = {chunk_id: rank for rank, chunk_id in enumerate(left[:k], start=1)}
    right_rank = {chunk_id: rank for rank, chunk_id in enumerate(right[:k], start=1)}
    x = np.asarray([left_rank.get(item, missing_rank) for item in universe], dtype="float64")
    y = np.asarray([right_rank.get(item, missing_rank) for item in universe], dtype="float64")
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float((x @ y) / denom) if denom else 1.0


def overlap_at_k(left: list[str], right: list[str], k: int) -> float:
    denominator = min(k, len(left), len(right))
    if denominator == 0:
        return 1.0 if not left and not right else 0.0
    return len(set(left[:k]) & set(right[:k])) / denominator


def load_collection_records(collection: Any) -> dict[str, Any]:
    return collection.get(include=["documents", "metadatas", "embeddings"])


def validate_records(
    source: ChromaSourceIndex,
    collection: Any,
    *,
    embedding_sample_size: int,
) -> tuple[Counter[str], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    counts: Counter[str] = Counter()
    payload = load_collection_records(collection)
    ids = [str(item) for item in payload.get("ids", [])]
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    embeddings = payload.get("embeddings")
    embedding_rows = list(embeddings) if embeddings is not None else []
    counts["collection_count_mismatch"] += int(collection.count() != len(source.chunk_ids))
    counts["duplicate_record_id"] += len(ids) - len(set(ids))
    source_ids = set(source.chunk_ids)
    chroma_ids = set(ids)
    counts["missing_record_id"] += len(source_ids - chroma_ids)
    counts["stale_record_id"] += len(chroma_ids - source_ids)
    source_by_id = {str(doc["chunk_id"]): doc for doc in source.documents}
    vector_by_id = {chunk_id: source.vectors[index] for index, chunk_id in enumerate(source.chunk_ids)}
    record_by_id = {
        chunk_id: (document, metadata, embedding_rows[index] if index < len(embedding_rows) else None)
        for index, (chunk_id, document, metadata) in enumerate(zip(ids, documents, metadatas))
    }
    record_audit: list[dict[str, Any]] = []
    for chunk_id in source.chunk_ids:
        source_doc = source_by_id[chunk_id]
        stored = record_by_id.get(chunk_id)
        if stored is None:
            continue
        stored_document, stored_metadata, _ = stored
        decoded = decode_record_metadata(stored_metadata)
        document_ok = str(stored_document or "") == str(source_doc.get("text", ""))
        content_hash_ok = decoded.get("content_hash") == source_doc.get("content_hash")
        metadata_hash_ok = decoded.get("metadata_hash") == source_doc.get("metadata_hash")
        metadata_fields_ok = all(decoded.get(field) == source_doc.get(field) for field in LIST_FIELDS)
        scalar_fields_ok = all(
            decoded.get(field) == source_doc.get(field)
            for field in ("doc_id", "section", "page_start", "page_end", "chunk_type")
        )
        fingerprint_ok = decoded.get("corpus_fingerprint") == source.corpus_fingerprint
        counts["document_mismatch"] += int(not document_ok)
        counts["content_hash_mismatch"] += int(not content_hash_ok)
        counts["metadata_hash_mismatch"] += int(not metadata_hash_ok or not metadata_fields_ok or not scalar_fields_ok)
        counts["corpus_fingerprint_mismatch"] += int(not fingerprint_ok)
        record_audit.append(
            {
                "chunk_id": chunk_id,
                "document_ok": document_ok,
                "content_hash_ok": content_hash_ok,
                "metadata_hash_ok": metadata_hash_ok,
                "list_metadata_ok": metadata_fields_ok,
                "scalar_metadata_ok": scalar_fields_ok,
                "corpus_fingerprint_ok": fingerprint_ok,
            }
        )

    rng = random.Random("chroma-v1-embedding-parity")
    sampled_ids = rng.sample(source.chunk_ids, min(embedding_sample_size, len(source.chunk_ids)))
    embedding_audit: list[dict[str, Any]] = []
    absolute_errors: list[float] = []
    for chunk_id in sampled_ids:
        stored_embedding = record_by_id.get(chunk_id, (None, None, None))[2]
        if stored_embedding is None:
            counts["invalid_embedding"] += 1
            continue
        actual = np.asarray(stored_embedding, dtype="float32")
        expected = np.asarray(vector_by_id[chunk_id], dtype="float32")
        dimension_ok = actual.shape == expected.shape
        finite = bool(np.isfinite(actual).all())
        norm = float(np.linalg.norm(actual)) if finite else math.nan
        valid = dimension_ok and finite and norm > 0
        counts["embedding_dimension_mismatch"] += int(not dimension_ok)
        counts["invalid_embedding"] += int(not valid)
        errors = np.abs(actual - expected) if dimension_ok else np.asarray([math.inf])
        absolute_errors.extend(float(item) for item in errors)
        embedding_audit.append(
            {
                "chunk_id": chunk_id,
                "dimension": int(actual.shape[0]) if actual.ndim == 1 else None,
                "dimension_ok": dimension_ok,
                "finite": finite,
                "norm": norm,
                "max_absolute_error": float(np.max(errors)),
                "mean_absolute_error": float(np.mean(errors)),
            }
        )
    embedding_summary = {
        "sample_size": len(embedding_audit),
        "max_absolute_error": max(absolute_errors, default=0.0),
        "mean_absolute_error": float(np.mean(absolute_errors)) if absolute_errors else 0.0,
    }
    return counts, record_audit, embedding_audit, embedding_summary


def validate_persistence(
    settings: ChromaSettings,
    source: ChromaSourceIndex,
) -> tuple[Counter[str], dict[str, Any]]:
    counts: Counter[str] = Counter()
    audit: dict[str, Any] = {"reload_success": False, "query_success": False}
    try:
        _, collection = ChromaIndexManager(settings).get_collection(strict_fingerprint=True, source=source)
        audit["reload_success"] = collection.count() == len(source.chunk_ids)
        counts["persistence_reload_failure"] += int(not audit["reload_success"])
        result = collection.query(
            query_embeddings=[source.vectors[0].tolist()],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        audit["query_success"] = bool((result.get("ids") or [[]])[0])
        audit["query_ids"] = (result.get("ids") or [[]])[0]
        counts["query_failure"] += int(not audit["query_success"])
    except Exception as exc:
        counts["persistence_reload_failure"] += 1
        counts["query_failure"] += 1
        audit["error"] = f"{type(exc).__name__}: {exc}"
    return counts, audit


def validate_filters(
    source: ChromaSourceIndex,
    backend: ChromaDenseRetrieverBackend,
    query_vector: np.ndarray,
) -> tuple[Counter[str], dict[str, Any]]:
    counts: Counter[str] = Counter()
    cases = {
        "single_doc": MetadataFilter(doc_ids=("paper_rag",)),
        "single_section": MetadataFilter(sections=("method",)),
        "multi_doc": MetadataFilter(doc_ids=("paper_rag", "paper_agent")),
        "equation": MetadataFilter(has_equation=True),
        "table": MetadataFilter(has_table=True),
        "caption": MetadataFilter(has_caption=True),
        "exclude_references": MetadataFilter(exclude_references=True),
        "page_range": MetadataFilter(page_start_min=2, page_end_max=5),
        "combined": MetadataFilter(doc_ids=("paper_agent",), sections=("method",), exclude_references=True),
        "no_match": MetadataFilter(doc_ids=("__missing_document__",)),
    }
    audit: dict[str, Any] = {}
    for name, filters in cases.items():
        expected = {
            str(document["chunk_id"])
            for document in source.documents
            if metadata_matches(document, filters)
        }
        try:
            candidates, trace = backend.search(query_vector, candidate_k=len(source.documents), filters=filters)
            actual = {str(item["chunk_id"]) for item in candidates}
            passed = actual == expected
            counts["filter_correctness_failure"] += int(not passed)
            audit[name] = {
                "passed": passed,
                "expected_count": len(expected),
                "actual_count": len(actual),
                "missing_ids": sorted(expected - actual),
                "unexpected_ids": sorted(actual - expected),
                "trace": trace,
            }
        except Exception as exc:
            counts["filter_correctness_failure"] += 1
            audit[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    return counts, audit


def validate_search_parity(
    config: dict[str, Any],
    source: ChromaSourceIndex,
    *,
    chroma_config_path: str | Path,
) -> tuple[Counter[str], list[dict[str, Any]], dict[str, Any], ChromaDenseRetrieverBackend, np.ndarray]:
    counts: Counter[str] = Counter()
    validation_cfg = config.get("validation", {}) or {}
    benchmark_path = Path(validation_cfg.get("benchmark_path", "data/eval/retrieval_v1_queries.jsonl"))
    if not benchmark_path.is_absolute():
        benchmark_path = PROJECT_ROOT / benchmark_path
    query_limit = max(30, int(validation_cfg.get("comparison_queries", 30)))
    benchmark = read_jsonl(benchmark_path)[:query_limit]
    engine = RetrievalEngine.from_config(RETRIEVAL_CONFIG, dense_backend_override="numpy")
    numpy_backend = NumpyDenseRetrieverBackend(engine.bundle)
    chroma_backend = ChromaDenseRetrieverBackend(engine.bundle, chroma_config_path=chroma_config_path)
    started = time.perf_counter()
    vectors = engine.embedding_provider.embed_documents([str(item["query"]) for item in benchmark])
    embedding_latency_ms = (time.perf_counter() - started) * 1000.0
    query_vectors = [np.asarray(vector, dtype="float32") for vector in vectors]
    for vector in query_vectors:
        validate_vector(vector.tolist(), dimensions=source.vectors.shape[1])

    rows: list[dict[str, Any]] = []
    numpy_latencies: list[float] = []
    chroma_latencies: list[float] = []
    score_errors: list[float] = []
    for case, query_vector in zip(benchmark, query_vectors):
        filters = MetadataFilter.from_mapping(case.get("filters"))
        started = time.perf_counter()
        numpy_hits, _ = numpy_backend.search(query_vector, candidate_k=10, filters=filters)
        numpy_latency = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        chroma_hits, chroma_trace = chroma_backend.search(query_vector, candidate_k=10, filters=filters)
        chroma_latency = (time.perf_counter() - started) * 1000.0
        numpy_latencies.append(numpy_latency)
        chroma_latencies.append(chroma_latency)
        numpy_ids = [str(item["chunk_id"]) for item in numpy_hits]
        chroma_ids = [str(item["chunk_id"]) for item in chroma_hits]
        numpy_scores = {str(item["chunk_id"]): float(item["dense_score"]) for item in numpy_hits}
        chroma_scores = {str(item["chunk_id"]): float(item["dense_score"]) for item in chroma_hits}
        common_ids = set(numpy_scores) & set(chroma_scores)
        query_score_errors = [abs(numpy_scores[item] - chroma_scores[item]) for item in common_ids]
        score_errors.extend(query_score_errors)
        rows.append(
            {
                "query_id": case.get("query_id"),
                "query": case.get("query"),
                "filters": filters.to_dict(),
                "numpy_ids": numpy_ids,
                "chroma_ids": chroma_ids,
                "top1_agreement": bool(numpy_ids and chroma_ids and numpy_ids[0] == chroma_ids[0]),
                "top3_overlap": overlap_at_k(numpy_ids, chroma_ids, 3),
                "top5_overlap": overlap_at_k(numpy_ids, chroma_ids, 5),
                "top10_overlap": overlap_at_k(numpy_ids, chroma_ids, 10),
                "rank_correlation": rank_correlation(numpy_ids, chroma_ids, 10),
                "max_score_error": max(query_score_errors, default=0.0),
                "numpy_latency_ms": numpy_latency,
                "chroma_latency_ms": chroma_latency,
                "chroma_trace": chroma_trace,
            }
        )
    summary = {
        "query_count": len(rows),
        "top1_agreement": sum(int(row["top1_agreement"]) for row in rows) / max(len(rows), 1),
        "top3_overlap": float(np.mean([row["top3_overlap"] for row in rows])) if rows else 0.0,
        "top5_overlap": float(np.mean([row["top5_overlap"] for row in rows])) if rows else 0.0,
        "top10_overlap": float(np.mean([row["top10_overlap"] for row in rows])) if rows else 0.0,
        "rank_correlation": float(np.mean([row["rank_correlation"] for row in rows])) if rows else 0.0,
        "max_score_error": max(score_errors, default=0.0),
        "mean_score_error": float(np.mean(score_errors)) if score_errors else 0.0,
        "query_embedding_latency_ms": embedding_latency_ms,
        "numpy_average_latency_ms": float(np.mean(numpy_latencies)) if numpy_latencies else 0.0,
        "chroma_average_latency_ms": float(np.mean(chroma_latencies)) if chroma_latencies else 0.0,
        "numpy_p95_latency_ms": float(np.percentile(numpy_latencies, 95)) if numpy_latencies else 0.0,
        "chroma_p95_latency_ms": float(np.percentile(chroma_latencies, 95)) if chroma_latencies else 0.0,
    }
    return counts, rows, summary, chroma_backend, query_vectors[0]


def synthetic_document(chunk_id: str, text: str, *, section: str = "method") -> dict[str, Any]:
    document = {
        "chunk_id": chunk_id,
        "doc_id": "synthetic_doc",
        "title": "Synthetic",
        "section": section,
        "section_heading": section.title(),
        "heading_path": [section.title()],
        "chunk_type": "text",
        "page_start": 1,
        "page_end": 1,
        "source_block_ids": [f"{chunk_id}_block"],
        "heading_block_ids": [],
        "overlap_source_block_ids": [],
        "content_types": ["paragraph"],
        "has_equation": False,
        "has_table": False,
        "has_caption": False,
        "short_chunk": False,
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split()),
        "schema_version": "corpus_manifest_v1",
    }
    document["content_hash"] = stable_json_hash({"text": text})
    metadata_fields = {key: value for key, value in document.items() if key not in {"text", "content_hash"}}
    document["metadata_hash"] = stable_json_hash(metadata_fields)
    return document


def synthetic_source(documents: list[dict[str, Any]], vectors: list[list[float]]) -> ChromaSourceIndex:
    fingerprint = corpus_fingerprint(documents)
    dimension = len(vectors[0])
    manifest = {
        "build_status": "complete",
        "corpus_fingerprint": fingerprint,
        "embedding_provider": "synthetic",
        "embedding_model": "synthetic-v1",
        "embedding_dimensions": dimension,
        "dense_metric": "cosine",
        "dense_backend": "numpy",
        "build_timestamp": "synthetic",
        "chunk_count": len(documents),
    }
    return ChromaSourceIndex(
        documents=documents,
        chunk_ids=[str(item["chunk_id"]) for item in documents],
        vectors=np.asarray(vectors, dtype="float32"),
        index_manifest=manifest,
        dense_manifest={"dimension": dimension, "corpus_fingerprint": fingerprint},
        hard_checks={},
    )


def run_incremental_synthetic_tests(output_dir: Path) -> tuple[Counter[str], dict[str, Any]]:
    counts: Counter[str] = Counter()
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    persist_directory = output_dir / "synthetic" / suffix
    settings = ChromaSettings(
        persist_directory=persist_directory,
        collection_name="researchguard_chroma_incremental_synthetic",
        distance_metric="cosine",
        batch_size=2,
        allow_reset=False,
        source_index_directory=persist_directory,
        corpus_manifest_path=persist_directory / "unused.jsonl",
        vectors_path=persist_directory / "unused.npy",
        ids_path=persist_directory / "unused.json",
        dense_manifest_path=persist_directory / "unused-dense.json",
        index_manifest_path=persist_directory / "unused-index.json",
        delete_stale=True,
        large_delete_ratio=0.10,
        validation_output_directory=output_dir,
    )
    manager = ChromaIndexManager(settings)
    try:
        first_source = synthetic_source(
            [synthetic_document("a", "alpha"), synthetic_document("b", "beta")],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        first = manager.sync(first_source)
        unchanged = manager.sync(first_source)
        changed_source = synthetic_source(
            [
                synthetic_document("a", "alpha", section="results"),
                synthetic_document("b", "beta changed"),
                synthetic_document("c", "gamma"),
            ],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.7, 0.7, 0.0]],
        )
        changed = manager.sync(changed_source)
        removed_source = synthetic_source(
            [
                synthetic_document("a", "alpha", section="results"),
                synthetic_document("b", "beta changed"),
            ],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        )
        removed = manager.sync(removed_source, allow_large_delete=True)
        final_unchanged = manager.sync(removed_source)
        _, collection = manager.get_collection(strict_fingerprint=True, source=removed_source)
        query_one = collection.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=2, include=["distances"])
        query_two = collection.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=2, include=["distances"])
        tests = {
            "add": first["added"] == 2 and first["collection_count"] == 2,
            "unchanged_rerun": unchanged["added"] == unchanged["updated"] == unchanged["deleted"] == 0
            and unchanged["reused"] == 2,
            "update_metadata": changed["metadata_updated"] >= 1,
            "update_embedding": changed["updated"] == 1,
            "add_incremental": changed["added"] == 1,
            "delete_stale": removed["deleted"] == 1 and removed["collection_count"] == 2,
            "final_unchanged": final_unchanged["added"] == final_unchanged["updated"] == final_unchanged["deleted"] == 0,
            "deterministic_results": query_one["ids"] == query_two["ids"]
            and query_one["distances"] == query_two["distances"],
        }
        counts["incremental_sync_failure"] += sum(int(not passed) for passed in tests.values())
        return counts, {
            "passed": all(tests.values()),
            "tests": tests,
            "first": first,
            "unchanged": unchanged,
            "changed": changed,
            "removed": removed,
            "final_unchanged": final_unchanged,
            "persist_directory": str(persist_directory),
        }
    except Exception as exc:
        counts["incremental_sync_failure"] += 1
        return counts, {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def write_report(summary: dict[str, Any], output_dir: Path) -> None:
    parity = summary.get("search_parity", {})
    lines = [
        "# Chroma Vector Database Backend v1 Validation",
        "",
        f"Conclusion: **{summary['conclusion']}**",
        "",
        "## Integrity",
        "",
        f"- source_count: `{summary['source_count']}`",
        f"- collection_count: `{summary['collection_count']}`",
        f"- corpus_fingerprint: `{summary['corpus_fingerprint']}`",
        f"- persistence_reload: `{summary['persistence']['reload_success']}`",
        "",
        "## Hard Checks",
        "",
    ]
    lines.extend(f"- {key}: `{summary['hard_checks'].get(key, 0)}`" for key in HARD_CHECKS)
    lines.extend(
        [
            "",
            "## NumPy / Chroma Search Parity",
            "",
            f"- queries: `{parity.get('query_count')}`",
            f"- Top-1 agreement: `{parity.get('top1_agreement', 0):.6f}`",
            f"- Top-3 overlap: `{parity.get('top3_overlap', 0):.6f}`",
            f"- Top-5 overlap: `{parity.get('top5_overlap', 0):.6f}`",
            f"- Top-10 overlap: `{parity.get('top10_overlap', 0):.6f}`",
            f"- rank correlation: `{parity.get('rank_correlation', 0):.6f}`",
            f"- NumPy average latency ms: `{parity.get('numpy_average_latency_ms', 0):.4f}`",
            f"- Chroma average latency ms: `{parity.get('chroma_average_latency_ms', 0):.4f}`",
            "",
            "## Incremental Synthetic Tests",
            "",
            f"- passed: `{summary['incremental_tests'].get('passed')}`",
        ]
    )
    (output_dir / "chroma_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config, settings = load_chroma_settings(args.config)
    output_dir = settings.validation_output_directory
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter({key: 0 for key in HARD_CHECKS})
    failure_cases: list[dict[str, Any]] = []
    try:
        source = load_source_index(settings, strict=True)
        counts.update(source.hard_checks)
        manager = ChromaIndexManager(settings)
        _, collection = manager.get_collection(strict_fingerprint=True, source=source)
        expected_collection_metadata = build_collection_metadata(source.index_manifest)
        collection_metadata_mismatches = validate_collection_metadata(
            collection.metadata,
            expected_collection_metadata,
            check_fingerprint=True,
        )
        counts["corpus_fingerprint_mismatch"] += int(bool(collection_metadata_mismatches))

        record_counts, record_audit, embedding_audit, embedding_summary = validate_records(
            source,
            collection,
            embedding_sample_size=int((config.get("validation", {}) or {}).get("embedding_sample_size", 40)),
        )
        counts.update(record_counts)
        persistence_counts, persistence_audit = validate_persistence(settings, source)
        counts.update(persistence_counts)
        parity_counts, parity_rows, parity_summary, chroma_backend, query_vector = validate_search_parity(
            config,
            source,
            chroma_config_path=args.config,
        )
        counts.update(parity_counts)
        filter_counts, filter_audit = validate_filters(source, chroma_backend, query_vector)
        counts.update(filter_counts)
        incremental_counts, incremental_audit = run_incremental_synthetic_tests(output_dir)
        counts.update(incremental_counts)

        minimum_overlap = float((config.get("validation", {}) or {}).get("minimum_top10_overlap", 0.95))
        hard_failure = any(counts.get(key, 0) for key in HARD_CHECKS)
        conclusion = "FAIL" if hard_failure else (
            "PASS_WITH_MINOR_ISSUES" if parity_summary["top10_overlap"] < minimum_overlap else "PASS"
        )
        for key in HARD_CHECKS:
            if counts.get(key, 0):
                failure_cases.append({"type": key, "count": counts[key]})
        if parity_summary["top10_overlap"] < minimum_overlap:
            failure_cases.append(
                {
                    "type": "top10_overlap_below_target",
                    "actual": parity_summary["top10_overlap"],
                    "minimum": minimum_overlap,
                }
            )
        summary = {
            "conclusion": conclusion,
            "hard_checks": {key: int(counts.get(key, 0)) for key in HARD_CHECKS},
            "source_count": len(source.documents),
            "source_vector_count": len(source.vectors),
            "collection_count": collection.count(),
            "unique_chroma_ids": len({row["chunk_id"] for row in record_audit}),
            "collection_name": settings.collection_name,
            "persist_directory": str(settings.persist_directory),
            "corpus_fingerprint": source.corpus_fingerprint,
            "collection_metadata_mismatches": collection_metadata_mismatches,
            "embedding_model": source.index_manifest.get("embedding_model"),
            "embedding_dimensions": source.index_manifest.get("embedding_dimensions"),
            "embedding_parity": embedding_summary,
            "persistence": persistence_audit,
            "search_parity": parity_summary,
            "filter_tests_passed": all(item.get("passed", False) for item in filter_audit.values()),
            "incremental_tests": incremental_audit,
            "failure_case_count": len(failure_cases),
        }
        write_json(output_dir / "chroma_validation_summary.json", summary)
        write_jsonl(output_dir / "record_audit.jsonl", record_audit)
        write_jsonl(output_dir / "embedding_parity_audit.jsonl", embedding_audit)
        write_jsonl(output_dir / "search_parity_results.jsonl", parity_rows)
        write_json(output_dir / "filter_validation.json", filter_audit)
        write_json(output_dir / "incremental_test_results.json", incremental_audit)
        write_jsonl(output_dir / "failure_cases.jsonl", failure_cases)
        write_report(summary, output_dir)
        print(json.dumps({"conclusion": conclusion, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
        return 0 if conclusion != "FAIL" else 2
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        write_jsonl(output_dir / "failure_cases.jsonl", [failure])
        print(json.dumps({"conclusion": "FAIL", **failure}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
