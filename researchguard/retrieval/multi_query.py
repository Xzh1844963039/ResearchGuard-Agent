# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\multi_query.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from researchguard.retrieval.query_rewriter import QueryRewriteResult, normalize_query_text


@dataclass(frozen=True)
class QueryVariant:
    variant_id: str
    variant_types: tuple[str, ...]
    query: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "variant_types": list(self.variant_types),
            "query": self.query,
        }


def build_query_variants(result: QueryRewriteResult, *, multi_query: bool) -> tuple[list[QueryVariant], int]:
    candidates: list[tuple[str, str]] = []
    if multi_query:
        candidates.append(("original", normalize_query_text(result.original_query)))
    candidates.append(("normalized", normalize_query_text(result.normalized_query)))
    if multi_query:
        candidates.extend(
            (f"expansion_{index}", normalize_query_text(query))
            for index, query in enumerate(result.expansion_queries, start=1)
        )

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    duplicate_count = 0
    for variant_type, query in candidates:
        if not query:
            continue
        key = query.casefold()
        if key not in merged:
            merged[key] = {"query": query, "types": [variant_type]}
            order.append(key)
        else:
            duplicate_count += 1
            merged[key]["types"].append(variant_type)
    variants = [
        QueryVariant(
            variant_id="+".join(merged[key]["types"]),
            variant_types=tuple(merged[key]["types"]),
            query=str(merged[key]["query"]),
        )
        for key in order
    ]
    return variants, duplicate_count


def fuse_query_rankings(
    rankings: list[tuple[QueryVariant, list[dict[str, Any]]]],
    *,
    rrf_k: float,
    candidate_k: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for variant_order, (variant, candidates) in enumerate(rankings):
        for rank, candidate in enumerate(candidates, start=1):
            chunk_id = str(candidate["chunk_id"])
            contribution = {
                "variant_id": variant.variant_id,
                "variant_types": list(variant.variant_types),
                "query": variant.query,
                "rank": rank,
                "dense_rank": candidate.get("dense_rank"),
                "sparse_rank": candidate.get("sparse_rank"),
                "fusion_rank": candidate.get("fusion_rank"),
                "fusion_score": candidate.get("fusion_score"),
            }
            if chunk_id not in merged:
                row = dict(candidate)
                row["multi_query_fusion_score"] = 0.0
                row["query_variant_hits"] = []
                row["_best_variant_key"] = (rank, variant_order)
                merged[chunk_id] = row
            row = merged[chunk_id]
            row["multi_query_fusion_score"] = float(row["multi_query_fusion_score"]) + 1.0 / (rrf_k + rank)
            row["query_variant_hits"].append(contribution)
            if (rank, variant_order) < tuple(row["_best_variant_key"]):
                preserved = {
                    "multi_query_fusion_score": row["multi_query_fusion_score"],
                    "query_variant_hits": row["query_variant_hits"],
                    "_best_variant_key": (rank, variant_order),
                }
                row.clear()
                row.update(candidate)
                row.update(preserved)

    candidates = list(merged.values())
    for row in candidates:
        variant_types = {
            variant_type
            for hit in row["query_variant_hits"]
            for variant_type in hit["variant_types"]
        }
        row["original_query_recalled"] = "original" in variant_types
        row["rewrite_query_recalled"] = "normalized" in variant_types
        row["expansion_query_recalled"] = any(item.startswith("expansion_") for item in variant_types)
    candidates.sort(
        key=lambda item: (
            -float(item["multi_query_fusion_score"]),
            tuple(item["_best_variant_key"]),
            str(item["chunk_id"]),
        )
    )
    for rank, row in enumerate(candidates, start=1):
        row["multi_query_fusion_rank"] = rank
        row.pop("_best_variant_key", None)
    return candidates[:candidate_k]
