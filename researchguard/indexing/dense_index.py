# C:\Users\18449\Desktop\researchguard_workspace\researchguard\indexing\dense_index.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class DenseIndexError(RuntimeError):
    pass


class DenseNumpyIndex:
    def __init__(
        self,
        *,
        chunk_ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]],
        metric: str = "cosine",
    ):
        if len(chunk_ids) != len(vectors) or len(chunk_ids) != len(metadata):
            raise DenseIndexError("Dense index inputs have inconsistent lengths.")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise DenseIndexError("Dense index received duplicate chunk_id values.")
        if metric not in {"cosine", "dot"}:
            raise DenseIndexError(f"Unsupported dense metric: {metric}")
        if vectors.ndim != 2:
            raise DenseIndexError("Dense vectors must be a 2D array.")
        if not np.isfinite(vectors).all():
            raise DenseIndexError("Dense vectors contain NaN or Inf.")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise DenseIndexError("Dense vectors contain all-zero rows.")
        self.chunk_ids = list(chunk_ids)
        self.vectors = vectors.astype("float32")
        self.metadata = list(metadata)
        self.metric = metric
        self.id_to_pos = {chunk_id: index for index, chunk_id in enumerate(self.chunk_ids)}

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def save(self, output_dir: Path, *, manifest: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "vectors.npy", self.vectors)
        (output_dir / "ids.json").write_text(
            json.dumps(self.chunk_ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (output_dir / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in self.metadata:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        payload = {
            **manifest,
            "backend": "numpy",
            "metric": self.metric,
            "vector_count": len(self.chunk_ids),
            "dimension": self.dimension,
            "schema_version": "dense_numpy_v1",
        }
        (output_dir / "dense_manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: Path) -> "DenseNumpyIndex":
        vectors_path = output_dir / "vectors.npy"
        ids_path = output_dir / "ids.json"
        metadata_path = output_dir / "metadata.jsonl"
        manifest_path = output_dir / "dense_manifest.json"
        for path in (vectors_path, ids_path, metadata_path, manifest_path):
            if not path.exists():
                raise DenseIndexError(f"Dense index file missing: {path}")
        vectors = np.load(vectors_path)
        chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        metadata: list[dict[str, Any]] = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    metadata.append(json.loads(line))
        dense_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(
            chunk_ids=[str(item) for item in chunk_ids],
            vectors=vectors,
            metadata=metadata,
            metric=str(dense_manifest.get("metric", "cosine")),
        )

    def search_vector(self, query_vector: list[float] | np.ndarray, *, top_k: int = 5) -> list[dict[str, Any]]:
        query = np.asarray(query_vector, dtype="float32")
        if query.ndim != 1 or query.shape[0] != self.dimension:
            raise DenseIndexError("Query vector dimension does not match dense index.")
        if self.metric == "cosine":
            q_norm = np.linalg.norm(query)
            if q_norm == 0:
                raise DenseIndexError("Query vector has zero norm.")
            scores = self.vectors @ (query / q_norm)
        else:
            scores = self.vectors @ query
        order = np.argsort(-scores)[:top_k]
        return [
            {
                "rank": rank,
                "chunk_id": self.chunk_ids[int(pos)],
                "score": float(scores[int(pos)]),
                "metadata": self.metadata[int(pos)],
            }
            for rank, pos in enumerate(order, start=1)
        ]
