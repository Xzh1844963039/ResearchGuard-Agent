# C:\Users\18449\Desktop\researchguard_workspace\researchguard\indexing\sparse_index.py
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]*|\d+(?:\.\d+)*|[A-Za-z0-9]+")


class SparseIndexError(RuntimeError):
    pass


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


class LocalBM25Index:
    def __init__(
        self,
        *,
        chunk_ids: list[str],
        metadata: list[dict[str, Any]],
        doc_term_freqs: list[dict[str, int]],
        doc_lengths: list[int],
        df: dict[str, int],
        avgdl: float,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not (len(chunk_ids) == len(metadata) == len(doc_term_freqs) == len(doc_lengths)):
            raise SparseIndexError("Sparse index inputs have inconsistent lengths.")
        self.chunk_ids = chunk_ids
        self.metadata = metadata
        self.doc_term_freqs = doc_term_freqs
        self.doc_lengths = doc_lengths
        self.df = df
        self.avgdl = avgdl
        self.k1 = k1
        self.b = b

    @classmethod
    def build(cls, documents: list[dict[str, Any]]) -> "LocalBM25Index":
        chunk_ids = [str(doc["chunk_id"]) for doc in documents]
        metadata = [
            {
                "chunk_id": doc.get("chunk_id"),
                "doc_id": doc.get("doc_id"),
                "title": doc.get("title"),
                "section": doc.get("section"),
                "chunk_type": doc.get("chunk_type"),
                "page_start": doc.get("page_start"),
                "page_end": doc.get("page_end"),
            }
            for doc in documents
        ]
        doc_term_freqs: list[dict[str, int]] = []
        doc_lengths: list[int] = []
        df_counter: Counter[str] = Counter()
        for doc in documents:
            tokens = tokenize(str(doc.get("text", "")))
            counts = Counter(tokens)
            doc_term_freqs.append(dict(counts))
            doc_lengths.append(len(tokens))
            df_counter.update(counts.keys())
        avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
        return cls(
            chunk_ids=chunk_ids,
            metadata=metadata,
            doc_term_freqs=doc_term_freqs,
            doc_lengths=doc_lengths,
            df=dict(df_counter),
            avgdl=avgdl,
        )

    def save(self, output_dir: Path, *, manifest: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "local_bm25_v1",
            "backend": "local_bm25",
            "chunk_ids": self.chunk_ids,
            "metadata": self.metadata,
            "doc_term_freqs": self.doc_term_freqs,
            "doc_lengths": self.doc_lengths,
            "df": self.df,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
            "manifest": manifest,
        }
        (output_dir / "bm25_index.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: Path) -> "LocalBM25Index":
        path = output_dir / "bm25_index.json"
        if not path.exists():
            raise SparseIndexError(f"Sparse index file missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            chunk_ids=[str(item) for item in payload.get("chunk_ids", [])],
            metadata=list(payload.get("metadata", [])),
            doc_term_freqs=[{str(k): int(v) for k, v in row.items()} for row in payload.get("doc_term_freqs", [])],
            doc_lengths=[int(value) for value in payload.get("doc_lengths", [])],
            df={str(k): int(v) for k, v in payload.get("df", {}).items()},
            avgdl=float(payload.get("avgdl", 0.0)),
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
        )

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        query_terms = Counter(tokens)
        n_docs = len(self.chunk_ids)
        scores = [0.0 for _ in self.chunk_ids]
        for term, query_count in query_terms.items():
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for index, tf_map in enumerate(self.doc_term_freqs):
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                length = self.doc_lengths[index]
                denom = tf + self.k1 * (1 - self.b + self.b * length / max(self.avgdl, 1.0))
                scores[index] += query_count * idf * (tf * (self.k1 + 1) / denom)
        order = sorted(range(len(scores)), key=lambda idx: (-scores[idx], self.chunk_ids[idx]))[:top_k]
        return [
            {
                "rank": rank,
                "chunk_id": self.chunk_ids[index],
                "score": float(scores[index]),
                "metadata": self.metadata[index],
            }
            for rank, index in enumerate(order, start=1)
            if scores[index] > 0
        ]
