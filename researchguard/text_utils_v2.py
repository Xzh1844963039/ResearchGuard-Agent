# C:\Users\18449\Desktop\researchguard_workspace\researchguard\text_utils_v2.py
from __future__ import annotations

import re


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compact_text(text: str, limit: int = 160) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def tokenize(text: str) -> set[str]:
    lowered = (text or "").lower()
    words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+|\d+(?:\.\d+)?%?", lowered))
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
    for term in chinese_terms:
        words.add(term)
        for i in range(max(0, len(term) - 1)):
            words.add(term[i : i + 2])
    return words


def detect_section(text: str) -> str | None:
    head = (text or "").strip().lower()[:80]
    section_map = {
        "abstract": ["abstract", "摘要"],
        "introduction": ["introduction", "引言", "背景"],
        "related work": ["related work", "相关工作"],
        "method": ["method", "methods", "方法"],
        "experiments": ["experiment", "experiments", "实验", "results", "结果"],
        "limitations": ["limitation", "limitations", "局限"],
        "references": ["references", "参考文献"],
    }
    for section, keys in section_map.items():
        if any(key in head for key in keys):
            return section
    return None


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?", text or "")


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[。！？?!])\s+|\n+|；|;", text or "")
    return [piece.strip(" -\t\r\n") for piece in pieces if len(piece.strip()) >= 8]


def ordered_tokenize(text: str) -> list[str]:
    """Like tokenize() but returns tokens in order of first appearance.

    Uses the same tokenization regex and Chinese-bigram logic as tokenize(),
    but returns a list deduplicated by first occurrence rather than an
    unordered set.  This guarantees deterministic iteration order, which is
    critical for keyword extraction and query generation.
    """
    lowered = (text or "").lower()
    seen: set[str] = set()
    ordered: list[str] = []
    for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+|\d+(?:\.\d+)?%?", lowered):
        if word not in seen:
            seen.add(word)
            ordered.append(word)
    chinese_terms = re.findall(r"[一-鿿]{2,}", lowered)
    for term in chinese_terms:
        if term not in seen:
            seen.add(term)
            ordered.append(term)
        for i in range(max(0, len(term) - 1)):
            bigram = term[i : i + 2]
            if bigram not in seen:
                seen.add(bigram)
                ordered.append(bigram)
    return ordered
