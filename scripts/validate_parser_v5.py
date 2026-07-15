# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_parser_v5.py
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


PROJECT_ROOT = Path(r"C:\Users\18449\Desktop\researchguard_workspace")
INPUT_ROOT = PROJECT_ROOT / "data" / "parsed" / "parser_eval_v5"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "parser_validation_v5"

PAPERS = [
    "paper_rag",
    "paper_agent",
    "paper_hallucination",
    "paper_corrective_rag",
    "paper_citation",
]

SECTION_ORDER = [
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiment",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "references",
    "appendix",
]

CORE_TEXT_BLOCK_TYPES = {"paragraph", "heading", "heading_candidate", "reference_entry"}
NON_TEXT_CONTEXT_TYPES = {"caption", "table"}
SUSPICIOUS_HEADING_RE = re.compile(
    r"\b(19|20)\d{2}\b|conference|proceedings|arxiv|auc|acc|score|precision|recall|\b\d+(?:\.\d+)?\b",
    flags=re.I,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_preview(text: str, limit: int = 120) -> str:
    return re.sub(r"\s+", " ", text.strip())[:limit]


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    return int(ordered[idx])


def is_reference_like_text(text: str) -> bool:
    lowered = text.lower()
    markers = [
        r"\b(19|20)\d{2}[a-z]?\.",
        r"\barxiv\b",
        r"\bproceedings\b",
        r"\bconference\b",
        r"\bpages?\s+\d+",
        r"\bassociation for computational linguistics\b",
        r"\bjournal\b",
        r"\btransactions\b",
        r"\bneurips\b|\bemnlp\b|\bacl\b|\biclr\b|\bicml\b",
    ]
    hits = sum(1 for pat in markers if re.search(pat, lowered, flags=re.I))
    author_year = bool(re.search(r"[A-Z][A-Za-z\-]+,\s+[A-Z].*?\b(19|20)\d{2}\b", text))
    return hits >= 1 or author_year


def reference_like_ratio(blocks: list[dict[str, Any]]) -> float:
    text_blocks = [b for b in blocks if b.get("block_type") not in {"equation", "table", "caption"}]
    if not text_blocks:
        return 0.0
    return round(sum(1 for b in text_blocks if is_reference_like_text(str(b.get("text", "")))) / len(text_blocks), 4)


def is_heading_only_chunk(chunk: dict[str, Any], block_by_id: dict[str, dict[str, Any]]) -> bool:
    ids = chunk.get("block_ids", [])
    if not ids:
        return False
    linked = [block_by_id[i] for i in ids if i in block_by_id]
    if not linked:
        return False
    return all(b.get("block_type") == "heading" for b in linked) or (
        len(linked) == 1 and linked[0].get("block_type") in {"heading", "heading_candidate"} and chunk.get("char_count", 0) < 150
    )


def choose_reading_order_pages(
    *,
    report: dict[str, Any],
    blocks: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, int]:
    pages = sorted({int(b["page"]) for b in blocks})
    first = pages[0]
    middle = pages[len(pages) // 2]

    ref_pages = [
        int(h["page"])
        for h in report.get("detected_headings", [])
        if h.get("section") == "references"
    ]
    if not ref_pages:
        ref_pages = [
            int(row["page"])
            for row in report.get("page_sections", [])
            if row.get("section") == "references"
        ]
    references_first = min(ref_pages) if ref_pages else pages[-1]

    page_noise_counts: Counter[int] = Counter()
    for b in blocks:
        if b.get("block_type") in {"caption", "table", "equation"}:
            page_noise_counts[int(b["page"])] += 1
    heavy_page = page_noise_counts.most_common(1)[0][0] if page_noise_counts else middle

    return {
        "first_page": first,
        "middle_page": middle,
        "references_first_page": references_first,
        "table_or_figure_heavy_page": heavy_page,
    }


def audit_reading_order(paper: str, report: dict[str, Any], blocks: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = choose_reading_order_pages(report=report, blocks=blocks, chunks=chunks)
    rows: list[dict[str, Any]] = []
    page_findings: dict[str, Any] = {}

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for b in blocks:
        by_page[int(b["page"])].append(b)

    for reason, page_no in selected.items():
        page_blocks = by_page.get(page_no, [])
        column_sequence = [int(b.get("column", 0)) for b in page_blocks]
        column_backtracks = sum(
            1
            for left, right in zip(column_sequence, column_sequence[1:])
            if left > right
        )
        y_backtracks_within_column = 0
        last_y_by_col: dict[int, float] = {}
        for b in page_blocks:
            col = int(b.get("column", 0))
            y0 = float(b.get("y0", 0.0))
            if col in last_y_by_col and y0 < last_y_by_col[col] - 3:
                y_backtracks_within_column += 1
            last_y_by_col[col] = y0

        page_findings[reason] = {
            "page": page_no,
            "block_count": len(page_blocks),
            "column_sequence": column_sequence,
            "column_backtracks": column_backtracks,
            "y_backtracks_within_column": y_backtracks_within_column,
            "needs_manual_review": True,
        }

        for b in page_blocks:
            rows.append(
                {
                    "paper": paper,
                    "sample_reason": reason,
                    "page": int(b.get("page", 0)),
                    "block_id": b.get("block_id"),
                    "column": b.get("column"),
                    "y0": b.get("y0"),
                    "block_type": b.get("block_type"),
                    "section": b.get("section"),
                    "text_preview": text_preview(str(b.get("text", "")), 120),
                }
            )

    return rows, page_findings


def audit_headings(paper: str, blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_page: Counter[int] = Counter()
    suspicious: list[dict[str, Any]] = []
    unmapped = 0

    for b in blocks:
        if b.get("block_type") != "heading":
            continue

        pred = b.get("heading_prediction", {})
        section = b.get("section") or pred.get("section")
        text = str(b.get("text", ""))
        is_suspicious = bool(SUSPICIOUS_HEADING_RE.search(text)) or looks_table_numeric(text)
        row = {
            "paper": paper,
            "page": b.get("page"),
            "block_id": b.get("block_id"),
            "text": text,
            "section": section,
            "score": pred.get("score"),
            "confidence": pred.get("confidence"),
            "reasons": pred.get("reasons", []),
            "suspicious": is_suspicious,
        }
        rows.append(row)
        by_page[int(b.get("page", 0))] += 1
        if not section:
            unmapped += 1
        if is_suspicious:
            suspicious.append(row)

    anomaly_pages = {str(page): count for page, count in by_page.items() if count > 5}
    summary = {
        "heading_count": len(rows),
        "unmapped_heading_count": unmapped,
        "pages_with_more_than_5_headings": anomaly_pages,
        "suspicious_heading_count": len(suspicious),
        "suspicious_headings": suspicious,
    }
    return rows, summary


def looks_table_numeric(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    numeric = sum(1 for line in lines if re.search(r"\d", line))
    short = sum(1 for line in lines if len(line.split()) <= 4)
    return numeric >= 3 and short / len(lines) >= 0.5


def audit_sections(paper: str, blocks: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    text_blocks = [
        b
        for b in blocks
        if b.get("block_type") in CORE_TEXT_BLOCK_TYPES and str(b.get("section", "")).strip()
    ]

    transitions: list[dict[str, Any]] = []
    compressed: list[str] = []
    same_page_transitions: list[dict[str, Any]] = []
    frequent_short_jumps: list[dict[str, Any]] = []
    previous_section: str | None = None
    previous_index = -999
    previous_page: int | None = None
    references_started = False
    references_regressions: list[dict[str, Any]] = []

    for idx, b in enumerate(text_blocks):
        section = str(b.get("section"))
        page = int(b.get("page", 0))

        if section == "references":
            references_started = True
        elif references_started and section in {"main_text", "method"}:
            references_regressions.append(
                {
                    "page": page,
                    "block_id": b.get("block_id"),
                    "section": section,
                    "text_preview": text_preview(str(b.get("text", ""))),
                }
            )

        if section != previous_section:
            transition = {
                "index": idx,
                "page": page,
                "block_id": b.get("block_id"),
                "from": previous_section,
                "to": section,
                "block_type": b.get("block_type"),
                "text_preview": text_preview(str(b.get("text", ""))),
            }
            transitions.append(transition)
            compressed.append(section)

            if previous_page == page and previous_section is not None:
                same_page_transitions.append(transition)

            if previous_section is not None and idx - previous_index <= 3:
                frequent_short_jumps.append(transition)

            previous_section = section
            previous_index = idx
            previous_page = page

    abstract_pages = sorted({int(b["page"]) for b in text_blocks if b.get("section") == "abstract"})
    introduction_present = any(b.get("section") == "introduction" for b in text_blocks)

    audit = {
        "paper": paper,
        "section_transition_sequence": compressed,
        "transitions": transitions,
        "same_page_transitions": same_page_transitions,
        "references_regressions": references_regressions,
        "abstract_pages": abstract_pages,
        "abstract_crosses_many_pages": len(abstract_pages) > 2,
        "introduction_missing": not introduction_present,
        "frequent_short_jumps": frequent_short_jumps,
    }
    summary = {
        "transition_count": len(transitions),
        "same_page_transition_count": len(same_page_transitions),
        "references_regression_count": len(references_regressions),
        "abstract_crosses_many_pages": len(abstract_pages) > 2,
        "introduction_missing": not introduction_present,
        "frequent_short_jump_count": len(frequent_short_jumps),
    }
    return audit, summary


def audit_references(paper: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    ref_heading_indices = [
        idx
        for idx, b in enumerate(blocks)
        if b.get("block_type") == "heading"
        and str(b.get("section")) == "references"
        and re.search(r"references|bibliography", str(b.get("text", "")), flags=re.I)
    ]
    first_ref_idx = ref_heading_indices[0] if ref_heading_indices else None

    if first_ref_idx is None:
        return {
            "paper": paper,
            "references_heading_found": False,
            "post_references_reference_like_ratio": 0.0,
            "possible_body_marked_references": [],
            "possible_appendix_locked_as_references": [],
            "first_page_samples": [],
            "last_page_samples": [],
        }

    post_ref_blocks = blocks[first_ref_idx + 1 :]
    ratio = reference_like_ratio(post_ref_blocks)
    possible_body_refs = [
        {
            "page": b.get("page"),
            "block_id": b.get("block_id"),
            "block_type": b.get("block_type"),
            "section": b.get("section"),
            "text_preview": text_preview(str(b.get("text", ""))),
        }
        for b in post_ref_blocks
        if b.get("section") == "references"
        and b.get("block_type") in CORE_TEXT_BLOCK_TYPES
        and len(str(b.get("text", ""))) > 400
        and not is_reference_like_text(str(b.get("text", "")))
    ]
    appendix_locked = [
        {
            "page": b.get("page"),
            "block_id": b.get("block_id"),
            "block_type": b.get("block_type"),
            "section": b.get("section"),
            "heading_section": b.get("heading_prediction", {}).get("section"),
            "text_preview": text_preview(str(b.get("text", ""))),
        }
        for b in post_ref_blocks
        if b.get("section") == "references"
        and re.search(r"\b(appendix|appendices|supplementary|additional results|experiment details)\b", str(b.get("text", "")), flags=re.I)
    ]

    ref_pages = sorted({int(b["page"]) for b in post_ref_blocks if b.get("section") == "references"})
    first_page = ref_pages[0] if ref_pages else int(blocks[first_ref_idx]["page"])
    last_page = ref_pages[-1] if ref_pages else first_page

    def page_sample(page_no: int) -> list[dict[str, Any]]:
        return [
            {
                "page": b.get("page"),
                "block_id": b.get("block_id"),
                "block_type": b.get("block_type"),
                "section": b.get("section"),
                "reference_like": is_reference_like_text(str(b.get("text", ""))),
                "text_preview": text_preview(str(b.get("text", "")), 180),
            }
            for b in blocks
            if int(b.get("page", 0)) == page_no
        ][:8]

    return {
        "paper": paper,
        "references_heading_found": True,
        "references_heading_page": blocks[first_ref_idx].get("page"),
        "post_references_block_count": len(post_ref_blocks),
        "post_references_reference_like_ratio": ratio,
        "possible_body_marked_references": possible_body_refs,
        "possible_appendix_locked_as_references": appendix_locked,
        "first_page_samples": page_sample(first_page),
        "last_page_samples": page_sample(last_page),
    }


def audit_chunks(paper: str, chunks: list[dict[str, Any]], blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    block_by_id = {str(b["block_id"]): b for b in blocks}
    block_ids = set(block_by_id)
    chunk_rows: list[dict[str, Any]] = []
    used_block_ids: list[str] = []
    chunk_lengths = [int(c.get("char_count", len(str(c.get("text", ""))))) for c in chunks]
    duplicate_block_ids: list[str] = []
    seen_ids: set[str] = set()

    for chunk in chunks:
        ids = [str(i) for i in chunk.get("block_ids", [])]
        used_block_ids.extend(ids)
        for block_id in ids:
            if block_id in seen_ids:
                duplicate_block_ids.append(block_id)
            seen_ids.add(block_id)

        linked_blocks = [block_by_id[i] for i in ids if i in block_by_id]
        linked_sections = sorted({str(b.get("section")) for b in linked_blocks if b.get("section")})
        linked_types = Counter(str(b.get("block_type")) for b in linked_blocks)
        chunk_section = str(chunk.get("section", ""))
        section_mismatch = bool(linked_sections and chunk_section not in linked_sections)
        multi_section = len(linked_sections) > 1
        row = {
            "paper": paper,
            "chunk_id": chunk.get("chunk_id"),
            "section": chunk_section,
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "char_count": int(chunk.get("char_count", 0)),
            "block_ids": ids,
            "linked_sections": linked_sections,
            "linked_block_types": dict(linked_types),
            "multi_section": multi_section,
            "section_mismatch": section_mismatch,
            "heading_only": is_heading_only_chunk(chunk, block_by_id),
            "cross_page": chunk.get("page_start") != chunk.get("page_end"),
            "has_caption_or_table": any(t in linked_types for t in NON_TEXT_CONTEXT_TYPES),
            "text_preview": text_preview(str(chunk.get("text", "")), 220),
        }
        chunk_rows.append(row)

    used_set = set(used_block_ids)
    omitted_blocks = sorted(block_ids - used_set)
    lost_core_blocks = [
        {
            "block_id": b["block_id"],
            "page": b.get("page"),
            "block_type": b.get("block_type"),
            "section": b.get("section"),
            "text_preview": text_preview(str(b.get("text", ""))),
        }
        for bid in omitted_blocks
        for b in [block_by_id[bid]]
        if b.get("block_type") in CORE_TEXT_BLOCK_TYPES and str(b.get("text", "")).strip()
    ]
    equation_blocks = [b for b in blocks if b.get("block_type") == "equation"]
    equation_in_chunks = [b["block_id"] for b in equation_blocks if b["block_id"] in used_set]
    caption_table_blocks = [b for b in blocks if b.get("block_type") in NON_TEXT_CONTEXT_TYPES]
    caption_table_in_chunks = [b for b in caption_table_blocks if b["block_id"] in used_set]
    isolated_caption_table = [
        {
            "block_id": b["block_id"],
            "page": b.get("page"),
            "block_type": b.get("block_type"),
            "section": b.get("section"),
            "text_preview": text_preview(str(b.get("text", ""))),
        }
        for b in caption_table_in_chunks
        if not caption_table_context_ok(b, chunks, block_by_id)
    ]

    summary = {
        "chunk_count": len(chunks),
        "length_stats": {
            "min": min(chunk_lengths) if chunk_lengths else 0,
            "p10": percentile(chunk_lengths, 0.10),
            "median": int(median(chunk_lengths)) if chunk_lengths else 0,
            "p90": percentile(chunk_lengths, 0.90),
            "max": max(chunk_lengths) if chunk_lengths else 0,
        },
        "small_chunk_count_lt_150": sum(1 for c in chunk_rows if c["char_count"] < 150),
        "large_chunk_count_gt_1600": sum(1 for c in chunk_rows if c["char_count"] > 1600),
        "multi_section_chunk_count": sum(1 for c in chunk_rows if c["multi_section"]),
        "heading_only_chunk_count": sum(1 for c in chunk_rows if c["heading_only"]),
        "section_mismatch_chunk_count": sum(1 for c in chunk_rows if c["section_mismatch"]),
        "cross_page_chunk_count": sum(1 for c in chunk_rows if c["cross_page"]),
        "duplicate_block_id_count": len(duplicate_block_ids),
        "duplicate_block_ids": sorted(set(duplicate_block_ids)),
        "missing_block_count": len(omitted_blocks),
        "lost_core_block_count": len(lost_core_blocks),
        "lost_core_blocks": lost_core_blocks[:50],
        "equation_block_count": len(equation_blocks),
        "equation_block_in_chunk_count": len(equation_in_chunks),
        "equation_blocks_in_chunks": equation_in_chunks,
        "caption_table_block_count": len(caption_table_blocks),
        "caption_table_in_chunk_count": len(caption_table_in_chunks),
        "isolated_caption_table_count": len(isolated_caption_table),
        "isolated_caption_table": isolated_caption_table[:30],
        "small_chunks": [c for c in chunk_rows if c["char_count"] < 150],
        "large_chunks": [c for c in chunk_rows if c["char_count"] > 1600],
        "multi_section_chunks": [c for c in chunk_rows if c["multi_section"]],
        "section_mismatch_chunks": [c for c in chunk_rows if c["section_mismatch"]],
    }
    return chunk_rows, summary


def caption_table_context_ok(block: dict[str, Any], chunks: list[dict[str, Any]], block_by_id: dict[str, dict[str, Any]]) -> bool:
    block_id = str(block["block_id"])
    for chunk in chunks:
        ids = [str(i) for i in chunk.get("block_ids", [])]
        if block_id not in ids:
            continue
        linked = [block_by_id[i] for i in ids if i in block_by_id]
        non_caption = [
            b
            for b in linked
            if b.get("block_type") not in {"caption", "table", "equation"}
            and len(str(b.get("text", "")).strip()) >= 80
        ]
        return bool(non_caption) or len(ids) > 1
    return False


def select_chunk_samples(paper: str, chunks: list[dict[str, Any]], chunk_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["chunk_id"]: row for row in chunk_rows}
    samples: dict[str, dict[str, Any]] = {}
    rng = random.Random(f"parser-v5-{paper}")

    ordinary = [
        c
        for c in chunks
        if 150 <= int(c.get("char_count", 0)) <= 1600
        and str(c.get("section")) not in {"references"}
    ]
    for c in rng.sample(ordinary, min(10, len(ordinary))):
        samples[str(c["chunk_id"])] = {"reason": "random普通chunk", "chunk": c, "audit": by_id.get(c["chunk_id"], {})}

    selectors = [
        ("method chunk", lambda c: c.get("section") == "method", 3),
        ("result/experiment chunk", lambda c: c.get("section") in {"results", "experiment"}, 3),
        ("references chunk", lambda c: c.get("section") == "references", 3),
        ("small chunk <150", lambda c: int(c.get("char_count", 0)) < 150, 999),
        ("large chunk >1600", lambda c: int(c.get("char_count", 0)) > 1600, 999),
    ]
    for reason, predicate, limit in selectors:
        selected = [c for c in chunks if predicate(c)]
        if limit != 999:
            selected = selected[:limit]
        for c in selected:
            existing = samples.get(str(c["chunk_id"]))
            if existing:
                existing["reason"] += f"; {reason}"
            else:
                samples[str(c["chunk_id"])] = {"reason": reason, "chunk": c, "audit": by_id.get(c["chunk_id"], {})}

    return list(samples.values())


def write_chunk_samples_md(all_samples: dict[str, list[dict[str, Any]]], path: Path) -> None:
    lines = ["# Parser v5 Chunk Samples", ""]
    for paper, samples in all_samples.items():
        lines.extend([f"## {paper}", ""])
        for sample in samples:
            c = sample["chunk"]
            audit = sample.get("audit", {})
            lines.extend(
                [
                    f"### {c.get('chunk_id')} ({sample['reason']})",
                    "",
                    f"- section: `{c.get('section')}`",
                    f"- pages: `{c.get('page_start')}` to `{c.get('page_end')}`",
                    f"- chars: `{c.get('char_count')}`",
                    f"- block_ids: `{', '.join(map(str, c.get('block_ids', [])))}`",
                    f"- linked_sections: `{audit.get('linked_sections', [])}`",
                    f"- linked_block_types: `{audit.get('linked_block_types', {})}`",
                    "",
                    "```text",
                    str(c.get("text", "")).strip(),
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def decide_conclusion(summary_by_paper: dict[str, Any]) -> str:
    hard_fail = False
    chunk_fixes = False

    for paper_summary in summary_by_paper.values():
        heading = paper_summary["heading"]
        sections = paper_summary["sections"]
        refs = paper_summary["references"]
        chunks = paper_summary["chunks"]
        reading = paper_summary["reading_order"]

        if heading["suspicious_heading_count"] > 0 or heading["unmapped_heading_count"] > 0:
            hard_fail = True
        if sections["introduction_missing"] or sections["abstract_crosses_many_pages"]:
            hard_fail = True
        if sections["references_regression_count"] > 0:
            hard_fail = True
        if not refs["references_heading_found"]:
            hard_fail = True
        if refs["possible_appendix_locked_as_references"]:
            hard_fail = True
        if any(item["column_backtracks"] > 0 or item["y_backtracks_within_column"] > 0 for item in reading.values()):
            hard_fail = True

        if chunks["large_chunk_count_gt_1600"] > 0:
            chunk_fixes = True
        if chunks["small_chunk_count_lt_150"] > 0:
            chunk_fixes = True
        if chunks["multi_section_chunk_count"] > 0:
            chunk_fixes = True
        if chunks["heading_only_chunk_count"] > 0:
            chunk_fixes = True
        if chunks["section_mismatch_chunk_count"] > 0:
            chunk_fixes = True
        if chunks["duplicate_block_id_count"] > 0:
            chunk_fixes = True
        if chunks["lost_core_block_count"] > 0:
            chunk_fixes = True
        if chunks["equation_block_count"] > 0 and chunks["equation_block_in_chunk_count"] == 0:
            chunk_fixes = True

    if hard_fail:
        return "FAIL"
    if chunk_fixes:
        return "PASS_WITH_CHUNK_FIXES"
    return "PASS"


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Parser v5 Validation Report",
        "",
        f"Conclusion: **{summary['conclusion']}**",
        "",
        "This validation audits parser output only. It does not modify parser, embedding, or indexing code.",
        "",
        "## Per-paper Summary",
        "",
    ]

    for paper, s in summary["papers"].items():
        lines.extend(
            [
                f"### {paper}",
                "",
                f"- parse status: `{s['parse_status']}`",
                f"- pages/chars: `{s['parsed_pages']}` / `{s['total_chars']}`",
                f"- sections: `{s['section_counts']}`",
                f"- reading order sample pages: `{s['reading_order']}`",
                f"- headings: `{s['heading']['heading_count']}` total, `{s['heading']['suspicious_heading_count']}` suspicious, `{s['heading']['unmapped_heading_count']}` unmapped",
                f"- section transitions: `{s['sections']['transition_count']}` total, `{s['sections']['same_page_transition_count']}` same-page, `{s['sections']['frequent_short_jump_count']}` short-distance jumps",
                f"- references: found=`{s['references']['references_heading_found']}`, post-ref ratio=`{s['references']['post_references_reference_like_ratio']}`",
                f"- chunks: `{s['chunks']['chunk_count']}` total, length stats `{s['chunks']['length_stats']}`",
                f"- chunk issues: small=`{s['chunks']['small_chunk_count_lt_150']}`, large=`{s['chunks']['large_chunk_count_gt_1600']}`, multi-section=`{s['chunks']['multi_section_chunk_count']}`, heading-only=`{s['chunks']['heading_only_chunk_count']}`, cross-page=`{s['chunks']['cross_page_chunk_count']}`",
                f"- block integrity: duplicate block refs=`{s['chunks']['duplicate_block_id_count']}`, lost core blocks=`{s['chunks']['lost_core_block_count']}`, equation blocks in chunks=`{s['chunks']['equation_block_in_chunk_count']}/{s['chunks']['equation_block_count']}`",
                "",
            ]
        )

        if s["heading"]["suspicious_headings"]:
            lines.extend(["Suspicious headings:", ""])
            for h in s["heading"]["suspicious_headings"][:10]:
                lines.append(f"- p{h['page']} `{h['text']}` section=`{h['section']}`")
            lines.append("")

        if s["chunks"]["small_chunks"]:
            lines.extend(["Small chunks (<150 chars):", ""])
            for c in s["chunks"]["small_chunks"][:10]:
                lines.append(f"- `{c['chunk_id']}` chars={c['char_count']} section={c['section']} preview={c['text_preview']!r}")
            lines.append("")

        if s["chunks"]["large_chunks"]:
            lines.extend(["Large chunks (>1600 chars):", ""])
            for c in s["chunks"]["large_chunks"][:10]:
                lines.append(f"- `{c['chunk_id']}` chars={c['char_count']} section={c['section']} preview={c['text_preview']!r}")
            lines.append("")

        if s["chunks"]["multi_section_chunks"]:
            lines.extend(["Multi-section chunks:", ""])
            for c in s["chunks"]["multi_section_chunks"][:10]:
                lines.append(f"- `{c['chunk_id']}` section={c['section']} linked={c['linked_sections']} preview={c['text_preview']!r}")
            lines.append("")

    lines.extend(
        [
            "## Output Files",
            "",
            "- `reading_order_samples.jsonl`: selected page block order for manual inspection.",
            "- `heading_audit.jsonl`: all detected headings with scores and reasons.",
            "- `section_transition_audit.jsonl`: full block-level section transition traces.",
            "- `chunk_audit.jsonl`: chunk-level structural checks.",
            "- `chunk_samples.md`: random and targeted chunk samples for manual review.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_paper(paper: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    paper_dir = INPUT_ROOT / paper
    layout = read_json(paper_dir / "layout.json")
    blocks = read_jsonl(paper_dir / "blocks.jsonl")
    pages = read_jsonl(paper_dir / "parsed_pages.jsonl")
    chunks = read_jsonl(paper_dir / "chunks.jsonl")
    report = read_json(paper_dir / "parse_quality_report.json")

    reading_rows, reading_summary = audit_reading_order(paper, report, blocks, chunks)
    heading_rows, heading_summary = audit_headings(paper, blocks)
    section_audit, section_summary = audit_sections(paper, blocks)
    reference_summary = audit_references(paper, blocks)
    chunk_rows, chunk_summary = audit_chunks(paper, chunks, blocks)
    chunk_samples = select_chunk_samples(paper, chunks, chunk_rows)

    paper_summary = {
        "doc_id": layout.get("doc_id"),
        "parse_status": report.get("status"),
        "parsed_pages": report.get("parsed_pages", len(pages)),
        "total_chars": report.get("total_chars"),
        "section_counts": report.get("section_counts", {}),
        "block_type_counts": report.get("block_type_counts", {}),
        "reading_order": reading_summary,
        "heading": heading_summary,
        "sections": section_summary,
        "references": reference_summary,
        "chunks": chunk_summary,
    }
    outputs = {
        "reading_rows": reading_rows,
        "heading_rows": heading_rows,
        "section_rows": [section_audit],
        "chunk_rows": chunk_rows,
    }
    return paper_summary, outputs, chunk_samples


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary_by_paper: dict[str, Any] = {}
    reading_rows_all: list[dict[str, Any]] = []
    heading_rows_all: list[dict[str, Any]] = []
    section_rows_all: list[dict[str, Any]] = []
    chunk_rows_all: list[dict[str, Any]] = []
    samples_by_paper: dict[str, list[dict[str, Any]]] = {}

    for paper in PAPERS:
        paper_summary, outputs, chunk_samples = validate_paper(paper)
        summary_by_paper[paper] = paper_summary
        reading_rows_all.extend(outputs["reading_rows"])
        heading_rows_all.extend(outputs["heading_rows"])
        section_rows_all.extend(outputs["section_rows"])
        chunk_rows_all.extend(outputs["chunk_rows"])
        samples_by_paper[paper] = chunk_samples

    summary = {
        "input_root": str(INPUT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "papers_validated": PAPERS,
        "papers": summary_by_paper,
    }
    summary["conclusion"] = decide_conclusion(summary_by_paper)

    write_json(summary, OUTPUT_ROOT / "parser_validation_summary.json")
    append_jsonl(reading_rows_all, OUTPUT_ROOT / "reading_order_samples.jsonl")
    append_jsonl(heading_rows_all, OUTPUT_ROOT / "heading_audit.jsonl")
    append_jsonl(section_rows_all, OUTPUT_ROOT / "section_transition_audit.jsonl")
    append_jsonl(chunk_rows_all, OUTPUT_ROOT / "chunk_audit.jsonl")
    write_chunk_samples_md(samples_by_paper, OUTPUT_ROOT / "chunk_samples.md")
    write_report(summary, OUTPUT_ROOT / "parser_validation_report.md")

    print(json.dumps({"conclusion": summary["conclusion"], "output_root": str(OUTPUT_ROOT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
