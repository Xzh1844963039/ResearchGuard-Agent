# C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_chunking_v1.py
from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from researchguard.ingestion.chunk_builder import build_chunks  # noqa: E402


PARSER_DIR = ROOT / "data" / "parsed" / "parser_eval_v5"
CHUNK_DIR = ROOT / "data" / "parsed" / "chunk_eval_v1"
OUTPUT_DIR = ROOT / "outputs" / "chunk_validation_v1"
MAX_CHARS = 1600
MIN_SMALL_CHARS = 150
CORE_TYPES = {"paragraph", "heading", "heading_candidate", "reference_entry"}
SPECIAL_TYPES = {"equation", "caption", "table"}
BINDING_BODY_TYPES = {"paragraph", "reference_entry"}
TARGETED_SECTIONS = {
    "method": {"method"},
    "result_experiment": {"results", "experiment"},
    "references": {"references"},
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def length_stats(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = sorted(int(chunk.get("char_count", 0)) for chunk in chunks)
    if not lengths:
        return {"min": 0, "p10": 0, "median": 0, "p90": 0, "max": 0, "avg": 0}

    def percentile(q: float) -> int:
        return lengths[round((len(lengths) - 1) * q)]

    return {
        "min": lengths[0],
        "p10": percentile(0.10),
        "median": int(statistics.median(lengths)),
        "p90": percentile(0.90),
        "max": lengths[-1],
        "avg": round(sum(lengths) / len(lengths), 2),
    }


def title_for_paper(paper_dir: Path, fallback: str) -> str:
    for name in ("parse_quality_report.json", "layout.json"):
        title = str(read_json(paper_dir / name).get("title", "")).strip()
        if title:
            return title
    return fallback


def normalized_json_hash(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ids_from_chunks(chunks: list[dict[str, Any]], field: str) -> set[str]:
    return {
        str(block_id)
        for chunk in chunks
        for block_id in chunk.get(field, [])
        if str(block_id).strip()
    }


def source_ids(chunk: dict[str, Any]) -> list[str]:
    return [str(block_id) for block_id in chunk.get("source_block_ids", chunk.get("block_ids", [])) if str(block_id).strip()]


def linked_values(chunk: dict[str, Any], block_by_id: dict[str, dict[str, Any]], key: str) -> list[str]:
    values = []
    for block_id in source_ids(chunk):
        block = block_by_id.get(block_id)
        if block:
            values.append(str(block.get(key, "")))
    return values


def block_mid_y(block: dict[str, Any]) -> float:
    return (float(block.get("y0", 0.0)) + float(block.get("y1", block.get("y0", 0.0)))) / 2


def block_text_len(block: dict[str, Any]) -> int:
    return len(str(block.get("text", "")).strip())


def nearest_candidate(
    source_blocks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    order_index: dict[str, int],
    max_combined_chars: int | None = None,
) -> dict[str, Any] | None:
    sections = {str(block.get("section", "")) for block in source_blocks}
    pages = {int(block.get("page", 0)) for block in source_blocks}
    source_mid = sum(block_mid_y(block) for block in source_blocks) / len(source_blocks)
    source_index = sum(order_index.get(str(block.get("block_id", "")), 0) for block in source_blocks) / len(source_blocks)
    source_len = sum(block_text_len(block) for block in source_blocks)
    filtered = [
        candidate
        for candidate in candidates
        if str(candidate.get("section", "")) in sections
        and (
            max_combined_chars is None
            or source_len + block_text_len(candidate) + 2 <= max_combined_chars
        )
    ]
    if not filtered:
        return None
    same_page = [candidate for candidate in filtered if int(candidate.get("page", 0)) in pages]
    pool = same_page if same_page else filtered
    return min(
        pool,
        key=lambda candidate: (
            0 if int(candidate.get("page", 0)) in pages else 1,
            abs(block_mid_y(candidate) - source_mid),
            abs(order_index.get(str(candidate.get("block_id", "")), 0) - source_index),
            order_index.get(str(candidate.get("block_id", "")), 0),
        ),
    )


def is_heading_only(chunk: dict[str, Any], block_by_id: dict[str, dict[str, Any]]) -> bool:
    ids = source_ids(chunk)
    types = [str(block_by_id.get(block_id, {}).get("block_type", "")) for block_id in ids]
    has_heading_context = bool(chunk.get("heading_block_ids"))
    if not ids and has_heading_context:
        return True
    if ids and all(block_type in {"heading", "heading_candidate"} for block_type in types):
        return int(chunk.get("char_count", 0)) < 250
    return False


def duplicate_source_analysis(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        for block_id in source_ids(chunk):
            occurrences[block_id].append(chunk)

    repeated = {block_id: rows for block_id, rows in occurrences.items() if len(rows) > 1}
    unexplained: dict[str, list[str]] = {}
    split_invalid = []
    for block_id, rows in repeated.items():
        split_entries = []
        missing_split_meta = []
        for row in rows:
            row_entries = [
                entry
                for entry in row.get("split_blocks", [])
                if str(entry.get("block_id", "")) == block_id
            ]
            if not row_entries:
                missing_split_meta.append(str(row.get("chunk_id")))
            split_entries.extend(row_entries)

        if missing_split_meta:
            unexplained[block_id] = [str(row.get("chunk_id")) for row in rows]
            continue

        total_values = {int(entry.get("split_total", 0)) for entry in split_entries}
        part_values = {int(entry.get("split_part", 0)) for entry in split_entries}
        expected_parts = set(range(1, next(iter(total_values)) + 1)) if len(total_values) == 1 else set()
        if (
            len(total_values) != 1
            or not part_values
            or min(part_values) < 1
            or max(part_values) > next(iter(total_values))
            or part_values != expected_parts
        ):
            split_invalid.append(
                {
                    "block_id": block_id,
                    "chunk_ids": [row.get("chunk_id") for row in rows],
                    "split_parts": sorted(part_values),
                    "split_totals": sorted(total_values),
                }
            )
    return {
        "repeated_source_block_ids": len(repeated),
        "unexplained_repeated_source_block_ids": len(unexplained),
        "unexplained_examples": dict(list(unexplained.items())[:20]),
        "split_metadata_invalid_count": len(split_invalid),
        "split_metadata_invalid_examples": split_invalid[:20],
    }


def special_binding_analysis(chunks: list[dict[str, Any]], blocks: list[dict[str, Any]]) -> dict[str, Any]:
    block_by_id = {str(block.get("block_id")): block for block in blocks if str(block.get("block_id", "")).strip()}
    order_index = {str(block.get("block_id")): index for index, block in enumerate(blocks)}
    chunk_by_block_id: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        for block_id in source_ids(chunk):
            chunk_by_block_id[block_id] = chunk

    body_blocks = [
        block
        for block in blocks
        if str(block.get("block_type", "")) in BINDING_BODY_TYPES
        and str(block.get("text", "")).strip()
    ]
    table_blocks = [block for block in blocks if str(block.get("block_type", "")) == "table"]
    special_blocks = [block for block in blocks if str(block.get("block_type", "")) in SPECIAL_TYPES]

    isolated_bindable: list[dict[str, Any]] = []
    caption_table_misses: list[dict[str, Any]] = []

    for block in special_blocks:
        bid = str(block.get("block_id"))
        chunk = chunk_by_block_id.get(bid)
        if not chunk:
            continue
        chunk_ids = set(source_ids(chunk))
        linked_types = {
            str(block_by_id.get(block_id, {}).get("block_type", ""))
            for block_id in chunk_ids
        }

        if str(block.get("block_type")) == "caption":
            nearest_table = nearest_candidate([block], table_blocks, order_index, max_combined_chars=MAX_CHARS)
            if nearest_table is not None and str(nearest_table.get("block_id")) not in chunk_ids:
                caption_table_misses.append(
                    {
                        "caption_block_id": bid,
                        "expected_table_block_id": nearest_table.get("block_id"),
                        "chunk_id": chunk.get("chunk_id"),
                    }
                )

        nearest_body = nearest_candidate([block], body_blocks, order_index, max_combined_chars=MAX_CHARS)
        has_body = bool(linked_types & BINDING_BODY_TYPES)
        if nearest_body is not None and not has_body:
            body_chunk = chunk_by_block_id.get(str(nearest_body.get("block_id")))
            body_chunk_types = {
                str(block_by_id.get(block_id, {}).get("block_type", ""))
                for block_id in source_ids(body_chunk or {})
            }
            body_capacity_blocked = bool(body_chunk_types & SPECIAL_TYPES) and (
                int((body_chunk or {}).get("char_count", 0)) + block_text_len(block) + 2 > MAX_CHARS
            )
            if body_capacity_blocked:
                continue
            isolated_bindable.append(
                {
                    "block_id": bid,
                    "block_type": block.get("block_type"),
                    "expected_body_block_id": nearest_body.get("block_id"),
                    "chunk_id": chunk.get("chunk_id"),
                }
            )

    return {
        "isolated_bindable_special_count": len(isolated_bindable),
        "isolated_bindable_special_examples": isolated_bindable[:20],
        "caption_nearest_table_miss_count": len(caption_table_misses),
        "caption_nearest_table_miss_examples": caption_table_misses[:20],
    }


def validate_overlap_implementation() -> dict[str, Any]:
    doc_id = "overlap_unit"
    sentence_a = "Alpha first sentence has enough detail for overlap testing."
    sentence_b = "Alpha second sentence has enough detail for overlap testing."
    sentence_c = "Alpha third sentence has enough detail for overlap testing."
    blocks = [
        {
            "block_id": f"{doc_id}_p0001_b0001",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "paragraph",
            "text": f"{sentence_a} {sentence_b} {sentence_c}",
            "section": "introduction",
            "column": 0,
            "y0": 10,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_p0001_b0002",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "paragraph",
            "text": "Beta first sentence has enough detail for direct-neighbor overlap testing. Beta second sentence closes this paragraph.",
            "section": "introduction",
            "column": 0,
            "y0": 80,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_p0001_b0003",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "paragraph",
            "text": "Method first sentence deliberately interrupts the introduction sequence. Method second sentence closes this paragraph.",
            "section": "method",
            "column": 0,
            "y0": 150,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_p0002_b0001",
            "doc_id": doc_id,
            "page": 2,
            "block_type": "paragraph",
            "text": "Gamma first sentence should not receive overlap from the earlier introduction block. Gamma second sentence closes this paragraph.",
            "section": "introduction",
            "column": 0,
            "y0": 10,
            "x0": 10,
        },
    ]

    def make(overlap_sentences: int) -> list[dict[str, Any]]:
        return build_chunks(
            doc_id=doc_id,
            title="Overlap Unit",
            blocks=blocks,
            max_chars=1000,
            target_chars=100,
            min_chars=0,
            overlap_sentences=overlap_sentences,
        )

    chunks_0 = make(0)
    chunks_1 = make(1)
    chunks_2 = make(2)
    failures: list[str] = []

    if any(chunk.get("overlap_from_chunk_id") for chunk in chunks_0):
        failures.append("overlap_sentences=0 produced overlap metadata")

    if len(chunks_1) < 4 or len(chunks_2) < 4:
        failures.append("synthetic overlap fixture did not produce expected chunk count")
    else:
        if chunks_1[1].get("overlap_from_chunk_id") != chunks_1[0].get("chunk_id"):
            failures.append("overlap_sentences=1 did not overlap from direct previous same-section chunk")
        if chunks_2[1].get("overlap_from_chunk_id") != chunks_2[0].get("chunk_id"):
            failures.append("overlap_sentences=2 did not overlap from direct previous same-section chunk")
        if int(chunks_2[1].get("overlap_char_count", 0)) <= int(chunks_1[1].get("overlap_char_count", 0)):
            failures.append("overlap_sentences=2 did not include more context than overlap_sentences=1")
        if sentence_b not in str(chunks_2[1].get("text", ""))[:180] or sentence_c not in str(chunks_2[1].get("text", ""))[:220]:
            failures.append("overlap_sentences=2 did not prepend the previous two sentences")
        if chunks_1[3].get("overlap_from_chunk_id") or chunks_2[3].get("overlap_from_chunk_id"):
            failures.append("overlap crossed over an intervening method section")

    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "overlap_0_count": sum(1 for chunk in chunks_0 if chunk.get("overlap_from_chunk_id")),
        "overlap_1_second_chunk_chars": chunks_1[1].get("overlap_char_count") if len(chunks_1) > 1 else None,
        "overlap_2_second_chunk_chars": chunks_2[1].get("overlap_char_count") if len(chunks_2) > 1 else None,
    }


def repeated_text(prefix: str, target_len: int) -> str:
    sentence = f"{prefix} contains enough stable words for deterministic synthetic validation. "
    text = ""
    while len(text) < target_len:
        text += sentence
    return text[:target_len].rstrip() + "."


def test_heading_not_in_overlap() -> dict[str, Any]:
    doc_id = "synthetic_heading_overlap"
    heading = "METHOD HEADING SHOULD NOT ENTER OVERLAP"
    previous_tail = "The previous body sentence should be the only copied overlap sentence."
    blocks = [
        {
            "block_id": f"{doc_id}_h1",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "heading",
            "text": heading,
            "section": "method",
            "column": 0,
            "y0": 10,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_p1",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "paragraph",
            "text": "The first body sentence gives context for this method chunk. " + previous_tail,
            "section": "method",
            "column": 0,
            "y0": 50,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_p2",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "paragraph",
            "text": "The next method paragraph receives overlap without receiving heading text.",
            "section": "method",
            "column": 0,
            "y0": 160,
            "x0": 10,
        },
    ]
    chunks = build_chunks(doc_id=doc_id, title="Synthetic", blocks=blocks, max_chars=500, target_chars=120, min_chars=0)
    target = next((chunk for chunk in chunks if f"{doc_id}_p2" in source_ids(chunk)), {})
    failures = []
    if not target.get("overlap_from_chunk_id"):
        failures.append("second body chunk did not receive overlap")
    if heading in str(target.get("text", "")).split("The next method paragraph", 1)[0]:
        failures.append("heading text leaked into overlap prefix")
    if previous_tail not in str(target.get("text", "")):
        failures.append("expected previous body sentence missing from overlap")
    return {"name": "heading_not_in_overlap", "passed": not failures, "failures": failures}


def test_overlap_two_sentence_provenance() -> dict[str, Any]:
    doc_id = "synthetic_overlap_provenance"
    block1_sentence = (
        "Block one contributes the first copied overlap sentence with stable wording and enough extra detail "
        "to pass the target threshold without creating multiple sentence records."
    )
    block2_sentence = (
        "Block two contributes the second copied overlap sentence with stable wording and enough extra detail "
        "to force the receiving chunk to record both source blocks."
    )
    blocks = [
        {
            "block_id": f"{doc_id}_p1",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "paragraph",
            "text": block1_sentence,
            "section": "introduction",
            "column": 0,
            "y0": 10,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_p2",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "paragraph",
            "text": block2_sentence,
            "section": "introduction",
            "column": 0,
            "y0": 80,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_p3",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "paragraph",
            "text": "The receiving paragraph should record two overlap source block ids.",
            "section": "introduction",
            "column": 0,
            "y0": 160,
            "x0": 10,
        },
    ]
    chunks = build_chunks(
        doc_id=doc_id,
        title="Synthetic",
        blocks=blocks,
        max_chars=800,
        target_chars=200,
        min_chars=0,
        overlap_sentences=2,
    )
    target = next((chunk for chunk in chunks if f"{doc_id}_p3" in source_ids(chunk)), {})
    expected_ids = [f"{doc_id}_p1", f"{doc_id}_p2"]
    failures = []
    if target.get("overlap_source_block_ids") != expected_ids:
        failures.append(
            f"expected overlap_source_block_ids={expected_ids}, got {target.get('overlap_source_block_ids')}"
        )
    return {"name": "overlap_two_sentence_provenance", "passed": not failures, "failures": failures}


def test_special_heading_budget() -> dict[str, Any]:
    doc_id = "synthetic_special_budget"
    blocks = [
        {
            "block_id": f"{doc_id}_h1",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "heading",
            "text": repeated_text("A long method heading prefix", 180),
            "section": "method",
            "column": 0,
            "y0": 10,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_body",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "paragraph",
            "text": repeated_text("The main body paragraph is close to the chunk budget", 1260),
            "section": "method",
            "column": 0,
            "y0": 80,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_table",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "table",
            "text": repeated_text("Table values compare method scores", 120),
            "section": "method",
            "column": 0,
            "y0": 260,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_caption",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "caption",
            "text": repeated_text("Table caption describes the comparison", 80),
            "section": "method",
            "column": 0,
            "y0": 300,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_equation",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "equation",
            "text": repeated_text("Equation alpha plus beta equals score", 80),
            "section": "method",
            "column": 0,
            "y0": 340,
            "x0": 10,
        },
    ]
    chunks = build_chunks(doc_id=doc_id, title="Synthetic", blocks=blocks, max_chars=1600, target_chars=1200, min_chars=0)
    covered = ids_from_chunks(chunks, "source_block_ids")
    failures = []
    over = [chunk.get("chunk_id") for chunk in chunks if int(chunk.get("char_count", 0)) > MAX_CHARS]
    if over:
        failures.append(f"chunks exceeded max_chars after heading prefix: {over}")
    for bid in (f"{doc_id}_table", f"{doc_id}_caption", f"{doc_id}_equation"):
        if bid not in covered:
            failures.append(f"special block lost: {bid}")
    return {"name": "special_heading_budget", "passed": not failures, "failures": failures}


def test_caption_nearest_table_choice() -> dict[str, Any]:
    doc_id = "synthetic_caption_table"
    blocks = [
        {
            "block_id": f"{doc_id}_table_far",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "table",
            "text": repeated_text("Far table contents", 250),
            "section": "results",
            "column": 0,
            "y0": 80,
            "y1": 120,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_caption",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "caption",
            "text": repeated_text("Caption closest to the lower table", 80),
            "section": "results",
            "column": 0,
            "y0": 270,
            "y1": 290,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_table_near",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "table",
            "text": repeated_text("Near table contents", 80),
            "section": "results",
            "column": 0,
            "y0": 300,
            "y1": 340,
            "x0": 10,
        },
    ]
    chunks = build_chunks(doc_id=doc_id, title="Synthetic", blocks=blocks, max_chars=400, target_chars=120, min_chars=0)
    caption_chunk = next((chunk for chunk in chunks if f"{doc_id}_caption" in source_ids(chunk)), {})
    caption_ids = set(source_ids(caption_chunk))
    failures = []
    if f"{doc_id}_table_near" not in caption_ids:
        failures.append("caption did not bind to nearest same-page table")
    if f"{doc_id}_table_far" in caption_ids:
        failures.append("caption also bound to farther same-page table")
    return {"name": "caption_nearest_table_choice", "passed": not failures, "failures": failures}


def test_special_no_cross_section_binding() -> dict[str, Any]:
    doc_id = "synthetic_special_section"
    blocks = [
        {
            "block_id": f"{doc_id}_intro_body",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "paragraph",
            "text": "Intro body is visually close but section-incompatible with the equation.",
            "section": "introduction",
            "column": 0,
            "y0": 100,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_equation",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "equation",
            "text": "score = alpha + beta + gamma",
            "section": "method",
            "column": 0,
            "y0": 120,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_method_body",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "paragraph",
            "text": "Method body is the valid section-compatible attachment target for the equation.",
            "section": "method",
            "column": 0,
            "y0": 400,
            "x0": 10,
        },
    ]
    chunks = build_chunks(doc_id=doc_id, title="Synthetic", blocks=blocks, max_chars=600, target_chars=120, min_chars=0)
    equation_chunk = next((chunk for chunk in chunks if f"{doc_id}_equation" in source_ids(chunk)), {})
    ids = set(source_ids(equation_chunk))
    failures = []
    if f"{doc_id}_intro_body" in ids:
        failures.append("special block crossed section to bind introduction body")
    if f"{doc_id}_method_body" not in ids:
        failures.append("special block did not bind same-section method body")
    return {"name": "special_no_cross_section_binding", "passed": not failures, "failures": failures}


def test_deterministic_synthetic_run() -> dict[str, Any]:
    doc_id = "synthetic_deterministic"
    blocks = [
        {
            "block_id": f"{doc_id}_h1",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "heading",
            "text": "Results",
            "section": "results",
            "column": 0,
            "y0": 10,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_p1",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "paragraph",
            "text": repeated_text("Deterministic result paragraph", 260),
            "section": "results",
            "column": 0,
            "y0": 80,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_caption",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "caption",
            "text": "Table 1: Deterministic caption.",
            "section": "results",
            "column": 0,
            "y0": 200,
            "x0": 10,
        },
        {
            "block_id": f"{doc_id}_table",
            "doc_id": doc_id,
            "page": 1,
            "block_type": "table",
            "text": "Metric A 0.90\nMetric B 0.88",
            "section": "results",
            "column": 0,
            "y0": 220,
            "x0": 10,
        },
    ]
    first = build_chunks(doc_id=doc_id, title="Synthetic", blocks=blocks, max_chars=700, target_chars=180, min_chars=0)
    second = build_chunks(doc_id=doc_id, title="Synthetic", blocks=blocks, max_chars=700, target_chars=180, min_chars=0)
    failures = []
    if normalized_json_hash(first) != normalized_json_hash(second):
        failures.append("same input produced different chunk output hashes")
    return {"name": "deterministic_synthetic_run", "passed": not failures, "failures": failures}


def validate_synthetic_tests() -> dict[str, Any]:
    overlap = validate_overlap_implementation()
    tests = [
        {"name": "overlap_parameter_and_direct_previous", "passed": overlap["passed"], "failures": overlap["failures"]},
        test_heading_not_in_overlap(),
        test_overlap_two_sentence_provenance(),
        test_special_heading_budget(),
        test_caption_nearest_table_choice(),
        test_special_no_cross_section_binding(),
        test_deterministic_synthetic_run(),
    ]
    failures = [failure for test in tests for failure in test.get("failures", [])]
    return {
        "passed": all(bool(test.get("passed")) for test in tests),
        "failure_count": len(failures),
        "failures": failures,
        "tests": tests,
        "overlap_0_count": overlap.get("overlap_0_count"),
        "overlap_1_second_chunk_chars": overlap.get("overlap_1_second_chunk_chars"),
        "overlap_2_second_chunk_chars": overlap.get("overlap_2_second_chunk_chars"),
    }


def validate_paper(paper: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    parser_paper_dir = PARSER_DIR / paper
    chunk_path = CHUNK_DIR / paper / "chunks.jsonl"
    blocks = read_jsonl(parser_paper_dir / "blocks.jsonl")
    chunks = read_jsonl(chunk_path)
    block_by_id = {str(block.get("block_id")): block for block in blocks if str(block.get("block_id", "")).strip()}
    report_title = title_for_paper(parser_paper_dir, paper)
    regenerated = build_chunks(doc_id=paper, title=report_title, blocks=blocks)

    source_covered = ids_from_chunks(chunks, "source_block_ids") | ids_from_chunks(chunks, "block_ids")
    heading_covered = ids_from_chunks(chunks, "heading_block_ids")
    all_covered = source_covered | heading_covered
    block_ids_by_type = {
        block_type: {
            str(block.get("block_id"))
            for block in blocks
            if str(block.get("block_type", "")) == block_type and str(block.get("block_id", "")).strip()
        }
        for block_type in sorted(CORE_TYPES | SPECIAL_TYPES)
    }

    large_chunks = [chunk for chunk in chunks if int(chunk.get("char_count", 0)) > MAX_CHARS]
    small_chunks = [chunk for chunk in chunks if int(chunk.get("char_count", 0)) < MIN_SMALL_CHARS]
    heading_only = [chunk for chunk in chunks if is_heading_only(chunk, block_by_id)]
    empty_text = [chunk for chunk in chunks if not str(chunk.get("text", "")).strip()]
    chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
    duplicate_chunk_ids = [chunk_id for chunk_id, count in Counter(chunk_ids).items() if count > 1]

    multi_section = []
    section_mismatch = []
    page_order_invalid = []
    char_count_invalid = []
    word_count_invalid = []
    cross_page = []
    refs_overlap = []
    overlap_cross_section = []
    overlap_over_max = []
    overlap_not_direct_previous = []
    missing_heading_path = []
    audit_rows: list[dict[str, Any]] = []
    chunk_by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunks}

    for position, chunk in enumerate(chunks):
        linked_sections = sorted({value for value in linked_values(chunk, block_by_id, "section") if value})
        linked_types = Counter(linked_values(chunk, block_by_id, "block_type"))
        chunk_id = str(chunk.get("chunk_id", ""))
        section = str(chunk.get("section", ""))
        char_count = int(chunk.get("char_count", 0))
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")

        is_multi = len(linked_sections) > 1
        is_mismatch = bool(linked_sections and section not in linked_sections)
        if is_multi:
            multi_section.append(chunk)
        if is_mismatch:
            section_mismatch.append(chunk)
        if page_start is not None and page_end is not None and int(page_start) > int(page_end):
            page_order_invalid.append(chunk)
        if page_start is not None and page_end is not None and int(page_start) != int(page_end):
            cross_page.append(chunk)
        if char_count != len(str(chunk.get("text", ""))):
            char_count_invalid.append(chunk)
        if int(chunk.get("word_count", 0)) != len(str(chunk.get("text", "")).split()):
            word_count_invalid.append(chunk)
        if section == "references" and chunk.get("overlap_from_chunk_id"):
            refs_overlap.append(chunk)
        if chunk.get("overlap_from_chunk_id"):
            previous = chunk_by_id.get(str(chunk.get("overlap_from_chunk_id")))
            direct_previous = chunks[position - 1] if position > 0 else None
            if not direct_previous or str(direct_previous.get("chunk_id", "")) != str(chunk.get("overlap_from_chunk_id")):
                overlap_not_direct_previous.append(chunk)
            if previous and str(previous.get("section", "")) != section:
                overlap_cross_section.append(chunk)
            if char_count > MAX_CHARS:
                overlap_over_max.append(chunk)
        if section not in {"abstract", "references"} and not chunk.get("heading_path"):
            missing_heading_path.append(chunk)

        audit_rows.append(
            {
                "paper": paper,
                "chunk_id": chunk_id,
                "section": section,
                "page_start": page_start,
                "page_end": page_end,
                "char_count": char_count,
                "source_block_ids": source_ids(chunk),
                "heading_block_ids": chunk.get("heading_block_ids", []),
                "linked_sections": linked_sections,
                "linked_block_types": dict(linked_types),
                "multi_section": is_multi,
                "section_mismatch": is_mismatch,
                "heading_only": is_heading_only(chunk, block_by_id),
                "cross_page": bool(page_start is not None and page_end is not None and int(page_start) != int(page_end)),
                "has_equation": bool(chunk.get("has_equation")),
                "has_caption": bool(chunk.get("has_caption")),
                "has_table": bool(chunk.get("has_table")),
                "split_reason": chunk.get("split_reason"),
                "split_part": chunk.get("split_part"),
                "split_total": chunk.get("split_total"),
                "split_blocks": chunk.get("split_blocks", []),
                "overlap_from_chunk_id": chunk.get("overlap_from_chunk_id"),
                "overlap_char_count": chunk.get("overlap_char_count"),
                "text_preview": str(chunk.get("text", ""))[:220],
            }
        )

    lost_core: dict[str, list[str]] = {}
    for block_type in sorted(CORE_TYPES):
        ids = block_ids_by_type.get(block_type, set())
        covered = all_covered if block_type in {"heading", "heading_candidate"} else source_covered
        lost_core[block_type] = sorted(ids - covered)

    special_coverage: dict[str, dict[str, Any]] = {}
    for block_type in sorted(SPECIAL_TYPES):
        ids = block_ids_by_type.get(block_type, set())
        lost = sorted(ids - source_covered)
        special_coverage[block_type] = {
            "total": len(ids),
            "covered": len(ids) - len(lost),
            "lost": len(lost),
            "lost_block_ids": lost,
        }

    special_binding = special_binding_analysis(chunks, blocks)
    dup = duplicate_source_analysis(chunks)
    deterministic = normalized_json_hash(chunks) == normalized_json_hash(regenerated)
    hard_fail_counts = {
        "chunks_over_1600": len(large_chunks),
        "multi_section_chunks": len(multi_section),
        "heading_only_chunks": len(heading_only),
        "section_mismatch_chunks": len(section_mismatch),
        "unexplained_repeated_source_block_ids": dup["unexplained_repeated_source_block_ids"],
        "split_metadata_invalid_count": dup["split_metadata_invalid_count"],
        "lost_core_blocks": sum(len(values) for values in lost_core.values()),
        "lost_equation_blocks": special_coverage["equation"]["lost"],
        "lost_caption_blocks": special_coverage["caption"]["lost"],
        "lost_table_blocks": special_coverage["table"]["lost"],
        "isolated_bindable_special_blocks": special_binding["isolated_bindable_special_count"],
        "caption_nearest_table_misses": special_binding["caption_nearest_table_miss_count"],
        "duplicate_chunk_ids": len(duplicate_chunk_ids),
        "references_overlap_chunks": len(refs_overlap),
        "overlap_cross_section_chunks": len(overlap_cross_section),
        "overlap_not_direct_previous_chunks": len(overlap_not_direct_previous),
        "overlap_over_max_chunks": len(overlap_over_max),
        "empty_text_chunks": len(empty_text),
        "char_count_invalid": len(char_count_invalid),
        "word_count_invalid": len(word_count_invalid),
        "page_order_invalid": len(page_order_invalid),
        "determinism_mismatch": 0 if deterministic else 1,
    }

    if any(hard_fail_counts.values()):
        conclusion = "FAIL"
    elif small_chunks or missing_heading_path:
        conclusion = "PASS_WITH_MINOR_ISSUES"
    else:
        conclusion = "PASS"

    metrics = {
        "paper": paper,
        "chunk_path": str(chunk_path),
        "chunk_count": len(chunks),
        "length_stats": length_stats(chunks),
        "section_counts": dict(Counter(str(chunk.get("section", "unknown")) for chunk in chunks)),
        "chunk_type_counts": dict(Counter(str(chunk.get("chunk_type", "unknown")) for chunk in chunks)),
        "small_chunks_lt_150": len(small_chunks),
        "large_chunks_gt_1600": len(large_chunks),
        "heading_only_chunks": len(heading_only),
        "multi_section_chunks": len(multi_section),
        "section_mismatch_chunks": len(section_mismatch),
        "cross_page_chunks": len(cross_page),
        "empty_text_chunks": len(empty_text),
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "source_duplicate_analysis": dup,
        "lost_core_blocks": {key: {"count": len(value), "block_ids": value} for key, value in lost_core.items()},
        "special_coverage": special_coverage,
        "special_binding": special_binding,
        "references_overlap_chunks": len(refs_overlap),
        "overlap_cross_section_chunks": len(overlap_cross_section),
        "overlap_not_direct_previous_chunks": len(overlap_not_direct_previous),
        "overlap_over_max_chunks": len(overlap_over_max),
        "heading_path_missing_non_abstract_non_refs": len(missing_heading_path),
        "char_count_invalid": len(char_count_invalid),
        "word_count_invalid": len(word_count_invalid),
        "page_order_invalid": len(page_order_invalid),
        "deterministic": deterministic,
        "stored_hash": normalized_json_hash(chunks),
        "regenerated_hash": normalized_json_hash(regenerated),
        "hard_fail_counts": hard_fail_counts,
        "conclusion": conclusion,
        "examples": {
            "small_chunks": sample_brief(small_chunks, 20),
            "large_chunks": sample_brief(large_chunks, 20),
            "heading_only": sample_brief(heading_only, 20),
            "multi_section": sample_brief(multi_section, 20),
            "section_mismatch": sample_brief(section_mismatch, 20),
        },
    }

    sample_sections = build_sample_markdown(paper, chunks)
    section_distribution_rows = [
        {"paper": paper, "section": section, "chunk_count": count}
        for section, count in sorted(Counter(str(chunk.get("section", "unknown")) for chunk in chunks).items())
    ]
    return metrics, audit_rows, sample_sections, section_distribution_rows


def sample_brief(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.get("chunk_id"),
            "section": chunk.get("section"),
            "char_count": chunk.get("char_count"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "source_block_ids": source_ids(chunk),
            "text_preview": str(chunk.get("text", ""))[:260],
        }
        for chunk in chunks[:limit]
    ]


def chunk_markdown(chunk: dict[str, Any]) -> str:
    meta = {
        key: chunk.get(key)
        for key in (
            "chunk_id",
            "section",
            "chunk_type",
            "char_count",
            "page_start",
            "page_end",
            "source_block_ids",
            "heading_block_ids",
            "heading_path",
            "split_reason",
            "split_part",
            "split_total",
            "split_blocks",
            "overlap_from_chunk_id",
            "has_equation",
            "has_caption",
            "has_table",
        )
    }
    return (
        "```json\n"
        + json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n\n"
        + "```text\n"
        + str(chunk.get("text", "")).strip()
        + "\n```\n"
    )


def build_sample_markdown(paper: str, chunks: list[dict[str, Any]]) -> list[str]:
    rng = random.Random(f"chunk-validation-v1:{paper}")
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(label: str, candidates: list[dict[str, Any]], limit: int | None = None, shuffle: bool = False) -> None:
        nonlocal selected, selected_ids
        pool = list(candidates)
        if shuffle:
            rng.shuffle(pool)
        if limit is not None:
            pool = pool[:limit]
        for chunk in pool:
            chunk_id = str(chunk.get("chunk_id"))
            if chunk_id not in selected_ids:
                cloned = dict(chunk)
                cloned["_sample_label"] = label
                selected.append(cloned)
                selected_ids.add(chunk_id)

    ordinary = [
        chunk
        for chunk in chunks
        if str(chunk.get("section")) not in {"references"}
        and not chunk.get("has_equation")
        and not chunk.get("has_table")
        and not chunk.get("has_caption")
        and MIN_SMALL_CHARS <= int(chunk.get("char_count", 0)) <= MAX_CHARS
    ]
    add("random ordinary", ordinary, limit=10, shuffle=True)
    for label, sections in TARGETED_SECTIONS.items():
        add(label, [chunk for chunk in chunks if str(chunk.get("section")) in sections], limit=3)
    add("table", [chunk for chunk in chunks if chunk.get("has_table")], limit=3)
    add("equation", [chunk for chunk in chunks if chunk.get("has_equation")], limit=3)
    add("caption", [chunk for chunk in chunks if chunk.get("has_caption")], limit=3)
    add("small <150", [chunk for chunk in chunks if int(chunk.get("char_count", 0)) < MIN_SMALL_CHARS])
    add("large >1600", [chunk for chunk in chunks if int(chunk.get("char_count", 0)) > MAX_CHARS])

    lines = [f"## {paper}", ""]
    for chunk in selected:
        lines.append(f"### {chunk.get('_sample_label')} / {chunk.get('chunk_id')}")
        lines.append("")
        lines.append(chunk_markdown(chunk))
        lines.append("")
    return lines


def overall_conclusion(paper_metrics: list[dict[str, Any]]) -> str:
    if any(metric["conclusion"] == "FAIL" for metric in paper_metrics):
        return "FAIL"
    if any(metric["conclusion"] == "PASS_WITH_MINOR_ISSUES" for metric in paper_metrics):
        return "PASS_WITH_MINOR_ISSUES"
    return "PASS"


def write_report(summary: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    lines = [
        "# Chunk Validation v1",
        "",
        f"Conclusion: **{summary['conclusion']}**",
        "",
        "## Synthetic Tests",
        "",
        f"- passed: `{summary.get('synthetic_tests', {}).get('passed')}`",
        f"- failure_count: `{summary.get('synthetic_tests', {}).get('failure_count')}`",
        f"- failures: `{summary.get('synthetic_tests', {}).get('failures', [])}`",
        f"- overlap_sentences=0 overlap count: `{summary.get('synthetic_tests', {}).get('overlap_0_count')}`",
        f"- overlap_sentences=1 second chunk overlap chars: `{summary.get('synthetic_tests', {}).get('overlap_1_second_chunk_chars')}`",
        f"- overlap_sentences=2 second chunk overlap chars: `{summary.get('synthetic_tests', {}).get('overlap_2_second_chunk_chars')}`",
        "",
        "| Test | Passed | Failures |",
        "| --- | --- | --- |",
    ]
    for test in summary.get("synthetic_tests", {}).get("tests", []):
        lines.append(f"| {test.get('name')} | `{test.get('passed')}` | `{test.get('failures', [])}` |")

    lines.extend([
        "",
        "## Per-paper Summary",
        "",
        "| Paper | Conclusion | Chunks | Min | P10 | Median | P90 | Max | <150 | >1600 | Multi-section | Heading-only | Lost core | Lost eq/cap/table |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in metrics:
        stats = item["length_stats"]
        lost_core = sum(value["count"] for value in item["lost_core_blocks"].values())
        lost_special = sum(value["lost"] for value in item["special_coverage"].values())
        lines.append(
            "| {paper} | {conclusion} | {chunk_count} | {min} | {p10} | {median} | {p90} | {max} | {small} | {large} | {multi} | {heading_only} | {lost_core} | {lost_special} |".format(
                paper=item["paper"],
                conclusion=item["conclusion"],
                chunk_count=item["chunk_count"],
                min=stats["min"],
                p10=stats["p10"],
                median=stats["median"],
                p90=stats["p90"],
                max=stats["max"],
                small=item["small_chunks_lt_150"],
                large=item["large_chunks_gt_1600"],
                multi=item["multi_section_chunks"],
                heading_only=item["heading_only_chunks"],
                lost_core=lost_core,
                lost_special=lost_special,
            )
        )

    lines.extend(["", "## Hard Checks", ""])
    for item in metrics:
        lines.append(f"### {item['paper']}")
        lines.append("")
        for key, value in item["hard_fail_counts"].items():
            lines.append(f"- {key}: {value}")
        lines.append(f"- deterministic: {item['deterministic']}")
        lines.append(f"- heading_path_missing_non_abstract_non_refs: {item['heading_path_missing_non_abstract_non_refs']}")
        lines.append("")

    (OUTPUT_DIR / "chunk_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_tests = validate_synthetic_tests()
    papers = sorted(path.name for path in PARSER_DIR.iterdir() if path.is_dir())
    all_metrics: list[dict[str, Any]] = []
    all_audit_rows: list[dict[str, Any]] = []
    all_sample_lines: list[str] = ["# Chunk Samples", ""]
    section_distribution: list[dict[str, Any]] = []

    for paper in papers:
        metrics, audit_rows, sample_lines, section_rows = validate_paper(paper)
        all_metrics.append(metrics)
        all_audit_rows.extend(audit_rows)
        all_sample_lines.extend(sample_lines)
        all_sample_lines.append("")
        section_distribution.extend(section_rows)

    summary = {
        "conclusion": "FAIL" if not synthetic_tests["passed"] else overall_conclusion(all_metrics),
        "paper_count": len(all_metrics),
        "synthetic_tests": synthetic_tests,
        "overlap_implementation": synthetic_tests,
        "papers": all_metrics,
        "hard_fail_totals": dict(
            Counter(
                {
                    key: sum(metric["hard_fail_counts"].get(key, 0) for metric in all_metrics)
                    for key in all_metrics[0]["hard_fail_counts"]
                }
            )
        )
        if all_metrics
        else {},
    }

    write_json(OUTPUT_DIR / "chunk_validation_summary.json", summary)
    write_jsonl(OUTPUT_DIR / "chunk_audit.jsonl", all_audit_rows)
    write_json(OUTPUT_DIR / "section_distribution.json", section_distribution)
    write_json(
        OUTPUT_DIR / "block_coverage_report.json",
        [
            {
                "paper": metric["paper"],
                "lost_core_blocks": metric["lost_core_blocks"],
                "special_coverage": metric["special_coverage"],
                "source_duplicate_analysis": metric["source_duplicate_analysis"],
            }
            for metric in all_metrics
        ],
    )
    (OUTPUT_DIR / "chunk_samples.md").write_text("\n".join(all_sample_lines), encoding="utf-8")
    write_report(summary, all_metrics)
    print(json.dumps({"conclusion": summary["conclusion"], "output_dir": str(OUTPUT_DIR)}, ensure_ascii=False, indent=2))
    return 0 if summary["conclusion"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
