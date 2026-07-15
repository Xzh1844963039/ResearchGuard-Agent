# C:\Users\18449\Desktop\researchguard_workspace\researchguard\ingestion\chunk_builder.py
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


TEXT_BLOCK_TYPES = {"paragraph", "heading_candidate", "reference_entry"}
ATTACHABLE_BLOCK_TYPES = {"caption", "table", "equation"}
BODY_BINDING_TYPES = {"paragraph", "reference_entry"}
CORE_COVERAGE_TYPES = TEXT_BLOCK_TYPES | {"heading"}


@dataclass
class ChunkDraft:
    doc_id: str
    title: str
    section: str
    parts: list[str] = field(default_factory=list)
    body_segments: list[dict[str, Any]] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    heading_block_ids: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    content_types: set[str] = field(default_factory=set)
    section_heading: str | None = None
    heading_path: list[str] = field(default_factory=list)
    chunk_type: str = "text"
    split_reason: str | None = None
    split_part: int | None = None
    split_total: int | None = None
    split_blocks: list[dict[str, Any]] = field(default_factory=list)
    short_chunk: bool = False
    short_chunk_reason: str | None = None
    overlap_from_chunk_id: str | None = None
    overlap_char_count: int = 0
    overlap_source_block_ids: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n\n".join(part.strip() for part in self.parts if part.strip()).strip()

    def char_count(self) -> int:
        return len(self.text())


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_to_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    protected = text
    protected = re.sub(r"\b(et al)\.", r"\1<prd>", protected, flags=re.I)
    protected = re.sub(r"\b(Fig|Eq|Sec|Ref|No|vol|pp)\.", r"\1<prd>", protected, flags=re.I)
    protected = re.sub(r"\b([A-Z])\.", r"\1<prd>", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|(?<=\n)\s*(?=[A-Z0-9(\[])", protected)
    return [part.replace("<prd>", ".").strip() for part in parts if part.replace("<prd>", ".").strip()]


def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    remaining = sentence.strip()

    while len(remaining) > max_chars:
        cut = max(
            remaining.rfind("; ", 0, max_chars),
            remaining.rfind(", ", 0, max_chars),
            remaining.rfind(" ", 0, max_chars),
        )
        if cut < max_chars * 0.45:
            cut = max_chars
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        pieces.append(remaining)

    return pieces


def split_text_by_sentences(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []

    for sentence in split_text_to_sentences(text):
        sentence_pieces = split_long_sentence(sentence, max_chars) if len(sentence) > max_chars else [sentence]
        for piece in sentence_pieces:
            if current and len(" ".join(current + [piece])) > max_chars:
                pieces.append(" ".join(current).strip())
                current = [piece]
            else:
                current.append(piece)

    if current:
        pieces.append(" ".join(current).strip())

    if not pieces and text.strip():
        return split_long_sentence(text.strip(), max_chars)

    return [piece for piece in pieces if piece]


def split_reference_entries(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    if re.search(r"(?m)^\s*\[\d+\]\s+", text):
        parts = re.split(r"(?m)(?=^\s*\[\d+\]\s+)", text)
        return [part.strip() for part in parts if part.strip()]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return [text]

    entries: list[str] = []
    current: list[str] = []
    entry_start = re.compile(
        r"^[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+(?:,| and | [A-Z]\.)|^[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+ et al\.",
    )
    year_pat = re.compile(r"\b(19|20)\d{2}[a-z]?\.")

    for line in lines:
        starts_new = bool(current and entry_start.search(line) and year_pat.search(line))
        if starts_new:
            entries.append(" ".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        entries.append(" ".join(current).strip())

    return entries or [text]


def split_reference_text(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    for entry in split_reference_entries(text):
        if len(entry) > max_chars:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
            chunks.extend(split_text_by_sentences(entry, max_chars=max_chars))
            continue

        prospective = "\n\n".join(current + [entry]).strip()
        if current and len(prospective) > max_chars:
            chunks.append("\n\n".join(current).strip())
            current = [entry]
        else:
            current.append(entry)

    if current:
        chunks.append("\n\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def make_chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}_chunk_{index:05d}"


def numbered_heading_level(text: str) -> int:
    stripped = text.strip()
    match = re.match(r"^(\d{1,2}(?:\.\d{1,2})*)\b", stripped)
    if match:
        return match.group(1).count(".") + 1
    match = re.match(r"^([A-Z])(?:\.(\d{1,2}(?:\.\d{1,2})*)?)?\b", stripped)
    if match and len(stripped.split()) <= 12:
        return 1 if not match.group(2) else match.group(2).count(".") + 2
    return 2


def update_heading_path(path: list[str], heading_text: str) -> list[str]:
    level = numbered_heading_level(heading_text)
    new_path = path[: max(level - 1, 0)]
    new_path.append(heading_text.strip())
    return new_path


def block_content_type(block_type: str) -> str:
    if block_type == "reference_entry":
        return "reference"
    if block_type in {"caption", "table", "equation"}:
        return block_type
    return "text"


def infer_chunk_type(section: str, content_types: set[str]) -> str:
    if section == "references":
        return "references"
    non_text = content_types - {"text", "reference"}
    if content_types == {"equation"}:
        return "equation"
    if content_types == {"table"} or content_types == {"caption", "table"}:
        return "table"
    if non_text:
        return "mixed"
    return "text"


def text_with_optional_prefix(prefix_parts: list[str], text: str) -> str:
    parts = [part for part in prefix_parts if part.strip()]
    if text.strip():
        parts.append(text.strip())
    return "\n\n".join(parts).strip()


def block_id(block: dict[str, Any]) -> str:
    return str(block.get("block_id", ""))


def block_order_key(block: dict[str, Any]) -> tuple[int, int, float, float, str]:
    return (
        int(block.get("page", 0)),
        int(block.get("column", 0)),
        float(block.get("y0", 0.0)),
        float(block.get("x0", 0.0)),
        block_id(block),
    )


def block_mid_y(block: dict[str, Any]) -> float:
    return (float(block.get("y0", 0.0)) + float(block.get("y1", block.get("y0", 0.0)))) / 2


def block_text_len(block: dict[str, Any]) -> int:
    return len(normalize_whitespace(str(block.get("text", ""))))


def unit_text(blocks: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        normalize_whitespace(str(block.get("text", "")))
        for block in blocks
        if normalize_whitespace(str(block.get("text", "")))
    ).strip()


def body_segments_from_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for block in blocks:
        text = normalize_whitespace(str(block.get("text", "")))
        bid = block_id(block)
        if text and bid:
            segments.append({"text": text, "source_block_ids": [bid]})
    return segments


def nearest_block(
    *,
    source_blocks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    order_index: dict[str, int],
    prefer_same_page: bool,
    max_combined_chars: int | None = None,
) -> dict[str, Any] | None:
    if not candidates:
        return None

    source_pages = {int(block.get("page", 0)) for block in source_blocks}
    source_sections = {str(block.get("section", "")) for block in source_blocks}
    source_mid = sum(block_mid_y(block) for block in source_blocks) / len(source_blocks)
    source_index = sum(order_index.get(block_id(block), 0) for block in source_blocks) / len(source_blocks)
    source_len = len(unit_text(source_blocks))

    filtered = [
        candidate
        for candidate in candidates
        if str(candidate.get("section", "")) in source_sections
        and (
            max_combined_chars is None
            or source_len + block_text_len(candidate) + 2 <= max_combined_chars
        )
    ]
    if not filtered:
        return None

    same_page = [candidate for candidate in filtered if int(candidate.get("page", 0)) in source_pages]
    pool = same_page if prefer_same_page and same_page else filtered

    return min(
        pool,
        key=lambda candidate: (
            0 if int(candidate.get("page", 0)) in source_pages else 1,
            abs(block_mid_y(candidate) - source_mid),
            abs(order_index.get(block_id(candidate), 0) - source_index),
            order_index.get(block_id(candidate), 0),
        ),
    )


def build_binding_units(blocks: list[dict[str, Any]], max_chars: int) -> list[list[dict[str, Any]]]:
    order_index = {block_id(block): index for index, block in enumerate(blocks)}
    by_id = {block_id(block): block for block in blocks}
    body_blocks = [
        block
        for block in blocks
        if str(block.get("block_type", "")) in BODY_BINDING_TYPES
        and normalize_whitespace(str(block.get("text", "")))
    ]
    table_blocks = [block for block in blocks if str(block.get("block_type", "")) == "table"]
    caption_blocks = [block for block in blocks if str(block.get("block_type", "")) == "caption"]
    equation_blocks = [block for block in blocks if str(block.get("block_type", "")) == "equation"]

    caption_by_table: dict[str, list[dict[str, Any]]] = {}
    caption_ids_bound_to_table: set[str] = set()
    for caption in caption_blocks:
        table = nearest_block(
            source_blocks=[caption],
            candidates=table_blocks,
            order_index=order_index,
            prefer_same_page=True,
            max_combined_chars=max_chars,
        )
        if table is not None:
            caption_by_table.setdefault(block_id(table), []).append(caption)
            caption_ids_bound_to_table.add(block_id(caption))

    special_groups: list[list[dict[str, Any]]] = []
    for table in table_blocks:
        group = [table] + caption_by_table.get(block_id(table), [])
        special_groups.append(sorted(group, key=lambda item: order_index[block_id(item)]))

    for caption in caption_blocks:
        if block_id(caption) not in caption_ids_bound_to_table:
            special_groups.append([caption])

    for equation in equation_blocks:
        special_groups.append([equation])

    groups_by_target: dict[str, list[dict[str, Any]]] = {}
    target_unit_chars: dict[str, int] = {}
    standalone_by_first_id: dict[str, list[dict[str, Any]]] = {}
    consumed_special_ids: set[str] = set()

    for group in special_groups:
        group = sorted(group, key=lambda item: order_index[block_id(item)])
        target = nearest_block(
            source_blocks=group,
            candidates=body_blocks,
            order_index=order_index,
            prefer_same_page=True,
            max_combined_chars=max_chars,
        )
        if target is not None:
            target_id = block_id(target)
            current_chars = target_unit_chars.setdefault(target_id, block_text_len(target))
            group_chars = len(unit_text(group))
            if current_chars + group_chars + 2 <= max_chars:
                groups_by_target.setdefault(target_id, []).extend(group)
                target_unit_chars[target_id] = current_chars + group_chars + 2
            else:
                standalone_by_first_id[block_id(group[0])] = group
        else:
            standalone_by_first_id[block_id(group[0])] = group
        consumed_special_ids.update(block_id(block) for block in group)

    units: list[list[dict[str, Any]]] = []
    emitted_standalone: set[str] = set()
    for block in blocks:
        bid = block_id(block)
        if bid in groups_by_target:
            unit = [block] + groups_by_target[bid]
            units.append(sorted(unit, key=lambda item: order_index[block_id(item)]))
            continue

        if bid in standalone_by_first_id and bid not in emitted_standalone:
            group = standalone_by_first_id[bid]
            units.append(group)
            emitted_standalone.update(block_id(item) for item in group)
            continue

        if bid in consumed_special_ids:
            continue

        if bid in emitted_standalone:
            continue

        units.append([block])

    return units


def max_piece_size(max_chars: int, heading_path: list[str], include_heading_prefix: bool) -> int:
    if not include_heading_prefix or not heading_path:
        return max_chars
    prefix = "\n\n".join(heading_path)
    return max(200, max_chars - len(prefix) - 2)


def split_bound_unit_for_budget(unit: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if len(unit) <= 1:
        return [unit]
    return [[block] for block in unit]


def create_draft(
    *,
    doc_id: str,
    title: str,
    section: str,
    text: str,
    block: dict[str, Any],
    heading_path: list[str],
    pending_heading_ids: list[str],
    include_heading_prefix: bool,
    split_reason: str | None = None,
    split_part: int | None = None,
    split_total: int | None = None,
) -> ChunkDraft:
    block_type = str(block.get("block_type", "paragraph"))
    content_type = block_content_type(block_type)
    prefix_parts = heading_path if include_heading_prefix else []
    final_text = text_with_optional_prefix(prefix_parts, text)
    page = int(block.get("page", 0))

    draft = ChunkDraft(
        doc_id=doc_id,
        title=title,
        section=section,
        parts=[final_text],
        body_segments=[{"text": text, "source_block_ids": [str(block.get("block_id", ""))]}],
        source_block_ids=[str(block.get("block_id", ""))],
        heading_block_ids=list(pending_heading_ids) if include_heading_prefix else [],
        pages=[page],
        content_types={content_type},
        section_heading=heading_path[-1] if heading_path else None,
        heading_path=list(heading_path),
        split_reason=split_reason,
        split_part=split_part,
        split_total=split_total,
        split_blocks=[
            {
                "block_id": str(block.get("block_id", "")),
                "split_reason": split_reason,
                "split_part": split_part,
                "split_total": split_total,
            }
        ]
        if split_reason and split_part and split_total
        else [],
    )
    draft.chunk_type = infer_chunk_type(section, draft.content_types)
    return draft


def create_draft_from_unit(
    *,
    doc_id: str,
    title: str,
    section: str,
    blocks: list[dict[str, Any]],
    heading_path: list[str],
    pending_heading_ids: list[str],
    include_heading_prefix: bool,
) -> ChunkDraft:
    first = blocks[0]
    prefix_parts = heading_path if include_heading_prefix else []
    text = text_with_optional_prefix(prefix_parts, unit_text(blocks))
    content_types = {block_content_type(str(block.get("block_type", "paragraph"))) for block in blocks}
    source_ids = [block_id(block) for block in blocks]
    pages = [int(block.get("page", 0)) for block in blocks]

    draft = ChunkDraft(
        doc_id=doc_id,
        title=title,
        section=section,
        parts=[text],
        body_segments=body_segments_from_blocks(blocks),
        source_block_ids=source_ids,
        heading_block_ids=list(pending_heading_ids) if include_heading_prefix else [],
        pages=pages,
        content_types=content_types,
        section_heading=heading_path[-1] if heading_path else None,
        heading_path=list(heading_path),
    )
    draft.chunk_type = infer_chunk_type(section, draft.content_types)
    if not draft.pages:
        draft.pages = [int(first.get("page", 0))]
    return draft


def append_to_draft(draft: ChunkDraft, other: ChunkDraft) -> None:
    draft.parts.extend(other.parts)
    draft.body_segments.extend(other.body_segments)
    draft.source_block_ids.extend(other.source_block_ids)
    draft.heading_block_ids.extend([bid for bid in other.heading_block_ids if bid not in draft.heading_block_ids])
    draft.pages.extend(other.pages)
    draft.content_types.update(other.content_types)
    draft.split_blocks.extend(other.split_blocks)
    if other.heading_path:
        draft.heading_path = other.heading_path
        draft.section_heading = other.section_heading
    draft.chunk_type = infer_chunk_type(draft.section, draft.content_types)


def can_merge(left: ChunkDraft, right: ChunkDraft, max_chars: int) -> bool:
    if left.section != right.section:
        return False
    return len("\n\n".join([left.text(), right.text()]).strip()) <= max_chars


def merge_drafts(left: ChunkDraft, right: ChunkDraft) -> ChunkDraft:
    append_to_draft(left, right)
    return left


def chunk_from_blocks(
    *,
    doc_id: str,
    title: str,
    blocks: list[dict[str, Any]],
    max_chars: int,
    target_chars: int,
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    current: ChunkDraft | None = None
    active_section: str | None = None
    heading_path: list[str] = []
    pending_heading_ids: list[str] = []
    heading_prefix_consumed = False
    orphan_heading_ids: list[str] = []

    def flush_current() -> None:
        nonlocal current
        if current and current.text():
            current.chunk_type = infer_chunk_type(current.section, current.content_types)
            drafts.append(current)
        current = None

    def flush_section() -> None:
        nonlocal heading_path, pending_heading_ids, heading_prefix_consumed, orphan_heading_ids
        flush_current()
        if pending_heading_ids and drafts:
            drafts[-1].heading_block_ids.extend([bid for bid in pending_heading_ids if bid not in drafts[-1].heading_block_ids])
        elif pending_heading_ids:
            orphan_heading_ids.extend(pending_heading_ids)
        heading_path = []
        pending_heading_ids = []
        heading_prefix_consumed = False

    units = build_binding_units(blocks, max_chars=max_chars)
    unit_index = 0
    while unit_index < len(units):
        unit = units[unit_index]
        block = unit[0]
        block_type = str(block.get("block_type", "paragraph"))
        section = str(block.get("section", "main_text") or "main_text")
        text = unit_text(unit)
        block_id = str(block.get("block_id", ""))
        if not text:
            unit_index += 1
            continue

        if active_section is None:
            active_section = section
        elif section != active_section:
            flush_section()
            active_section = section

        if len(unit) == 1 and block_type == "heading":
            heading_path = update_heading_path(heading_path, text)
            if block_id:
                pending_heading_ids.append(block_id)
            heading_prefix_consumed = False
            unit_index += 1
            continue

        if len(unit) == 1 and block_type == "heading_candidate" and heading_path and (
            len(text) <= 80 or re.fullmatch(r"[A-Z](?:\.\d+)*", text)
        ):
            if re.fullmatch(r"[A-Z](?:\.\d+)*", text):
                heading_path = update_heading_path(heading_path, text)
            else:
                heading_path[-1] = f"{heading_path[-1]} {text}".strip()
            if block_id:
                pending_heading_ids.append(block_id)
            heading_prefix_consumed = False
            unit_index += 1
            continue

        include_heading = bool(heading_path and not heading_prefix_consumed)
        if (
            len(unit) > 1
            and include_heading
            and len(text_with_optional_prefix(heading_path, text)) > max_chars
        ):
            replacement_units = split_bound_unit_for_budget(unit)
            if replacement_units != [unit]:
                units[unit_index : unit_index + 1] = replacement_units
                continue

        available = max_piece_size(max_chars, heading_path, include_heading)

        if len(unit) > 1:
            pieces = [text]
        elif section == "references":
            pieces = split_reference_text(text, max_chars=available)
        elif len(text_with_optional_prefix(heading_path if include_heading else [], text)) > max_chars:
            pieces = split_text_by_sentences(text, max_chars=available)
        else:
            pieces = [text]

        split_total = len(pieces)
        for index, piece in enumerate(pieces, start=1):
            use_heading = include_heading and index == 1
            if len(unit) > 1:
                draft = create_draft_from_unit(
                    doc_id=doc_id,
                    title=title,
                    section=section,
                    blocks=unit,
                    heading_path=heading_path,
                    pending_heading_ids=pending_heading_ids,
                    include_heading_prefix=use_heading,
                )
            else:
                draft = create_draft(
                    doc_id=doc_id,
                    title=title,
                    section=section,
                    text=piece,
                    block=block,
                    heading_path=heading_path,
                    pending_heading_ids=pending_heading_ids,
                    include_heading_prefix=use_heading,
                    split_reason="block_exceeds_max_chars" if split_total > 1 else None,
                    split_part=index if split_total > 1 else None,
                    split_total=split_total if split_total > 1 else None,
                )

            if orphan_heading_ids:
                draft.heading_block_ids.extend([bid for bid in orphan_heading_ids if bid not in draft.heading_block_ids])
                orphan_heading_ids = []

            if current is None:
                current = draft
            elif can_merge(current, draft, max_chars) and (
                current.char_count() < target_chars
                or bool(draft.content_types & {"caption", "table", "equation"})
            ):
                append_to_draft(current, draft)
            else:
                flush_current()
                current = draft

            if use_heading:
                pending_heading_ids = []
                heading_prefix_consumed = True

        unit_index += 1

    flush_section()
    return drafts


def merge_short_chunks(drafts: list[ChunkDraft], *, max_chars: int, min_chars: int) -> list[ChunkDraft]:
    merged = list(drafts)
    changed = True

    while changed:
        changed = False
        i = 0
        while i < len(merged):
            draft = merged[i]
            if draft.char_count() >= min_chars:
                i += 1
                continue

            if i > 0 and can_merge(merged[i - 1], draft, max_chars):
                merge_drafts(merged[i - 1], draft)
                merged.pop(i)
                changed = True
                continue

            if i + 1 < len(merged) and can_merge(draft, merged[i + 1], max_chars):
                merge_drafts(draft, merged[i + 1])
                merged.pop(i + 1)
                changed = True
                continue

            i += 1

    for draft in merged:
        if draft.char_count() < min_chars:
            draft.short_chunk = True
            draft.short_chunk_reason = (
                "no same-section neighbor can merge without exceeding max_chars"
                if draft.char_count() > 0
                else "empty after normalization"
            )

    return merged


def last_sentences_for_overlap(text: str, sentence_count: int) -> str:
    if sentence_count <= 0:
        return ""

    sentences = split_text_to_sentences(text)
    if not sentences:
        return ""

    selected = sentences[-sentence_count:]
    return " ".join(sentence.strip() for sentence in selected if sentence.strip()).strip()


def overlap_sentence_records(draft: ChunkDraft, sentence_count: int) -> list[dict[str, Any]]:
    if sentence_count <= 0:
        return []

    records: list[dict[str, Any]] = []
    for segment in draft.body_segments:
        text = str(segment.get("text", "")).strip()
        source_ids = [str(item) for item in segment.get("source_block_ids", []) if str(item).strip()]
        for sentence in split_text_to_sentences(text):
            if sentence.strip():
                records.append({"text": sentence.strip(), "source_block_ids": source_ids})

    return records[-sentence_count:]


def apply_overlap(drafts: list[ChunkDraft], *, max_chars: int, overlap_sentences: int) -> None:
    if overlap_sentences <= 0:
        return

    previous: ChunkDraft | None = None
    previous_id: str | None = None

    for index, draft in enumerate(drafts, start=1):
        if (
            previous is not None
            and previous_id is not None
            and draft.section != "references"
            and previous.section == draft.section
        ):
            overlap_records = overlap_sentence_records(previous, overlap_sentences)
            overlap_text = " ".join(record["text"] for record in overlap_records if record.get("text")).strip()
            if overlap_text and len(overlap_text) >= 30:
                prospective = f"{overlap_text}\n\n{draft.text()}".strip()
                if len(prospective) <= max_chars:
                    draft.parts.insert(0, overlap_text)
                    draft.overlap_from_chunk_id = previous_id
                    draft.overlap_char_count = len(overlap_text)
                    draft.overlap_source_block_ids = stable_unique(
                        [
                            block_id
                            for record in overlap_records
                            for block_id in record.get("source_block_ids", [])
                        ]
                    )

        previous = draft
        previous_id = make_chunk_id(draft.doc_id, index)


def draft_to_row(draft: ChunkDraft, chunk_id: str) -> dict[str, Any]:
    text = draft.text()
    source_block_ids = stable_unique(draft.source_block_ids)
    heading_block_ids = stable_unique(draft.heading_block_ids)
    content_types = sorted(draft.content_types)
    equation_ids = [bid for bid in source_block_ids if bid and bid_type_marker(bid, draft, "equation")]
    table_ids = [bid for bid in source_block_ids if bid and bid_type_marker(bid, draft, "table")]
    caption_ids = [bid for bid in source_block_ids if bid and bid_type_marker(bid, draft, "caption")]

    return {
        "chunk_id": chunk_id,
        "doc_id": draft.doc_id,
        "title": draft.title,
        "section": draft.section,
        "section_heading": draft.section_heading,
        "heading_path": draft.heading_path,
        "heading_block_ids": heading_block_ids,
        "chunk_type": draft.chunk_type,
        "page_start": min(draft.pages) if draft.pages else None,
        "page_end": max(draft.pages) if draft.pages else None,
        "source_block_ids": source_block_ids,
        "block_ids": source_block_ids,
        "overlap_source_block_ids": stable_unique(draft.overlap_source_block_ids),
        "content_types": content_types,
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split()),
        "has_equation": "equation" in draft.content_types,
        "equation_block_ids": equation_ids,
        "has_table": "table" in draft.content_types,
        "table_block_ids": table_ids,
        "has_caption": "caption" in draft.content_types,
        "caption_block_ids": caption_ids,
        "short_chunk": draft.short_chunk,
        "short_chunk_reason": draft.short_chunk_reason,
        "split_reason": draft.split_reason,
        "split_part": draft.split_part,
        "split_total": draft.split_total,
        "split_blocks": draft.split_blocks,
        "overlap_from_chunk_id": draft.overlap_from_chunk_id,
        "overlap_char_count": draft.overlap_char_count,
    }


def stable_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def bid_type_marker(block_id: str, draft: ChunkDraft, block_type: str) -> bool:
    marker = f"__{block_type}__"
    return marker in block_id


def annotate_special_block_ids(rows: list[dict[str, Any]], block_type_by_id: dict[str, str]) -> None:
    for row in rows:
        source_ids = row.get("source_block_ids", [])
        row["equation_block_ids"] = [bid for bid in source_ids if block_type_by_id.get(bid) == "equation"]
        row["table_block_ids"] = [bid for bid in source_ids if block_type_by_id.get(bid) == "table"]
        row["caption_block_ids"] = [bid for bid in source_ids if block_type_by_id.get(bid) == "caption"]
        row["has_equation"] = bool(row["equation_block_ids"])
        row["has_table"] = bool(row["table_block_ids"])
        row["has_caption"] = bool(row["caption_block_ids"])
        row["content_types"] = sorted(
            {
                block_content_type(block_type_by_id.get(bid, "paragraph"))
                for bid in source_ids
            }
        )
        row["chunk_type"] = infer_chunk_type(str(row.get("section", "")), set(row["content_types"]))


def build_chunks(
    *,
    doc_id: str,
    title: str,
    blocks: list[dict[str, Any]],
    max_chars: int = 1600,
    target_chars: int = 1200,
    min_chars: int = 250,
    overlap_sentences: int = 1,
) -> list[dict[str, Any]]:
    ordered_blocks = sorted(
        blocks,
        key=lambda item: (
            int(item.get("page", 0)),
            int(item.get("column", 0)),
            float(item.get("y0", 0.0)),
            float(item.get("x0", 0.0)),
            str(item.get("block_id", "")),
        ),
    )
    drafts = chunk_from_blocks(
        doc_id=doc_id,
        title=title,
        blocks=ordered_blocks,
        max_chars=max_chars,
        target_chars=target_chars,
    )
    drafts = merge_short_chunks(drafts, max_chars=max_chars, min_chars=min_chars)
    apply_overlap(drafts, max_chars=max_chars, overlap_sentences=overlap_sentences)

    rows: list[dict[str, Any]] = []
    block_type_by_id = {str(block.get("block_id", "")): str(block.get("block_type", "paragraph")) for block in ordered_blocks}
    for index, draft in enumerate(drafts, start=1):
        rows.append(draft_to_row(draft, make_chunk_id(doc_id, index)))

    annotate_special_block_ids(rows, block_type_by_id)
    return rows


def chunk_length_stats(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = sorted(int(chunk.get("char_count", 0)) for chunk in chunks)
    if not lengths:
        return {"min": 0, "p10": 0, "median": 0, "p90": 0, "max": 0, "avg": 0}

    def pct(q: float) -> int:
        return lengths[round((len(lengths) - 1) * q)]

    return {
        "min": lengths[0],
        "p10": pct(0.10),
        "median": lengths[len(lengths) // 2],
        "p90": pct(0.90),
        "max": lengths[-1],
        "avg": round(sum(lengths) / len(lengths), 2),
    }


def summarize_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    section_counts = Counter(str(chunk.get("section", "unknown")) for chunk in chunks)
    type_counts = Counter(str(chunk.get("chunk_type", "unknown")) for chunk in chunks)
    return {
        "chunk_count": len(chunks),
        "length_stats": chunk_length_stats(chunks),
        "section_counts": dict(section_counts),
        "chunk_type_counts": dict(type_counts),
        "equation_chunks": sum(1 for chunk in chunks if chunk.get("has_equation")),
        "table_chunks": sum(1 for chunk in chunks if chunk.get("has_table")),
        "caption_chunks": sum(1 for chunk in chunks if chunk.get("has_caption")),
    }
