# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\filters.py
from __future__ import annotations

from typing import Any

from researchguard.retrieval.models import MetadataFilter


def metadata_matches(document: dict[str, Any], filters: MetadataFilter) -> bool:
    if filters.doc_ids and str(document.get("doc_id", "")) not in filters.doc_ids:
        return False
    if filters.sections and str(document.get("section", "")) not in filters.sections:
        return False
    if filters.chunk_types and str(document.get("chunk_type", "")) not in filters.chunk_types:
        return False
    if filters.exclude_references and str(document.get("section", "")) == "references":
        return False
    if filters.page_start_min is not None:
        page_end = document.get("page_end")
        if page_end is None or int(page_end) < int(filters.page_start_min):
            return False
    if filters.page_end_max is not None:
        page_start = document.get("page_start")
        if page_start is None or int(page_start) > int(filters.page_end_max):
            return False
    if filters.has_equation is not None and bool(document.get("has_equation")) is not bool(filters.has_equation):
        return False
    if filters.has_table is not None and bool(document.get("has_table")) is not bool(filters.has_table):
        return False
    if filters.has_caption is not None and bool(document.get("has_caption")) is not bool(filters.has_caption):
        return False
    return True


def apply_metadata_filters(documents: list[dict[str, Any]], filters: MetadataFilter) -> list[dict[str, Any]]:
    return [document for document in documents if metadata_matches(document, filters)]
