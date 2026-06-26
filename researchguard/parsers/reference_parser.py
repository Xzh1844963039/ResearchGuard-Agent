# C:\Users\18449\Desktop\researchguard_workspace\researchguard\parsers\reference_parser.py
from __future__ import annotations

import re


def extract_reference_section(text: str) -> str:
    match = re.search(r"(references|参考文献)\s*(.*)$", text or "", flags=re.I | re.S)
    return match.group(2).strip() if match else ""


def citation_patterns(text: str) -> list[str]:
    patterns = []
    patterns.extend(re.findall(r"\[[0-9]{1,3}\]", text or ""))
    patterns.extend(re.findall(r"\b[A-Z][A-Za-z-]+ et al\.?\s*\(\d{4}\)", text or ""))
    patterns.extend(re.findall(r"\([A-Z][A-Za-z-]+,\s*\d{4}\)", text or ""))
    return sorted(set(patterns))

