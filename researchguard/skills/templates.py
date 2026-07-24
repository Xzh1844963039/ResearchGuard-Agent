# C:\Users\18449\Desktop\researchguard_workspace\researchguard\skills\templates.py
from __future__ import annotations


WORKFLOW_SKILL_TEMPLATES: dict[str, tuple[str, ...]] = {
    "literature_review": (
        "search_papers",
        "retrieve_evidence",
        "analyze_claims",
        "generate_report",
    ),
    "paper_comparison": (
        "search_papers",
        "retrieve_evidence",
        "compare_evidence",
        "generate_report",
    ),
    "claim_audit": (
        "retrieve_evidence",
        "analyze_claims",
        "audit_claims",
    ),
}


TASK_SKILL_TEMPLATES: dict[str, tuple[str, ...]] = {
    "qa": ("retrieve_evidence", "generate_report"),
    "comparison": (
        "retrieve_evidence",
        "compare_evidence",
        "generate_report",
    ),
    "literature_search": ("search_papers",),
    **WORKFLOW_SKILL_TEMPLATES,
}
