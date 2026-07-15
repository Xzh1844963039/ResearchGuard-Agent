# C:\Users\18449\Desktop\researchguard_workspace\researchguard\reporting\audit_report.py
from __future__ import annotations

from typing import Any


def render_audit_markdown(question: str, answer: str, audit_result: dict[str, Any]) -> str:
    summary = audit_result.get("summary", {})
    records = audit_result.get("records", [])

    lines = [
        "# ResearchGuard Audit Report",
        "",
        "## Question",
        "",
        question.strip(),
        "",
        "## Generated Answer",
        "",
        answer.strip(),
        "",
        "## Audit Summary",
        "",
        f"- Total claims: {summary.get('total_claims', 0)}",
        f"- Supported: {summary.get('supported', 0)}",
        f"- Partial: {summary.get('partial', 0)}",
        f"- Unsupported: {summary.get('unsupported', 0)}",
        f"- Support rate: {summary.get('support_rate', 0.0)}",
        "",
        "## Claim-level Audit",
        "",
    ]

    for record in records:
        lines.extend(
            [
                f"### {record.get('claim_id', 'C???')} - {record.get('verdict', 'unknown')}",
                "",
                f"**Claim:** {record.get('claim', '')}",
                "",
                f"**Evidence IDs:** {', '.join(record.get('evidence_ids', [])) or 'None'}",
                "",
                f"**Overlap score:** {record.get('overlap_score', 0.0)}",
                "",
                f"**Risk type:** {record.get('risk_type', 'unknown')}",
                "",
                f"**Explanation:** {record.get('explanation', '')}",
                "",
            ]
        )

    return "\n".join(lines)