# C:\Users\18449\Desktop\researchguard_workspace\researchguard\reporting\markdown_renderer.py
from __future__ import annotations

from collections import defaultdict


STATUS_EMOJI = {
    "supported": "🟢",
    "inferred": "🟡",
    "insufficient": "🟡",
    "hypothesis": "🟡",
    "contradicted": "🔴",
    "invalid_citation": "🔴",
    "mis_citation": "🔴",
    "number_error": "🔴",
    "scope_overclaim": "🔴",
}


def render_markdown_report(ctx: dict) -> str:
    reviews = ctx["reviews"]
    evidence_records = ctx["evidence_records"]
    claims = {claim["claim_id"]: claim for claim in ctx["claims"]}
    used_by = defaultdict(list)
    for review in reviews:
        for eid in review.get("evidence_ids", []):
            used_by[eid].append(review["claim_id"])

    lines = [
        "# EvidenceClaw Final Report",
        "",
        "## 1. Task Information",
        "",
        f"* Case ID: `{ctx['case_id']}`",
        f"* Source PDF / Source File: `{ctx['source_path']}`",
        f"* Research Topic: {ctx['topic']}",
        "",
        "## 2. First-stage Model Output",
        "",
        ctx["generated_report"],
        "",
        "## 3. Reviewed Output with Evidence Marks",
        "",
    ]
    for review in reviews:
        emoji = STATUS_EMOJI.get(review["status"], "⚪")
        evidence = ", ".join(f"<sup>[{eid}]</sup>" for eid in review.get("evidence_ids", [])) or "<sup>[no evidence]</sup>"
        lines.append(f"- {emoji} **{review['status']}**: {review['claim_text']} {evidence}")
        lines.append(f"  - Explanation: {review['explanation']}")
    lines.extend(["", "## 4. Claim Review Table", "", "| Claim ID | Claim | Status | Color | Evidence IDs | Counter Evidence | Explanation |", "|---|---|---|---|---|---|---|"])
    for review in reviews:
        lines.append(
            "| {claim_id} | {claim} | {status} | {color} | {eids} | {ceids} | {explanation} |".format(
                claim_id=review["claim_id"],
                claim=escape_table(review["claim_text"]),
                status=review["status"],
                color=review["color"],
                eids=", ".join(review.get("evidence_ids", [])),
                ceids=", ".join(review.get("counter_evidence_ids", [])),
                explanation=escape_table(review["explanation"]),
            )
        )

    lines.extend(["", "## 5. Evidence Table", "", "| Evidence ID | Source | Location | Content Summary | Used By Claim |", "|---|---|---|---|---|"])
    for evidence in evidence_records:
        lines.append(f"| {evidence['evidence_id']} | {evidence['source_name']} | {escape_table(evidence['location_text'])} | {escape_table(evidence['content_summary'])} | {', '.join(used_by[evidence['evidence_id']])} |")

    lines.extend(["", "## 6. Citation Check Table", ""])
    if ctx["citation_checks"]:
        lines.extend(["| Citation | Exists in Input | Supports Claim | Status | Explanation |", "|---|---|---|---|---|"])
        for citation in ctx["citation_checks"]:
            lines.append(f"| {escape_table(citation['raw_citation_text'])} | {citation['exists_in_input_references']} | {citation.get('supports_claim')} | {citation['status']} | matched reference: {escape_table(citation.get('matched_reference_text') or '')} |")
    else:
        lines.append("No explicit citation detected.")

    lines.extend(["", "## 7. Number / Scope / Strength Check Summary", "", "| Claim ID | Number Check | Scope Check | Strength Check |", "|---|---|---|---|"])
    for row in ctx["number_scope_strength"]:
        lines.append(f"| {row['claim_id']} | {row['number_check']['status']} | {row['scope_check']['status']} | {row['strength_check']['status']} |")

    coverage = ctx["coverage"]
    lines.extend(["", "## 8. Evidence Coverage Score", "", "```json", json_block(coverage), "```"])

    snapshot = ctx["memory_snapshot"]
    lines.extend(
        [
            "",
            "## 9. Memory Table",
            "",
            "| Memory Type | Records Added | File | Purpose |",
            "|---|---:|---|---|",
            f"| Source Memory | {snapshot['source_memory_count']} | memory/source_memory.json | 保存输入论文来源 |",
            f"| Evidence Memory | {snapshot['evidence_memory_count']} | memory/evidence_memory.json | 保存证据条目 |",
            f"| Review Memory | {snapshot['review_memory_count']} | memory/review_memory.json | 保存 claim 审查结果 |",
            f"| Tool Trace Memory | {snapshot['tool_trace_count']} | memory/tool_trace.jsonl | 保存工具调用日志 |",
            f"| Failure Memory | {snapshot['failure_memory_count']} | memory/failure_memory.json | 保存错误或边界情况 |",
            "",
            "## 10. Tool Trace Summary",
            "",
            "| Tool | Status | Output | Duration ms |",
            "|---|---|---|---:|",
        ]
    )
    for trace in ctx["tool_traces"]:
        lines.append(f"| {trace['tool_name']} | {trace['status']} | {escape_table(trace['output_summary'])} | {trace['duration_ms']} |")

    lines.extend(
        [
            "",
            "## 11. Limitations of This Run",
            "",
            "- This system cannot fully prove scientific truth automatically.",
            "- PDF parsing may miss figures, tables, equations, or layout-dependent context.",
            "- Citation existence does not guarantee that the citation supports the claim.",
            "- LLM-generated research ideas are hypotheses, not paper-proven conclusions.",
            "- Red/yellow/green labels support human review and do not replace expert peer review.",
        ]
    )
    return "\n".join(lines) + "\n"


def escape_table(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def json_block(data: dict) -> str:
    import json

    return json.dumps(data, ensure_ascii=False, indent=2)

