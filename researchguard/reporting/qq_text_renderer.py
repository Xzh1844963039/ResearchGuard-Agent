# C:\Users\18449\Desktop\researchguard_workspace\researchguard\reporting\qq_text_renderer.py
from __future__ import annotations

from collections import Counter

from researchguard.reporting.markdown_renderer import STATUS_EMOJI


def render_qq_summary(ctx: dict) -> str:
    reviews = ctx["reviews"]
    counter = Counter(review["status"] for review in reviews)
    risk = [r for r in reviews if r["color"] == "red"][:2]
    yellow = [r for r in reviews if r["color"] == "yellow"][:2]
    examples = risk or yellow or reviews[:2]
    lines = [
        f"EvidenceClaw 审查完成：{ctx['case_id']}",
        f"Claims: {len(reviews)}",
        f"🟢 Supported: {counter['supported']}",
        f"🟡 Inferred / Insufficient / Hypothesis: {counter['inferred'] + counter['insufficient'] + counter['hypothesis']}",
        f"🔴 Risk / Wrong: {sum(1 for r in reviews if r['color'] == 'red')}",
        "",
        "关键审查结果：",
    ]
    for review in examples:
        emoji = STATUS_EMOJI.get(review["status"], "⚪")
        eids = ", ".join(review.get("evidence_ids", [])) or "no evidence"
        lines.append(f"{review['claim_id']} {emoji} {review['status']}: {review['claim_text'][:70]} [{eids}]")
    lines.extend(["", f"完整报告：outputs/{ctx['case_id']}/final_report.md"])
    return "\n".join(lines)

