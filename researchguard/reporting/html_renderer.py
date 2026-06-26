# C:\Users\18449\Desktop\researchguard_workspace\researchguard\reporting\html_renderer.py
from __future__ import annotations

import html

from researchguard.reporting.markdown_renderer import STATUS_EMOJI


COLOR = {"green": "green", "yellow": "#b58900", "red": "red", "gray": "gray"}


def render_html_report(ctx: dict) -> str:
    items = []
    for review in ctx["reviews"]:
        color = COLOR.get(review["color"], "gray")
        eids = " ".join(f"<sup>[{html.escape(eid)}]</sup>" for eid in review.get("evidence_ids", []))
        emoji = STATUS_EMOJI.get(review["status"], "⚪")
        items.append(
            f'<li><span style="color:{color}"><strong>{emoji} {html.escape(review["status"])}</strong></span>: '
            f'{html.escape(review["claim_text"])} {eids}<br><small>{html.escape(review["explanation"])}</small></li>'
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>EvidenceClaw Final Report - {html.escape(ctx['case_id'])}</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.55; max-width: 1100px; margin: 32px auto; padding: 0 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
    pre {{ white-space: pre-wrap; background: #f7f7f7; padding: 12px; }}
  </style>
</head>
<body>
  <h1>EvidenceClaw Final Report</h1>
  <h2>Task Information</h2>
  <ul>
    <li>Case ID: {html.escape(ctx['case_id'])}</li>
    <li>Topic: {html.escape(ctx['topic'])}</li>
    <li>Source: {html.escape(ctx['source_path'])}</li>
  </ul>
  <h2>First-stage Model Output</h2>
  <pre>{html.escape(ctx['generated_report'])}</pre>
  <h2>Reviewed Output with Evidence Marks</h2>
  <ul>{''.join(items)}</ul>
  <p>See <code>final_report.md</code> for full Claim Review Table, Evidence Table, Memory Table, and Tool Trace Summary.</p>
</body>
</html>
"""

