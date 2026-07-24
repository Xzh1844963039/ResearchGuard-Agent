# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tracing\formatter.py
from __future__ import annotations

import json
from typing import Any, Mapping

from researchguard.tracing.trace import AgentTrace


def format_trace_json(trace: AgentTrace, *, indent: int = 2) -> str:
    return trace.to_json(indent=indent)


def format_trace_markdown(trace: AgentTrace) -> str:
    rows = [
        "# ResearchGuard Agent Trace",
        "",
        f"- Run ID: `{trace.run_id}`",
        f"- Task: `{trace.task_type}`",
        f"- Workflow: `{trace.workflow_name or 'none'}`",
        f"- Status: `{trace.status}`",
        "",
        "## Timeline",
        "",
        "| Stage | Status | Summary | Latency (ms) |",
        "|---|---|---|---:|",
    ]
    for event in trace.timeline:
        summary = str(event.get("summary", "")).replace("|", "\\|")
        latency = event.get("latency_ms")
        rows.append(
            f"| {event.get('stage', '')} | {event.get('status', '')} | "
            f"{summary} | {latency if latency is not None else ''} |"
        )
    return "\n".join(rows) + "\n"


def trace_display_payload(trace: AgentTrace) -> Mapping[str, Any]:
    return json.loads(format_trace_json(trace))
