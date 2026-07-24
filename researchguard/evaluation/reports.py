# C:\Users\18449\Desktop\researchguard_workspace\researchguard\evaluation\reports.py
from __future__ import annotations

from pathlib import Path

from researchguard.evaluation.schemas import AgentEvaluationReport


def render_evaluation_markdown(report: AgentEvaluationReport) -> str:
    lines = [
        "# ResearchGuard Agent Evaluation Report",
        "",
        f"- Generated: `{report.generated_at}`",
        f"- Cases: `{report.case_count}`",
        f"- Passed: `{report.passed_count}`",
        f"- Final status: `{'PASS' if report.passed else 'FAIL'}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in sorted(report.aggregate_metrics.items()):
        rendered = "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| `{name}` | {rendered} |")
    lines.extend(["", "## Cases", ""])
    for result in report.results:
        lines.extend(
            [
                f"### {result.case_id}",
                "",
                f"- Status: `{'PASS' if result.passed else 'FAIL'}`",
                f"- Issues: `{', '.join(result.issues) if result.issues else 'none'}`",
                f"- Task: `{result.observed.get('task_type')}`",
                f"- Workflow: `{result.observed.get('workflow') or 'none'}`",
                f"- Tools: `{', '.join(result.observed.get('tools', [])) or 'none'}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_evaluation_report(
    report: AgentEvaluationReport,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "agent_evaluation_report.json"
    markdown_path = target / "agent_evaluation_report.md"
    json_path.write_text(report.to_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_evaluation_markdown(report),
        encoding="utf-8",
        newline="\n",
    )
    return json_path, markdown_path
