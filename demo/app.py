# C:\Users\18449\Desktop\researchguard_workspace\demo\app.py
from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.utils import (
    STAGE_LABELS,
    answer_view,
    audit_claims,
    audit_summary,
    evidence_sufficiency_view,
    evidence_view,
    final_status_view,
    retrieval_hits,
    safe_error_message,
    sanitize_for_display,
    stage_rows,
    validate_pipeline_result,
)
from researchguard.pipeline import ResearchGuardPipeline


PIPELINE_CONFIG = PROJECT_ROOT / "configs" / "pipeline_v1.yaml"
EXAMPLE_QUERIES = (
    "How does CRAG reduce hallucination?",
    "What is the difference between RAG-Sequence and RAG-Token?",
    "Does any indexed paper describe quantum error correction for superconducting qubits?",
)


PAGE_CSS = """
<style>
    :root {
        --rg-ink: #18201c;
        --rg-muted: #657069;
        --rg-line: #d9dedb;
        --rg-paper: #f7f8f6;
        --rg-green: #16704a;
        --rg-green-soft: #e7f3ec;
        --rg-amber: #9a5b12;
        --rg-amber-soft: #fff2dc;
        --rg-red: #a33a32;
        --rg-red-soft: #fbe9e7;
        --rg-blue: #216789;
    }
    .stApp { background: #ffffff; color: var(--rg-ink); }
    [data-testid="stHeader"] { background: rgba(255,255,255,.92); }
    [data-testid="stSidebar"] { background: var(--rg-paper); border-right: 1px solid var(--rg-line); }
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span { color: var(--rg-ink) !important; }
    .block-container { max-width: 1180px; padding-top: 2.25rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: var(--rg-ink); letter-spacing: 0; }
    h1 { font-size: 2.25rem !important; font-weight: 720 !important; margin-bottom: .15rem !important; }
    h2 { font-size: 1.35rem !important; margin-top: 2rem !important; }
    h3 { font-size: 1.05rem !important; }
    p, label, li { letter-spacing: 0; }
    .rg-subtitle { color: var(--rg-muted); font-size: 1rem; margin: 0 0 1.65rem 0; }
    .rg-rule { border-top: 1px solid var(--rg-line); margin: 0 0 1.5rem 0; }
    .rg-final { border-left: 4px solid var(--rg-green); background: var(--rg-green-soft); padding: .85rem 1rem; border-radius: 4px; }
    .rg-final.warning { border-left-color: var(--rg-amber); background: var(--rg-amber-soft); }
    .rg-final.error { border-left-color: var(--rg-red); background: var(--rg-red-soft); }
    .rg-final.neutral { border-left-color: var(--rg-blue); background: #eaf3f7; }
    .rg-final-label { color: var(--rg-muted); font-size: .72rem; font-weight: 700; text-transform: uppercase; }
    .rg-final-value { color: var(--rg-ink); font-size: 1.05rem; font-weight: 700; margin-top: .15rem; }
    .rg-stage-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .65rem; margin: .5rem 0 1.25rem; }
    .rg-stage { border: 1px solid var(--rg-line); border-radius: 6px; padding: .75rem .8rem; min-height: 82px; background: #fff; }
    .rg-stage-top { display: flex; align-items: center; justify-content: space-between; gap: .5rem; }
    .rg-stage-name { font-size: .86rem; font-weight: 700; color: var(--rg-ink); }
    .rg-stage-status { font-size: .68rem; font-weight: 750; text-transform: uppercase; color: var(--rg-muted); }
    .rg-stage-status.completed { color: var(--rg-green); }
    .rg-stage-status.fallback, .rg-stage-status.skipped { color: var(--rg-amber); }
    .rg-stage-status.failed { color: var(--rg-red); }
    .rg-stage-latency { color: var(--rg-muted); font-size: .73rem; margin-top: .8rem; }
    .rg-section-kicker { color: var(--rg-muted); font-size: .72rem; font-weight: 700; text-transform: uppercase; margin-bottom: .2rem; }
    .rg-answer { font-size: 1.02rem; line-height: 1.72; border-left: 3px solid var(--rg-blue); padding-left: 1rem; margin: .75rem 0 1.25rem; }
    .rg-answer.refused { border-left-color: var(--rg-amber); color: #68420f; }
    .rg-claim-status { font-size: .72rem; font-weight: 750; text-transform: uppercase; }
    .rg-claim-status.supported { color: var(--rg-green); }
    .rg-claim-status.partial { color: var(--rg-amber); }
    .rg-claim-status.unsupported { color: var(--rg-red); }
    div[data-testid="stExpander"] { border-color: var(--rg-line); border-radius: 6px; }
    div[data-testid="stButton"] > button { border-radius: 5px; min-height: 2.65rem; font-weight: 700; }
    div[data-testid="stButton"] > button[kind="primary"] { background: var(--rg-green); border-color: var(--rg-green); color: #fff; }
    div[data-testid="stButton"] > button[kind="primary"] p { color: #fff !important; }
    div[data-testid="stButton"] > button[kind="secondary"] { background: #fff; border-color: #aeb7b2; color: var(--rg-ink); }
    div[data-testid="stTextArea"] textarea { border-radius: 5px; background: #fff; color: var(--rg-ink); caret-color: var(--rg-ink); }
    div[data-testid="stTextArea"] textarea::placeholder { color: #7b8580; }
    [data-baseweb="select"] > div { background: #fff; border-color: #aeb7b2; color: var(--rg-ink); }
    [data-baseweb="select"] svg { fill: var(--rg-ink); }
    div[data-testid="stMetric"] { background: var(--rg-paper); border: 1px solid var(--rg-line); border-radius: 6px; padding: .65rem .75rem; }
    @media (max-width: 760px) {
        .block-container { padding-top: 1.25rem; }
        .rg-stage-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 480px) {
        .rg-stage-grid { grid-template-columns: 1fr; }
    }
</style>
"""


@st.cache_resource(show_spinner=False)
def get_pipeline() -> ResearchGuardPipeline:
    return ResearchGuardPipeline.from_config(PIPELINE_CONFIG)


def run_researchguard(query: str) -> dict[str, Any]:
    return validate_pipeline_result(get_pipeline().run(query))


def render_stage_status(result: dict[str, Any]) -> None:
    rows = stage_rows(result)
    cards = []
    for row in rows:
        reason = html.escape(row["reason"])
        title = f' title="{reason}"' if reason else ""
        cards.append(
            f'<div class="rg-stage"{title}>'
            f'<div class="rg-stage-top">'
            f'<span class="rg-stage-name">{html.escape(row["label"])}</span>'
            f'<span class="rg-stage-status {html.escape(row["status"])}">'
            f'{html.escape(row["status_label"])}</span>'
            f'</div><div class="rg-stage-latency">{row["latency_ms"]:.1f} ms</div></div>'
        )
    st.markdown('<div class="rg-stage-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_evidence(result: dict[str, Any]) -> None:
    hits = [evidence_view(hit) for hit in retrieval_hits(result)]
    if not hits:
        st.info("No retrieval evidence was returned.")
        return
    for item in hits:
        score = "n/a" if item["score"] is None else f"{item['score']:.4f}"
        label = f"Evidence {item['rank']} | {item['document']} | {item['section']}"
        with st.expander(label, expanded=item["rank"] <= 2):
            columns = st.columns((1.2, 1.5, 1, 1.15))
            columns[0].metric("Rank", item["rank"])
            columns[1].metric("Page", item["page"])
            columns[2].metric(item["score_name"], score)
            columns[3].metric("Section", item["section_id"] or "Unknown")
            st.markdown("**Document**")
            st.write(item["document"])
            st.markdown("**Chunk text**")
            st.write(item["text"] or "No text available.")
            st.caption(f"Chunk ID: {item['chunk_id']}")


def render_sufficiency(result: dict[str, Any]) -> None:
    view = evidence_sufficiency_view(result)
    columns = st.columns((1.1, 1, 2.9))
    columns[0].metric("Decision", view["label"])
    columns[1].metric("Confidence", f"{view['confidence']:.2f}")
    with columns[2]:
        st.markdown("**Reason**")
        st.write(view["reason"])


def render_answer(result: dict[str, Any]) -> None:
    view = answer_view(result)
    answer = html.escape(view["answer"] or "No answer was generated.")
    css_class = "rg-answer refused" if view["refused"] else "rg-answer"
    st.markdown(f'<div class="{css_class}">{answer}</div>', unsafe_allow_html=True)
    if view["citations"]:
        st.dataframe(
            view["citations"],
            column_order=("chunk_id", "doc_id", "section", "page"),
            hide_index=True,
            use_container_width=True,
        )


def render_audit(result: dict[str, Any]) -> None:
    summary = audit_summary(result)
    if summary["status"] in {"skipped", "disabled"}:
        st.info(summary["reason"] or "Citation audit was not run because no answer was generated.")
        return
    columns = st.columns(3)
    columns[0].metric("Grounding", f"{summary['grounding_score']:.0%}")
    columns[1].metric("Unsupported claims", summary["unsupported_claim_count"])
    columns[2].metric("Partial claims", summary["partial_claim_count"])
    claims = audit_claims(result)
    if not claims:
        st.info(summary["reason"] or "No claims were returned by the citation audit.")
        return
    for index, claim in enumerate(claims, start=1):
        with st.expander(f"Claim {index}: {claim['text']}", expanded=index == 1):
            status = html.escape(claim["support_level"])
            st.markdown(
                f'<div class="rg-claim-status {status}">{status.upper()}</div>',
                unsafe_allow_html=True,
            )
            st.write(claim["reason"])
            if claim["citations"]:
                st.dataframe(
                    claim["citations"],
                    column_order=("chunk_id", "doc_id", "section", "page"),
                    hide_index=True,
                    use_container_width=True,
                )


def render_result(result: dict[str, Any]) -> None:
    status = final_status_view(result)
    latency = float((result.get("pipeline") or {}).get("latency_ms") or 0.0)
    st.markdown(
        f"""
        <div class="rg-final {status['tone']}">
            <div class="rg-final-label">Final status</div>
            <div class="rg-final-value">{html.escape(status['label'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Total pipeline latency: {latency:.1f} ms")

    st.markdown("## Pipeline Status")
    render_stage_status(result)

    st.markdown("## Evidence")
    render_evidence(result)

    st.markdown("## Evidence Sufficiency")
    render_sufficiency(result)

    st.markdown("## Generated Answer")
    render_answer(result)

    st.markdown("## Citation Audit")
    render_audit(result)

    with st.expander("Show Pipeline Details"):
        details = {
            "query": result.get("query"),
            "final_status": result.get("final_status"),
            "pipeline": result.get("pipeline"),
            "stages": {
                key: {
                    "status": result[key].get("status"),
                    "latency_ms": result[key].get("latency_ms"),
                    "model": result[key].get("model"),
                    "config_version": result[key].get("config_version"),
                    "reason": result[key].get("reason"),
                }
                for key in STAGE_LABELS
            },
            "result": result,
        }
        st.json(sanitize_for_display(details), expanded=2)


def main() -> None:
    st.set_page_config(page_title="ResearchGuard", layout="wide", initial_sidebar_state="auto")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Demo Queries")
        selected = st.selectbox("Example", EXAMPLE_QUERIES, label_visibility="collapsed")
        if st.button("Use Example", use_container_width=True):
            st.session_state["query"] = selected
        st.divider()
        st.markdown("### Runtime")
        st.caption("Pipeline v1")
        st.caption("Local Chroma index")
        st.caption("Evidence gate enabled")

    st.title("ResearchGuard")
    st.markdown('<p class="rg-subtitle">Evidence-grounded RAG for Scientific Papers</p>', unsafe_allow_html=True)
    st.markdown('<div class="rg-rule"></div>', unsafe_allow_html=True)

    if "query" not in st.session_state:
        st.session_state["query"] = EXAMPLE_QUERIES[0]
    query = st.text_area(
        "Research question",
        key="query",
        height=110,
        placeholder="Ask a question about the indexed scientific papers...",
    )
    run_clicked = st.button("Run ResearchGuard", type="primary", use_container_width=False)

    if run_clicked:
        if not query.strip():
            st.warning("Enter a research question before running the pipeline.")
        else:
            with st.spinner("Running ResearchGuard pipeline..."):
                try:
                    st.session_state["pipeline_result"] = run_researchguard(query.strip())
                    st.session_state.pop("pipeline_error", None)
                except Exception as exc:
                    st.session_state.pop("pipeline_result", None)
                    st.session_state["pipeline_error"] = safe_error_message(exc)

    if st.session_state.get("pipeline_error"):
        st.error("ResearchGuard could not complete this request.")
        st.code(st.session_state["pipeline_error"], language="text")
    result = st.session_state.get("pipeline_result")
    if isinstance(result, dict):
        render_result(result)


if __name__ == "__main__":
    main()
