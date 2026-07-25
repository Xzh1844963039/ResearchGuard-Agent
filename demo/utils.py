# C:\Users\18449\Desktop\researchguard_workspace\demo\utils.py
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


STAGE_LABELS = {
    "rewrite": "Query Rewrite",
    "retrieval": "Retrieval",
    "reranking": "Reranker",
    "evidence_check": "Evidence Gate",
    "answer_generation": "Answer Generation",
    "citation_audit": "Citation Audit",
}
STAGE_STATUS_LABELS = {
    "completed": "Completed",
    "fallback": "Degraded",
    "failed": "Failed",
    "skipped": "Skipped",
    "disabled": "Disabled",
    "pending": "Pending",
}
FINAL_STATUS_LABELS = {
    "grounded": "Grounded answer",
    "needs_review": "Review required",
    "rejected": "Insufficient evidence",
    "answered": "Answer generated",
    "evidence_sufficient": "Evidence sufficient",
    "retrieved": "Retrieval complete",
    "disabled": "Pipeline disabled",
    "failed": "Pipeline failed",
}
SENSITIVE_KEYS = {
    "index_dir",
    "cache_directory",
    "model_path",
    "output_directory",
    "config_path",
    "benchmark_path",
}
WINDOWS_PATH_RE = re.compile(r"(?i)[A-Z]:\\(?:[^\s\"']+\\)*[^\s\"']*")
UNIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9])/(?:home|users|var|tmp|opt)/[^\s\"']+")
SECRET_RE = re.compile(r"(?i)(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{8,}")


class DemoResultError(ValueError):
    pass


def validate_pipeline_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise DemoResultError("Pipeline returned an invalid result object.")
    required = {"query", "final_status", "pipeline", *STAGE_LABELS}
    missing = sorted(required - set(result))
    if missing:
        raise DemoResultError("Pipeline result is missing required fields.")
    for name in STAGE_LABELS:
        stage = result.get(name)
        if not isinstance(stage, dict) or "status" not in stage:
            raise DemoResultError(f"Pipeline stage '{name}' has an invalid schema.")
    return result


def stage_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, label in STAGE_LABELS.items():
        stage = result.get(key) or {}
        status = str(stage.get("status", "failed"))
        rows.append(
            {
                "key": key,
                "label": label,
                "status": status,
                "status_label": STAGE_STATUS_LABELS.get(status, status.replace("_", " ").title()),
                "latency_ms": float(stage.get("latency_ms") or 0.0),
                "model": str(stage.get("model") or "Not used"),
                "config_version": str(stage.get("config_version") or "Unknown"),
                "reason": str(stage.get("reason") or ""),
            }
        )
    return rows


def retrieval_hits(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = (result.get("retrieval") or {}).get("output") or {}
    hits = output.get("hits") or []
    return [dict(hit) for hit in hits if isinstance(hit, Mapping)]


def evidence_score(hit: Mapping[str, Any]) -> tuple[str, float | None]:
    for key, label in (
        ("rerank_score", "Rerank"),
        ("multi_query_fusion_score", "Multi-query RRF"),
        ("fusion_score", "Hybrid RRF"),
        ("dense_score", "Dense"),
        ("sparse_score", "BM25"),
    ):
        value = hit.get(key)
        if value is not None:
            try:
                return label, float(value)
            except (TypeError, ValueError):
                continue
    return "Score", None


def page_label(hit: Mapping[str, Any]) -> str:
    start = hit.get("page_start")
    end = hit.get("page_end")
    if start is None:
        return "Unknown"
    return str(start) if end in (None, start) else f"{start}-{end}"


def document_label(hit: Mapping[str, Any]) -> str:
    title = str(hit.get("title") or "").strip()
    return title or str(hit.get("doc_id") or "Unknown document")


def evidence_view(hit: Mapping[str, Any]) -> dict[str, Any]:
    score_name, score = evidence_score(hit)
    return {
        "rank": int(hit.get("rank") or 0),
        "document": document_label(hit),
        "doc_id": str(hit.get("doc_id") or ""),
        "section": str(hit.get("section_heading") or hit.get("section") or "Unknown"),
        "section_id": str(hit.get("section") or ""),
        "page": page_label(hit),
        "score_name": score_name,
        "score": score,
        "chunk_id": str(hit.get("chunk_id") or ""),
        "text": str(hit.get("text") or "").strip(),
        "content_types": [str(item) for item in hit.get("content_types", [])],
    }


def evidence_sufficiency_view(result: Mapping[str, Any]) -> dict[str, Any]:
    stage = result.get("evidence_check") or {}
    output = stage.get("output") or {}
    support = str(output.get("support_level") or "unavailable")
    labels = {
        "strong": "SUPPORTED",
        "partial": "PARTIAL EVIDENCE",
        "unsupported": "INSUFFICIENT EVIDENCE",
        "unavailable": "NOT EVALUATED",
    }
    return {
        "support_level": support,
        "label": labels.get(support, support.replace("_", " ").upper()),
        "answerable": bool(output.get("answerable", False)),
        "confidence": float(output.get("confidence") or 0.0),
        "reason": str(output.get("reason") or stage.get("reason") or "No evidence decision was returned."),
        "supporting_chunk_ids": [str(item) for item in output.get("supporting_chunk_ids", [])],
    }


def answer_view(result: Mapping[str, Any]) -> dict[str, Any]:
    stage = result.get("answer_generation") or {}
    output = stage.get("output") or {}
    answer = str(output.get("answer") or "").strip()
    refused = bool(output.get("refused", False)) or str(result.get("final_status")) == "rejected"
    if refused and not answer:
        answer = "Insufficient evidence in the current corpus."
    return {
        "answer": answer,
        "refused": refused,
        "confidence": float(output.get("confidence") or 0.0),
        "citations": [dict(item) for item in output.get("citations", []) if isinstance(item, Mapping)],
        "status": str(stage.get("status") or "unknown"),
        "reason": str(output.get("refusal_reason") or stage.get("reason") or ""),
    }


def audit_claims(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = (result.get("citation_audit") or {}).get("output") or {}
    claims: list[dict[str, Any]] = []
    for claim in output.get("claims", []):
        if not isinstance(claim, Mapping):
            continue
        claims.append(
            {
                "id": str(claim.get("id") or ""),
                "text": str(claim.get("text") or "").strip(),
                "support_level": str(claim.get("support_level") or "unknown"),
                "confidence": float(claim.get("confidence") or 0.0),
                "reason": str(claim.get("reason") or ""),
                "citations": [dict(item) for item in claim.get("citations", []) if isinstance(item, Mapping)],
            }
        )
    return claims


def audit_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    stage = result.get("citation_audit") or {}
    output = stage.get("output") or {}
    return {
        "status": str(stage.get("status") or "unknown"),
        "audit_completed": bool(output.get("audit_completed", False)),
        "overall_grounded": bool(output.get("overall_grounded", False)),
        "grounding_score": float(output.get("grounding_score") or 0.0),
        "unsupported_claim_count": int(output.get("unsupported_claim_count") or 0),
        "partial_claim_count": int(output.get("partial_claim_count") or 0),
        "reason": str(output.get("audit_reason") or stage.get("reason") or ""),
    }


def final_status_view(result: Mapping[str, Any]) -> dict[str, str]:
    status = str(result.get("final_status") or "failed")
    tone = {
        "grounded": "success",
        "answered": "success",
        "evidence_sufficient": "success",
        "needs_review": "warning",
        "rejected": "warning",
        "failed": "error",
    }.get(status, "neutral")
    return {"status": status, "label": FINAL_STATUS_LABELS.get(status, status.title()), "tone": tone}


def agent_plan_rows(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "step": index,
            "skill": str(item.get("skill") or item.get("tool") or "unknown"),
            "purpose": str(item.get("purpose") or ""),
            "expected_observation": str(
                item.get("expected_observation") or ""
            ),
            "max_retry": item.get("max_retry"),
        }
        for index, item in enumerate(state.get("plan") or (), start=1)
        if isinstance(item, Mapping)
    ]


def skill_timeline_rows(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "step": index,
            "skill": str(item.get("skill_name") or "unknown"),
            "status": str(item.get("output_status") or "unknown"),
            "evidence": int(item.get("evidence_count") or 0),
            "latency_ms": float(item.get("latency_ms") or 0.0),
            "trace_id": str(item.get("trace_id") or ""),
        }
        for index, item in enumerate(
            state.get("skill_history") or (),
            start=1,
        )
        if isinstance(item, Mapping)
    ]


def tool_call_rows(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "call": index,
            "tool": str(item.get("tool_name") or "unknown"),
            "status": str(item.get("output_status") or "unknown"),
            "api_calls": int(item.get("api_call_count") or 0),
            "latency_ms": float(item.get("latency_ms") or 0.0),
            "trace_id": str(item.get("trace_id") or ""),
        }
        for index, item in enumerate(state.get("tool_history") or (), start=1)
        if isinstance(item, Mapping)
    ]


def evidence_bundle_view(state: Mapping[str, Any]) -> dict[str, Any]:
    bundle = state.get("evidence_bundle")
    if not isinstance(bundle, Mapping):
        return {
            "bundle_id": None,
            "evidence_count": len(state.get("evidence") or ()),
            "query": str(state.get("query") or ""),
            "provenance": {},
        }
    records = bundle.get("evidence_records") or ()
    return {
        "bundle_id": bundle.get("bundle_id"),
        "evidence_count": len(records) if isinstance(records, list) else 0,
        "query": str(bundle.get("query") or state.get("query") or ""),
        "provenance": dict(bundle.get("provenance") or {}),
    }


def agent_answer_view(state: Mapping[str, Any]) -> dict[str, Any]:
    answer = state.get("answer")
    if not isinstance(answer, Mapping):
        return {
            "generated": False,
            "answer": "",
            "citations": [],
        }
    return {
        "generated": True,
        "answer": str(answer.get("answer") or ""),
        "citations": [
            dict(item)
            for item in answer.get("citations", ())
            if isinstance(item, Mapping)
        ],
    }


def agent_audit_view(state: Mapping[str, Any]) -> dict[str, Any]:
    audit = state.get("audit_result")
    if not isinstance(audit, Mapping):
        return {"completed": False, "claims": [], "summary": {}}
    claims = audit.get("claims", audit.get("claim_results", ()))
    return {
        "completed": bool(audit.get("audit_completed", True)),
        "claims": [
            dict(item)
            for item in claims
            if isinstance(item, Mapping)
        ]
        if isinstance(claims, list)
        else [],
        "summary": {
            key: audit.get(key)
            for key in (
                "overall_grounded",
                "grounding_score",
                "unsupported_claim_count",
                "partial_claim_count",
                "audit_reason",
            )
            if key in audit
        },
    }


def memory_trace_view(state: Mapping[str, Any]) -> dict[str, Any]:
    context = state.get("memory_context")
    snapshot = state.get("memory_snapshot")
    status = state.get("memory_status")
    return {
        "status": dict(status) if isinstance(status, Mapping) else {},
        "context": dict(context) if isinstance(context, Mapping) else {},
        "snapshot": dict(snapshot) if isinstance(snapshot, Mapping) else {},
    }


def sanitize_for_display(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_for_display(item)
            for key, item in value.items()
            if str(key).casefold() not in SENSITIVE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_for_display(item) for item in value]
    if isinstance(value, str):
        text = SECRET_RE.sub("[REDACTED]", value)
        text = WINDOWS_PATH_RE.sub("[LOCAL_PATH]", text)
        return UNIX_PATH_RE.sub("[LOCAL_PATH]", text)
    return value


def safe_error_message(exc: Exception) -> str:
    message = str(exc).strip() or "The pipeline did not return a result."
    sanitized = str(sanitize_for_display(message))
    return f"{type(exc).__name__}: {sanitized}"
