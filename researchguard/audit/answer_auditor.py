# C:\Users\18449\Desktop\researchguard_workspace\researchguard\audit\answer_auditor.py
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

from researchguard.text_utils_v2 import tokenize


@dataclass
class ClaimAuditRecord:
    claim_id: str
    claim: str
    verdict: str
    evidence_ids: list[str]
    overlap_score: float
    risk_type: str
    explanation: str


class AnswerAuditor:
    """Lightweight claim-level auditor for generated RAG answers.

    This module is the first bridge between rag_agent_harness and EvidenceClaw:
    - input: generated answer + retrieved evidence nodes
    - output: claim-level support records
    """

    def __init__(self, min_support_overlap: float = 0.18, partial_support_overlap: float = 0.10) -> None:
        self.min_support_overlap = min_support_overlap
        self.partial_support_overlap = partial_support_overlap

    def audit(self, answer: str, evidence_nodes: list[dict[str, Any]]) -> dict[str, Any]:
        claims = self.extract_claims(answer)
        records: list[ClaimAuditRecord] = []

        for idx, claim in enumerate(claims, start=1):
            record = self.audit_one_claim(idx, claim, evidence_nodes)
            records.append(record)

        summary = self.summarize(records)

        return {
            "summary": summary,
            "records": [asdict(record) for record in records],
        }

    def extract_claims(self, answer: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", answer or "").strip()

        if not cleaned:
            return []

        parts = re.split(r"(?<=[.!?。！？])\s+", cleaned)
        claims = []

        for part in parts:
            claim = part.strip()

            if len(claim) < 20:
                continue

            if self._is_non_claim(claim):
                continue

            claims.append(claim)

        return claims

    def audit_one_claim(self, idx: int, claim: str, evidence_nodes: list[dict[str, Any]]) -> ClaimAuditRecord:
        claim_tokens = tokenize(claim)

        best_score = 0.0
        best_evidence_ids: list[str] = []

        for node in evidence_nodes:
            evidence_text = (
                node.get("text")
                or node.get("content")
                or node.get("clean_content")
                or node.get("content_summary")
                or ""
            )
            evidence_id = (
                node.get("evidence_id")
                or node.get("node_id")
                or node.get("chunk_id")
                or node.get("id")
                or "unknown"
            )

            evidence_tokens = tokenize(evidence_text)

            if not claim_tokens or not evidence_tokens:
                continue

            score = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))

            if score > best_score:
                best_score = score
                best_evidence_ids = [str(evidence_id)]
            elif score == best_score and score > 0:
                best_evidence_ids.append(str(evidence_id))

        if best_score >= self.min_support_overlap:
            verdict = "supported"
            risk_type = "none"
            explanation = "The claim has sufficient lexical overlap with retrieved evidence."
        elif best_score >= self.partial_support_overlap:
            verdict = "partial"
            risk_type = "weak_support"
            explanation = "The claim has limited evidence support and should be checked."
        else:
            verdict = "unsupported"
            risk_type = "missing_evidence"
            explanation = "No retrieved evidence provides enough support for this claim."

        return ClaimAuditRecord(
            claim_id=f"C{idx:03d}",
            claim=claim,
            verdict=verdict,
            evidence_ids=best_evidence_ids[:3],
            overlap_score=round(best_score, 4),
            risk_type=risk_type,
            explanation=explanation,
        )

    def summarize(self, records: list[ClaimAuditRecord]) -> dict[str, Any]:
        total = len(records)
        supported = sum(1 for record in records if record.verdict == "supported")
        partial = sum(1 for record in records if record.verdict == "partial")
        unsupported = sum(1 for record in records if record.verdict == "unsupported")

        return {
            "total_claims": total,
            "supported": supported,
            "partial": partial,
            "unsupported": unsupported,
            "support_rate": round(supported / total, 4) if total else 0.0,
        }

    def _is_non_claim(self, text: str) -> bool:
        lower = text.lower()

        noise_terms = [
            "as an ai",
            "i cannot",
            "i don't have access",
            "according to the context",
            "based on the provided context",
        ]

        return any(term in lower for term in noise_terms)