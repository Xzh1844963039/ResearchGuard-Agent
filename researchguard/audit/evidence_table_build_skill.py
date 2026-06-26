# C:\Users\18449\Desktop\researchguard_workspace\researchguard\audit\evidence_table_build_skill.py
from __future__ import annotations

from researchguard.audit.base_skill import BaseSkill


class EvidenceTableBuildSkill(BaseSkill):
    name = "evidence_table_build_skill"
    description = "Build combined E-id and R-id evidence table."
    input_schema = {"source_evidence": "list", "references": "list", "audits": "list"}
    output_schema = {"evidence_table": "list"}

    def run(self, case_id: str, payload: dict) -> dict:
        rows = []
        for e in payload.get("source_evidence", []):
            rows.append({"id": e["evidence_id"], "type": "input_pdf", "source": e.get("source_name"), "location": e.get("location_text"), "content_summary": e.get("content_summary"), "used_for": ", ".join(e.get("used_for", [])), "status": "source_evidence"})
        audit_by_ref = {a.get("ref_id"): a for a in payload.get("audits", []) if a.get("ref_id")}
        for r in payload.get("references", []):
            audit = audit_by_ref.get(r["ref_id"], {})
            rows.append({"id": r["ref_id"], "type": "retrieved_reference", "source": r.get("source_api"), "location": r.get("doi") or r.get("url") or r.get("arxiv_id") or r.get("pmid"), "content_summary": r.get("abstract") or r.get("title"), "used_for": audit.get("support_claim_id", ""), "status": audit.get("final_status", "retrieved")})
        return {"evidence_table": rows}

