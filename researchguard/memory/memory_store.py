# C:\Users\18449\Desktop\researchguard_workspace\researchguard\memory\memory_store.py
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from researchguard.schemas import MemorySnapshot, SkillTraceRecord, ToolTraceRecord, to_dict
from researchguard.text_utils_v2 import now_iso
from researchguard.utils_ids import next_id


_TRACE_WRITE_LOCK = threading.RLock()


class MemoryStore:
    """JSON/JSONL memory backend for one EvidenceClaw run.

    Example:
        store = MemoryStore(Path("memory"), Path("outputs/sample"))
        store.append_source({"source_id": "S001", ...})
        snapshot = store.snapshot("sample")
    """

    def __init__(self, memory_dir: Path, case_output_dir: Path | None = None) -> None:
        self.memory_dir = memory_dir
        self.case_output_dir = case_output_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if self.case_output_dir:
            self.case_output_dir.mkdir(parents=True, exist_ok=True)

        self.source_file = self.memory_dir / "source_memory.json"
        self.evidence_file = self.memory_dir / "evidence_memory.json"
        self.review_file = self.memory_dir / "review_memory.json"
        self.literature_file = self.memory_dir / "literature_memory.json"
        self.hypothesis_file = self.memory_dir / "hypothesis_memory.json"
        self.failure_file = self.memory_dir / "failure_memory.json"
        self.tool_trace_file = self.memory_dir / "tool_trace.jsonl"
        self.skill_trace_file = self.memory_dir / "skill_trace.jsonl"

        for file in [self.source_file, self.evidence_file, self.review_file, self.literature_file, self.hypothesis_file, self.failure_file]:
            if not file.exists():
                file.write_text("[]", encoding="utf-8")
        if not self.tool_trace_file.exists():
            self.tool_trace_file.write_text("", encoding="utf-8")
        if not self.skill_trace_file.exists():
            self.skill_trace_file.write_text("", encoding="utf-8")
        self._baseline_counts = {
            "source": len(self.read_json_list(self.source_file)),
            "evidence": len(self.read_json_list(self.evidence_file)),
            "literature": len(self.read_json_list(self.literature_file)),
            "hypothesis": len(self.read_json_list(self.hypothesis_file)),
            "review": len(self.read_json_list(self.review_file)),
            "failure": len(self.read_json_list(self.failure_file)),
            "tool_trace": len(self.read_tool_traces()),
            "skill_trace": len(self.read_skill_traces()),
        }

    def read_json_list(self, path: Path) -> list[dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def write_json_list(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    def append_json(self, path: Path, record: Any) -> dict[str, Any]:
        row = to_dict(record)
        rows = self.read_json_list(path)
        rows.append(row)
        self.write_json_list(path, rows)
        return row

    def append_source(self, record: Any) -> dict[str, Any]:
        return self.append_json(self.source_file, record)

    def append_evidence(self, record: Any) -> dict[str, Any]:
        return self.append_json(self.evidence_file, record)

    def append_review(self, record: Any) -> dict[str, Any]:
        return self.append_json(self.review_file, record)

    def append_literature(self, record: Any) -> dict[str, Any]:
        return self.append_json(self.literature_file, record)

    def append_hypothesis(self, record: Any) -> dict[str, Any]:
        return self.append_json(self.hypothesis_file, record)

    def append_failure(self, case_id: str, failure_type: str, message: str, details: dict | None = None) -> dict:
        record = {
            "failure_id": next_id("F", self.read_json_list(self.failure_file), "failure_id"),
            "case_id": case_id,
            "failure_type": failure_type,
            "message": message,
            "details": details or {},
            "created_at": now_iso(),
        }
        return self.append_json(self.failure_file, record)

    def append_tool_trace(self, record: ToolTraceRecord | dict) -> dict:
        row = to_dict(record)
        line = json.dumps(row, ensure_ascii=False)
        with _TRACE_WRITE_LOCK:
            with self.tool_trace_file.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
            if self.case_output_dir:
                with (self.case_output_dir / "tool_trace.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                    f.write(line + "\n")
        return row

    def append_skill_trace(self, record: SkillTraceRecord | dict) -> dict:
        row = to_dict(record)
        line = json.dumps(row, ensure_ascii=False)
        with _TRACE_WRITE_LOCK:
            with self.skill_trace_file.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
            if self.case_output_dir:
                with (self.case_output_dir / "skill_trace.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                    f.write(line + "\n")
        return row

    def read_tool_traces(self, case_id: str | None = None) -> list[dict]:
        return self.read_jsonl_rows(self.tool_trace_file, case_id)

    def read_skill_traces(self, case_id: str | None = None) -> list[dict]:
        return self.read_jsonl_rows(self.skill_trace_file, case_id)

    def read_jsonl_rows(self, path: Path, case_id: str | None = None) -> list[dict]:
        """Read valid JSONL records while tolerating an interrupted trailing write."""
        rows: list[dict] = []
        if not path.exists():
            return rows
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if case_id is None or row.get("case_id") == case_id:
                rows.append(row)
        return rows

    def trace_tool(self, case_id: str, tool_name: str, input_summary: str):
        return ToolTraceContext(self, case_id, tool_name, input_summary)

    def trace_skill(self, case_id: str, skill_name: str, input_summary: str, cached: bool = False):
        return SkillTraceContext(self, case_id, skill_name, input_summary, cached=cached)

    def snapshot(self, case_id: str) -> dict:
        all_source_rows = self.read_json_list(self.source_file)
        all_evidence_rows = self.read_json_list(self.evidence_file)
        all_literature_rows = self.read_json_list(self.literature_file)
        all_hypothesis_rows = self.read_json_list(self.hypothesis_file)
        all_review_rows = self.read_json_list(self.review_file)
        all_failure_rows = self.read_json_list(self.failure_file)
        all_tool_trace_rows = self.read_tool_traces()
        all_skill_trace_rows = self.read_skill_traces()
        source_rows = [r for r in all_source_rows if r.get("case_id") == case_id]
        evidence_rows = [r for r in all_evidence_rows if r.get("case_id") == case_id]
        literature_rows = [r for r in all_literature_rows if r.get("case_id") == case_id]
        hypothesis_rows = [r for r in all_hypothesis_rows if r.get("case_id") == case_id]
        review_rows = [r for r in all_review_rows if r.get("case_id") == case_id]
        failure_rows = [r for r in all_failure_rows if r.get("case_id") == case_id]
        trace_rows = self.read_tool_traces(case_id)
        skill_trace_rows = self.read_skill_traces(case_id)
        snapshot = MemorySnapshot(
            case_id=case_id,
            source_memory_count=len(source_rows),
            evidence_memory_count=len(evidence_rows),
            literature_memory_count=len(literature_rows),
            hypothesis_memory_count=len(hypothesis_rows),
            review_memory_count=len(review_rows),
            tool_trace_count=len(trace_rows),
            skill_trace_count=len(skill_trace_rows),
            failure_memory_count=len(failure_rows),
            updated_files=[
                str(self.source_file),
                str(self.evidence_file),
                str(self.literature_file),
                str(self.hypothesis_file),
                str(self.review_file),
                str(self.tool_trace_file),
                str(self.skill_trace_file),
                str(self.failure_file),
            ],
            created_at=now_iso(),
        )
        row = to_dict(snapshot)
        row.update(
            {
                "this_run_added_source": max(0, len(all_source_rows) - self._baseline_counts["source"]),
                "this_run_added_evidence": max(0, len(all_evidence_rows) - self._baseline_counts["evidence"]),
                "this_run_added_literature": max(0, len(all_literature_rows) - self._baseline_counts["literature"]),
                "this_run_added_hypothesis": max(0, len(all_hypothesis_rows) - self._baseline_counts["hypothesis"]),
                "this_run_added_review": max(0, len(all_review_rows) - self._baseline_counts["review"]),
                "this_run_added_tool_trace": max(0, len(all_tool_trace_rows) - self._baseline_counts["tool_trace"]),
                "this_run_added_skill_trace": max(0, len(all_skill_trace_rows) - self._baseline_counts["skill_trace"]),
                "this_run_added_failure": max(0, len(all_failure_rows) - self._baseline_counts["failure"]),
                "global_total_source": len(all_source_rows),
                "global_total_evidence": len(all_evidence_rows),
                "global_total_literature": len(all_literature_rows),
                "global_total_hypothesis": len(all_hypothesis_rows),
                "global_total_review": len(all_review_rows),
                "global_total_tool_trace": len(all_tool_trace_rows),
                "global_total_skill_trace": len(all_skill_trace_rows),
                "global_total_failure": len(all_failure_rows),
            }
        )
        return row


class ToolTraceContext:
    def __init__(self, store: MemoryStore, case_id: str, tool_name: str, input_summary: str) -> None:
        self.store = store
        self.case_id = case_id
        self.tool_name = tool_name
        self.input_summary = input_summary
        self.started_at = now_iso()
        self.start = time.perf_counter()

    def __enter__(self):
        return self

    def success(self, output_summary: str) -> None:
        self._finish("success", output_summary, None)

    def failed(self, output_summary: str, error_message: str) -> None:
        self._finish("failed", output_summary, error_message)

    def skipped(self, output_summary: str) -> None:
        self._finish("skipped", output_summary, None)

    def _finish(self, status: str, output_summary: str, error_message: str | None) -> None:
        finished_at = now_iso()
        duration_ms = int((time.perf_counter() - self.start) * 1000)
        with _TRACE_WRITE_LOCK:
            trace_id = next_id("T", self.store.read_tool_traces(), "trace_id")
            self.store.append_tool_trace(
                ToolTraceRecord(
                    trace_id=trace_id,
                    case_id=self.case_id,
                    tool_name=self.tool_name,
                    input_summary=self.input_summary,
                    output_summary=output_summary,
                    status=status,
                    error_message=error_message,
                    started_at=self.started_at,
                    finished_at=finished_at,
                    duration_ms=duration_ms,
                )
            )

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.failed("tool raised exception", str(exc))
            return False
        return False


class SkillTraceContext:
    def __init__(self, store: MemoryStore, case_id: str, skill_name: str, input_summary: str, cached: bool = False) -> None:
        self.store = store
        self.case_id = case_id
        self.skill_name = skill_name
        self.input_summary = input_summary
        self.cached = cached
        self.started_at = now_iso()
        self.start = time.perf_counter()

    def __enter__(self):
        return self

    def success(self, output_summary: str) -> None:
        self._finish("success", output_summary, None)

    def failed(self, output_summary: str, error_message: str) -> None:
        self._finish("failed", output_summary, error_message)

    def skipped(self, output_summary: str) -> None:
        self._finish("skipped", output_summary, None)

    def _finish(self, status: str, output_summary: str, error_message: str | None) -> None:
        finished_at = now_iso()
        duration_ms = int((time.perf_counter() - self.start) * 1000)
        with _TRACE_WRITE_LOCK:
            trace_id = next_id("ST", self.store.read_skill_traces(), "trace_id")
            self.store.append_skill_trace(
                SkillTraceRecord(
                    trace_id=trace_id,
                    case_id=self.case_id,
                    skill_name=self.skill_name,
                    input_summary=self.input_summary,
                    output_summary=output_summary,
                    status=status,
                    error_message=error_message,
                    started_at=self.started_at,
                    finished_at=finished_at,
                    duration_ms=duration_ms,
                    cached=self.cached,
                )
            )

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.failed("skill raised exception", str(exc))
            return False
        return False
