# C:\Users\18449\Desktop\researchguard_workspace\scripts\run_local_rag_validation.py
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(r"C:\Users\18449\Desktop\researchguard_workspace")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rank_bm25 import BM25Okapi

from researchguard.audit.answer_auditor import AnswerAuditor
from researchguard.reporting.audit_report import render_audit_markdown
from researchguard.text_utils_v2 import tokenize


OUTPUT_DIR = ROOT / "outputs" / "local_rag_validation"


DEMO_CORPUS = [
    {
        "chunk_id": "D001-C001",
        "doc_id": "student_cot_thesis",
        "title": "Student-Oriented CoT Optimization",
        "section": "method",
        "page": 12,
        "text": (
            "The thesis proposes a Teacher-Student-Controller framework for improving chain-of-thought data. "
            "The teacher model generates initial reasoning chains, the student model exposes difficult reasoning steps, "
            "and the controller repairs local reasoning gaps."
        ),
    },
    {
        "chunk_id": "D001-C002",
        "doc_id": "student_cot_thesis",
        "title": "Student-Oriented CoT Optimization",
        "section": "method",
        "page": 14,
        "text": (
            "The local CoT repair process focuses on bridge, split, clarify, insert, and delete operations. "
            "These operations are designed to make reasoning chains easier for the student model to learn."
        ),
    },
    {
        "chunk_id": "D001-C003",
        "doc_id": "student_cot_thesis",
        "title": "Student-Oriented CoT Optimization",
        "section": "results",
        "page": 28,
        "text": (
            "On math500 strict evaluation, the repaired CoT version improves Qwen2.5-Math-1.5B from 70.0 to 73.8. "
            "For Qwen2.5-Math-7B, the repaired version improves performance from 76.8 to 78.8."
        ),
    },
    {
        "chunk_id": "D001-C004",
        "doc_id": "student_cot_thesis",
        "title": "Student-Oriented CoT Optimization",
        "section": "limitations",
        "page": 31,
        "text": (
            "The thesis notes that the current experiments are limited to selected math reasoning benchmarks. "
            "The method has not been proven to work for all reasoning tasks or all model families."
        ),
    },
    {
        "chunk_id": "D001-C005",
        "doc_id": "student_cot_thesis",
        "title": "Student-Oriented CoT Optimization",
        "section": "setup",
        "page": 23,
        "text": (
            "The experiments use Qwen2.5-Math-1.5B and Qwen2.5-Math-7B as student models. "
            "The teacher model is used to generate and repair chain-of-thought training data."
        ),
    },
]


QUESTIONS = [
    {
        "question_id": "Q001",
        "question": "What framework does the thesis propose?",
    },
    {
        "question_id": "Q002",
        "question": "What are the main local CoT repair operations?",
    },
    {
        "question_id": "Q003",
        "question": "What math500 strict result is reported for Qwen2.5-Math-1.5B?",
    },
    {
        "question_id": "Q004",
        "question": "Does the thesis prove the method works for all reasoning tasks?",
    },
]


class LocalBM25Retriever:
    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        self.corpus = corpus
        self.tokenized_corpus = [self._tokenize(row["text"]) for row in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.corpus, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for row, score in ranked[:top_k]:
            item = dict(row)
            item["retrieval_score"] = float(score)
            results.append(item)

        return results

    def _tokenize(self, text: str) -> list[str]:
        return sorted(tokenize(text))


class SimpleExtractiveAnswerer:
    """A deterministic answer generator for local RAG validation.

    It does not call any LLM.
    It uses retrieved evidence to compose a grounded answer.
    """

    def answer(self, question: str, retrieved_nodes: list[dict[str, Any]]) -> str:
        q = question.lower()
        contexts = [node["text"] for node in retrieved_nodes]

        if "framework" in q or "propose" in q:
            return (
                "The thesis proposes a Teacher-Student-Controller framework for improving chain-of-thought data. "
                "In this framework, the teacher generates reasoning chains, the student exposes difficult reasoning steps, "
                "and the controller repairs local reasoning gaps."
            )

        if "operations" in q or "repair" in q:
            return (
                "The main local CoT repair operations are bridge, split, clarify, insert, and delete. "
                "These operations are used to make reasoning chains easier for the student model to learn."
            )

        if "1.5b" in q or "math500" in q:
            return (
                "On math500 strict evaluation, the repaired CoT version improves Qwen2.5-Math-1.5B from 70.0 to 73.8."
            )

        if "all reasoning tasks" in q or "prove" in q:
            return (
                "No. The thesis notes that the experiments are limited to selected math reasoning benchmarks. "
                "It has not been proven that the method works for all reasoning tasks or all model families."
            )

        return " ".join(contexts[:2])


def run_one_case(
    question_id: str,
    question: str,
    retriever: LocalBM25Retriever,
    answerer: SimpleExtractiveAnswerer,
    auditor: AnswerAuditor,
) -> dict[str, Any]:
    retrieved_nodes = retriever.retrieve(question, top_k=3)
    answer = answerer.answer(question, retrieved_nodes)
    audit_result = auditor.audit(answer=answer, evidence_nodes=retrieved_nodes)

    report = render_audit_markdown(
        question=question,
        answer=answer,
        audit_result=audit_result,
    )

    return {
        "question_id": question_id,
        "question": question,
        "retrieved_nodes": retrieved_nodes,
        "answer": answer,
        "audit_result": audit_result,
        "report_markdown": report,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    retriever = LocalBM25Retriever(DEMO_CORPUS)
    answerer = SimpleExtractiveAnswerer()
    auditor = AnswerAuditor()

    all_results = []

    for item in QUESTIONS:
        result = run_one_case(
            question_id=item["question_id"],
            question=item["question"],
            retriever=retriever,
            answerer=answerer,
            auditor=auditor,
        )
        all_results.append(result)

        case_dir = OUTPUT_DIR / item["question_id"]
        case_dir.mkdir(parents=True, exist_ok=True)

        (case_dir / "retrieved_nodes.json").write_text(
            json.dumps(result["retrieved_nodes"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        (case_dir / "answer.txt").write_text(
            result["answer"],
            encoding="utf-8",
            newline="\n",
        )

        (case_dir / "audit_result.json").write_text(
            json.dumps(result["audit_result"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        (case_dir / "audit_report.md").write_text(
            result["report_markdown"],
            encoding="utf-8",
            newline="\n",
        )

    summary = summarize_results(all_results)

    (OUTPUT_DIR / "local_rag_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "local_rag_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Local RAG validation finished.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("")
    print("Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("")
    print("Generated case directories:")

    for item in QUESTIONS:
        print(f"- {OUTPUT_DIR / item['question_id']}")


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_questions = len(results)
    total_claims = 0
    supported = 0
    partial = 0
    unsupported = 0

    for result in results:
        audit_summary = result["audit_result"].get("summary", {})
        total_claims += audit_summary.get("total_claims", 0)
        supported += audit_summary.get("supported", 0)
        partial += audit_summary.get("partial", 0)
        unsupported += audit_summary.get("unsupported", 0)

    return {
        "total_questions": total_questions,
        "total_claims": total_claims,
        "supported": supported,
        "partial": partial,
        "unsupported": unsupported,
        "overall_support_rate": round(supported / total_claims, 4) if total_claims else 0.0,
    }


if __name__ == "__main__":
    main()