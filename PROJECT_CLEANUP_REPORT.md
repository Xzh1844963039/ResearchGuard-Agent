# ResearchGuard Final Cleanup v1

## Agent Intelligence Upgrade v1 Addendum

The two orphan Git links recorded by this historical audit have now been remediated:

- `EvidenceClaw` and `rag_agent_harness` were removed from the parent repository index with `git rm --cached`.
- Their local nested repositories were not deleted and are now excluded by root `.gitignore` rules.
- Fresh GitHub checkouts no longer contain inaccessible mode `160000` entries without `.gitmodules` mappings.

The remaining classifications below describe the repository at the time of the original cleanup audit.

## 1. Audit Scope and Conclusion

本报告基于当前 working tree、Git tracked files、Python AST import、CLI 入口、Pipeline 入口、Demo 入口、validation/build scripts、YAML 配置引用和 README 引用进行静态审计。

结论：**保留现有目录结构，不删除或移动任何源码。** 当前没有 Python 文件同时满足以下全部安全删除条件：无 import、无 Pipeline 引用、无 README 引用、无 validation 引用。部分模块不在 v1 主流程内，但仍被历史 CLI、验证工具或同组 legacy 模块引用；这些文件统一标记为 `LEGACY`，等待独立迁移任务处理。

因此本轮不创建 `DELETE_PLAN.md`，也没有执行文件删除。

分类定义：

- `KEEP`：当前 Parser/Chunking/Index/Retrieval/Pipeline/CLI/Demo 的运行依赖或正式 build 入口。
- `LEGACY`：不在 v1 统一 Pipeline 主链，仍有历史入口、内部引用、兼容价值或未来迁移用途。
- `TEST_ONLY`：只用于 synthetic、regression、benchmark 或本地验证。
- `UNUSED`：经所有入口和文档审计后可确认没有用途；本轮为 0。

## 2. Directory Inventory

```text
researchguard/                    87 tracked Python files
  ingestion/                     Parser v5 and section-aware chunking
  indexing/                      Dense/BM25/Chroma index stack
  retrieval/                     Retrieval through citation audit v1
  agent/, audit/, memory/        Retained legacy implementation
  evaluation/, reporting/        Retained legacy evaluation/reporting
configs/                         11 tracked YAML files
scripts/                         19 tracked Python entry points
demo/                            2 tracked Python files
tests/                           Empty; stage validators currently live in scripts/
data/                            7 tracked evaluation JSONL files; generated assets ignored
outputs/                         Generated validation reports; ignored by Git
frontend/streamlit_app.py        Retained legacy frontend entry
main.py                          Retained legacy root entry
EvidenceClaw                     Historical orphan Git link; removed from public index in Agent Intelligence Upgrade v1
rag_agent_harness                Historical orphan Git link; removed from public index in Agent Intelligence Upgrade v1
```

本地规模快照：

| Directory | Files | Approximate size | Git policy |
|---|---:|---:|---|
| `researchguard/` | 174（含 `__pycache__`） | 1.46 MiB | 87 `.py` tracked；cache ignored |
| `configs/` | 11 | 0.01 MiB | tracked |
| `scripts/` | 38（含 `__pycache__`） | 0.81 MiB | 19 `.py` tracked；cache ignored |
| `demo/` | 4（含 `__pycache__`） | 0.05 MiB | 2 `.py` tracked；cache ignored |
| `tests/` | 0 | 0 | tracked directory has no files |
| `data/` | 2401 | 144.43 MiB | only 7 `data/eval/*.jsonl` tracked |
| `outputs/` | 162 | 8.52 MiB | ignored |

`data/` 的本地大头是 cache、indexes、parsed 和 raw PDFs；这些资产没有纳入提交。`outputs/` 全部是可再生成的验证结果。

## 3. Core Module Responsibilities

| Area | Responsibility | Primary files |
|---|---|---|
| Parser | PyMuPDF layout extraction、noise filtering、column/reading order、block/heading/section recovery | `ingestion/layout_extractor.py`, `block_detector.py`, `heading_classifier.py`, `section_recovery.py`, `parse_pdf.py` |
| Chunking | section boundary、heading prefix、overlap provenance、special block binding、length limits | `ingestion/chunk_builder.py`, `scripts/build_chunks_v1.py` |
| Indexing | corpus loading、embedding、dense matrix、BM25、Chroma metadata and persistence | `indexing/corpus_loader.py`, `embedding_provider.py`, `dense_index.py`, `sparse_index.py`, `chroma_index.py` |
| Retrieval | Dense/BM25 search、filters、RRF、query variants、candidate provenance | `retrieval/retrieval_v1.py`, `multi_query.py`, `filters.py`, `models.py` |
| Reranking | Cross-Encoder scoring、cache、candidate-to-top-k pipeline | `retrieval/reranker.py`, `rerank_pipeline.py`, `rerank_cache.py` |
| Rewrite | Entity-preserving rewrite、expansion、fallback、cache | `retrieval/query_rewriter.py`, `query_rewrite_pipeline.py`, `rewrite_cache.py` |
| Evidence | Strong/partial/unsupported judgment and fail-closed gate | `retrieval/evidence_judge.py`, `evidence_pipeline.py`, `evidence_cache.py` |
| Answer | Supporting-evidence-only generation、citation validation、refusal | `retrieval/answer_generator.py`, `answer_pipeline.py`, `answer_cache.py` |
| Citation audit | Atomic claim extraction、verification、canonical citations、cache | `retrieval/claim_extractor.py`, `claim_verifier.py`, `citation_audit.py`, `citation_cache.py` |
| Integration | Stage orchestration、dependency rules、stable result schema | `pipeline.py`, `configs/pipeline_v1.yaml` |
| Presentation | CLI JSON output and Streamlit evidence/audit views | `cli.py`, `demo/app.py`, `demo/utils.py` |

## 4. Dependency Analysis

AST import closure（只计项目内 Python 文件）：

| Entry set | Reachable project files | Notes |
|---|---:|---|
| CLI | 35 | Includes unified Pipeline plus retained legacy status/import/smoke commands |
| Unified Pipeline | 31 | Current v1 production path |
| Streamlit Demo | 33 | `demo/app.py -> demo/utils.py + researchguard.pipeline` |
| Validation | 58 | Current stage validators plus retained legacy validators |
| Build scripts | 37 | Parser/chunk/index/Chroma dependencies |

关键调用关系：

```text
researchguard.cli run
  -> researchguard.pipeline.run_pipeline
  -> RetrievalEngine
     -> QueryRewritePipeline
     -> Dense/BM25/Chroma + RRF
     -> RerankPipeline
  -> EvidenceSufficiencyPipeline
  -> AnswerGenerationPipeline
  -> CitationAuditPipeline

demo.app
  -> demo.utils (schema validation and presentation adapters)
  -> ResearchGuardPipeline
```

Parser 与 build scripts 是命令行模块入口，不能仅因“没有被其他模块 import”而判断为 unused。`researchguard/ingestion/parse_pdf.py`、`scripts/build_*_v1.py` 和 `scripts/retrieve_v1.py` 均属于这种情况。

配置审计：

- Active：`indexing_v1.yaml`, `chroma_v1.yaml`, `retrieval_v1.yaml`, `reranker_v1.yaml`, `query_rewrite_v1.yaml`, `evidence_sufficiency_v1.yaml`, `answer_generation_v1.yaml`, `citation_audit_v1.yaml`, `pipeline_v1.yaml`。
- Legacy：`baseline.yaml`, `corpus_profile.yaml`，由早期模块使用或保留作迁移参考。
- Portability issue：部分 active YAML 使用本机绝对路径。本轮不改变运行配置，只在 README 与限制中明确记录。

## 5. Python File Classification

以下覆盖任务指定目录中的全部 108 个 Python 文件。

### KEEP (49)

Current package and integration entry points:

```text
researchguard/__init__.py
researchguard/cli.py
researchguard/pipeline.py
```

Parser and chunking:

```text
researchguard/ingestion/__init__.py
researchguard/ingestion/block_detector.py
researchguard/ingestion/chunk_builder.py
researchguard/ingestion/heading_classifier.py
researchguard/ingestion/layout_extractor.py
researchguard/ingestion/parse_pdf.py
researchguard/ingestion/section_recovery.py
```

Indexing:

```text
researchguard/indexing/__init__.py
researchguard/indexing/chroma_index.py
researchguard/indexing/chroma_metadata.py
researchguard/indexing/corpus_loader.py
researchguard/indexing/dense_index.py
researchguard/indexing/embedding_cache.py
researchguard/indexing/embedding_provider.py
researchguard/indexing/index_v1.py
researchguard/indexing/sparse_index.py
```

Retrieval through audit:

```text
researchguard/retrieval/__init__.py
researchguard/retrieval/answer_cache.py
researchguard/retrieval/answer_generator.py
researchguard/retrieval/answer_pipeline.py
researchguard/retrieval/chroma_retriever.py
researchguard/retrieval/citation_audit.py
researchguard/retrieval/citation_cache.py
researchguard/retrieval/claim_extractor.py
researchguard/retrieval/claim_verifier.py
researchguard/retrieval/dense_backend.py
researchguard/retrieval/evidence_cache.py
researchguard/retrieval/evidence_judge.py
researchguard/retrieval/evidence_pipeline.py
researchguard/retrieval/filters.py
researchguard/retrieval/index_loader.py
researchguard/retrieval/models.py
researchguard/retrieval/multi_query.py
researchguard/retrieval/query_rewrite_pipeline.py
researchguard/retrieval/query_rewriter.py
researchguard/retrieval/rerank_cache.py
researchguard/retrieval/rerank_pipeline.py
researchguard/retrieval/reranker.py
researchguard/retrieval/retrieval_v1.py
researchguard/retrieval/rewrite_cache.py
```

Build, retrieval, and presentation entry points:

```text
scripts/build_chroma_v1.py
scripts/build_chunks_v1.py
scripts/build_index_v1.py
scripts/retrieve_v1.py
demo/app.py
demo/utils.py
```

### TEST_ONLY (14)

These scripts are intentionally kept outside `tests/` because they are executable stage-level benchmark/report generators:

```text
scripts/run_functional_validation.py
scripts/run_local_rag_validation.py
scripts/validate_answer_generation_v1.py
scripts/validate_chroma_v1.py
scripts/validate_chunking_v1.py
scripts/validate_citation_audit_v1.py
scripts/validate_demo_v1.py
scripts/validate_evidence_sufficiency_v1.py
scripts/validate_index_v1.py
scripts/validate_parser_v5.py
scripts/validate_pipeline_v1.py
scripts/validate_query_rewrite_v1.py
scripts/validate_reranker_v1.py
scripts/validate_retrieval_v1.py
```

### LEGACY (45)

Agent and API placeholders:

```text
researchguard/agent/__init__.py
researchguard/agent/legacy_agentic_rag.py
researchguard/api/__init__.py
```

Earlier rule-based audit stack, still referenced by legacy CLI/validation:

```text
researchguard/audit/__init__.py
researchguard/audit/answer_auditor.py
researchguard/audit/base_skill.py
researchguard/audit/citation_in_paper_check_skill.py
researchguard/audit/evidence_table_build_skill.py
researchguard/audit/evidence_verdict_validator.py
researchguard/audit/internal_consistency_check_skill.py
researchguard/audit/numerical_claim_check_skill.py
researchguard/audit/paper_claim_extraction_skill.py
researchguard/audit/paper_error_audit_skill.py
researchguard/audit/reference_support_check_skill.py
researchguard/audit/source_evidence_extract_skill.py
```

Earlier evaluation and index builder:

```text
researchguard/evaluation/__init__.py
researchguard/evaluation/agentic_judge.py
researchguard/evaluation/evaluate_agentic_rag.py
researchguard/evaluation/evaluate_answer.py
researchguard/evaluation/evaluate_retrieval.py
researchguard/indexing/index_builder.py
```

Earlier ingestion utilities:

```text
researchguard/ingestion/check_clean_quality.py
researchguard/ingestion/parse_hybrid_pdf_ocr.py
researchguard/ingestion/parse_pdf_old.py
```

Memory, parser, and reporting abstractions retained for migration:

```text
researchguard/memory/__init__.py
researchguard/memory/evidence_memory.py
researchguard/memory/failure_memory.py
researchguard/memory/hypothesis_memory.py
researchguard/memory/literature_memory.py
researchguard/memory/memory_store.py
researchguard/memory/review_memory.py
researchguard/memory/skill_trace_memory.py
researchguard/memory/source_memory.py
researchguard/memory/tool_trace_memory.py
researchguard/parsers/__init__.py
researchguard/parsers/reference_parser.py
researchguard/reporting/__init__.py
researchguard/reporting/audit_report.py
researchguard/reporting/html_renderer.py
researchguard/reporting/markdown_renderer.py
researchguard/reporting/qq_text_renderer.py
```

Earlier shared utilities and repair entry point:

```text
researchguard/schemas.py
researchguard/text_utils_v2.py
researchguard/utils_ids.py
scripts/repair_researchguard.py
```

### UNUSED (0)

No file was classified as confidently unused. A missing incoming AST import was not treated as proof of non-use when the file is a CLI/module entry point, validation target, plugin-style component, or retained migration asset.

### Supplementary tracked Python outside the requested roots

```text
frontend/streamlit_app.py    LEGACY - earlier frontend entry, not used by Demo v1
main.py                      LEGACY - earlier root entry, not used by unified CLI
```

## 6. Retained Legacy and Structural Risks

- `researchguard/audit/` and `researchguard/reporting/audit_report.py` remain reachable from `researchguard.cli smoke-audit`.
- `researchguard/agent/`, `memory/`, old evaluation/reporting and old parser helpers form a historical implementation set with internal dependencies.
- `scripts/run_functional_validation.py` and `run_local_rag_validation.py` exercise that historical set.
- `EvidenceClaw` and `rag_agent_harness` were tracked as Git mode `160000` without `.gitmodules` mappings. Agent Intelligence Upgrade v1 completed the confirmed migration by removing only the parent-index entries while retaining and ignoring the local nested repositories.
- Empty `tests/` is not deleted because current executable validations intentionally live under `scripts/`; a later test-runner migration should be atomic.
- Some active YAML paths are machine-specific; changing them without a path migration test could break current local validation.

## 7. Cleanup Actions

Completed in this phase:

1. Inventoried code, configs, data and generated outputs.
2. Traced current and legacy dependency closures.
3. Classified every Python file in the requested roots.
4. Replaced the development-log README with a concise GitHub project page.
5. Documented evaluation scope, limitations and retained legacy assets.

Not performed:

- No source deletion.
- No directory move.
- No import rewrite.
- No config, benchmark, data, cache or output edit.
- No Pipeline behavior change.
- No push.

## 8. Validation Record

文档编辑完成后、cleanup commit 前，使用项目虚拟环境执行：

```text
python -m compileall researchguard
python -m researchguard.cli --help
python scripts/validate_pipeline_v1.py
python scripts/validate_demo_v1.py
```

Acceptance requires compile success, CLI help success, a grounded strong case with all stages completed, an unsupported case with answer/audit skipped, and Streamlit startup/render contracts passing.

实际结果：

| Check | Result | Details |
|---|---|---|
| `python -m compileall -q researchguard` | PASS | no syntax/import compilation error |
| `python -m researchguard.cli --help` | PASS | `status`, `check-imports`, `smoke-audit`, `run` available |
| `scripts/validate_pipeline_v1.py` | PASS | 6/6 synthetic tests passed; all 8 hard-check counters are 0 |
| Strong pipeline case | PASS | `grounded`; rewrite/retrieval/reranking/evidence/answer/audit all completed; 1 answer citation; audit grounded |
| Partial pipeline case | PASS | `rejected`; answer generation and citation audit skipped |
| Unsupported pipeline case | PASS | `rejected`; 0 citations; answer generation and citation audit skipped |
| Retrieval regression | PASS | direct retrieval and Pipeline returned identical chunk ID order |
| Cache stability | PASS | rewrite, rerank, evidence, answer and audit repeat reads hit cache |
| `scripts/validate_demo_v1.py` | PASS | startup, strong render, unsupported render and display sanitization passed; all 4 hard-check counters are 0 |

Streamlit validator emitted the expected `missing ScriptRunContext` warning because it runs the app in bare test mode; no startup exception was recorded.
