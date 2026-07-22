# ResearchGuard

ResearchGuard 是面向科研论文的多文档 Agentic RAG 与 Evidence Audit 平台原型。它不是普通的 PDF 问答 demo：项目目标是把论文 PDF 解析成可追溯的结构化 evidence，再服务于后续 indexing、retrieval、Agentic RAG、claim-level audit、报告生成和可复现实验。

当前工程重点已经从早期迁移代码，推进到真实论文解析、section-aware chunking、Embedding + Persistent Indexing v1、Retrieval v1、Chroma Vector Database Backend v1、Reranker v1、Query Rewrite v1、Evidence Sufficiency v1、Answer Generation v1 和 Citation Audit v1 验收阶段。Parser v5 已完成严格验收，结论为 `PASS_WITH_CHUNK_FIXES`；Chunking v1 已完成最终边界修复，当前为 `APPROVED_FOR_INDEXING`；Indexing、Retrieval、Chroma、Reranker、Query Rewrite、Evidence Sufficiency 和 Answer Generation 均通过独立验证。Claim Verification / Citation Audit v1 已能把答案拆成 atomic claims，并逐 claim 校验 answer-time evidence；12 个答案、26 个 claims 的 8 项 hard checks 全部为 0，结论为 `PASS`。Retrieval 当前默认使用 Chroma dense backend；reranker、query rewrite、evidence check、answer generation 和 citation audit 默认关闭，可通过 CLI 显式启用。Agent workflow、LangGraph、多 Agent 和产品化 API/UI 仍未接入新主流程。

## 1. 项目背景与目标

ResearchGuard 关注的是科研论文场景中 RAG 的可信性问题：

```text
论文 PDF
  -> 结构化解析
  -> section-aware chunks
  -> embedding / indexing
  -> query analysis / normalized rewrite / optional expansions
  -> per-query dense + BM25 + RRF candidate retrieval
  -> cross-query RRF fusion
  -> Cross-Encoder reranking
  -> Evidence Sufficiency / Answerability Detection
  -> strong: Evidence-grounded Answer Generation + citation metadata
  -> partial / unsupported: reject or request more evidence
  -> claim extraction
  -> claim verification against answer-time evidence
  -> Citation Audit: supported / partial / unsupported per claim
  -> JSON / Markdown report
  -> memory / trace
```

项目最终希望支持以下能力：

- 对多篇科研论文构建结构化语料；
- 检索时保留 page、section、heading、block_id 等来源信息；
- 生成回答后拆分事实 claim；
- 将 claim 与 evidence 对齐，判断是否 supported、partial 或 unsupported；
- 输出可复现的 audit trace 和报告；
- 为后续多文档 Agentic RAG 提供可靠输入。

当前已经重点完成的是 `PDF -> layout/block/section -> section-aware chunk -> index_v1 -> query rewrite/multi-query -> NumPy/Chroma dense backend -> Dense + BM25 + RRF -> Cross-Encoder rerank -> Evidence Sufficiency -> Evidence-grounded Answer Generation -> Claim Verification / Citation Audit` 这段主链路。后续 Agent workflow、自动 retry/evidence expansion 和产品化 API/UI 还需要继续接入。

## 2. 当前完整数据流

### 2.1 PDF -> PyMuPDF layout extraction

- 输入：`data/raw_docs/parser_eval/*.pdf` 中的原始论文 PDF。
- 输出：每页的 line-level layout，写入 `layout.json`。
- 方法：使用 PyMuPDF `page.get_text("dict")` 读取 text block、line、span、font、bbox、font size、font name，再合并 span 为 line。
- 为什么这样做：普通纯文本抽取会丢失栏、标题、caption、表格和页内位置；layout 信息是后续 reading order、heading、section 判断的基础。
- 在 RAG 中的作用：为 chunk metadata 提供 page、bbox、font、column 等可追溯依据。
- 核心文件：`researchguard/ingestion/layout_extractor.py`。

### 2.2 Noise filtering

- 输入：`layout.json` 中的 page lines。
- 输出：过滤页码、空行、arXiv 页眉、preprint / under review 等噪声后的 lines。
- 方法：`is_noise_line()` 对纯数字、空白、常见页眉文本做规则过滤。
- 为什么这样做：噪声进入 block 会造成 heading false positive、section 抖动和无意义短 chunk。
- 在 RAG 中的作用：减少 retrieval 召回页眉页脚、编号碎片的概率。
- 核心文件：`researchguard/ingestion/block_detector.py`。

### 2.3 Column detection

- 输入：过滤后的 page lines、页面宽度、body font size。
- 输出：每行增加 `column` 字段。
- 方法：先用接近 body font 的长正文行判断是否双栏；双栏页按 x 坐标分左栏/右栏，跨栏宽行保留为 column 0。
- 为什么这样做：论文常见双栏排版，如果只按 y 坐标排序，会出现左右栏交叉、右栏提前等错误。
- 在 RAG 中的作用：恢复接近人类阅读顺序的 block 序列，避免一个 chunk 混入无关上下文。
- 核心文件：`researchguard/ingestion/block_detector.py`。

### 2.4 Reading-order recovery

- 输入：带 column 的 lines。
- 输出：按阅读顺序排列的 lines。
- 方法：双栏页按 `(column, y0, x0)` 排序，单栏页按 `(y0, x0)` 排序。
- 为什么这样做：保证左栏正文先于右栏正文，跨栏标题和宽块尽量在合理位置出现。
- 在 RAG 中的作用：保证后续 block、chunk 的上下文顺序可用。
- 核心文件：`researchguard/ingestion/block_detector.py`。

### 2.5 Line-to-block

- 输入：排序后的 lines。
- 输出：`blocks.jsonl` 中的 block-level 结构。
- 方法：按 column、y gap、font size continuity 合并相邻 lines；遇到 standalone heading candidate 时强制切块。
- 为什么这样做：RAG chunk 不应该直接基于碎 line，而应基于语义较稳定的 paragraph / caption / equation / table / heading block。
- 在 RAG 中的作用：提供 `source_block_ids`，支持 chunk 可追溯和后续 evidence audit。
- 核心文件：`researchguard/ingestion/block_detector.py`。

### 2.6 Block type classification

- 输入：TextBlock 文本和 layout 特征。
- 输出：`paragraph`、`heading_candidate`、`caption`、`table`、`equation`、`reference_entry` 等 block type。
- 方法：规则识别 Figure/Table caption、reference 编号、符号密度较高的 equation、数字短行密集的 table、疑似独立标题行。
- 为什么这样做：不同 block 类型在 chunking 中处理方式不同，尤其 equation/caption/table 不能被静默丢弃。
- 在 RAG 中的作用：决定 chunk type、special block coverage 和上下文绑定策略。
- 核心文件：`researchguard/ingestion/block_detector.py`。

### 2.7 Heading classification

- 输入：block、body font size、page size。
- 输出：heading prediction，包括 `is_heading`、`section`、`score`、`confidence`、`reasons`、`normalized`。
- 方法：结合字体大小、bold、编号、全大写、短文本、靠近页顶、左对齐、section alias 命中等正向特征；用句子形态、reference entry、bibliographic venue、caption、table/chart numeric block、第一页 title area 等负向特征降权或排除。
- 为什么这样做：科研论文标题形式多样，单靠字号或正则都会误判。
- 在 RAG 中的作用：为 section recovery 和 chunk heading_path 提供结构锚点。
- 核心文件：`researchguard/ingestion/heading_classifier.py`。

### 2.8 Section recovery

- 输入：layout、blocks、heading predictions。
- 输出：`parsed_pages.jsonl` 和带 block-level section 的 `blocks.jsonl`。
- 方法：使用 heading classifier 的 section mapping，加上 page text implicit section 检测；通过 `soft_transition_score()` 限制 section 回跳，尤其防止 References 之后错误回到 main_text/method。
- 为什么这样做：论文页级 dominant section 不足以指导 chunking；chunk 必须依赖 block-level section。
- 在 RAG 中的作用：保证 embedding 片段有稳定 section metadata，后续检索和 audit 可以按 abstract/method/results/references 等过滤或解释。
- 核心文件：`researchguard/ingestion/section_recovery.py`。

### 2.9 Section-aware chunking

- 输入：parser v5 的 `blocks.jsonl`。
- 输出：`data/parsed/chunk_eval_v1/<paper>/chunks.jsonl`。
- 方法：严格按 block-level section 切断；heading 绑定正文；长 block 按句子拆；短 chunk 只与同 section 邻居合并；overlap 只允许来自文档顺序中直接前一个且同 section 的 chunk，并支持 `overlap_sentences=0/1/2/...`；overlap 只从前一个 chunk 的正文 body segments 提取，不包含 heading prefix，且 `overlap_source_block_ids` 覆盖每句 overlap 的真实来源；equation/table/caption 会按同 section、优先同页、y 坐标和 block 顺序绑定到最近正文，caption 优先绑定邻近 table；special block 与正文绑定后，加入 heading prefix 的最终文本仍必须不超过 `max_chars`；References 使用特殊切分；所有 special block 保留进 chunk。
- 为什么这样做：embedding 输入必须同时满足长度、section 边界、来源可追溯和特殊块不丢失。
- 在 RAG 中的作用：这是 indexing/retrieval 的稳定输入层。
- 核心文件：`researchguard/ingestion/chunk_builder.py`、`scripts/build_chunks_v1.py`。

### 2.10 Embedding + Persistent Indexing

- 输入：`data/parsed/chunk_eval_v1/<paper>/chunks.jsonl`。
- 输出：`data/indexes/index_v1/` 下的 corpus manifest、embedding cache、dense index、sparse BM25 index 和 index manifest。
- 方法：先生成 deterministic corpus manifest，为每个 chunk 计算 `content_hash` 和 `metadata_hash`；使用 OpenAI `text-embedding-3-small` 生成 1536 维向量；按 `provider + model + content_hash` 持久化 embedding cache；dense index 使用本地 NumPy `.npy` + JSON/JSONL 保存；sparse index 使用本地 BM25 统计 JSON 保存。
- 为什么这样做：NumPy exact search 是透明、可复现的精确基线，BM25 提供词法召回；两者先建立稳定索引语义，再由独立 Chroma 阶段复用同一批 vectors，而不是把 embedding 生成和数据库接入混在一起。
- 在 RAG 中的作用：为后续 retrieval 提供稳定的向量和稀疏召回输入，同时保留 chunk metadata、source provenance、overlap provenance 和 corpus fingerprint。
- 核心文件：`researchguard/indexing/corpus_loader.py`、`researchguard/indexing/embedding_provider.py`、`researchguard/indexing/embedding_cache.py`、`researchguard/indexing/dense_index.py`、`researchguard/indexing/sparse_index.py`、`researchguard/indexing/index_v1.py`、`scripts/build_index_v1.py`、`scripts/validate_index_v1.py`。

### 2.11 Chroma persistent vector backend

- 输入：`data/indexes/index_v1` 中已经生成并验证的 `corpus_manifest.jsonl`、`vectors.npy`、`ids.json` 和 manifests。
- 输出：`data/indexes/chroma_v1` 下的本地 Chroma PersistentClient 数据库，collection 为 `researchguard_papers_v1`。
- 方法：显式传入已有 337 个 `text-embedding-3-small` 1536 维向量、完整 chunk text 和可过滤 metadata；禁用 Chroma 默认 embedding function；以 `chunk_id` 作为 record ID；按 `content_hash` / `metadata_hash` 增量 add、upsert、metadata update 和 stale delete。
- 为什么这样做：在保留 NumPy 精确基线的同时，提供可重启加载、原生 metadata filtering 和可增量维护的向量数据库后端，不重复调用文档 embedding API。
- 在 RAG 中的作用：作为 Retrieval v1 可配置的 dense candidate source；BM25 和 RRF 逻辑不变。
- 核心文件：`researchguard/indexing/chroma_index.py`、`researchguard/indexing/chroma_metadata.py`、`researchguard/retrieval/chroma_retriever.py`、`scripts/build_chroma_v1.py`、`scripts/validate_chroma_v1.py`。

### 2.12 Retrieval v1

- 输入：`data/indexes/index_v1/` 中的 `index_manifest.json`、`corpus_manifest.jsonl`、dense NumPy index、BM25 sparse index，以及 `configs/retrieval_v1.yaml`。
- 输出：query 的 ranked chunks，包含 `chunk_id`、doc/page/section、heading_path、source block provenance、dense_score、sparse_score、fusion_score、dense_rank、sparse_rank 和 retrieval_sources。
- 方法：loader 先严格校验 index build status、corpus fingerprint、dense/sparse chunk_id 顺序、embedding dimension、metadata 完整性和 schema；dense candidate source 可配置为 `numpy` 或 `chroma`，当前默认 Chroma，NumPy 保留为精确基线；sparse 使用已持久化 BM25；hybrid 使用 Reciprocal Rank Fusion 合并 dense/BM25 排名；metadata filters 在三种模式中保持相同语义。
- 为什么这样做：Retrieval 必须只消费已验收的 `index_v1`，不能重新读 PDF 或临时重建 vector DB；dense 负责语义召回，BM25 负责术语、缩写和数值/表格词召回，RRF hybrid 在不训练 reranker 的前提下提供稳定融合。
- 在 RAG 中的作用：为后续 Agentic RAG 和 Evidence Audit 提供可追溯 evidence candidates，但当前不生成答案、不判断证据充分性、不做 citation audit。
- 核心文件：`researchguard/retrieval/index_loader.py`、`researchguard/retrieval/dense_backend.py`、`researchguard/retrieval/chroma_retriever.py`、`researchguard/retrieval/retrieval_v1.py`、`researchguard/retrieval/models.py`、`researchguard/retrieval/filters.py`、`scripts/retrieve_v1.py`、`scripts/validate_retrieval_v1.py`。

### 2.13 Cross-Encoder reranking

- 输入：Hybrid RRF 已排好序的 Top-N candidates；每个候选继续携带 chunk text、doc/section/page、dense/BM25/RRF score 与 rank、source provenance。
- 输出：最终 Top-K ranked chunks；保留原字段并增加 `rerank_score`、`rerank_rank`、`pre_rerank_rank`、`reranker_backend` 和 `reranker_model`。
- 方法：当前 backend 为 `cross-encoder/ms-marco-MiniLM-L6-v2`，固定 revision `c5ee24cb16019beea0893ab7796b1df96625c6b8`。模型直接对 query 与 `Title/Section/Heading/Content` 文本对打分；默认精排 20 个候选并返回 10 个。结果按 query、chunk content hash、模型、配置版本和输入模板版本缓存。
- 为什么这样做：Hybrid `Recall@10=0.9444` 说明正确证据通常已进入候选集，但 `MRR@10=0.6847` 仍说明第一条正确证据不总在前列。RRF 负责扩大和稳定候选召回，Cross-Encoder 负责 query-chunk 交互式相关性判断。
- 在 RAG 中的作用：提高最终 evidence list 的前排精度，为后续 evidence sufficiency 和 answer generation 减少排序噪声；它不负责 no-answer 判断。
- 核心文件：`researchguard/retrieval/reranker.py`、`researchguard/retrieval/rerank_cache.py`、`researchguard/retrieval/rerank_pipeline.py`、`researchguard/retrieval/retrieval_v1.py`、`scripts/retrieve_v1.py`、`scripts/validate_reranker_v1.py`、`configs/reranker_v1.yaml`。

### 2.14 Query analysis、rewrite 与跨查询融合

- 输入：用户原始 query；原始文本始终保留，不接收 benchmark labels、`query_id` 或 `relevant_chunk_ids`。
- 输出：1 条 normalized query、最多 2 条合法 expansion queries、preserved entities/constraints、fallback/cache trace；多查询检索后为每个 chunk 增加 query variant provenance 和 multi-query RRF score/rank。
- 方法：本地规则先抽取模型名、数据集、数字、Table/Figure/Equation 编号、缩写、否定、时间和比较关系；`gpt-4.1-mini` 以 temperature 0 和 strict JSON schema 生成 rewrite；非法 expansion 被丢弃并记录，normalized query 非法或 API/JSON 失败时回退原始 query。multi-query 模式保留原始 query，与 normalized/expansions 分别执行 Hybrid Retrieval，再以 `chunk_id` 去重做跨查询 RRF；最后使用原始 query 进行 Cross-Encoder rerank。
- 为什么这样做：单条 query 可能过长、缺术语、混合精确实体和语义描述；扩展可增加表达覆盖，但必须通过原始 query、实体校验和最终 reranker 抑制漂移。
- 在 RAG 中的作用：改善查询表达、术语覆盖和候选召回；不判断 corpus 是否有答案、evidence 是否充分或最终回答是否可信。
- 核心文件：`researchguard/retrieval/query_rewriter.py`、`researchguard/retrieval/query_rewrite_pipeline.py`、`researchguard/retrieval/rewrite_cache.py`、`researchguard/retrieval/multi_query.py`、`researchguard/retrieval/retrieval_v1.py`、`scripts/validate_query_rewrite_v1.py`、`configs/query_rewrite_v1.yaml`。

### 2.15 Evidence Sufficiency 与 Answerability Detection

- 输入：原始用户 query 和经过 Query Rewrite、Hybrid Retrieval、跨查询 RRF、Cross-Encoder rerank 后的最终 Top-k chunks；不输入 benchmark answer、`query_id`、expected label 或 `relevant_chunk_ids`。
- 输出：`answerable`、`support_level`、`confidence`、`reason` 和真实 `supporting_chunk_ids`，另带 model/prompt/config、cache、token 和 latency trace。
- 方法：`gpt-4.1-mini`、temperature 0、OpenAI strict JSON schema；模型内部先列出已支持和缺失的 material requirements，再由确定性规则映射 `strong/partial/unsupported`。任何 API/JSON/schema/provenance 失败都保守回退为 `unsupported`，禁止 answer generation。
- 为什么这样做：Retriever 只负责找到相关内容，Reranker 只负责把更相关的内容排到前面；两者都会为 no-answer query 返回最近邻。Evidence Judge 单独判断这些 passages 是否足够支持回答，避免把“相关”误当作“有答案”。
- 在 RAG 中的作用：作为当前 Answer Generation v1 的强制 gate；本模块本身只输出 verdict，不生成答案、不提取 claim、不做 citation audit，也不能证明整个 corpus 中绝对没有答案。
- 核心文件：`researchguard/retrieval/evidence_judge.py`、`researchguard/retrieval/evidence_pipeline.py`、`researchguard/retrieval/evidence_cache.py`、`scripts/validate_evidence_sufficiency_v1.py`、`configs/evidence_sufficiency_v1.yaml`。

### 2.16 Evidence-grounded Answer Generation

- 输入：原始用户 query、最终 reranked hits，以及 Evidence Sufficiency v1 的 `answerable/support_level/supporting_chunk_ids`；benchmark labels、标准答案、`relevant_chunk_ids` 和 expected output 不进入生成模型。
- 输出：严格 JSON 核心字段 `answer`、`citations`、`confidence`；每条 citation 包含真实 `chunk_id/doc_id/section/page`，另附 cache、fallback、token 和 latency trace。
- 方法：仅当 `answerable=true` 且 `support_level=strong` 时调用 `gpt-4.1-mini`；生成器只读取 `supporting_chunk_ids` 对应的 passages。Prompt 明确要求 “If evidence does not support a claim, do not include it.”，禁止外部知识和无证据补全。Pipeline 会校验 citation ID 属于允许 evidence 且 metadata 与原 hit 完全一致；API/JSON/schema/provenance 失败统一 fail closed。
- Gate：`partial/unsupported` 或 Judge fallback 不调用 generator，固定返回 `Insufficient evidence in the current corpus.`、空 citations 和 confidence 0。
- Cache：key 覆盖 normalized query、按顺序排列的 evidence chunk IDs、完整 input hash、model、prompt/config version 和 generation config；目录为 `data/cache/answer_generation_v1`，不提交 Git。
- 在 RAG 中的作用：把已判定充分的证据转换为简洁回答，并保留 source provenance；它不做 claim extraction、claim verification 或 citation entailment/completeness audit。
- 核心文件：`researchguard/retrieval/answer_generator.py`、`researchguard/retrieval/answer_pipeline.py`、`researchguard/retrieval/answer_cache.py`、`scripts/validate_answer_generation_v1.py`、`configs/answer_generation_v1.yaml`。

### 2.17 Claim Verification / Citation Audit

- 输入：Answer Generation v1 的原始 answer、answer citations、generation-time `evidence_chunk_ids`，以及这些 IDs 在当前 hits 中对应的 text/metadata；不输入 benchmark labels。
- 输出：原答案、按顺序排列的 atomic claims、每个 claim 的 `supported/partial/unsupported`、confidence、canonical citations、reason，以及 `overall_grounded`、unsupported/partial count 和 grounding score。
- 方法：`gpt-4.1-mini` 先用 strict JSON 拆 claim，再逐 claim 独立验证。候选 evidence 先放答案 citations，再按 claim 词项重合度补充同次 Answer Generation 已见 evidence；审计层不重新调用 Retriever。数字、单位、模型名、否定和比较方向必须精确匹配。
- Provenance：Verifier 只能返回当前 claim candidate 中的 chunk IDs；Pipeline 使用原 evidence 重建 `chunk_id/doc_id/section/page`，unsupported claim 必须为空 citations，越界 ID 和 schema 错误 fail closed。
- Cache：key 覆盖 answer hash、claim extraction/verification prompt versions、generation evidence IDs、model、config 和完整 input hash；目录为 `data/cache/citation_audit_v1`，不提交 Git。
- 在 RAG 中的作用：Citation Coverage 只回答“答案是否附带引用”，Citation Audit 进一步回答“每个事实 claim 是否真的被所引 evidence 支持”。本阶段不是 Agent，也不证明论文来源本身为真。
- 核心文件：`researchguard/retrieval/claim_extractor.py`、`researchguard/retrieval/claim_verifier.py`、`researchguard/retrieval/citation_audit.py`、`researchguard/retrieval/citation_cache.py`、`scripts/validate_citation_audit_v1.py`、`configs/citation_audit_v1.yaml`。

### 2.18 Unified ResearchGuard Pipeline v1

- 输入：一条原始用户 query 和 `configs/pipeline_v1.yaml`；不读取 benchmark label、标准答案或 relevant chunk IDs。
- 输出：统一 JSON，包含 `query`、六个阶段 envelope（`rewrite/retrieval/reranking/evidence_check/answer_generation/citation_audit`）、`final_status` 和 pipeline 级 start/end/latency。
- 方法：`researchguard.pipeline.ResearchGuardPipeline` 只编排现有模块。Rewrite、跨查询 RRF 和 Reranker 仍由 `RetrievalEngine` 执行；Evidence Judge 是不可绕过的生成 gate。只有 `answerable=true && support_level=strong` 才调用 Answer Generation，`partial/unsupported` 直接结束为 `rejected`；已生成答案再进入 Citation Audit。
- 为什么这样做：独立模块已经分别通过验证，但手工拼 CLI flags 容易漏 gate、漏 provenance 或混用配置。统一编排层固定依赖关系、状态、错误边界和可观测性，不改变任何检索或判断算法。
- 在 RAG 中的作用：形成可程序调用和可命令行运行的 evidence-grounded 主链路；它仍不是 Agent、LangGraph workflow 或自动 retry 系统。
- 核心文件：`researchguard/pipeline.py`、`researchguard/cli.py`、`configs/pipeline_v1.yaml`、`scripts/validate_pipeline_v1.py`。

### 2.19 后续 agent/workflow

- 输入：计划使用当前 Retrieval/Evidence/Answer/Audit 主链路的结构化结果和 provenance。
- 输出：尚未正式接入新主流程。
- 当前状态：旧 agent 和 evaluation 模块存在，但尚未围绕新主链路重构成正式 Agentic RAG、自动 retry 或多 Agent workflow。
- 核心目录：`researchguard/agent`、`researchguard/audit`、`researchguard/evaluation`、`researchguard/reporting`。

## 3. Parser 详细说明

### 3.1 PyMuPDF layout extraction

`layout_extractor.py` 使用 PyMuPDF 打开 PDF，逐页调用 `page.get_text("dict")`。对每个 text block 内的 line 和 span，提取：

- span text；
- span font size；
- span font name；
- span bbox；
- line/page/block 序号；
- bold / italic 信号；
- line width / height。

多个 span 合并成一个 `LayoutLine`，bbox 使用所有 span 的 min/max，font size 使用 median，font name 使用众数。输出结构写入 `layout.json`。

### 3.2 Body font size 估计

`estimate_body_font_size()` 优先选择长度足够、词数足够的正文候选行，把字号 round 到 0.5 后取众数。若候选不足，则退回所有有效字号的众数；再失败时默认 `10.0`。

这样做的原因是 heading 判断和双栏判断都依赖 body font。直接使用 PDF metadata 不可靠，按正文行众数估计更稳。

### 3.3 单栏和双栏识别

`is_two_column_page()` 从接近 body font、长度足够的正文行中统计左右 x 分布。如果左右两侧都有足够正文行，认为该页是双栏。`detect_column()` 对双栏页按 x 坐标分 column；宽度超过页面约 68% 的行视为跨栏，保留 column 0。

当前双栏恢复是规则型 layout recovery，不是完整视觉版面分析。它在 v5 验收样本中没有发现 column backtrack，但复杂跨栏表格仍可能有限。

### 3.4 阅读顺序恢复

单栏页按 `(y0, x0)` 排序。双栏页按 `(column, y0, x0)` 排序，先左栏后右栏。parser v5 验收对每篇抽取第一页、中间页、References 首页、表格/图片较多页，输出 block 顺序供人工检查。

v5 结果显示抽样页 `column_backtracks=0`，`y_backtracks_within_column=0`。报告仍将这些样本标为 `needs_manual_review=True`，因为 reading order 本身需要人工浏览确认。

### 3.5 Line-to-block 合并

`detect_blocks()` 将排序后的 lines 合并为 block：

- column 改变时切块；
- y gap 过大时切块；
- font size 差异明显且间距较大时切块；
- 独立 heading candidate 单独成块；
- 行尾连字符会在 `clean_join_lines()` 中合并。

每个 block 记录 `block_id`、page、bbox、font、column、line_count、char_count、word_count。

### 3.6 Block type 识别

当前 block type 规则包括：

- `paragraph`：默认正文块；
- `heading_candidate`：layout 上像独立标题，但尚未经过 heading classifier 确认；
- `caption`：以 `Figure/Fig./Table + number` 开头；
- `table`：多行、数字多、短行比例高，并命中 Method/Score/Acc/AUC/Precision/Recall/F1/EM/PCC/SCC 等表格词；
- `equation`：符号密度高、公式符号或 LaTeX-like 标记明显；
- `reference_entry`：编号式 reference entry，例如 `[1] ...` 或 `1. Author, ...`。

### 3.7 Heading classifier

`heading_classifier.py` 首先做文本 normalization 和 section alias 映射。当前 alias 会统一到：

`abstract`、`introduction`、`related_work`、`method`、`experiment`、`results`、`discussion`、`limitations`、`conclusion`、`references`、`appendix`、`main_text`。

正向特征包括：

- font much larger / larger / slightly larger than body；
- bold font；
- numbered heading；
- all caps style；
- mapped to section；
- exact section heading；
- short heading-like length；
- near page top；
- left aligned。

负向或排除特征包括：

- reference entry；
- bibliographic venue text；
- caption；
- table/chart numeric block；
- sentence-like text；
- heading merged with paragraph sentence；
- first-page title area；
- paragraph-like starts such as `we` / `our` / `this paper`；
- sentence period ending；
- line hyphenation ending；
- broken citation or broken sentence。

classifier 输出 score、confidence 和 reasons，方便验收审查。

### 3.8 Soft transition section recovery

`section_recovery.py` 维护 `current_section`。显式 heading 和隐式 page text 都可以触发 section 转移，但必须通过 `soft_transition_score()`：

- 顺序向后转移轻微加分；
- `references` 后除 appendix 外的回跳会被强烈惩罚，低置信度时禁止；
- experiment/results/method 之间允许少量软回跳；
- 高置信 heading 可以突破部分回跳惩罚。

这比简单 page-level dominant section 更细，因为每个 block 都会带 section。page-level section 是该页主导 section，用于页面摘要；block-level section 是 chunking 的真实边界依据。

### 3.9 Parser v5 验收方式和结果

验收脚本：`scripts/validate_parser_v5.py`。

输入：`data/parsed/parser_eval_v5/<paper>/layout.json`、`blocks.jsonl`、`parsed_pages.jsonl`、`chunks.jsonl`、`parse_quality_report.json`。

输出：`outputs/parser_validation_v5/`。

主要检查：

- reading order 样本页；
- all heading audit，包括 score、confidence、reasons；
- block-level section transition trace；
- References heading 后 reference-like ratio；
- parser 旧 chunks 的长度、短 chunk、长 chunk、multi-section、heading-only、重复 block_id、公式覆盖；
- 人工抽样 chunk samples。

结论：`PASS_WITH_CHUNK_FIXES`。

含义：parser 的 reading order、heading、block-level section、references 基本可作为 chunking 输入；但 parser 阶段生成的旧 chunks 存在 chunking 问题，需要单独修 chunker。

Parser v5 主要结果：

| Paper | Pages | Headings suspicious/unmapped | Reading order backtracks | References found | Parser v5 conclusion source issue |
| --- | ---: | ---: | ---: | --- | --- |
| paper_rag | 19 | 0 / 0 | 0 | yes | chunk 仍有短块、heading-only、>1600、重复 block refs |
| paper_agent | 33 | 0 / 0 | 0 | yes | equation 0/4 进入旧 chunks，存在 >1600 和短块 |
| paper_hallucination | 14 | 0 / 0 | 0 | yes | equation 0/1 进入旧 chunks，heading-only |
| paper_corrective_rag | 16 | 0 / 0 | 0 | yes | equation 0/1 进入旧 chunks，heading-only |
| paper_citation | 24 | 0 / 0 | 0 | yes | equation 0/2 进入旧 chunks，且有 1 个 multi-section chunk |

### 3.10 Parser 当前真实限制

- OCR fallback 尚未接入当前 v5 主流程，扫描版或图片型 PDF 可能解析失败或文本很短。
- 复杂跨栏表格、跨页表格、图文环绕仍是规则式处理，不是完整视觉语义解析。
- caption/table/equation 的识别依赖启发式规则，可能误判或漏判。
- 当前 parser 输出中仍可见部分历史乱码，这会影响后续检索文本质量。
- page-level section 只是页面主导 section，不应单独作为 chunk 边界依据。

## 4. Chunking 详细说明

### 4.1 当前实现位置

核心实现：`researchguard/ingestion/chunk_builder.py`。

CLI：`scripts/build_chunks_v1.py`。

验收：`scripts/validate_chunking_v1.py`。

当前输入使用 parser v5 blocks：

```text
data/parsed/parser_eval_v5/<paper>/blocks.jsonl
```

当前输出写入：

```text
data/parsed/chunk_eval_v1/<paper>/chunks.jsonl
```

不会覆盖 `data/parsed/parser_eval_v5`。

### 4.2 Section 变化强制切断

`build_chunks()` 先按 `(page, column, y0, x0, block_id)` 排序。`chunk_from_blocks()` 维护 `active_section`，一旦 block-level section 改变，就 flush 当前 chunk 并重置 heading context。

这意味着即使当前 chunk 小于 `min_chars`，也不会跨 section 合并。`abstract -> introduction`、`method -> experiment`、`conclusion -> references`、`references -> appendix` 等边界都必须隔离。

### 4.3 Heading 绑定正文

遇到 `block_type=heading` 时，不立即输出 chunk，而是更新 `heading_path` 和 `pending_heading_ids`。后续第一个同 section 正文 chunk 会把 heading 文本作为开头 context，并记录：

- `section_heading`；
- `heading_path`；
- `heading_block_ids`。

连续 heading 会根据编号层级近似构建 hierarchy，例如 `3`、`3.1`、`A`、`A.1`。部分短 `heading_candidate` 在紧跟 heading 且像续标题时，会并入 heading context，避免产生 heading-only chunk。

### 4.4 长 block 拆分

默认参数：

```text
max_chars = 1600
target_chars = 1200
min_chars = 250
overlap_sentences = 1
```

长 block 处理规则：

- 优先按完整句子拆；
- 保护 `et al.`、`Fig.`、`Eq.`、`Sec.` 等常见缩写；
- 如果单句仍超过上限，再按分号、逗号、空格或安全字符切分；
- heading prefix 加入后仍要计入 1600 上限；
- 记录 `split_reason`、`split_part`、`split_total`；
- 对同一 chunk 内混入多个 split 来源的情况，记录 `split_blocks`，避免重复 source block 无法解释。

### 4.5 短 chunk 合并

`merge_short_chunks()` 对 `< min_chars` 的 chunk 先尝试与同 section 前一个 chunk 合并，再尝试与同 section 后一个 chunk 合并。合并后不得超过 `max_chars`。

如果无法合并，会保留短 chunk，并标记：

- `short_chunk: true`；
- `short_chunk_reason`。

当前验收中仍有少数 `<150` chunk，这些属于 v1 minor issues，未为了清零而跨 section 合并。

### 4.6 Overlap 当前实现

`apply_overlap()` 当前只在文档顺序中直接前一个 chunk 与当前 chunk 同 section 时使用 overlap，References 不使用 overlap。chunk 内部将最终展示文本 `parts` 与未带 heading 的 `body_segments` 分开保存，overlap 只从前一个 chunk 的 `body_segments` 取句，不会把 `heading_path` 或 heading prefix 复制到下一个 chunk。`overlap_sentences` 会真正控制复制句数：

- `overlap_sentences=0`：不生成 overlap；
- `overlap_sentences=1`：复制直接前一个同 section chunk 的最后 1 句；
- `overlap_sentences=2`：复制直接前一个同 section chunk 的最后 2 句；
- 更大的整数按同样规则取最后 N 句。

如果中间出现其他 section，即使更早之前有同 section chunk，也不能跨过去取 overlap。生成 overlap 时记录：

- `overlap_from_chunk_id`；
- `overlap_char_count`；
- `overlap_source_block_ids`。

当 `overlap_sentences=2` 且两句话来自不同 source block 时，`overlap_source_block_ids` 会按文档顺序稳定记录两个来源 block，而不是只记录最后一个 block。当前验收确认：`overlap_sentences=0/1/2` 合成用例通过，heading 不进入 overlap，References overlap 为 0，跨 section overlap 为 0，非直接前一个 chunk overlap 为 0，overlap 不导致 >1600。需要注意的是，overlap 是检索上下文复制，不是新的 source evidence；下游 audit 必须继续区分 `source_block_ids` 与 `overlap_source_block_ids`。

### 4.7 Equation、caption、table 保留

v1 chunker 不再跳过 equation。`block_content_type()` 将 `equation`、`caption`、`table` 显式加入 content_types。当前 special block 绑定策略是：

- 只在同 section 内绑定；
- 优先选择同页正文；
- 在候选正文中按 y 坐标距离、block 顺序距离选择最近者；
- caption 会先寻找同 section、优先同页的邻近 table，能放入同一 chunk 时先与 table 绑定；
- equation/table/caption 与正文合并前检查累计长度，合并后不得超过 `max_chars`；
- 如果 special block 与正文绑定后再加 heading prefix 会超过 `max_chars`，会把绑定 unit 拆回更小单位，让正文、table、caption、equation 在不越界的前提下合并或独立保留；
- 如果多个 special block 竞争同一个正文且累计会超过 `max_chars`，会保留无法容纳的 special block 独立或另寻可容纳正文，优先保证不越界和不丢失。

`annotate_special_block_ids()` 根据原始 block type 回填：

- `has_equation`、`equation_block_ids`；
- `has_caption`、`caption_block_ids`；
- `has_table`、`table_block_ids`。

当前绑定已经从单纯 reading order 合并升级为邻近绑定策略，但仍不是完整视觉语义绑定。复杂跨栏图表、跨页表格和图文环绕仍需要更强的视觉语义建模。

### 4.8 References 特殊切分

References section 使用 `split_reference_text()`：

- 如果存在 `[1]` 风格编号，按 reference entry 起点切分；
- 否则按 author/year-like 行近似拆分；
- 超长 reference entry 再回退到句子拆分；
- References 不加入 overlap。

### 4.9 Chunk metadata

当前 chunk 输出包含：

- identity：`chunk_id`、`doc_id`、`title`；
- section：`section`、`section_heading`、`heading_path`、`heading_block_ids`；
- type：`chunk_type`、`content_types`；
- page：`page_start`、`page_end`；
- provenance：`source_block_ids`、`block_ids`、`overlap_source_block_ids`；
- text stats：`text`、`char_count`、`word_count`；
- special blocks：`has_equation`、`equation_block_ids`、`has_table`、`table_block_ids`、`has_caption`、`caption_block_ids`；
- splitting：`split_reason`、`split_part`、`split_total`、`split_blocks`；
- short chunk：`short_chunk`、`short_chunk_reason`；
- overlap：`overlap_from_chunk_id`、`overlap_char_count`。

### 4.10 Chunk CLI

`scripts/build_chunks_v1.py` 支持：

```text
--blocks
--output
--doc_id
--title
--max_chars
--target_chars
--min_chars
--overlap_sentences
```

`--title` 可选；未传时会从同目录的 `parse_quality_report.json` 或 `layout.json` 读取。

### 4.11 Chunk validation

`scripts/validate_chunking_v1.py` 对 `data/parsed/chunk_eval_v1` 做独立验收，并与 `data/parsed/parser_eval_v5` 的 blocks 对齐。

检查包括：

- chunk 总数；
- min / p10 / median / p90 / max；
- `>1600`；
- `<150`；
- heading-only；
- multi-section；
- section metadata mismatch；
- source block duplicate 是否由 split metadata 解释；
- split metadata 完整性；
- paragraph/heading/reference 等 core block 是否丢失；
- equation/caption/table coverage；
- bindable special block 是否被孤立；
- caption 是否漏绑邻近 table；
- overlap 是否同 section；
- overlap 是否只来自直接前一个 chunk；
- `overlap_sentences=0/1/2` 是否按参数生效；
- heading 是否不会进入 overlap；
- `overlap_sentences=2` 跨两个 source block 时 provenance 是否准确；
- special block + 正文 + heading 接近 1600 时是否仍不越界；
- 同页两个 table 时 caption 是否选择 y 距离和顺序最近的 table；
- special block 是否不会跨 section 绑定；
- 相同 synthetic input 重复运行是否稳定；
- References 是否无 overlap；
- overlap 是否导致超长；
- heading_path / section_heading 缺失；
- cross-page chunk；
- chunk_id 唯一性；
- deterministic rebuild；
- empty text；
- char_count / word_count；
- page_start <= page_end；
- random 和 targeted samples。

### 4.12 Chunking v1 结果和含义

验收输出：`outputs/chunk_validation_v1/`。

验证脚本结论：`PASS_WITH_MINOR_ISSUES`。

Indexing 前置结论：`APPROVED_FOR_INDEXING`。

含义：synthetic tests 全部通过，五篇真实论文硬性检查全部通过，已经可以作为后续 embedding/indexing 的稳定输入；但仍有少数短 chunk、少数 heading_path 缺失和复杂视觉语义绑定优化空间，不应称为完全无缺。

主要指标：

| Paper | Chunks | Min | P10 | Median | P90 | Max | <150 | >1600 | Multi-section | Heading-only | Lost core | Lost eq/cap/table |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| paper_agent | 96 | 44 | 373 | 1454 | 1588 | 1598 | 3 | 0 | 0 | 0 | 0 | 0 |
| paper_citation | 80 | 57 | 682 | 1390 | 1588 | 1598 | 3 | 0 | 0 | 0 | 0 | 0 |
| paper_corrective_rag | 54 | 57 | 478 | 1378 | 1583 | 1596 | 1 | 0 | 0 | 0 | 0 | 0 |
| paper_hallucination | 46 | 95 | 335 | 1385 | 1578 | 1599 | 1 | 0 | 0 | 0 | 0 | 0 |
| paper_rag | 61 | 355 | 551 | 1343 | 1582 | 1599 | 0 | 0 | 0 | 0 | 0 | 0 |

Hard checks 全部为 0：

- `chunks_over_1600`；
- `multi_section_chunks`；
- `heading_only_chunks`；
- `section_mismatch_chunks`；
- `unexplained_repeated_source_block_ids`；
- `split_metadata_invalid_count`；
- `lost_core_blocks`；
- `lost_equation_blocks`；
- `lost_caption_blocks`；
- `lost_table_blocks`；
- `isolated_bindable_special_blocks`；
- `caption_nearest_table_misses`；
- `duplicate_chunk_ids`；
- `references_overlap_chunks`；
- `overlap_cross_section_chunks`；
- `overlap_not_direct_previous_chunks`；
- `overlap_over_max_chunks`；
- `empty_text_chunks`；
- `char_count_invalid`；
- `word_count_invalid`；
- `page_order_invalid`；
- `determinism_mismatch`。

Synthetic tests 全部通过：

- `overlap_parameter_and_direct_previous`；
- `heading_not_in_overlap`；
- `overlap_two_sentence_provenance`；
- `special_heading_budget`；
- `caption_nearest_table_choice`；
- `special_no_cross_section_binding`；
- `deterministic_synthetic_run`。

剩余问题：

- 8 个 `<150` chunk：paper_agent 3 个、paper_citation 3 个、paper_corrective_rag 1 个、paper_hallucination 1 个；
- paper_agent 有 4 个非 abstract / references chunk 缺 heading_path；
- overlap direct-previous 规则、`overlap_sentences=0/1/2` 参数语义、heading 不进入 overlap、跨 block provenance 已经补充验证；后续需要复核下游 indexing/audit 如何展示和消费 overlap provenance；
- equation/table/caption 邻近绑定已补充验证，覆盖率保持 100%，可绑定 special block 孤立数为 0，caption 漏绑邻近 table 数为 0；但它仍不是完整视觉语义绑定。

## 5. Retrieval v1 详细说明

### 5.1 当前实现位置

Retrieval v1 的配置、实现、CLI、benchmark 和验证脚本分别是：

- `configs/retrieval_v1.yaml`；
- `researchguard/retrieval/index_loader.py`；
- `researchguard/retrieval/dense_backend.py`；
- `researchguard/retrieval/chroma_retriever.py`；
- `researchguard/retrieval/retrieval_v1.py`；
- `researchguard/retrieval/models.py`；
- `researchguard/retrieval/filters.py`；
- `scripts/retrieve_v1.py`；
- `scripts/validate_retrieval_v1.py`；
- `data/eval/retrieval_v1_queries.jsonl`。

当前只读取：

```text
C:\Users\18449\Desktop\researchguard_workspace\data\indexes\index_v1
```

不会读取 PDF，不会重跑 parser/chunking/indexing，也不会绕过现有 index 临时重建 vector DB。

### 5.2 Index loader 和 hard checks

`load_index_bundle()` 会加载：

- `index_manifest.json`；
- `corpus_manifest.jsonl`；
- `dense/dense_manifest.json`、`dense/vectors.npy`、`dense/ids.json`、`dense/metadata.jsonl`；
- `sparse/bm25_index.json`；
- `configs/indexing_v1.yaml`。

加载时硬性检查：

- `build_status == complete`；
- `corpus_fingerprint` 与当前 corpus manifest 重新计算结果一致；
- dense manifest 和 sparse payload 中的 fingerprint 与 index manifest 一致；
- corpus、dense、sparse 三处 `chunk_id` 顺序完全一致；
- dense dimension 与 embedding config / manifest 一致；
- required metadata 不缺失；
- chunk_id 无重复；
- schema version 合法。

任何 hard check 非 0 都会使严格模式加载失败；validation 中也会将其计入 `FAIL`。

### 5.3 Dense retrieval

Dense retrieval 使用 `OpenAIEmbeddingProvider.embed_query()` 对真实 query 生成 `text-embedding-3-small` query embedding，并校验维度、有限值和非零向量。`DenseRetrieverBackend` 将 candidate source 抽象为两个可切换实现：

- `numpy`：对当前 337 个向量做 exact cosine search，作为精确基线和回退方案；
- `chroma`：把同一个 query vector 显式传给 Chroma `query_embeddings`，使用持久化 collection 和原生 metadata filter，当前为默认 backend。

两种 backend 都恢复为统一 candidate schema，按 `(score desc, chunk_id asc)` 稳定排序；上层 Retrieval API、BM25 和 RRF 不因 backend 变化而改变。可通过配置 `dense.backend` 或 CLI `--dense-backend numpy|chroma` 显式选择。

这样做的原因是 metadata filter 必须对所有候选一致生效；如果先截断 top-k 再过滤，可能因为过滤条件丢掉应召回的 chunk。

### 5.4 Sparse BM25 retrieval

Sparse retrieval 使用 `data/indexes/index_v1/sparse/bm25_index.json` 中已持久化的 token statistics、document frequency、document length 和 BM25 参数。query 使用和 indexing 相同的 tokenizer。BM25 同样对全 corpus 打分，应用相同 metadata filters，再按 `(score desc, chunk_id asc)` 稳定排序。

BM25 的作用是补 dense 的短术语、模型名、数据集名、缩写、公式/表格关键词和 reference-like 查询。

### 5.5 Hybrid retrieval

Hybrid retrieval 同时运行 dense 和 BM25，并对各自 filtered candidate list 使用 Reciprocal Rank Fusion：

```text
fusion_score = dense_weight / (rrf_k + dense_rank)
             + sparse_weight / (rrf_k + sparse_rank)
```

默认参数：

```text
rrf_k = 60
dense_weight = 1.0
sparse_weight = 1.0
candidate_k = 80
top_k = 10
```

同一 `chunk_id` 在 dense 和 sparse 中出现时会合并为一个 hit，并保留 `dense_score`、`sparse_score`、`dense_rank`、`sparse_rank` 和 `retrieval_sources`。最终排序按 `(fusion_score desc, chunk_id asc)`，保证重复运行结果稳定。

### 5.6 Metadata filters

当前 filter 支持：

- `doc_ids`；
- `sections`；
- `chunk_types`；
- `page_start_min`；
- `page_end_max`；
- `has_equation`；
- `has_table`；
- `has_caption`；
- `exclude_references`。

这些 filter 在 dense、sparse、hybrid 三种模式中使用同一套 `metadata_matches()` 语义。Chroma backend 将它们转换为原生 `where`：列表条件使用 `$in`，References 排除使用 `$ne`，页范围使用 `page_end >= page_start_min` 和 `page_start <= page_end_max`，布尔字段使用 `$eq`；查询结果仍做受控 post-filter 作为语义防线，并在 trace 中记录 native filter、候选数和过滤后数量。当前 filter 是硬过滤，不做 query rewrite，也不自动扩展邻居 chunk。

### 5.7 Result schema

每个 hit 保留：

- rank 和 chunk identity：`rank`、`chunk_id`、`doc_id`、`title`；
- structure metadata：`section`、`section_heading`、`heading_path`、`chunk_type`、`page_start`、`page_end`；
- provenance：`source_block_ids`、`overlap_source_block_ids`；
- special block flags：`content_types`、`has_equation`、`has_table`、`has_caption`；
- retrieval score：`dense_score`、`sparse_score`、`fusion_score`、`dense_rank`、`sparse_rank`、`retrieval_sources`；
- `text` 或 CLI 中的 `text_preview`。

后续 audit 仍必须区分 `source_block_ids` 和 `overlap_source_block_ids`。Overlap 是 chunking 阶段复制的上下文，不应被当成独立原文 evidence。

### 5.8 Benchmark 和 validation

Retrieval v1 benchmark 位于：

```text
data/eval/retrieval_v1_queries.jsonl
```

当前包含 40 条人工可核对 query：

- 36 条 answerable；
- 4 条 no-answer；
- 覆盖 fact、method、experiment、result、related_work、limitation、reference、special_block、multi_evidence、no_answer；
- 所有 `relevant_chunk_ids` 均来自当前 `corpus_manifest.jsonl`，无虚构 ID。

validation 会运行 dense、sparse、hybrid 三种模式，并输出：

```text
outputs/retrieval_validation_v1/retrieval_validation_summary.json
outputs/retrieval_validation_v1/retrieval_validation_report.md
outputs/retrieval_validation_v1/retrieval_results.jsonl
outputs/retrieval_validation_v1/failure_cases.jsonl
outputs/retrieval_validation_v1/query_type_metrics.json
outputs/retrieval_validation_v1/latency_report.json
```

指标包括：

- Recall@1/3/5/10；
- MRR@10；
- nDCG@10；
- document_hit@k；
- section_hit@k；
- multi_evidence_coverage@10；
- no_answer_false_positive_rate；
- average_latency_ms；
- p95_latency_ms。

需要特别说明：`no_answer_false_positive_rate` 在 Retrieval v1 中按 retrieval-only 行为诚实记录。检索器组件只返回最近 chunks，不做 evidence sufficiency 或 answerability 判断，所以 no-answer 查询仍会得到候选 chunk。当前系统已在其后增加独立 Evidence Sufficiency v1；这里的 `1.0` 仍保留为未经过 Judge 的历史基线。

### 5.9 当前边界

Retrieval v1 明确不包含：

- query rewrite；
- 内置精排；Cross-Encoder 精排由后续独立 Reranker v1 阶段提供；
- evidence sufficiency；
- retry；
- neighbor expansion；
- answer generation；
- citation audit；
- Agentic RAG。

这些都不是 Retrieval v1 组件内部职责。Query Rewrite、Reranker、Evidence Sufficiency 和 Answer Generation 已作为独立下游阶段接入；retry、neighbor expansion、Citation Audit 和 Agentic RAG 仍未实现。不能把检索 benchmark 当作端到端问答质量评估。

## 6. Chroma Vector Database Backend v1

### 6.1 定位与 collection 设计

Chroma v1 不是第二套 RAG demo，而是 Indexing v1 与 Retrieval v1 之间的可选 dense backend：

```text
chunk_eval_v1
  -> corpus_manifest.jsonl
  -> existing OpenAI vectors.npy
  -> Chroma PersistentClient
  -> Retrieval v1 dense / hybrid
```

正式 collection 名称为 `researchguard_papers_v1`，持久化目录为 `data/indexes/chroma_v1`。五篇论文使用同一个 embedding model，需要跨论文搜索，因此使用单 collection，通过 `doc_id`、`section` 等 metadata 区分记录。record ID 直接使用稳定 `chunk_id`，不生成第二套 UUID。

本地 `PersistentClient` 允许 Python 进程结束后重新加载数据库，不需要 Chroma Cloud、Docker 或外部 server。它适合当前开发、单机回归和演示，不代表已经具备分布式、高并发或生产部署能力；未来如需服务化，可在保持 record schema 的前提下评估 Chroma server/client 模式。

### 6.2 Embedding、document 和 metadata

构建过程只读取现有 `data/indexes/index_v1/dense/vectors.npy`，按 `ids.json` 与 `corpus_manifest.jsonl` 的严格顺序导入 337 个现有向量，不重新调用 OpenAI 文档 Embedding API。collection 创建和加载均显式使用 `embedding_function=None`，查询也显式传入由项目统一 provider 生成的 `query_embeddings`，避免 Chroma 默认模型与 `text-embedding-3-small` 混用。

每条 record 保存：

- ID：`chunk_id`；
- embedding：对应的 1536 维向量；
- document：完整 chunk text；
- scalar metadata：`chunk_id`、`doc_id`、`title`、`section`、`section_heading`、`chunk_type`、page range、special flags、`short_chunk`、`content_hash`、`metadata_hash`、`corpus_fingerprint`、`schema_version` 等；
- list metadata：`heading_path`、`source_block_ids`、`heading_block_ids`、`overlap_source_block_ids`、`content_types` 使用稳定 JSON string 字段保存，读取时可逆恢复。

`source_block_ids` 与 `overlap_source_block_ids` 始终分开保存，避免把 overlap 上下文误当作新的原始 evidence。

### 6.3 Fingerprint 与 fail-fast

Collection-level metadata 保存 schema version、corpus fingerprint、embedding provider/model/dimensions、distance metric、source index backend 和 build timestamp。Retrieval 加载 Chroma 时会与当前 Indexing v1 manifest 比较；fingerprint、model、dimension、metric 或 schema 不一致时拒绝查询。

同步流程允许在兼容 schema/model/metric 下核对 record hashes 并将 collection 更新到新的 corpus fingerprint；查询流程则必须使用已经同步完成且 fingerprint 完全一致的 collection。

### 6.4 增量同步和删除保护

同步按 `chunk_id`、`content_hash`、`metadata_hash` 分类：

- 新 ID：`add`，显式传入已有 embedding；
- content hash 改变：`upsert` document、metadata 和当前 Indexing v1 vector；
- metadata hash 改变：只更新 metadata，不触发 embedding function；
- 未变化：跳过写入并计为 `reused`；
- stale ID：只删除当前 collection 中确认不再属于 corpus 的 record。

若 stale 数量超过现有 collection 的 10%，构建脚本默认 fail fast，必须通过明确确认参数才能继续；不会 reset 数据库、删除 collection 或删除 persist directory。首次同步结果为 `inserted=337`，第二次无变化同步为 `added=0`、`updated=0`、`metadata_updated=0`、`deleted=0`、`reused=337`。

### 6.5 验证结果

`scripts/validate_chroma_v1.py` 全量验证 337 个 IDs/documents/hashes，抽样比较 40 个 Chroma/NumPy embeddings，重新创建 PersistentClient 验证 reload/query，并使用 30 条真实 Retrieval benchmark query 比较 NumPy exact dense 与 Chroma dense。还会验证 10 组真实 metadata filters，并在独立临时 collection 中测试 add、metadata update、embedding update、stale delete、unchanged rerun 和 deterministic results。

当前结论：`PASS`。

| Metric | Result |
| --- | ---: |
| source / collection count | 337 / 337 |
| missing / stale / duplicate IDs | 0 / 0 / 0 |
| embedding sample | 40 |
| embedding max absolute error | 7.45e-09 |
| persistence reload / query | passed / passed |
| filter cases | 10/10 passed |
| incremental synthetic invariants | 8/8 passed |
| Top-1 agreement | 1.0000 |
| Top-3 overlap | 1.0000 |
| Top-5 overlap | 1.0000 |
| Top-10 overlap | 1.0000 |
| rank correlation | 1.0000 |
| NumPy backend average latency | 0.8650 ms |
| Chroma backend average latency | 10.7916 ms |

上述 latency 只比较已有 query vector 的本地检索，不包含约 3.52 秒的 30-query OpenAI embedding 批处理。当前 337 条小语料中 NumPy exact search 更快；引入 Chroma 的理由是持久化数据库接口、metadata filtering 和增量维护能力，不是宣称 ANN 在这一规模一定更快。

验证输出位于 `outputs/chroma_validation_v1`，正式 Chroma 数据库与验证输出都被 Git 忽略。

### 6.6 当前限制与安全边界

- 当前只使用本地进程内 PersistentClient，不启动 Chroma HTTP server，也不连接 Chroma Cloud。
- `chromadb==1.5.9` 当前有一项针对 HTTP server 端点的公开未修复安全公告；本项目不暴露该入口，但若未来启用 server 模式必须重新做版本与安全评估。
- 当前 HNSW 与 NumPy 在 30 条 benchmark 上完全一致，但 corpus 扩大后仍需持续测量 ANN recall，不能假设永远完全一致。
- Chroma 数据库不提交 Git；可通过 NumPy index 和构建 CLI 重建。
- 环境 `pip check` 仍报告 legacy `langgraph-sdk 0.4.2` 要求 `websockets<16`、当前为 16.0；该冲突不在 Chroma/Retrieval v1 执行路径，本阶段未擅自调整。

## 7. Reranker v1 详细说明

### 7.1 为什么高 Recall 之后仍需要 rerank

Hybrid RRF baseline 的 `Recall@10=0.9444`，说明 36 条 answerable queries 中，大多数正确 evidence 已进入 Top-10；但 `MRR@10=0.6847`，说明第一条正确 evidence 的位置仍有明显提升空间。RRF 只组合 Dense 和 BM25 的名次，不直接建模 query 与一个完整 chunk 之间的交互，因此适合 candidate recall，不等同于最终精排。

当前职责划分为：

```text
Query
  -> Chroma/NumPy Dense Retrieval
  -> BM25 Sparse Retrieval
  -> RRF Hybrid Candidate Recall (20 candidates)
  -> Cross-Encoder Reranker
  -> Final Top 10 evidence candidates
```

RRF 保留语义召回和词法召回的互补性；Reranker 只在有限候选池中判断哪个 chunk 更直接回答当前 query。关闭 reranker 时，系统原样回退到 Retrieval v1，不改变 Dense、BM25 或 RRF 的 score 与排序逻辑。

### 7.2 Backend、模型和运行环境

- 统一接口：`RerankerBackend`，具体实现为 `CrossEncoderReranker`；pipeline 对候选打分、缓存和稳定排序。
- 模型：`cross-encoder/ms-marco-MiniLM-L6-v2`，revision `c5ee24cb16019beea0893ab7796b1df96625c6b8`。
- 模型规模：约 22.7M parameters；本地 `safetensors` 权重约 90.9 MB。
- 推理参数：`device=cpu`、`batch_size=8`、`max_length=512`。
- 当前环境已存在 `sentence-transformers==5.5.1`、`torch==2.12.0`、`transformers==5.9.0`；本阶段未安装或升级依赖。机器虽有 NVIDIA GPU，但当前 PyTorch 是 CPU build，因此没有擅自修改 CUDA 或重建虚拟环境。
- 模型由用户确认后下载到 `data/cache/reranker_models/`，只允许离线本地加载，权重和 Hugging Face/PyTorch cache 均不提交 Git。

### 7.3 模型输入与结果 schema

模型输入模板固定为：

```text
Title: ...
Section: ...
Heading: ...
Content: ...
```

输入不包含 `chunk_id`、`content_hash`、`query_id`、`relevant_chunk_ids` 或 benchmark label。chunk 自身的 overlap text 保持不变，但 `overlap_source_block_ids` 只作为返回 metadata，不会被当作第二条独立 evidence。

每个候选保留 `dense_score`、`sparse_score`、`fusion_score`、`dense_rank`、`sparse_rank`、`fusion_rank` 和原有 provenance，并增加：

- `rerank_score`：Cross-Encoder 原始相关性分数；
- `rerank_rank`：精排后的名次；
- `pre_rerank_rank`：进入精排前的 RRF 名次；
- `reranker_backend`：当前为 `cross_encoder`；
- `reranker_model`：模型名和固定 revision。

排序按 `rerank_score desc`，再以 `pre_rerank_rank` 和 `chunk_id` 作为稳定 tie-break。`RetrievalResponse` 分别记录 retrieval、rerank 和 total latency。

### 7.4 候选数、开关和缓存

`configs/reranker_v1.yaml` 默认配置为 `candidate_k=20`、`final_top_k=10`。`scripts/retrieve_v1.py` 支持 `--rerank`、`--no-rerank` 和 `--rerank-candidate-k`；Reranker v1 只接受 Hybrid candidates，不能用于伪装 Dense 或 Sparse 的排序结果。

cache 位于 `data/cache/reranker_v1`。key 包含 normalized query、chunk `content_hash`、`metadata_hash`、model name/revision、reranker config version 和 input template version；这样 title/section/heading metadata 改变时不会误用旧分数，cache payload 也不包含 API key。validator 每次先强制忽略已有 cache 读取，得到真正 cold-cache 推理数据并写回，再执行全命中 warm-cache pass，比较 chunk 顺序和每个 rerank score 是否完全一致。

### 7.5 正式验证结果

`scripts/validate_reranker_v1.py` 在未修改的 40 条 Retrieval v1 benchmark 上比较 Hybrid RRF baseline 与 Hybrid RRF + Reranker。36 条为 answerable，4 条为 no-answer。

| Metric | Hybrid baseline | + Reranker | Delta |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.5556 | 0.6111 | +0.0556 |
| Recall@3 | 0.7778 | 0.8056 | +0.0278 |
| Recall@5 | 0.9167 | 0.9722 | +0.0556 |
| Recall@10 | 0.9444 | 0.9722 | +0.0278 |
| MRR@10 | 0.6847 | 0.7394 | +0.0546 |
| nDCG@10 | 0.6197 | 0.6584 | +0.0388 |
| document_hit@10 | 1.0000 | 1.0000 | 0.0000 |
| section_hit@10 | 1.0000 | 1.0000 | 0.0000 |
| multi_evidence_any_hit@10 | 0.9394 | 0.9697 | +0.0303 |
| multi_evidence_all_hit@10 | 0.4242 | 0.4545 | +0.0303 |

所有 8 项 hard checks 均为 0，validation 结论为 `PASS`。当前报告记录 10 个 query 改善、7 个 query 退化；没有删除困难 query，也没有修改 benchmark labels。

- 最大改善：`rv1_q035` 的 reference query，正确 evidence 从 baseline Top-10 外提升到 rank 4；`rv1_q001` 从 rank 6 提升到 rank 2；`rv1_q005` 从 rank 4 提升到 rank 1。
- 最大退化：`rv1_q013` 的 equation-like chunk query，第一条 relevant evidence 从 rank 1 降到 rank 3。Cross-Encoder 对公式、表格和 citation-style 特殊块的排序仍弱于自然语言正文。
- no-answer：false positive rate 仍为 `1.0`。四条 no-answer query 的最高 rerank score 虽均为负数，但本阶段没有校准 threshold，也不声明已经具备 answerability detection。

同一正式验证进程中的延迟如下：

| Mode | Average | P95 | Cache |
| --- | ---: | ---: | --- |
| Hybrid baseline total | 709.12 ms | 1291.93 ms | query embedding 按现有实现缓存 |
| Reranked cold total | 1732.90 ms | 1889.17 ms | 0 hits / 683 misses |
| Reranked cold inference | 1706.54 ms | - | CPU batch inference |
| Reranked warm total | 37.94 ms | 47.65 ms | 683 hits / 0 misses |

baseline 在每个 query 上先运行，因此后续 reranked retrieval 可复用进程内 query embedding；这些数字用于当前机器的相对回归，不代表独立服务的端到端 SLA。独立 CLI 新进程还需承担依赖导入、模型加载和首轮推理开销，实测一次全冷启动约 29.48 秒；生产化前需要常驻进程、启动预热和独立性能基准。

验证输出位于 `outputs/reranker_validation_v1`，包括 summary/report、baseline/reranked results、improved/regressed/failure cases、query type metrics 和 latency report，均不提交 Git。

## 8. Query Rewrite v1 详细说明

### 8.1 阶段边界与执行流

Query Rewrite v1 只处理检索前的 query 表达和多查询候选融合：

```text
Raw Query
  -> local query analysis
  -> normalized rewrite + max 2 expansions
  -> each variant runs Hybrid Retrieval independently
  -> cross-query RRF, deduplicate by chunk_id
  -> Cross-Encoder rerank with the original query
  -> Final Top K
```

原始 query 始终保留。单 rewrite 模式只检索 normalized query；multi-query 模式检索 original、normalized 和合法 expansions。配置默认 `enabled: false`，`--no-rewrite` 可显式回到冻结的 Retrieval/Reranker baseline。Rewrite 只支持 Hybrid Retrieval，不能隐式改变 Dense-only 或 BM25-only 语义。

### 8.2 Query analysis 与实体保持

`analyze_query()` 先做空白归一化，再本地提取：

- 模型、数据集和论文缩写，例如 `GPT-4o`、`CRAG`、`RAG-Sequence`、`HotpotQA`；
- 数字、百分比、年份；
- `Table/Figure/Equation` 编号；
- 否定词、时间条件和比较关系。

具体实体要求在 normalized query 和每条保留的 expansion 中逐字出现；normalized query 还必须保留否定、时间和比较约束。可选 expansion 如果丢实体、包含 chunk identifier 或与已有 variant 重复，会被丢弃并记录 `dropped_expansion_reasons`，不会进入检索。normalized query 为空、泄漏标识或违反约束时，整个 rewrite 自动回退到原始 query。

### 8.3 Structured LLM rewrite 与 fallback

当前使用已有 OpenAI SDK 和 `gpt-4.1-mini`：

- `temperature=0`；
- `max_rewrites=3`，即 1 条 normalized query 加最多 2 条 expansions；
- `timeout=30` 秒；
- `max_retries=2`；
- `prompt_version=query_rewrite_v1.2`；
- `entity_rules_version=entity_preservation_v1.3`；
- Responses API strict JSON schema，字段只允许 `normalized_query` 和 `expansion_queries`。

模型输入只有 original query、preserved entities 和 preserved constraints，不包含 `query_id`、relevant chunks、expected documents/sections、答案或人工 notes。API key 缺失、API/JSON 失败、空结果或 normalized entity validation 失败都会返回原始 query，并记录 `fallback_used`、`fallback_reason`、timestamp 和 token/API usage；rewrite 服务失败不会让 Retrieval 整体失败。

### 8.4 Rewrite cache

cache 位于 `data/cache/query_rewrite_v1`，key 包含 normalized original query 和完整 rewrite config identity：backend、model、temperature、max rewrites、timeout、max retries、prompt version 和 entity-rules version。payload 保存 original/normalized/expansions、preserved entities/constraints、被丢弃 expansion 的原因、fallback、model、prompt version 和 timestamp，不保存 API key。

正式 validator 每次先 `read_cache=false` 强制 cold rewrite 并写回，再立即执行 warm-cache roundtrip。最终结果为 cold `0/40` hits、warm `40/40` hits，cache 前后语义 payload 完全一致。

### 8.5 跨查询 RRF 与 provenance

每条 query variant 独立执行当前 Dense + BM25 + RRF Hybrid Retrieval。第二层 RRF 以 `chunk_id` 去重并融合名次，默认 `rrf_k=60`。每个最终 candidate 保存：

- `query_variant_hits`：variant ID/type、query text、variant 内 rank、dense/sparse/fusion rank 和 score；
- `original_query_recalled`、`rewrite_query_recalled`、`expansion_query_recalled`；
- `multi_query_fusion_score`、`multi_query_fusion_rank`；
- 原有 dense/BM25/RRF、document metadata 和 source provenance；
- rerank 后的 `pre_rerank_rank`、`rerank_score`、`rerank_rank`。

Cross-Encoder 最终仍使用原始 query 打分，避免 expansion 自己改变最终任务。`RetrievalResponse` 和 trace 分别记录 rewrite、retrieval、rerank、total latency，以及 rewrite/query-embedding API calls。

### 8.6 五路正式验证

未修改的 40 条 benchmark 上比较 5 条链路，36 条 answerable、4 条 no-answer：

| Metric | Raw Hybrid | Raw + Reranker | Rewrite Hybrid | Rewrite + Reranker | Multi-query + Reranker |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recall@1 | 0.5556 | 0.6111 | 0.5556 | 0.6111 | 0.6111 |
| Recall@3 | 0.7778 | 0.8056 | 0.7778 | 0.8056 | 0.8056 |
| Recall@5 | 0.9167 | 0.9722 | 0.9167 | 0.9444 | 0.9722 |
| Recall@10 | 0.9444 | 0.9722 | 0.9444 | 0.9722 | 0.9722 |
| MRR@10 | 0.6847 | 0.7394 | 0.6861 | 0.7370 | 0.7394 |
| nDCG@10 | 0.6197 | 0.6584 | 0.6166 | 0.6545 | 0.6468 |
| document_hit@10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| section_hit@10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9722 |
| multi_evidence_coverage@10 | 0.7172 | 0.7348 | 0.6995 | 0.7273 | 0.7096 |
| no_answer_false_positive_rate | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Multi-query + Reranker 相比 Raw Hybrid 的 Recall@10 增加 `0.0278`；相比 Raw + Reranker，MRR@10 在四位小数下持平，nDCG@10 下降 `0.0117`，属于已记录的小幅波动，不隐去。正式报告记录 10 个相对 Raw Hybrid 改善案例和 7 个退化案例。

可归因的原始 miss 改善是 `rv1_q035`：相关 reference chunk 在 Raw Hybrid 为 rank 11，经过 expansions 和跨查询 RRF 后进入 pre-rerank rank 10，再被既有 Reranker 提升到 rank 4。另一个 miss `rv1_q031` 仍未解决；其两个分别只保留 ReAct/RAG 的 expansions 因实体不完整被安全丢弃。最大排名退化仍为特殊公式查询 `rv1_q013`，从 Raw rank 1 到最终 rank 3，主要与通用 Cross-Encoder 对 equation-like chunk 的限制一致。

40 条真实 rewrite 全部成功，fallback 0；生成后有 14 条不合规 expansions 被丢弃，平均保留 1.65 条 expansion。8 项 hard checks 和 8 个 synthetic tests 全部通过，结论为 `PASS`。

| Mode | Average total | P95 total |
| --- | ---: | ---: |
| Raw Hybrid | 764.94 ms | 1404.68 ms |
| Raw Hybrid + Reranker | 27.20 ms | 36.13 ms |
| Rewrite Hybrid | 2609.24 ms | 7582.32 ms |
| Rewrite Hybrid + Reranker | 2366.16 ms | 7216.80 ms |
| Multi-query Hybrid + Reranker | 3760.33 ms | 8129.59 ms |

冷 rewrite 平均 `2330.99 ms`、P95 `7188.02 ms`；共 40 次 rewrite API calls、9348 input tokens、2132 output tokens，以及 131 次 query-embedding API calls。调用顺序会使后运行模式复用进程内 embeddings/rerank cache，因此各模式 total latency 只用于当前机器的回归比较，不应解释成独立服务 SLA；实际美元成本未硬编码，避免随 API pricing 变化失真。

验证输出位于 `outputs/query_rewrite_validation_v1`，包含五路 results、rewrite audit、original miss analysis、improved/regressed/fallback/failure cases、query type metrics、latency 和 API/cache reports，均不提交 Git。

## 9. Evidence Sufficiency v1 详细说明

### 9.1 为什么需要独立 Evidence Judge

当前三个组件的职责不同：

- Retriever：从索引中找到与 query 相关的候选；即使 corpus 没有答案，也会返回最近的 chunks。
- Reranker：重排已有候选，提高正确 evidence 靠前的概率；它不判断候选是否足以回答。
- Evidence Judge：只看原始 query 与最终 Top-k passages，判断证据是否直接覆盖实际问题，并决定未来是否允许 answer generation。

因此，retrieval score 或 rerank score 不能直接当作 answerability probability。Evidence Sufficiency v1 是检索后的独立只读层，不修改 dense/BM25/RRF/Cross-Encoder 排名。

### 9.2 Judge 输入与防泄漏边界

模型输入由 `build_evidence_model_input()` 白名单构造：

```json
{
  "question": "user query",
  "evidence_passages": [
    {
      "chunk_id": "paper_x_chunk_00001",
      "metadata": {
        "doc_id": "paper_x",
        "section": "method",
        "page": 3,
        "page_start": 3,
        "page_end": 3
      },
      "text": "retrieved chunk text"
    }
  ]
}
```

最多输入 reranked Top-10，每个 chunk 最多 1800 字符。禁止进入模型的字段包括 benchmark answer、`answerable` label、expected support level、`query_id` 和 `relevant_chunk_ids`。Passage 被明确视为不可信引用文本，模型必须忽略论文文本中可能出现的指令。

### 9.3 Structured LLM Judge

配置位于 `configs/evidence_sufficiency_v1.yaml`：

- model：`gpt-4.1-mini`；
- temperature：`0`；
- timeout：`30` 秒；
- max retries：`2`；
- prompt version：`evidence_sufficiency_v1.6`；
- config version：`evidence_sufficiency_v1.5`；
- cache：默认启用；
- evidence limit：Top-10、每 chunk 1800 字符。

OpenAI Responses API 使用 strict JSON schema。模型内部先输出 `supported_requirements` 和 `missing_requirements` 作为 coverage 自检，再由 pipeline 确定性映射：

- `strong`：至少一项 material requirement 有直接证据，且没有实质缺失；`answerable=true`，必须引用真实 supporting chunks。
- `partial`：至少一项有直接证据，同时存在另一项实质缺失；`answerable=false`，只引用已覆盖部分的真实 chunks。
- `unsupported`：没有 material requirement 获得直接支持；`answerable=false`，`supporting_chunk_ids=[]`。

内部 coverage 字段不进入公开 verdict。它用于防止 reason 已承认“部分支持”但标签仍返回 unsupported，或 reason 指出缺失却返回 strong。对模型名、缩写、数字等可抽取实体还会做 evidence coverage 后校验；单一关系中缺少关键实体时不能凭“未提及”推导否定答案。

公开核心 schema 为：

```json
{
  "answerable": false,
  "support_level": "partial",
  "confidence": 0.9,
  "reason": "The evaluator is supported, but GPT-5 is not mentioned.",
  "supporting_chunk_ids": ["paper_corrective_rag_chunk_00007"]
}
```

运行结果另带 model、prompt/config version、timestamp、cache、fallback、API token 和 latency metadata。

### 9.4 Cache、provenance 与 fallback

Judge cache 位于 `data/cache/evidence_sufficiency_v1`，不提交 Git。cache key 至少包含 normalized query、顺序稳定的 chunk IDs、model、prompt version、config version 和完整输入内容哈希；修改 passage 文本、顺序或 Judge 配置都会生成新 key。cache payload 保存 input hash、结构化输出和 timestamp。

Pipeline 对 supporting IDs 做集合和顺序校验：ID 必须来自当前传入的 Top-k，重复 ID 被稳定去重，模型不得引用不存在的 chunk。API、JSON、空 evidence、schema 或 provenance 校验失败时，系统返回 `answerable=false / unsupported / confidence=0`，并记录 fallback reason；它不会因 Judge 服务失败而错误放行 answer generation。

### 9.5 Benchmark 与 validation

独立 benchmark 位于 `data/eval/evidence_sufficiency_v1_queries.jsonl`，没有修改原 40 条 Retrieval benchmark。文件共 44 条，每行严格只有：

```json
{"query":"...","answerable":true,"support_level":"strong"}
```

分布为 35 strong、5 partial、4 unsupported。前 40 条复用原 query 文本，但不复制 ID/filter/relevant chunks；另外 4 条人工设计的复合问题用于检查“部分有证据、部分缺失”。正式 flow 为 `original query -> Query Rewrite -> optional expansions -> Hybrid Retrieval -> cross-query RRF -> Reranker -> Top-10 -> Evidence Judge`。

`scripts/validate_evidence_sufficiency_v1.py` 包含 11 个 synthetic tests，覆盖标签隔离、三种 schema、伪造 chunk ID、缺实体否定推断、coverage canonicalization、API/JSON fallback、cache key 和磁盘 cache 确定性。正式 hard checks 全部为 0：

- `json_parse_failure`；
- `empty_result`；
- `benchmark_leakage`；
- `cache_inconsistency`；
- `missing_chunk_reference`；
- `retrieval_regression`；
- `schema_failure`。

### 9.6 正式结果与真实含义

最终结论：`PASS`。

| Metric | Value |
| --- | ---: |
| Answerability accuracy | 0.8182 |
| Precision | 1.0000 |
| Recall | 0.7714 |
| F1 | 0.8710 |
| Support-level accuracy | 0.8182 |
| Strong recall | 0.7714 |
| Partial recall | 1.0000 |
| Unsupported recall | 1.0000 |
| No-answer false positive rate | 0.0000 |
| All-unanswerable false positive rate | 0.0000 |

与 retrieval-only 的 no-answer FPR `1.0` 相比，四条 unrelated no-answer query 在本层均被拒绝。44 条中没有 false positive，但保留 8 个 false negatives：部分来自 equation/reference/multi-document evidence 未进入当前无 filter Top-10，部分来自 Judge 对证据完整度的保守判断。这个结果表示 v1 已能作为 answer-generation gate 的开发基线，不表示已经解决 corpus-level answerability。

冷运行 Judge 平均 `4646.71 ms`、P95 `8502.83 ms`；完整 retrieval + Judge 平均 `7696.17 ms`、P95 `17172.80 ms`。共 44 次 Judge API calls、211326 input tokens、9921 output tokens；cold cache 0/44，紧接着的 warm consistency check 为 44/44 hits。所有 fallback 和误分类均保存在 validation outputs。

当前 benchmark 在开发过程中用于 prompt 校准，因此它是 regression/development benchmark，不是独立 blind hold-out。样本仅来自五篇论文，指标不能外推到开放域论文库、跨语言 query 或生产流量。

## 10. Answer Generation v1 详细说明

### 10.1 Evidence-only 输入与 gate

Answer pipeline 不直接把 Retrieval Top-10 全量交给生成模型。它先读取 Evidence Sufficiency verdict；只有 `strong + answerable=true` 才继续，并按 Judge 返回的 `supporting_chunk_ids` 精确选择 chunks。任一 supporting ID 在当前 hits 中缺失时都会拒答，不会悄悄换用相似 chunk。

`partial`、`unsupported` 和 Judge fallback 都在调用 backend 前短路，因此 generator API call count 为 0。固定拒答 schema 为：

```json
{
  "answer": "Insufficient evidence in the current corpus.",
  "citations": [],
  "confidence": 0
}
```

### 10.2 Prompt、结构化输出与 provenance

当前使用 OpenAI `gpt-4.1-mini`、temperature 0、strict JSON schema。模型输入只含 question 和允许的 evidence passages，每个 passage 只带 `chunk_id/doc_id/section/page/text`。Prompt 禁止外部知识、常识补全、benchmark 信息和 passages 内嵌指令，并明确要求无 evidence 支持的 claim 不得写入答案。

正常输出核心 schema 为：

```json
{
  "answer": "...",
  "citations": [
    {
      "chunk_id": "paper_rag_chunk_00002",
      "doc_id": "paper_rag",
      "section": "abstract",
      "page": 1
    }
  ],
  "confidence": 0.91
}
```

Pipeline 不信任模型返回的 citation metadata：`chunk_id` 必须属于 Judge 已批准 evidence，`doc_id/section/page` 必须和当前 retrieval hit 完全一致，之后才构建 canonical citations。缺 citation、重复 citation、越界 citation、metadata mismatch、空答案、非法 confidence 或 API/JSON 失败都会 fail closed。

### 10.3 Cache、CLI 与验证边界

Answer cache 保存 input hash、生成结果和 timestamp；key 同时覆盖 query、evidence IDs、model、prompt version、config version 和完整 generation config。相同输入的 warm result 必须与 cold result 的稳定字段一致。CLI 必须同时传 `--evidence-check --generate-answer`，后一个参数不能绕过 gate。

独立 benchmark 为 22 条：14 strong、4 partial、4 unsupported。12/12 synthetic tests 覆盖 gate 零调用、supporting-only 输入、benchmark leakage、citation 越界/metadata mismatch/缺失、API/JSON failure、缺 evidence、cache consistency 和 schema。真实验证的 Citation Coverage、citation provenance validity、Refusal Accuracy 和 Answerability consistency 均为 1.0；validation-only answer-level grounding judge 估计 Unsupported Claim Rate 为 0.0。该 grounding 指标只是回归期启发式检查，不是生产 Citation Audit，也不证明每条 citation 的 entailment 与 completeness。

## 11. Claim Verification / Citation Audit v1 详细说明

### 11.1 为什么 Citation Coverage 不够

Answer Generation v1 的 `Citation Coverage=1.0` 只说明每个已生成答案至少带有一个来自允许 evidence 的 citation，不能说明 citation 对答案中每个事实都构成支持。例如答案写出“CRAG improves accuracy by exactly 50%”并引用一段 CRAG 方法介绍，provenance 合法，但该段没有 50% 这个结果。Citation Audit 必须先把答案拆成事实单位，再逐 claim 判断 entailment。

### 11.2 Atomic claim extraction

Extractor 只接收原始 answer，不接收 query labels、标准 claim、support label 或 expected citation。`gpt-4.1-mini` 以 temperature 0 和 strict JSON 输出顺序稳定的 `c1/c2/...`。规则要求：

- 一个 claim 只表达一个可独立验证的事实；
- 并列的数字、数据集、动作和比较两侧在可分时拆开；
- 不修正、不增强、不弱化、不总结原答案；
- 保留数字、单位、模型/论文/数据集名称、否定和比较方向；
- 已经 atomic 的句子尽量保留原措辞。

Pipeline 校验 claim ID 连续、文本非空且不重复，并检查答案中的数字、缩写和模型名没有在全部 claims 中丢失。解析、实体保持或 schema 失败会终止整次 audit，不输出看似完整的局部报告。

### 11.3 Claim-evidence matching 与 verification

每个 claim 独立验证，不能利用其他 claim 的 verdict。候选 evidence 顺序为：

1. Answer Generation 返回的 citations；
2. 同次生成已经读取、但未被答案全局 citation 选中的 `evidence_chunk_ids`；
3. 第二组按 claim 与 chunk 的词项重合度稳定排序。

审计层绝不重新调用完整 Retriever，也不读取 corpus 中任意新 chunk。这样可以避免 Answer 用 evidence A、Audit 再搜索 evidence B 为答案补证。Verifier 只看当前 claim 和候选 passages，用 strict JSON 返回 `supported/partial/unsupported`：完整直接支持为 supported；只覆盖实质一部分为 partial；无直接支持或证据冲突为 unsupported。精确数字、单位、实体、否定和比较方向必须一致，topic overlap 不算支持。

### 11.4 输出 schema 与 fail-closed

核心输出示例：

```json
{
  "answer": "...",
  "claims": [
    {
      "id": "c1",
      "text": "CRAG uses a lightweight retrieval evaluator.",
      "support_level": "supported",
      "confidence": 0.95,
      "citations": [
        {
          "chunk_id": "paper_corrective_rag_chunk_00007",
          "doc_id": "paper_corrective_rag",
          "section": "introduction",
          "page": 2
        }
      ],
      "reason": "..."
    }
  ],
  "overall_grounded": true,
  "unsupported_claim_count": 0,
  "partial_claim_count": 0,
  "grounding_score": 1.0
}
```

`overall_grounded=true` 仅当全部 claims 都是 supported。`grounding_score` 是 supported claims / all claims；partial 不按完整支持计分。Supported/partial 必须有真实 citation，unsupported 必须无 citation。模型返回的 IDs 必须属于该 claim 的 candidates，最终 metadata 由 Pipeline 从原 evidence canonicalize。拒答答案不会调用 extractor/verifier；API、JSON、schema、provenance 或 cache validation 失败时 audit fail closed，但不会篡改原答案。

### 11.5 Benchmark 与真实限制

独立 benchmark 包含 12 个答案、26 个人工标注 atomic claims，覆盖 fully supported、partially supported answer、unsupported、numerical 和 comparison。12/12 synthetic tests 验证输入隔离、拒答零调用、claim 顺序、citation 优先、partial/unsupported、canonical provenance、invalid schema、越界 citation 和 cache consistency。正式结果 hard checks 全 0，Claim Support Accuracy、Supported Claim Precision 和 Unsupported Claim Detection Recall 均为 1.0；Citation Precision/Recall 为 0.96/1.0，额外 citation 被如实保留；Answer Grounded Rate 为 0.5833，因为 benchmark 有意包含不 grounded 的测试答案。

这些指标来自同一五篇开发论文和 LLM verifier，不是人工生产评测，也不证明原论文内容真实。当前 v1 没有跨来源冲突消解、表格单元格级 entailment、公式推导验证、claim dependency graph 或 Agent retry。

## Pipeline Integration v1 / System Architecture

### 编排边界

`researchguard/pipeline.py` 提供 `ResearchGuardPipeline.from_config(...)`、实例方法 `run(query)` 和便捷函数 `run_pipeline(query, config_path=...)`。它不新增 prompt、LLM、Retriever、Reranker 或 Judge，只按以下固定顺序调用已经验收的实现：

```text
Raw Query
  -> Query Rewrite + optional multi-query expansion
  -> Hybrid Retrieval + cross-query RRF
  -> Cross-Encoder Reranker
  -> Evidence Sufficiency gate
       -> strong: Answer Generation -> Citation Audit
       -> partial/unsupported: rejected, generation/audit skipped
```

模块异常返回结构化 `failed` stage 和 `final_status=failed`；Evidence、Answer 或 Audit 自身的 fail-closed result 继续保留其 `fallback` 状态。编排器不会用异常或 fallback 绕过 gate，也不会在审计阶段重新检索证据。

### 统一配置与模块开关

`configs/pipeline_v1.yaml` 固定 schema/config version、cache 读取策略、是否在输出保留 retrieval text，并集中声明六个模块。默认路径使用当前 Chroma + Hybrid 主流程：Rewrite 和 multi-query 开启、candidate-k 80、Reranker Top-20 到 Top-10、Evidence/Answer/Audit 开启。

模块可以关闭，例如同时设置 `answer_generation.enabled: false` 和 `citation_audit.enabled: false` 时，strong evidence 的最终状态为 `evidence_sufficient`。配置加载时执行依赖检查：Answer 不能绕过 Evidence，Citation Audit 不能绕过 Answer，multi-query 不能绕过 Rewrite，Rewrite/Reranker v1 只能搭配 Hybrid Retrieval。当前 v1 没有“关闭 Evidence 但继续生成”的配置。

### 统一输出 schema

顶层 schema：

```json
{
  "query": "...",
  "rewrite": {"status": "completed", "latency_ms": 1.0, "model": "gpt-4.1-mini", "config_version": "query_rewrite_v1.2", "output": {}},
  "retrieval": {"status": "completed", "latency_ms": 120.0, "model": "text-embedding-3-small", "config_version": "retrieval_v1", "output": {}},
  "reranking": {"status": "completed", "latency_ms": 14.0, "model": "cross-encoder/ms-marco-MiniLM-L6-v2", "config_version": "reranker_v1.0", "output": {}},
  "evidence_check": {"status": "completed", "latency_ms": 2.0, "model": "gpt-4.1-mini", "config_version": "evidence_sufficiency_v1.5", "output": {}},
  "answer_generation": {"status": "completed", "latency_ms": 1.0, "model": "gpt-4.1-mini", "config_version": "answer_generation_v1.0", "output": {}},
  "citation_audit": {"status": "completed", "latency_ms": 1.0, "model": "gpt-4.1-mini", "config_version": "citation_audit_v1.1", "output": {}},
  "final_status": "grounded",
  "pipeline": {"schema_version": "researchguard_pipeline_v1", "config_version": "pipeline_v1.0", "start_time": "...", "end_time": "...", "latency_ms": 153.0}
}
```

每个 stage 都有 `status/start_time/end_time/latency_ms/model/config_version/reason/output`。可见状态包括 `pending/completed/disabled/skipped/fallback/failed`；正常返回时不会遗留 `pending`。`final_status` 当前包括 `disabled/retrieved/evidence_sufficient/rejected/answered/grounded/needs_review/failed`。

### CLI、延迟与验证

统一入口是 `python -m researchguard.cli run`。`--query` 必填，`--config` 默认 `configs/pipeline_v1.yaml`，`--output` 可选；不提供输出路径时 JSON 写到 stdout。CLI 只负责参数和序列化，gate 逻辑全部位于可复用的 Pipeline 类。

Pipeline 保存总 start/end 和总 latency；RetrievalEngine 内部提供 rewrite、retrieval、rerank 分段 latency，Evidence、Answer、Audit 使用各自原生 latency。由于前三段在一次 RetrievalEngine 调用内完成，其 stage timestamp 是统一调用窗口的可观测表示，精确耗时以各自 `latency_ms` 为准。

`scripts/validate_pipeline_v1.py` 包含 6 个 synthetic tests 和 3 条真实 E2E：strong query 必须生成并完成 grounded audit；unsupported 与 partial 必须在生成前拒绝；另验证模块关闭、异常 schema 和重复输入稳定性。验证同时将 Pipeline Top-10 与直接 Retrieval Top-10 比较，并检查 warm cache。当前结论 `PASS`，八项 hard checks 全为 0。

## Streamlit Demo v1 / Demo Usage

### 定位与调用边界

`demo/app.py` 是当前 Unified Pipeline v1 的本地可视化展示层。页面通过 `ResearchGuardPipeline.from_config(...)` 创建可复用实例，并且每次查询只调用公开的 `pipeline.run(query)`；它不复制 Rewrite、Retrieval、Reranker、Evidence、Answer 或 Citation Audit 的内部逻辑，也不新增模型、数据库或 prompt。

页面包含：

- query 输入和固定 demo questions；
- 六阶段 status、latency 和失败 reason；
- reranked evidence 的 rank、document、section、page、score、chunk text 和 chunk ID；
- Evidence Sufficiency 的 support level、confidence 和 reason；
- grounded answer、answer citations 或统一拒答；
- claim-level Citation Audit、support status、reason 和 canonical citations；
- `Show Pipeline Details` 调试面板，展示脱敏后的 stage status、latency、model、config version 和 Pipeline JSON。

`demo/utils.py` 只负责 result schema 检查、展示视图转换、score/page/document 格式化和输出脱敏。本地绝对路径、cache/model/index path、API key 模式和环境变量不会显示在页面。Pipeline/API/JSON 异常会转成页面错误状态，不打印 stack trace 或环境内容。

### 安装与启动

完整环境按项目依赖安装；`requirements.txt` 已固定 `streamlit==1.58.0`。只补装 Streamlit 时可运行：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" -m pip install streamlit
```

从项目根目录启动：

```powershell
cd C:\Users\18449\Desktop\researchguard_workspace
& ".\.venv\Scripts\python.exe" -m streamlit run demo\app.py
```

虚拟环境已激活时也可使用：

```powershell
streamlit run demo/app.py
```

默认本地地址为 `http://localhost:8501`。Dense/Hybrid Retrieval 和未命中 cache 的 LLM 阶段仍需要项目既有 `OPENAI_API_KEY`；Demo 不读取、记录或显示 key。

### Demo validation

`scripts/validate_demo_v1.py` 使用 Streamlit `AppTest` 检查页面可启动，再用现有 Pipeline 跑一条 strong 和一条 unsupported 真实查询。当前验证结论 `PASS`：strong 得到 `grounded`、10 条 evidence、1 条 answer citation 和 4 条 audited claims；unsupported 得到 `rejected`，Answer/Audit 均 `skipped`，没有 citation 或 claim。四项 hard checks 全为 0，包括页面启动、strong render contract、unsupported render contract 和 display sanitization。

页面还在 1440x900 与 390x844 视口完成浏览器检查：桌面为三列阶段状态，移动端为单列状态并自动收起 sidebar；未发现文字覆盖、横向溢出或状态卡布局漂移。本阶段只提供本地单用户 demo，不是生产 UI 或 API。

## 12. 目录结构

```text
researchguard/ingestion/      PDF layout extraction, block detection, heading classification, section recovery, chunking
researchguard/indexing/       Indexing/Chroma：corpus manifest、embedding cache、NumPy dense、BM25 sparse、Chroma persistent sync
researchguard/retrieval/      Retrieval/Rewrite/Reranker/Evidence/Answer/Audit：检索、gate、生成、claim verification 与 cache
researchguard/pipeline.py     Unified Pipeline v1：配置依赖、证据 gate、状态、schema 和阶段计时
researchguard/agent/          legacy Agentic RAG 模块存在，尚未重构接入新 chunks
researchguard/audit/          Evidence audit 相关能力，部分实现
researchguard/memory/         memory / trace 存储能力，部分实现
researchguard/evaluation/     legacy answer / agentic evaluation 模块；当前各 v1 validation 位于 scripts
researchguard/reporting/      audit report 渲染能力，部分实现
demo/                         Streamlit Demo v1 页面和纯展示/脱敏工具
scripts/                      验证、构建和本地功能测试脚本，包括 validate_pipeline_v1.py
configs/                      parser/index/retrieval/Chroma/reranker/rewrite/evidence/answer/audit/pipeline 等配置
data/eval/                    Retrieval、Evidence Sufficiency、Answer Generation 与 Citation Audit 独立 benchmark
data/raw_docs/                原始 PDF，GitHub 提交中排除
data/parsed/                  parser 和 chunking 输出
outputs/                      parser/chunk/index/retrieval validation、local rag、audit 等输出，GitHub 提交中排除
frontend/                     旧 Streamlit 文件存在，尚未产品化
```

当前重点数据目录：

```text
data/raw_docs/parser_eval/         五篇 benchmark PDF
data/parsed/parser_eval_v5/        parser v5 输出
data/parsed/chunk_eval_v1/         section-aware chunk v1 输出
data/indexes/index_v1/             Indexing v1 输出：corpus manifest、embedding cache、dense/sparse index、index manifest
data/indexes/chroma_v1/            Chroma PersistentClient 数据库，不提交 Git
data/cache/reranker_models/        本地 Cross-Encoder 模型，不提交 Git
data/cache/reranker_v1/            query-chunk rerank score cache，不提交 Git
data/cache/query_rewrite_v1/       query rewrite/fallback cache，不提交 Git
data/cache/evidence_sufficiency_v1/  Judge verdict/input-hash cache，不提交 Git
data/cache/answer_generation_v1/  answer/input-hash/citation cache，不提交 Git
data/cache/citation_audit_v1/  claim/audit/input-hash cache，不提交 Git
data/eval/retrieval_v1_queries.jsonl  Retrieval v1 benchmark queries
data/eval/evidence_sufficiency_v1_queries.jsonl  44 条 answerability/support-level benchmark
data/eval/answer_generation_v1_queries.jsonl  22 条 answer/refusal/grounding benchmark
data/eval/citation_audit_v1_queries.jsonl  12 个答案、26 个 atomic claims 的 audit benchmark
outputs/parser_validation_v5/      parser v5 验收报告
outputs/chunk_validation_v1/       chunking v1 验收报告
outputs/index_validation_v1/       indexing v1 验收报告
outputs/retrieval_validation_v1/   retrieval v1 验收报告
outputs/chroma_validation_v1/      Chroma 构建记录、一致性验证和独立 synthetic collection
outputs/reranker_validation_v1/    RRF baseline 与 Cross-Encoder rerank 正式对比报告
outputs/query_rewrite_validation_v1/  五路 query rewrite/multi-query 正式对比报告
outputs/evidence_sufficiency_validation_v1/  answerability/support-level/hard-check 正式报告
outputs/answer_generation_validation_v1/  answer/citation/refusal/grounding/hard-check 正式报告
outputs/citation_audit_validation_v1/  claim/citation/grounding/hard-check 正式报告
outputs/pipeline_validation_v1/  unified pipeline synthetic、E2E、schema、cache、regression 和 latency 报告
outputs/demo_validation_v1/   Streamlit startup、strong/unsupported render contract 和脱敏验证报告
```

## 13. 当前阶段状态

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| Parser | 已完成 | 已完成 v5，并通过严格验收；结论 `PASS_WITH_CHUNK_FIXES`，说明 parser 可作为 chunking 输入，但旧 chunks 需修。 |
| Chunking | 已完成并通过 indexing 前置条件 | v1 最终边界修复已完成，synthetic tests 全过，五篇硬性检查全 0；验证脚本仍标记 `PASS_WITH_MINOR_ISSUES`，因为保留短 chunk、少数 heading_path 缺失和复杂视觉语义绑定限制。当前结论为 `APPROVED_FOR_INDEXING`。 |
| Indexing | 已完成 v1 并通过验证 | 旧 LlamaIndex page-level builder 仍保留为迁移代码；新的 Indexing v1 已接入 `data/parsed/chunk_eval_v1`，使用 OpenAI embedding、持久化 NumPy dense index 和本地 BM25 sparse index，验证结论 `PASS`。 |
| Chroma backend | 已完成 v1 并通过验证 | 已复用 337 个现有 embeddings 建立本地 persistent collection；完整性、persistence、filter、增量和 NumPy parity 均通过，结论 `PASS`。数据库文件不提交 Git。 |
| Retrieval | 已完成 v1 并通过验证 | dense backend 支持 `numpy/chroma` 配置切换，当前默认 Chroma；BM25、RRF hybrid、统一 metadata filters、CLI 和 40 条 query benchmark 回归结论 `PASS`。 |
| Reranker | 已完成 v1 并通过验证 | Cross-Encoder 对 Hybrid Top-20 精排到 Top-10；hard checks 全 0，Recall@10、MRR@10、nDCG@10 均提升，结论 `PASS`；no-answer 尚未解决。 |
| Query Rewrite | 已完成 v1 并通过验证 | strict JSON normalized rewrite、最多 2 expansions、实体/约束保护、cache/fallback、跨查询 RRF、五路 40-query 对比完成；hard checks 全 0，结论 `PASS`。 |
| Evidence Sufficiency | 已完成 v1 并通过验证 | strict JSON LLM Judge、requirement coverage、自洽三分类、provenance validation、cache/fallback、CLI 和 44-query validation 已完成；hard checks 全 0，no-answer FPR 为 0，结论 `PASS`。 |
| Answer Generation | 已完成 v1 并通过验证 | strong gate、supporting-only evidence、strict JSON、citation provenance、cache/fallback、CLI 和 22-query validation 已完成；hard checks 全 0，结论 `PASS`。 |
| Claim Verification / Citation Audit | 已完成 v1 并通过验证 | atomic extraction、answer-time evidence matching、逐 claim 三分类、canonical citations、cache、CLI 和 12-answer/26-claim validation 已完成；hard checks 全 0，结论 `PASS`。 |
| Pipeline Integration | 已完成 v1 并通过验证 | 已提供统一 Python API、YAML、CLI、evidence gate、stage schema/latency 和三类真实 E2E；6/6 synthetic tests 与八项 hard checks 全过，结论 `PASS`。 |
| Streamlit Demo | 已完成 v1 并通过验证 | 已直接接入 Unified Pipeline，展示 stage/evidence/sufficiency/answer/audit/debug；startup、strong、unsupported、脱敏和桌面/移动浏览器检查通过，结论 `PASS`。 |
| Agentic RAG | 已迁移但未重构 | legacy 模块存在，尚未接入当前 Retrieval/Reranker/Evidence/Answer/Audit 主流程；当前没有 tool retry、LangGraph 或多 Agent workflow。 |
| Evidence Audit | v1 主链路已实现，legacy 能力未整合 | 新 Citation Audit v1 已接入；`researchguard/audit` 中旧模块仍未与新主流程产品化整合。 |
| Evaluation | 部分实现 | Retrieval、Reranker、Rewrite、Evidence、Answer 和 Citation Audit 均有独立 validation；agentic 与生产级 blind hold-out 尚未完成。 |
| Frontend/API | 本地 Demo 已完成，尚未产品化 | `demo/app.py` 已提供 Streamlit Demo v1；旧 `frontend` 和 `researchguard/api` 仍未重构为带鉴权、并发、部署和稳定 API contract 的产品。 |

## 14. 数据和 benchmark

当前 benchmark 使用五篇科研论文：

- `paper_rag`
- `paper_agent`
- `paper_hallucination`
- `paper_corrective_rag`
- `paper_citation`

用途：

- 覆盖单栏/双栏、References、多 section、图表/caption/table/equation、appendix 等论文结构；
- 用于 parser v5 和 chunking v1 的固定验收；
- Indexing、Retrieval、Reranker、Query Rewrite、Evidence Sufficiency、Answer Generation 和 Citation Audit v1 均使用这五篇语料做开发回归；它们不是独立 blind hold-out。

关键目录：

```text
data\raw_docs\parser_eval
```

存放五篇原始 PDF。

```text
data\parsed\parser_eval_v5
```

每篇包含：

- `layout.json`
- `blocks.jsonl`
- `parsed_pages.jsonl`
- `chunks.jsonl`（parser 阶段旧 chunks，仅用于历史对照，不作为后续推荐输入）
- `parse_quality_report.json`
- `parsed.md`

```text
data\parsed\chunk_eval_v1
```

存放基于 parser v5 blocks 重建的 section-aware chunks。

```text
outputs\parser_validation_v5
```

存放 parser 严格验收结果。

```text
outputs\chunk_validation_v1
```

存放 chunking v1 验收结果和人工抽样。

```text
data\eval\retrieval_v1_queries.jsonl
```

存放 Retrieval v1、Reranker v1 与 Query Rewrite v1 共用的 40 条人工可核对 benchmark queries。`relevant_chunk_ids` 必须来自当前 `data/indexes/index_v1/corpus_manifest.jsonl`；任何 rewrite prompt 或模型输入都禁止读取这些 labels。

```text
outputs\retrieval_validation_v1
```

存放 Retrieval v1 的三种模式指标、failure cases、latency report 和 validation summary。

```text
data\indexes\chroma_v1
```

存放正式 Chroma PersistentClient 数据库。它由现有 Indexing v1 vectors 重建，不提交 Git。

```text
outputs\chroma_validation_v1
```

存放 Chroma build history、完整性/embedding/search parity audits、filter validation、incremental synthetic tests 和 failure cases。

```text
data\cache\reranker_models
data\cache\reranker_v1
```

分别存放用户确认下载的 Cross-Encoder 模型和 rerank score cache，均被 Git 忽略。

```text
outputs\reranker_validation_v1
```

存放 RRF baseline / reranked results、改进/退化/失败案例、query type metrics、latency 和正式结论。

```text
data\cache\query_rewrite_v1
```

存放 original query、normalized query、expansions、实体/约束、fallback、模型/prompt version 和 timestamp。cache 与 API key 分离，不提交 Git。

```text
outputs\query_rewrite_validation_v1
```

存放五路正式对比、rewrite audit、original miss attribution、改进/退化/fallback/failure cases、latency 和 API/cache usage。

```text
data\eval\evidence_sufficiency_v1_queries.jsonl
```

存放独立 44-query benchmark：35 strong、5 partial、4 unsupported。每行只有 query、answerable 和 support level，不包含 relevant IDs、expected document、filter 或答案。

```text
data\cache\evidence_sufficiency_v1
```

存放 input hash、Judge verdict、prompt/config identity 和 timestamp；不保存 API key，不提交 Git。

```text
outputs\evidence_sufficiency_validation_v1
```

存放 summary/report、逐 query verdict、误分类、fallback、synthetic tests、latency 和 API/cache usage，均不提交 Git。

```text
data\eval\answer_generation_v1_queries.jsonl
```

存放独立 22-query benchmark：14 strong、4 partial、4 unsupported。标签只用于 validator 判分，不进入 Answer Generator 模型输入。

```text
data\cache\answer_generation_v1
```

存放 input hash、answer、citations、model/prompt/config identity 和 timestamp；不保存 API key，不提交 Git。

```text
outputs\answer_generation_validation_v1
```

存放 summary/report、逐 query answer/citations、fallback、synthetic tests、latency 和 API/cache usage，均不提交 Git。

```text
data\eval\citation_audit_v1_queries.jsonl
```

存放独立 12-answer benchmark：26 个 atomic claims，覆盖 fully supported、partially supported、unsupported、numerical 和 comparison。Expected claim/support/citation labels 只供 validator 使用，不进入 audit 模型输入。

```text
data\cache\citation_audit_v1
```

存放 answer/input hash、prompt/config identity、claim verdicts、canonical citations 和 timestamp；不保存 API key，不提交 Git。

```text
outputs\citation_audit_validation_v1
```

存放 summary/report、逐 answer claim audit、failure cases、synthetic tests、integration regression、latency 和 API/cache usage，均不提交 Git。

```text
outputs\pipeline_validation_v1
```

存放 Pipeline v1 的 synthetic tests、三类真实 E2E、统一 schema、cache、retrieval regression、hard checks 和分阶段 latency；不提交 Git。

```text
outputs\demo_validation_v1
```

存放 Streamlit startup、strong/unsupported 展示 contract、display sanitization 和本地 server log；不提交 Git。

## 15. 运行命令

所有命令固定使用项目虚拟环境，不使用 Codex bundled Python。

### 15.1 进入项目并激活虚拟环境

```powershell
cd C:\Users\18449\Desktop\researchguard_workspace
.\.venv\Scripts\activate
```

也可以不激活，直接使用绝对路径 Python：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" --version
```

当前验证过的 Python 版本：`Python 3.12.10`。

### 15.2 Parser 运行

单篇运行示例：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" -m researchguard.ingestion.parse_pdf `
  --input "C:\Users\18449\Desktop\researchguard_workspace\data\raw_docs\parser_eval\paper_rag.pdf" `
  --out_dir "C:\Users\18449\Desktop\researchguard_workspace\data\parsed\parser_eval_v5\paper_rag" `
  --max_chunk_chars 1600 `
  --min_chunk_chars 250
```

五篇批量运行示例：

```powershell
$papers = @("paper_rag", "paper_agent", "paper_hallucination", "paper_corrective_rag", "paper_citation")
foreach ($paper in $papers) {
  & "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" -m researchguard.ingestion.parse_pdf `
    --input "C:\Users\18449\Desktop\researchguard_workspace\data\raw_docs\parser_eval\$paper.pdf" `
    --out_dir "C:\Users\18449\Desktop\researchguard_workspace\data\parsed\parser_eval_v5\$paper" `
    --max_chunk_chars 1600 `
    --min_chunk_chars 250
}
```

注意：当前 README 记录的是已完成结果。本轮未重新运行 parser。

### 15.3 Parser validation

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_parser_v5.py"
```

输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\outputs\parser_validation_v5
```

### 15.4 Chunk build

单篇运行示例：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" "C:\Users\18449\Desktop\researchguard_workspace\scripts\build_chunks_v1.py" `
  --blocks "C:\Users\18449\Desktop\researchguard_workspace\data\parsed\parser_eval_v5\paper_rag\blocks.jsonl" `
  --output "C:\Users\18449\Desktop\researchguard_workspace\data\parsed\chunk_eval_v1\paper_rag\chunks.jsonl" `
  --doc_id paper_rag `
  --max_chars 1600 `
  --target_chars 1200 `
  --min_chars 250 `
  --overlap_sentences 1
```

五篇批量运行示例：

```powershell
$papers = Get-ChildItem -Path "C:\Users\18449\Desktop\researchguard_workspace\data\parsed\parser_eval_v5" -Directory | Select-Object -ExpandProperty Name
foreach ($paper in $papers) {
  & "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" "C:\Users\18449\Desktop\researchguard_workspace\scripts\build_chunks_v1.py" `
    --blocks "C:\Users\18449\Desktop\researchguard_workspace\data\parsed\parser_eval_v5\$paper\blocks.jsonl" `
    --output "C:\Users\18449\Desktop\researchguard_workspace\data\parsed\chunk_eval_v1\$paper\chunks.jsonl" `
    --doc_id $paper `
    --max_chars 1600 `
    --target_chars 1200 `
    --min_chars 250 `
    --overlap_sentences 1
}
```

### 15.5 Chunk validation

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_chunking_v1.py"
```

输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\outputs\chunk_validation_v1
```

### 15.6 Index dry-run

dry-run 只扫描 corpus、验证 manifest schema、计算增量计划，不调用 embedding API，不写正式索引。

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\build_index_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\indexing_v1.yaml" `
  --dry-run
```

### 15.7 Index build

正式构建会读取 `OPENAI_API_KEY` 环境变量。脚本不会打印或写出 API key；如果 key 缺失，会失败并拒绝生成假向量。

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\build_index_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\indexing_v1.yaml"
```

增量构建：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\build_index_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\indexing_v1.yaml" `
  --incremental
```

输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\data\indexes\index_v1
```

### 15.8 Index validation

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_index_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\indexing_v1.yaml"
```

输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\outputs\index_validation_v1
```

### 15.9 Chroma build 和 incremental sync

首次构建会复用 `index_v1` 的 vectors，不重新调用文档 Embedding API：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\build_chroma_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\chroma_v1.yaml"
```

无变化增量同步：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\build_chroma_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\chroma_v1.yaml" `
  --incremental
```

若确认删除的 stale IDs 超过 collection 的 10%，脚本会拒绝继续；只有完成独立审阅和明确授权后，才能使用 `--confirm-large-delete`。

### 15.10 Chroma validation

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_chroma_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\chroma_v1.yaml"
```

输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\outputs\chroma_validation_v1
```

### 15.11 Retrieval / Rewrite / Reranker / Evidence / Answer / Audit CLI

Retrieval v1 会读取 `data/indexes/index_v1`。dense 和 hybrid 模式会调用 OpenAI query embedding，因此需要 `OPENAI_API_KEY` 环境变量；如果 key 缺失，脚本会失败，不会生成假向量。

Hybrid 示例：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\retrieve_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\retrieval_v1.yaml" `
  --query "How does corrective retrieval reduce hallucinations?" `
  --mode hybrid `
  --dense-backend chroma `
  --top-k 5 `
  --candidate-k 40
```

带 metadata filter 示例：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\retrieve_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\retrieval_v1.yaml" `
  --query "citation precision and citation recall in ALCE" `
  --mode hybrid `
  --doc-id paper_citation `
  --section results `
  --top-k 8
```

BM25-only 示例不需要 query embedding：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\retrieve_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\retrieval_v1.yaml" `
  --query "ALFWorld WebShop HotpotQA FEVER ReAct" `
  --mode sparse `
  --top-k 10
```

NumPy 精确基线回退只需在同一 CLI 增加：

```text
--dense-backend numpy
```

启用 Reranker v1：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\retrieve_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\retrieval_v1.yaml" `
  --query "How does corrective retrieval reduce hallucinations?" `
  --mode hybrid `
  --dense-backend chroma `
  --candidate-k 20 `
  --rerank-candidate-k 20 `
  --top-k 10 `
  --rerank
```

CLI JSON 会同时输出 pre-rerank/rerank rank、fusion/rerank score 和 retrieval/rerank/total latency。增加 `--no-rerank` 可显式回退到原 Hybrid RRF；配置中的 `reranker.enabled` 当前保持 `false`，避免旧调用在没有本地模型时意外启动精排。

启用 normalized rewrite：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\retrieve_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\retrieval_v1.yaml" `
  --query "Return only references related to ReAct or RAG citations from the indexed papers." `
  --mode hybrid `
  --candidate-k 80 `
  --top-k 10 `
  --rewrite
```

启用 original + normalized + expansions、多查询融合和最终 rerank：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\retrieve_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\retrieval_v1.yaml" `
  --query "Return only references related to ReAct or RAG citations from the indexed papers." `
  --mode hybrid `
  --candidate-k 80 `
  --rerank-candidate-k 20 `
  --top-k 10 `
  --rewrite `
  --multi-query `
  --rerank
```

`--multi-query` 会隐式启用 rewrite；`--no-rewrite` 可显式冻结在原 query。Rewrite 和 Dense/Hybrid query embedding 均需要 `OPENAI_API_KEY`；任一 rewrite 服务失败会回退原 query，但 embedding 服务失败仍按 Retrieval v1 的既有规则报错，不生成假向量。

启用完整检索链路并执行 Evidence Sufficiency gate：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\retrieve_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\retrieval_v1.yaml" `
  --evidence-config "C:\Users\18449\Desktop\researchguard_workspace\configs\evidence_sufficiency_v1.yaml" `
  --query "Does CRAG use GPT-5?" `
  --mode hybrid `
  --candidate-k 80 `
  --rerank-candidate-k 20 `
  --top-k 10 `
  --rewrite `
  --multi-query `
  --rerank `
  --evidence-check
```

JSON 顶层增加 `evidence_sufficiency` 和 `evidence_check_latency_ms`。Judge API/JSON/schema 失败时输出保守的 `unsupported` fallback；它不会清空或重排原 `hits`。配置中的 Evidence Judge 默认关闭，只有显式传入 `--evidence-check` 才运行。

在 strong gate 后生成 evidence-grounded answer：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\retrieve_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\retrieval_v1.yaml" `
  --evidence-config "C:\Users\18449\Desktop\researchguard_workspace\configs\evidence_sufficiency_v1.yaml" `
  --answer-config "C:\Users\18449\Desktop\researchguard_workspace\configs\answer_generation_v1.yaml" `
  --query "What is the difference between RAG-Sequence and RAG-Token?" `
  --mode hybrid `
  --candidate-k 80 `
  --rerank-candidate-k 20 `
  --top-k 10 `
  --rewrite `
  --multi-query `
  --rerank `
  --evidence-check `
  --generate-answer
```

JSON 顶层增加 `answer_generation`、`answer_generation_latency_ms` 和 `answer_pipeline_total_latency_ms`。`--generate-answer` 强制要求 `--evidence-check`；partial/unsupported 不调用 generator，并返回固定拒答。配置默认关闭，必须显式启用 CLI flag。

在生成后执行 Claim Verification / Citation Audit：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\retrieve_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\retrieval_v1.yaml" `
  --evidence-config "C:\Users\18449\Desktop\researchguard_workspace\configs\evidence_sufficiency_v1.yaml" `
  --answer-config "C:\Users\18449\Desktop\researchguard_workspace\configs\answer_generation_v1.yaml" `
  --citation-audit-config "C:\Users\18449\Desktop\researchguard_workspace\configs\citation_audit_v1.yaml" `
  --query "What is the difference between RAG-Sequence and RAG-Token?" `
  --mode hybrid `
  --candidate-k 80 `
  --rerank-candidate-k 20 `
  --top-k 10 `
  --rewrite `
  --multi-query `
  --rerank `
  --evidence-check `
  --generate-answer `
  --citation-audit
```

JSON 顶层增加 `citation_audit`、`citation_audit_latency_ms` 和 `grounded_answer_pipeline_total_latency_ms`。`--citation-audit` 强制要求 `--generate-answer`；若前序 gate 拒答，审计层不调用 claim models 并返回 `audit_reason=answer_refused`。

### 15.12 Retrieval validation

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_retrieval_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\retrieval_v1.yaml"
```

输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\outputs\retrieval_validation_v1
```

### 15.13 Reranker validation

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_reranker_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\reranker_v1.yaml"
```

输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\outputs\reranker_validation_v1
```

### 15.14 Query Rewrite validation

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_query_rewrite_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\query_rewrite_v1.yaml"
```

输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\outputs\query_rewrite_validation_v1
```

### 15.15 Evidence Sufficiency validation

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_evidence_sufficiency_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\evidence_sufficiency_v1.yaml"
```

输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\outputs\evidence_sufficiency_validation_v1
```

### 15.16 Answer Generation validation

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_answer_generation_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\answer_generation_v1.yaml"
```

输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\outputs\answer_generation_validation_v1
```

### 15.17 Citation Audit validation

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_citation_audit_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\citation_audit_v1.yaml"
```

输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\outputs\citation_audit_validation_v1
```

### 15.18 Unified Pipeline CLI 和 validation

运行完整主链路并把统一 JSON 写入文件：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  -m researchguard.cli run `
  --query "What is the difference between RAG-Sequence and RAG-Token?" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\pipeline_v1.yaml" `
  --output "C:\Users\18449\Desktop\researchguard_workspace\outputs\run_result.json"
```

运行 Pipeline v1 严格验证：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_pipeline_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\pipeline_v1.yaml"
```

输出目录：`outputs/pipeline_validation_v1`。

### 15.19 Streamlit Demo 和 validation

启动本地 Demo：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  -m streamlit run `
  "C:\Users\18449\Desktop\researchguard_workspace\demo\app.py"
```

运行 Demo v1 验证：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_demo_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\pipeline_v1.yaml"
```

输出目录：`outputs/demo_validation_v1`。

### 15.20 Compile check

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  -m compileall `
  "C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts"
```

## 16. 当前指标

### 16.1 Parser v5

Parser v5 结论：`PASS_WITH_CHUNK_FIXES`。

不是因为 `status=ok` 才通过，而是经过以下验收：

- reading order：五篇均抽样检查第一页、中间页、References 首页和表格/图片较多页；抽样中 column backtrack 和同栏 y backtrack 均为 0；
- heading：五篇 heading suspicious = 0，unmapped = 0；
- section：输出完整 block-level section transition trace；无 References 后错误回到 main_text/method 的硬失败；
- References：均找到 References heading，并检查后续 reference-like ratio、首页/末页 sample；
- chunk audit：旧 parser chunks 暴露问题，包括短 chunk、heading-only、>1600、multi-section、重复 block refs、equation 未进 chunk。

Parser v5 的真正结论是：parser 层可以进入下一阶段；chunk 层必须修。

### 16.2 Parser 阶段旧 chunks 暴露的问题

| Paper | Old chunks | <150 | >1600 | Multi-section | Heading-only | Duplicate block refs | Equation in old chunks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| paper_rag | 69 | 7 | 1 | 0 | 4 | 25 | 0/0 |
| paper_agent | 106 | 17 | 1 | 0 | 4 | 36 | 0/4 |
| paper_hallucination | 53 | 9 | 0 | 0 | 5 | 17 | 0/1 |
| paper_corrective_rag | 65 | 9 | 0 | 0 | 5 | 22 | 0/1 |
| paper_citation | 90 | 11 | 0 | 1 | 3 | 38 | 0/2 |

这些问题来自旧 chunk_builder，不是 parser v5 的 reading order / heading / section 主体失败。

### 16.3 Chunking v1

Chunking v1 验证脚本结论：`PASS_WITH_MINOR_ISSUES`。

Indexing 前置结论：`APPROVED_FOR_INDEXING`。

Synthetic tests：7/7 passed，failure_count = 0。

硬性检查：

| Check | Result |
| --- | ---: |
| `chunks_over_1600` | 0 |
| `multi_section_chunks` | 0 |
| `heading_only_chunks` | 0 |
| `section_mismatch_chunks` | 0 |
| `unexplained_repeated_source_block_ids` | 0 |
| `split_metadata_invalid_count` | 0 |
| `lost_core_blocks` | 0 |
| `lost_equation_blocks` | 0 |
| `lost_caption_blocks` | 0 |
| `lost_table_blocks` | 0 |
| `isolated_bindable_special_blocks` | 0 |
| `caption_nearest_table_misses` | 0 |
| `duplicate_chunk_ids` | 0 |
| `references_overlap_chunks` | 0 |
| `overlap_cross_section_chunks` | 0 |
| `overlap_not_direct_previous_chunks` | 0 |
| `overlap_over_max_chunks` | 0 |
| `empty_text_chunks` | 0 |
| `determinism_mismatch` | 0 |

剩余 minor issues：

- `<150` chunk 总数为 8；
- paper_agent 有 4 个非 abstract / references chunk 缺 heading_path；
- overlap direct-previous 规则和 `overlap_sentences=0/1/2` 参数语义已通过验证；
- heading 不进入 overlap、跨两个 source block 的 overlap provenance、special block 加 heading 后不超限、caption 最近 table 绑定、special 不跨 section 绑定、deterministic synthetic run 均已通过验证；
- equation/table/caption 邻近绑定已通过验证，但仍不是完整视觉语义绑定。

### 16.4 Indexing v1

Indexing v1 构建结论：`complete`。

Index validation 结论：`PASS`。

正式输入：

```text
C:\Users\18449\Desktop\researchguard_workspace\data\parsed\chunk_eval_v1\<paper>\chunks.jsonl
```

正式输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\data\indexes\index_v1
C:\Users\18449\Desktop\researchguard_workspace\outputs\index_validation_v1
```

Corpus manifest：

| Metric | Value |
| --- | ---: |
| paper_count | 5 |
| chunk_count | 337 |
| paper_agent | 96 |
| paper_citation | 80 |
| paper_corrective_rag | 54 |
| paper_hallucination | 46 |
| paper_rag | 61 |
| total_char_count | 403493 |
| total_word_count | 60910 |
| metadata_missing_count | 0 |
| duplicate_chunk_id_count | 0 |
| content_hash_duplicate_count | 0 |

Section distribution：

| Section | Chunks |
| --- | ---: |
| abstract | 13 |
| conclusion | 7 |
| discussion | 7 |
| experiment | 57 |
| introduction | 34 |
| limitations | 2 |
| method | 25 |
| references | 107 |
| related_work | 32 |
| results | 53 |

Chunk type distribution：

| Type | Chunks |
| --- | ---: |
| text | 194 |
| references | 107 |
| mixed | 26 |
| equation | 8 |
| table | 2 |

Special chunk counts：

| Type | Chunks |
| --- | ---: |
| has_equation | 16 |
| has_table | 11 |
| has_caption | 19 |
| mixed | 26 |
| references | 107 |

Embedding and index：

| Metric | Value |
| --- | --- |
| provider | `openai` |
| model | `text-embedding-3-small` |
| dimensions | `1536` |
| normalize | `true` |
| dense_backend | `numpy` |
| dense_metric | `cosine` |
| sparse_backend | `local_bm25` |
| corpus_fingerprint | `c54338f9f3f063f0bed4cf2d4f2e825333b4a794bfa89fe0b4341c9f2f3a7d59` |

构建记录：

- 首次正式 build：`added=337`、`embedded=337`、`cache_hits=0`、`cache_misses=337`。
- 第二次 incremental build：`added=0`、`updated=0`、`removed=0`、`reused=337`、`embedded=0`、`cache_hits=337`、`cache_misses=0`。

Index validation hard checks：

- corpus fingerprint mismatch = 0；
- duplicate index id = 0；
- missing id mapping = 0；
- stale entries = 0；
- zero vector = 0；
- normalization mismatch = 0；
- source file hash mismatch = 0；
- reload failure = 0；
- deterministic incremental synthetic failure = 0；
- self-retrieval samples = 56；
- self-retrieval catastrophic mismatch = 0。

### 16.5 Chroma backend v1

Chroma validation 结论：`PASS`。

- 首次构建：`inserted=337`、`updated=0`、`deleted=0`、`collection_count=337`；
- 第二次无变化同步：`added=0`、`updated=0`、`metadata_updated=0`、`deleted=0`、`reused=337`；
- 所有 hard checks 为 0，missing/stale/duplicate IDs 均为 0；
- collection metadata 与 current corpus fingerprint、model、dimension、metric、schema 完全一致；
- persistence reload 和独立 query 通过；
- 40 个 embedding 抽样最大绝对误差 `7.450580596923828e-09`，平均绝对误差 `1.886318878874249e-10`；
- 10 类真实 metadata filter 全部通过；
- 8 个 incremental synthetic invariants 全部通过；
- 30 条真实 query：Top-1 agreement、Top-3/5/10 overlap、rank correlation 均为 `1.0000`；
- 不含 query embedding 的本地平均 latency：NumPy `0.8650 ms`，Chroma `10.7916 ms`。

### 16.6 Retrieval v1

Retrieval validation 结论：`PASS`。

Benchmark：

| Metric | Value |
| --- | ---: |
| query_count | 40 |
| answerable_query_count | 36 |
| no_answer_query_count | 4 |
| corpus_fingerprint_match | yes |
| synthetic_tests_passed | 10/10 |
| failure_case_count | 18 |

Hard checks 全部为 0：

- index_load_failure；
- fingerprint_mismatch；
- chunk_id_mapping_mismatch；
- metadata_missing；
- dense_dimension_mismatch；
- duplicate_chunk_id；
- schema_error；
- query_embedding_failure；
- non_deterministic_ranking；
- benchmark_invalid_chunk_id；
- benchmark_empty_query；
- benchmark_schema_error；
- result_schema_error。

Dense / BM25 / Hybrid 对比：

| Metric | Dense | BM25 | Hybrid |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.5000 | 0.4167 | 0.5556 |
| Recall@3 | 0.8333 | 0.6667 | 0.7778 |
| Recall@5 | 0.8889 | 0.8333 | 0.9167 |
| Recall@10 | 0.9444 | 0.9444 | 0.9444 |
| MRR@10 | 0.6579 | 0.5779 | 0.6847 |
| nDCG@10 | 0.6176 | 0.5369 | 0.6197 |
| document_hit@10 | 1.0000 | 0.9722 | 1.0000 |
| section_hit@10 | 1.0000 | 0.9722 | 1.0000 |
| multi_evidence_coverage@10 | 0.7348 | 0.6667 | 0.7172 |
| no_answer_false_positive_rate | 1.0000 | 1.0000 | 1.0000 |
| average_latency_ms | 432.3211 | 0.7899 | 11.1072 |
| p95_latency_ms | 765.7316 | 1.7658 | 15.0317 |

解释：

- Hybrid Recall@10 不低于 Dense 和 BM25，MRR@10 也没有回退，符合 Retrieval v1 `PASS` 标准。
- Hybrid Recall@3 低于 Dense，但 Recall@5、MRR@10 和 nDCG@10 更好，当前记录为真实指标，不为追求单项指标调参。
- no-answer 查询全部返回了候选 chunk，因此 false positive rate 为 1.0。这是 retrieval-only 组件的真实基线；当前 Evidence Sufficiency v1 在其后执行，并在独立 44-query benchmark 的四条 unrelated no-answer 上将 FPR 降到 0。
- Dense 平均延迟主要受 query embedding API 影响；validation 中同一 query 的 dense/hybrid embedding 会通过进程内 cache 复用，因此 hybrid 平均延迟远低于单独 dense benchmark 不是向量检索本身更快，而是验证执行顺序和缓存造成的测量结果。当前默认 dense backend 为 Chroma。

Synthetic tests：

- `index_loader_hard_checks_zero`；
- `deterministic_hybrid_ranking`；
- `deterministic_dense_ranking`；
- `deterministic_sparse_ranking`；
- `metadata_filter_doc_and_section`；
- `exclude_references_filter`；
- `empty_query_rejected`；
- `fingerprint_mismatch_detected`；
- `hybrid_schema_has_fusion_and_sources`；
- `benchmark_cases_loadable`。

### 16.7 Reranker v1

Reranker validation 结论：`PASS`。

| Metric | Hybrid baseline | Reranked | Delta |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.5556 | 0.6111 | +0.0556 |
| Recall@3 | 0.7778 | 0.8056 | +0.0278 |
| Recall@5 | 0.9167 | 0.9722 | +0.0556 |
| Recall@10 | 0.9444 | 0.9722 | +0.0278 |
| MRR@10 | 0.6847 | 0.7394 | +0.0546 |
| nDCG@10 | 0.6197 | 0.6584 | +0.0388 |
| document_hit@10 | 1.0000 | 1.0000 | 0.0000 |
| section_hit@10 | 1.0000 | 1.0000 | 0.0000 |
| multi_evidence_coverage@10 | 0.7172 | 0.7348 | +0.0177 |
| no_answer_false_positive_rate | 1.0000 | 1.0000 | 0.0000 |

Hard checks 全部为 0：

- `reranker_load_failure`；
- `candidate_count_mismatch`；
- `missing_chunk_id`；
- `result_schema_failure`；
- `non_deterministic_failure`；
- `cache_consistency_failure`；
- `benchmark_label_leakage`；
- `retrieval_baseline_changed_unexpectedly`。

同一运行内 baseline 重复检索完全一致。报告另外保留 1 条 information-only historical Chroma drift：这是当前结果相对旧输出的近似排序差异，不改变同一运行中的 RRF baseline，也不作为 hard failure 隐藏。原 Retrieval v1 validation 重新运行后仍为 `PASS`。

### 16.8 Query Rewrite v1

Query Rewrite validation 结论：`PASS`。

| Metric | Raw Hybrid | Raw + Reranker | Rewrite Hybrid | Rewrite + Reranker | Multi-query + Reranker |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recall@10 | 0.9444 | 0.9722 | 0.9444 | 0.9722 | 0.9722 |
| MRR@10 | 0.6847 | 0.7394 | 0.6861 | 0.7370 | 0.7394 |
| nDCG@10 | 0.6197 | 0.6584 | 0.6166 | 0.6545 | 0.6468 |
| document_hit@10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| section_hit@10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9722 |
| multi_evidence_coverage@10 | 0.7172 | 0.7348 | 0.6995 | 0.7273 | 0.7096 |

Hard checks 全部为 0：`empty_rewrite`、`entity_preservation_failure`、`invalid_json_failure`、`benchmark_leakage`、`fallback_failure`、`duplicate_query_failure`、`non_deterministic_cache_failure`、`result_schema_failure`。

- synthetic tests：8/8 passed；
- 真实 rewrite：40 success、0 fallback；
- cold/warm cache：0/40 和 40/40 hits；
- 平均保留 expansions：1.65；不合规 expansions 丢弃并记录：14；
- rewrite API：40 calls、9348 input tokens、2132 output tokens；
- query embedding API：131 calls；
- original Top-10 misses：2，其中 1 条有可归因 rewrite 改善；
- Query Rewrite 接入后，原 Retrieval v1 和 Reranker v1 回归均重新得到 `PASS`。

### 16.9 Evidence Sufficiency v1

Evidence Sufficiency validation 结论：`PASS`。

| Metric | Value |
| --- | ---: |
| Benchmark | 44（35 strong / 5 partial / 4 unsupported） |
| Answerability accuracy | 0.8182 |
| Precision | 1.0000 |
| Recall | 0.7714 |
| F1 | 0.8710 |
| Support-level accuracy | 0.8182 |
| Partial recall | 1.0000 |
| Unsupported recall | 1.0000 |
| No-answer false positive rate | 0.0000（retrieval-only 为 1.0000） |

7 项 hard checks 全部为 0；11/11 synthetic tests 通过；44 次冷 Judge 无 fallback，随后 warm cache 44/44 命中且输出一致。运行记录为 44 API calls、211326 input tokens、9921 output tokens；Judge 平均 `4646.71 ms`、P95 `8502.83 ms`，完整 retrieval + Judge 平均 `7696.17 ms`、P95 `17172.80 ms`。

代表性结果：

- strong：RAG 核心思想 query，`answerable=true`，引用 `paper_rag_chunk_00002/00006/00007`；
- partial：CRAG lightweight evaluator + GPT-5 复合 query，仅 evaluator 部分有证据，`answerable=false`，引用三个 CRAG chunks；
- unsupported：superconducting-qubit quantum error correction query，没有 supporting chunks，`answerable=false`。

最终仍有 8 个保守 false negatives，false positives 为 0。原 Retrieval v1 与 Reranker v1 独立 validator 在接入后均重新得到 `PASS`；Evidence layer 没有改变已有 ranking。

### 16.10 Answer Generation v1

Answer Generation validation 结论：`PASS`。

| Metric | Value |
| --- | ---: |
| Benchmark | 22（14 strong / 4 partial / 4 unsupported） |
| Citation Coverage | 1.0000 |
| Citation provenance validity | 1.0000 |
| Unsupported Claim Rate | 0.0000 |
| Refusal Accuracy | 1.0000 |
| Answerability consistency | 1.0000 |
| Required-concept lexical coverage | 0.9535 |
| Supported answer rate | 1.0000 |
| Fully grounded answer rate | 1.0000 |

8 项 hard checks 全部为 0：`json_parse_failure`、`missing_citation`、`unsupported_generation`、`evidence_leakage`、`benchmark_leakage`、`cache_inconsistency`、`schema_failure`、`retrieval_regression`。12/12 synthetic tests 通过；14 条 strong 全部生成，4 条 partial 和 4 条 unsupported 全部固定拒答，generator 零越权调用；14/14 warm answer cache 命中且稳定字段一致，无 fallback。

正式运行记录为 14 generator API calls、27287 input tokens、2699 output tokens；验证期 grounding evaluator 为 14 calls、27626 input tokens、1881 output tokens。Retrieval 平均/P95 为 `2993.58/6932.66 ms`，Evidence gate 因命中已有 cache 为 `2.33/2.96 ms`，实际生成平均/P95 为 `6221.65/12129.33 ms`，不含 validation-only grounding 的产品链路端到端平均/P95 为 `6955.15/15616.29 ms`。

Unsupported Claim Rate 和 Fully grounded rate 来自 validation-only answer-level LLM 评估，适合当前回归，不等于已经实现 claim verification 或 Citation Audit。当前 22 条数据来自同一五篇开发语料，也不是独立 blind hold-out。

### 16.11 Claim Verification / Citation Audit v1

Citation Audit validation 结论：`PASS`。

| Metric | Value |
| --- | ---: |
| Benchmark | 12 answers / 26 atomic claims |
| Claim Support Accuracy | 1.0000 |
| Supported Claim Precision | 1.0000 |
| Unsupported Claim Detection Recall | 1.0000 |
| Unsupported Claim Rate | 0.1923 |
| Citation Precision | 0.9600 |
| Citation Recall | 1.0000 |
| Answer Grounded Rate | 0.5833 |

8 项 hard checks 全部为 0：`claim_parse_failure`、`verification_schema_failure`、`citation_missing`、`unsupported_claim_leakage`、`benchmark_leakage`、`cache_inconsistency`、`retrieval_regression`、`answer_generation_regression`。12/12 synthetic tests 通过；12 个 cold audits 无 fallback，26 个 predicted claims 与 26 个 expected claims 对齐；随后 warm cache 12/12 命中且稳定字段一致。

正式 benchmark 使用 `gpt-4.1-mini`，Claim Extraction prompt 为 `claim_extraction_v1.1`，Verification prompt 为 `claim_verification_v1.2`，config version 为 `citation_audit_v1.1`。共 38 次 API calls、23429 input tokens、2859 output tokens。Extraction 平均/P95 为 `3720.11/12615.94 ms`，逐答案 Verification 平均/P95 为 `5162.61/8813.87 ms`，完整 audit 平均/P95 为 `8883.13/19313.85 ms`。

Citation Precision 的 `0.96` 表示一个已支持 claim 返回了额外 citation；该 citation 仍来自允许 evidence，但人工 gold 未把它列为该 claim 的必要支持。项目如实保留该冗余，不把 provenance 合法等同于 citation 最小性。Answer Grounded Rate 只有 `0.5833` 是因为 benchmark 有意放入 5 个含 unsupported claims 的答案，不是系统遗漏拒答。

### 16.12 Pipeline Integration v1

Pipeline validation 结论：`PASS`。

- 6/6 synthetic tests 通过：strong 完整执行、partial/unsupported 零生成、answer 模块可关闭、模块异常保持统一 schema、相同输入稳定字段一致。
- 真实 sufficient query：`final_status=grounded`，Evidence=`strong`，Answer 有 1 条 citation，Citation Audit completed 且 overall grounded；最终回归总 latency `113.37 ms`。
- 真实 unsupported query：Evidence=`unsupported`，Answer/Audit 均 `skipped`，`final_status=rejected`；最终回归总 latency `2127.19 ms`。
- 真实 partial query：Evidence=`partial`，Answer/Audit 均 `skipped`，`final_status=rejected`；最终回归总 latency `2284.89 ms`。
- Rewrite、Reranker、Evidence、Answer 和 Citation Audit 重复运行 cache 均命中；Pipeline Top-10 与直接 Retrieval Top-10 完全一致。
- 八项 hard checks 全为 0：`pipeline_schema_failure`、`module_failure`、`evidence_gate_bypass`、`unsupported_generation`、`citation_missing`、`cache_failure`、`retrieval_regression`、`result_schema_failure`。

这些是本地 warm-cache E2E 数值，不代表 cold API、模型首次加载、并发服务或生产网络延迟。Pipeline v1 验证编排不变量，不替代各模块独立 benchmark，也不是 blind hold-out。

### 16.13 Streamlit Demo v1

Demo validation 结论：`PASS`。

| Check | Result |
| --- | ---: |
| Streamlit startup | PASS |
| Strong render contract | PASS |
| Unsupported render contract | PASS |
| Display sanitization | PASS |
| Browser desktop 1440x900 | PASS |
| Browser mobile 390x844 | PASS |

Strong 真实案例返回 `final_status=grounded`、`support_level=strong`、10 条 evidence、370 字符 answer、1 条 answer citation 和 4 条 audited claims，六个 stage 均 completed。Unsupported 真实案例返回 `final_status=rejected`、`support_level=unsupported`、固定拒答、0 citation、0 claim，Answer Generation 与 Citation Audit 均 skipped。Demo 没有修改 benchmark，也没有形成独立模型质量指标；它验证的是页面启动、Pipeline 接入和展示 contract。

## 17. 已知限制

- OCR fallback 尚未接入当前 parser v5 主流程。
- 复杂跨栏表格、跨页表格、图文环绕和多区域布局仍有限。
- caption/table/equation 当前通过同 section、优先同页、y 坐标和 block 顺序的邻近策略绑定，不是完整视觉语义绑定。
- overlap 当前是直接前一个同 section chunk 的最后 N 句正文复制，不包含 heading prefix，并已经通过 direct-previous、`0/1/2` 参数、heading exclusion 和 provenance 验证；但它仍只是上下文复制，不应在下游 audit 中当作新的 source evidence。
- parser 输出中可能存在历史乱码，尤其来自 PDF 字体编码问题。
- Indexing v1 的 NumPy exact dense index 仍是可复现基线；Chroma v1 已作为本地持久化 backend 接入并成为 Retrieval 默认 dense source，但当前只有 337 records，不应从这一规模推断大规模 ANN 性能。
- Chroma 当前只使用本地进程内 PersistentClient，适用于开发和单机演示，不是 Chroma Cloud、独立 server、分布式数据库或高并发生产部署。
- Chroma 1.5.9 当前公开安全信息包含一项 HTTP server 端点风险；项目不启动该 server，但未来服务化前必须重新评估并升级到确认修复的版本。
- 当前 Chroma/NumPy Top-10 parity 为 1.0；语料增长、HNSW 参数或 Chroma 版本变化后必须重新验收，不能假设近似索引始终与 exact search 完全一致。
- 当前环境存在 legacy `langgraph-sdk 0.4.2` 与 `websockets 16.0` 的 `pip check` 冲突；它不在本阶段 Chroma/Retrieval 路径，尚未调整。
- Sparse index 当前是本地 BM25 JSON 持久化实现，适合当前五篇 benchmark 和后续小规模验证；大规模语料需要重新评估性能和存储格式。
- Retrieval/Query Rewrite/Reranker/Evidence Sufficiency/Answer Generation/Citation Audit 已形成受控问答与审计主流程，但仍不是完整 Agentic RAG 产品。
- Reranker v1 使用通用 MS MARCO Cross-Encoder，并未针对科研论文、公式、表格、caption 或 reference entry 微调；40-query benchmark 中有 7 条 query 排名退化，最严重的是 equation-like query 从 relevant rank 1 降到 rank 3。
- 当前 PyTorch 为 CPU build；正式 benchmark 冷推理平均约 1.71 秒/query，独立 CLI 新进程还存在明显模型加载/首轮推理开销。尚未实现常驻服务预热、GPU、ONNX、量化或并发优化。
- rerank cache 是本地 JSON file cache，适合当前单机回归，不是并发安全的分布式 cache。
- Cross-Encoder `max_length=512`，过长的 title/section/heading/content 组合会由 tokenizer 截断；当前没有长文档 sliding-window rerank。
- Query Rewrite 的 temperature 0 只降低随机性，不保证跨独立 API 调用逐字一致；项目依靠版本化 cache 保证同一已生成结果的重复运行确定性。
- 当前 query analysis 主要依靠英文规则抽取模型名、缩写、数字和约束；复杂中文实体、化学式、带空格的数据集名和领域专名仍需扩充规则或独立 NER。
- 为防实体丢失，14 条按子问题拆分但缺少部分实体的 expansions 被丢弃；这提高安全性，但也使 `rv1_q031` 这类双实体比较 query 仍未召回 relevant chunks。
- 单独 Rewrite Hybrid 的 MRR/nDCG 略低于 Raw Hybrid；多查询扩展不是默认必开能力，必须与原 query 和 Reranker 一起评估。
- Multi-query 会放大 query embedding calls 和延迟；当前 40-query cold run 共 40 次 rewrite calls、131 次 query-embedding calls，尚未做 batch rewrite、并发限流或服务级成本预算。
- Rewrite cache 是本地 JSON file cache，适合单机回归，不是并发安全的分布式 cache；fallback 也会按 prompt/config version 缓存。
- Query Rewrite 本身不判断 evidence sufficiency；该职责由检索后的独立 Judge 承担。当前仍未实现 neighbor expansion、自动 audit retry 或 Agentic RAG workflow。
- Retrieval 仍会为 no-answer query 返回最近 chunks；Evidence Judge 在当前 4 条 unrelated no-answer 上将 FPR 降到 0，但它只判断传入 Top-k，不能证明整个 corpus 中没有答案。
- Evidence Sufficiency 使用 `gpt-4.1-mini` LLM-as-a-Judge。temperature 0 不保证跨独立 API 调用逐字一致，项目依赖 model/prompt/config/input-hash cache 固化已经生成的 verdict。
- Judge 每个 chunk 最多读取 1800 字符、最多读取 Top-10；retrieval miss、chunk 截断、缺失 metadata filter 或跨 chunk 组合证据都可能造成 false negative。最终 44 条 benchmark 保留 8 个 false negatives。
- 当前 Judge benchmark 只有 44 条，来自同一五篇论文，并在开发中用于 prompt 校准；它是 regression/development benchmark，不是独立 blind hold-out，也不能证明跨领域、跨语言或开放语料泛化。
- 当前 4 条 unrelated no-answer 太少，`no_answer_false_positive_rate=0` 只能说明这组回归案例通过，不能作为生产拒答率估计。
- Pipeline 只保证 supporting chunk ID 来自当前 Top-k，并通过 LLM coverage 约束其语义用途；尚未有独立人工标注的 supporting-ID precision/recall benchmark。
- Judge cold run 平均约 4.65 秒并消耗约 4803 input tokens/query；尚未实现 batch、并发限流、预算路由、小模型级联或离线本地 Judge。
- Judge API/JSON/schema 失败会保守拒答。这避免错误放行，但在服务故障时会增加 false negatives；当前 cache 也是单机 JSON file cache，不适合多进程并发生产环境。
- Answer Generator 只读取 Judge 返回的 supporting chunks；Judge 漏掉必要证据、错误放行或 retrieval miss 会直接限制答案质量。Answer pipeline 不重新判断 corpus-level answerability。
- Answer Generation 使用 `gpt-4.1-mini` 和 temperature 0；独立 API 调用仍不保证逐字确定，当前依靠 model/prompt/config/input-hash cache 固化结果。
- Answer Generation 单独看只保证 citation ID 与 `doc_id/section/page` provenance 来自允许 evidence；当前 Citation Audit v1 已进一步做 atomic claim support verification，但仍不保证 claim extraction 完整、citation 最小充分或表格/公式级 entailment。
- 当前 Unsupported Claim Rate 为 validation-only LLM 对 14 个生成答案的 answer-level 估计，不是人工 claim-level 标注，也不能外推为生产 hallucination rate。
- Answer benchmark 只有 22 条，和前序 benchmark 使用同一五篇开发论文；尚缺独立 blind hold-out、跨语言问题、长答案、多文档冲突证据和更复杂数值/表格回答。
- Generator 每个 supporting chunk 最多读取 1800 字符、最多 10 chunks；截断可能遗漏细节。当前没有 citation-aware sentence alignment、流式生成、batch、并发限流或成本路由。
- Answer cache 是本地 JSON file cache，不是并发安全的分布式 cache；生成失败也会 fail closed，服务故障期间可能增加拒答。
- Citation Audit 使用 LLM 做 atomic extraction 和 support judgement；temperature 0 不保证跨独立 API 调用逐字一致，版本化 cache 才是当前重复运行稳定性的来源。
- Audit 候选只来自 answer citations 和同次 generation evidence，不重新检索。这能保持 provenance 一致，但无法为 answer-time retrieval miss 补证；此时 claim 可能被判 unsupported。
- Verifier 判断“evidence 是否支持 claim”，不判断论文来源本身是否真实、实验是否可复现或不同论文之间谁更可信。
- 当前没有表格单元格级对齐、公式推导验证、图片内容理解、跨来源冲突消解、claim dependency graph 或时序/因果图验证。
- Citation Precision `0.96` 暴露了冗余引用：v1 校验相关支持与 provenance，但没有最小 citation set 优化。
- Audit benchmark 仅 12 个答案、26 个 claims，来自同一五篇开发论文；尚缺独立 blind hold-out、中文 claims、长答案、复杂否定、跨文档冲突和领域专家人工复核。
- 当前逐 claim 串行调用 verifier，完整 audit 平均约 8.88 秒；尚未实现 batch verification、并发限流、成本路由或本地 NLI fallback。
- Citation audit cache 是本地 JSON file cache，不适合多进程并发或分布式部署；失败会 fail closed，但不会自动触发重新检索或答案重写。
- Pipeline v1 是同步单 query 编排器；没有 async/batch、并发限流、服务级 timeout/cancellation、预算路由、自动 retry 或持久化 distributed trace。
- Pipeline 的 Rewrite/Retrieval/Reranker 由同一次 `RetrievalEngine.retrieve` 执行；分段 latency 精确记录，但三个 stage 的 wall-clock start/end 不是独立执行窗口。
- `include_retrieval_text=true` 会在统一 JSON 中保留完整 Top-k 文本，便于审阅但会增大输出；对外服务化前需要明确数据最小化和访问控制。
- Pipeline 只保证调用顺序、gate、provenance 和 fail-closed 状态，不提高任何单模块的准确率，也不能证明 corpus 完整、论文结论真实或最终系统可安全用于生产。
- Streamlit Demo v1 只面向本地单用户展示；没有用户系统、鉴权、速率限制、并发任务队列、取消/恢复、在线部署、HTTPS、审计日志或 API SLA。
- Demo 使用进程内 `st.cache_resource` 复用 Pipeline 实例；这降低重复模型加载开销，但不是多进程或多租户资源管理方案。
- Evidence 文本可能保留 PDF parser 的历史乱码；Demo 如实展示当前 chunk，不在 UI 层修正 source text。
- Debug JSON 已脱敏本地路径和 secret-like 字符串，但正式对外部署前仍需更严格的数据分级、内容访问控制和安全审查。
- Retrieval benchmark 目前只有五篇论文、40 条人工可核对 query；它适合当前回归验证，不代表大规模真实科研检索质量。
- Dense retrieval 依赖 OpenAI query embedding 和 `OPENAI_API_KEY`；sparse BM25 可离线运行。
- Agentic RAG 仍是 legacy 模块，尚未围绕新 chunks 重构。
- Evidence Audit 部分能力存在，但尚未和新 parser/chunk/index/retrieval 主链路完整产品化串联。
- Frontend/API 尚未完成产品级实现。

## 18. 下一步计划

当前 Retrieval、Chroma、Reranker、Query Rewrite、Evidence Sufficiency、Answer Generation、Citation Audit、Pipeline Integration 和 Streamlit Demo v1 均已通过独立验证。后续若进入 Agentic RAG，必须把当前 gate 和 audit 作为不可绕过的边界，而不是让 Agent 用重试掩盖 unsupported claims。

1. 冻结 Pipeline v1 schema/config、Demo display contract、stage status、gate 语义和 E2E regression outputs。
2. 建立独立 blind hold-out，增加中文、长答案、复杂数字/表格/公式、否定、跨论文冲突和 citation redundancy 标注。
3. 为 claim 保存 answer span offsets，减少 paraphrase 后人工定位困难，并评估 extractor 的 claim completeness，而不只评估已匹配 claims。
4. 单独评估 citation necessity/minimality、跨多个 chunks 的联合 entailment 和冲突 evidence，不能用 provenance validity 代替。
5. 研究 async/batch、并发限流、统一 timeout/cancellation、预算路由和本地 fallback，降低完整链路延迟。
6. 只有在新的 blind hold-out 通过后，才设计 Agent workflow、受控 retrieval retry 和 Evidence Graph；任何 retry 仍需重新经过 Evidence Sufficiency、Answer Generation 和 Citation Audit。
7. 语料或任一 parser/chunk/index/retrieval/rewrite/rerank/Judge/Answer/Audit/Pipeline config 变化时，重新运行对应上游和完整下游回归。

## 19. Development Documentation Rule

今后每次新增功能、修改功能、调整目录、增加脚本、改变运行命令、更新验证结果或推进项目阶段时，都必须同步更新 `README.md`。

每次 README 更新至少要同步：

- 当前状态；
- 数据流；
- 核心文件；
- 方法说明；
- 运行命令；
- 输出目录；
- 验证结果；
- 已知限制；
- 下一步计划。

以后在本项目中完成任何代码修改时，默认执行这条规则。若某个任务明确禁止修改 README，则应先提示：README 尚未同步，并说明需要在允许时补充更新。
