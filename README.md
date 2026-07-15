# ResearchGuard

ResearchGuard 是面向科研论文的多文档 Agentic RAG 与 Evidence Audit 平台原型。它不是普通的 PDF 问答 demo：项目目标是把论文 PDF 解析成可追溯的结构化 evidence，再服务于后续 indexing、retrieval、Agentic RAG、claim-level audit、报告生成和可复现实验。

当前工程重点已经从早期迁移代码，推进到真实论文解析、section-aware chunking 和 Embedding + Persistent Indexing v1 验收阶段。Parser v5 已完成严格验收，结论为 `PASS_WITH_CHUNK_FIXES`；Chunking v1 已完成最终边界修复，synthetic tests 和五篇真实论文硬性检查均通过，当前可标记为 `APPROVED_FOR_INDEXING`；Indexing v1 已基于 `data/parsed/chunk_eval_v1` 完成真实 OpenAI embedding 构建、本地持久化 dense/sparse index 和严格验证，结论为 `PASS`。Retrieval/Agentic RAG/answer generation/evidence audit 主流程尚未正式接入新索引。

## 1. 项目背景与目标

ResearchGuard 关注的是科研论文场景中 RAG 的可信性问题：

```text
论文 PDF
  -> 结构化解析
  -> section-aware chunks
  -> embedding / indexing
  -> retrieval
  -> Agentic RAG answer
  -> claim extraction
  -> evidence audit
  -> supported / partial / unsupported verdict
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

当前已经重点完成的是 `PDF -> layout/block/section -> section-aware chunk` 这段底座。后续 embedding/indexing/retrieval/agent/audit 的产品级主链路还需要继续接入。

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
- 为什么这样做：当前环境没有 FAISS/Chroma，但已有 `openai`、`numpy`、`rank_bm25` 和 `scikit-learn`。本地 NumPy + BM25 能在不新增依赖的前提下完成可重载、可验证、可增量的索引底座。
- 在 RAG 中的作用：为后续 retrieval 提供稳定的向量和稀疏召回输入，同时保留 chunk metadata、source provenance、overlap provenance 和 corpus fingerprint。
- 核心文件：`researchguard/indexing/corpus_loader.py`、`researchguard/indexing/embedding_provider.py`、`researchguard/indexing/embedding_cache.py`、`researchguard/indexing/dense_index.py`、`researchguard/indexing/sparse_index.py`、`researchguard/indexing/index_v1.py`、`scripts/build_index_v1.py`、`scripts/validate_index_v1.py`。

### 2.11 后续 retrieval/agent/audit

- 输入：计划使用 `data/indexes/index_v1/`。
- 输出：尚未正式接入新主流程。
- 当前状态：旧 retrieval、agent、audit、evaluation 模块存在，但尚未围绕 index_v1 重构成正式 pipeline。当前只做了 index-level self-retrieval sanity check，不是正式问答或 retrieval benchmark。
- 核心目录：`researchguard/retrieval`、`researchguard/agent`、`researchguard/audit`、`researchguard/evaluation`。

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

## 5. 目录结构

```text
researchguard/ingestion/      PDF layout extraction, block detection, heading classification, section recovery, chunking
researchguard/indexing/       旧索引模块存在，尚未接入 chunk_eval_v1
researchguard/retrieval/      检索层目录存在，尚未形成新主流程
researchguard/agent/          legacy Agentic RAG 模块存在，尚未重构接入新 chunks
researchguard/audit/          Evidence audit 相关能力，部分实现
researchguard/memory/         memory / trace 存储能力，部分实现
researchguard/evaluation/     retrieval / answer / agentic evaluation 相关脚本，部分实现
researchguard/reporting/      audit report 渲染能力，部分实现
scripts/                      验证、构建和本地功能测试脚本
configs/                      baseline 和 corpus profile 配置
data/raw_docs/                原始 PDF
data/parsed/                  parser 和 chunking 输出
outputs/                      parser/chunk validation、local rag、audit 等输出
frontend/                     旧 Streamlit 文件存在，尚未产品化
```

当前重点数据目录：

```text
data/raw_docs/parser_eval/         五篇 benchmark PDF
data/parsed/parser_eval_v5/        parser v5 输出
data/parsed/chunk_eval_v1/         section-aware chunk v1 输出
data/indexes/index_v1/             Indexing v1 输出：corpus manifest、embedding cache、dense/sparse index、index manifest
outputs/parser_validation_v5/      parser v5 验收报告
outputs/chunk_validation_v1/       chunking v1 验收报告
outputs/index_validation_v1/       indexing v1 验收报告
```

## 6. 当前阶段状态

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| Parser | 已完成 | 已完成 v5，并通过严格验收；结论 `PASS_WITH_CHUNK_FIXES`，说明 parser 可作为 chunking 输入，但旧 chunks 需修。 |
| Chunking | 已完成并通过 indexing 前置条件 | v1 最终边界修复已完成，synthetic tests 全过，五篇硬性检查全 0；验证脚本仍标记 `PASS_WITH_MINOR_ISSUES`，因为保留短 chunk、少数 heading_path 缺失和复杂视觉语义绑定限制。当前结论为 `APPROVED_FOR_INDEXING`。 |
| Indexing | 已完成 v1 并通过验证 | 旧 LlamaIndex page-level builder 仍保留为迁移代码；新的 Indexing v1 已接入 `data/parsed/chunk_eval_v1`，使用 OpenAI embedding、持久化 NumPy dense index 和本地 BM25 sparse index，验证结论 `PASS`。 |
| Retrieval | 尚未实现新主流程 | 目录存在，正式 retrieval pipeline 尚未围绕 `data/indexes/index_v1` 建立；目前只完成 index-level self-retrieval sanity check。 |
| Agentic RAG | 已迁移但未重构 | legacy 模块存在，尚未接入新 indexing/retrieval 主流程。 |
| Evidence Audit | 部分实现 | `researchguard/audit` 中已有 claim/evidence/audit 相关模块，但未与新主流程产品化串联。 |
| Evaluation | 部分实现 | 评测脚本和数据基础存在，正式新链路评测尚未完成。 |
| Frontend/API | 尚未完成产品化 | `frontend` 和 `researchguard/api` 存在，但未完成产品级 UI/API。 |

## 7. 数据和 benchmark

当前 benchmark 使用五篇科研论文：

- `paper_rag`
- `paper_agent`
- `paper_hallucination`
- `paper_corrective_rag`
- `paper_citation`

用途：

- 覆盖单栏/双栏、References、多 section、图表/caption/table/equation、appendix 等论文结构；
- 用于 parser v5 和 chunking v1 的固定验收；
- Indexing v1 已完成后，五篇语料继续作为 retrieval 接入前的固定回归集。

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

## 8. 运行命令

所有命令固定使用项目虚拟环境，不使用 Codex bundled Python。

### 8.1 进入项目并激活虚拟环境

```powershell
cd C:\Users\18449\Desktop\researchguard_workspace
.\.venv\Scripts\activate
```

也可以不激活，直接使用绝对路径 Python：

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" --version
```

当前验证过的 Python 版本：`Python 3.12.10`。

### 8.2 Parser 运行

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

### 8.3 Parser validation

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_parser_v5.py"
```

输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\outputs\parser_validation_v5
```

### 8.4 Chunk build

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

### 8.5 Chunk validation

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_chunking_v1.py"
```

输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\outputs\chunk_validation_v1
```

### 8.6 Index dry-run

dry-run 只扫描 corpus、验证 manifest schema、计算增量计划，不调用 embedding API，不写正式索引。

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\build_index_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\indexing_v1.yaml" `
  --dry-run
```

### 8.7 Index build

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

### 8.8 Index validation

```powershell
& "C:\Users\18449\Desktop\researchguard_workspace\.venv\Scripts\python.exe" `
  "C:\Users\18449\Desktop\researchguard_workspace\scripts\validate_index_v1.py" `
  --config "C:\Users\18449\Desktop\researchguard_workspace\configs\indexing_v1.yaml"
```

输出：

```text
C:\Users\18449\Desktop\researchguard_workspace\outputs\index_validation_v1
```

## 9. 当前指标

### 9.1 Parser v5

Parser v5 结论：`PASS_WITH_CHUNK_FIXES`。

不是因为 `status=ok` 才通过，而是经过以下验收：

- reading order：五篇均抽样检查第一页、中间页、References 首页和表格/图片较多页；抽样中 column backtrack 和同栏 y backtrack 均为 0；
- heading：五篇 heading suspicious = 0，unmapped = 0；
- section：输出完整 block-level section transition trace；无 References 后错误回到 main_text/method 的硬失败；
- References：均找到 References heading，并检查后续 reference-like ratio、首页/末页 sample；
- chunk audit：旧 parser chunks 暴露问题，包括短 chunk、heading-only、>1600、multi-section、重复 block refs、equation 未进 chunk。

Parser v5 的真正结论是：parser 层可以进入下一阶段；chunk 层必须修。

### 9.2 Parser 阶段旧 chunks 暴露的问题

| Paper | Old chunks | <150 | >1600 | Multi-section | Heading-only | Duplicate block refs | Equation in old chunks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| paper_rag | 69 | 7 | 1 | 0 | 4 | 25 | 0/0 |
| paper_agent | 106 | 17 | 1 | 0 | 4 | 36 | 0/4 |
| paper_hallucination | 53 | 9 | 0 | 0 | 5 | 17 | 0/1 |
| paper_corrective_rag | 65 | 9 | 0 | 0 | 5 | 22 | 0/1 |
| paper_citation | 90 | 11 | 0 | 1 | 3 | 38 | 0/2 |

这些问题来自旧 chunk_builder，不是 parser v5 的 reading order / heading / section 主体失败。

### 9.3 Chunking v1

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

### 9.4 Indexing v1

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

## 10. 已知限制

- OCR fallback 尚未接入当前 parser v5 主流程。
- 复杂跨栏表格、跨页表格、图文环绕和多区域布局仍有限。
- caption/table/equation 当前通过同 section、优先同页、y 坐标和 block 顺序的邻近策略绑定，不是完整视觉语义绑定。
- overlap 当前是直接前一个同 section chunk 的最后 N 句正文复制，不包含 heading prefix，并已经通过 direct-previous、`0/1/2` 参数、heading exclusion 和 provenance 验证；但它仍只是上下文复制，不应在下游 audit 中当作新的 source evidence。
- parser 输出中可能存在历史乱码，尤其来自 PDF 字体编码问题。
- Indexing v1 已正式接入 `data/parsed/chunk_eval_v1` 并通过验证，但 dense backend 当前是本地 NumPy brute-force search，不是 FAISS/Chroma 等专用向量数据库。
- Sparse index 当前是本地 BM25 JSON 持久化实现，适合当前五篇 benchmark 和后续小规模验证；大规模语料需要重新评估性能和存储格式。
- Self-retrieval sanity check 只用于确认索引没有错位、metadata 没串、向量映射没损坏，不是正式 retrieval benchmark。
- Retrieval 尚未形成新主流程，尚未把 dense/sparse hybrid ranking、metadata filters、rerank 或 query rewrite 产品化接入。
- Agentic RAG 仍是 legacy 模块，尚未围绕新 chunks 重构。
- Evidence Audit 部分能力存在，但尚未和新 parser/chunk/index/retrieval 主链路完整产品化串联。
- Frontend/API 尚未完成产品级实现。

## 11. 下一步计划

当前 Embedding + Persistent Indexing v1 已完成并通过验证，下一步可以进入 retrieval 主流程接入；同时保留以下后续优化项：

1. 基于 `data/indexes/index_v1` 建立正式 retrieval v1，不进入 answer generation 或 audit 前先完成独立检索验收。
2. 明确 dense/sparse hybrid 策略、metadata filters、section filters、top-k 和 score normalization。
3. 在 retrieval/audit 中继续区分 `source_block_ids` 与 `overlap_source_block_ids`，避免复制上下文被误当作新的 source evidence。
4. 继续评估 caption/table/equation 的复杂视觉语义绑定，尤其是跨栏、跨页和图文环绕场景。
5. 复核剩余 `<150` chunk 和 paper_agent 的 heading_path 缺失场景，但不得为了检索指标擅自修改已冻结的 `chunk_eval_v1`。
6. 后续若 corpus 扩大，再评估是否需要引入 FAISS/Chroma 等专用向量后端；安装或升级依赖前必须先确认。

## 12. Development Documentation Rule

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
