# ResearchGuard-Agent

**Evidence-Grounded Adaptive Research Agent for Scientific Papers**

## Overview

ResearchGuard-Agent 是一个面向科研论文的 bounded、adaptive、
evidence-grounded research agent。它以已经验收的多文档 Evidence Engine
为底座，在其上增加受 schema 和预算约束的 Hybrid Planner、动态 Skill
Composition、单 Agent Controller、Research Memory、Evidence Ledger 与
完整 Trace。系统目标是完成可复核的科研检索、比较和 claim audit，而不是把
开放式生成包装成普通 PDF 问答。

当前系统不是 fully autonomous research agent，也不宣称 AGI。LLM Planner
只能提出有限的结构化计划；确定性 Validator、Controller policy、Evidence
Gate 和 Citation Audit 保留最终执行与发布控制。系统没有 Multi-Agent、
无限 ReAct、运行时代码生成或不受约束的 autonomous browsing。

ResearchGuard 是一个面向科研论文的多文档 RAG 工程。它不止返回与问题相似的段落，而是把版面解析、section-aware chunking、混合检索、Cross-Encoder 精排、查询改写、证据充分性判断、受证据约束的回答生成和逐 claim 引用核验串成一条可审计流程。

当前 Evidence Engine 主流程已经完成本地集成与验证；v2 在其上增加了 Tool Facade、受约束的单 Agent Controller、Scholarly Discovery、Research Workflow Skills、Research Memory 与 Evidence Ledger、Agent Trace、Agent Evaluation、Agent Intelligence Upgrade v1、Hybrid Planner，以及 **Dynamic Skill Composition Upgrade v1**。最新 Agent 路径使用单一 Evidence Artifact，允许 LLM 提议严格结构化的 skill plan，并由确定性 Policy Validator、Skill Executor 与执行预算共同控制；Planner 不可用或计划非法时自动回退到原确定性 Planner。历史 Agent 和早期 Audit 代码仍保留用于兼容；历史 `MemoryStore` 也继续服务迁移代码，但不承担当前 Controller 的科研过程记忆。

最终工程验证状态来自完整真实运行：Evidence Pipeline 回归、API smoke test、
编译和 `127/127` 单元测试通过；2026-07-27 的独立 holdout v1 与 Agent
Benchmark v3 均为 `PASS`；Planner A/B 仍为
`INCONCLUSIVE_FALLBACK_ONLY`。Benchmark v3 的优化前失败基线和 holdout
剩余失败案例继续保留，README 不以 synthetic PASS 替代真实 Agent 任务结果。

## Architecture

```mermaid
flowchart LR
    PDF[Scientific PDFs] --> Parser[Layout-aware Parser v5]
    Parser --> Chunks[Section-aware Chunks]
    Chunks --> Index[Dense / BM25 / Chroma Indexes]
    Query[User Query] --> Rewrite[Query Rewrite]
    Rewrite --> Retrieve[Hybrid Retrieval + RRF]
    Index --> Retrieve
    Retrieve --> Rerank[Cross-Encoder Reranker]
    Rerank --> Gate[Evidence Sufficiency Gate]
    Gate -->|Strong evidence| Answer[Grounded Answer]
    Gate -->|Partial / unsupported| Reject[Refusal]
    Answer --> Audit[Claim Verification / Citation Audit]
    Audit --> Report[Final Structured Report]
```

统一入口是 `researchguard.pipeline.ResearchGuardPipeline`。Pipeline 复用各阶段实现，不在 CLI 或 Demo 中复制检索、判断、生成或审计逻辑。

## Agent-ready Architecture

ResearchGuard v2 已完成 Evidence Contract、Tool Facade、Bounded Controller、Scholarly Discovery、Research Workflow Skills、Research Memory、Agent Evaluation、Agent Intelligence Upgrade v1，以及 **Dynamic Skill Composition Upgrade v1**。升级没有改变 Parser、Chunking、Indexing、Retrieval 算法或统一 Evidence Pipeline，而是在 Agent 层增加可验证、可执行、可追踪的 skill composition。

```mermaid
flowchart TD
    Query["User Query"] --> Planner["Hybrid Planner"]
    Planner --> Validator["Planner Policy Validator"]
    SkillRegistry["Skill Registry"] --> Planner
    Validator --> Controller["Single-Agent Controller"]
    Planner -->|failure or invalid plan| Fallback["Deterministic Planner"]
    Fallback --> Validator
    Policy["Step / Call / Retry / Timeout Policy"] --> Controller
    State["Serializable Agent State"] <--> Controller
    Controller --> Memory["Research Memory"]
    Memory --> Context["Bounded Advisory Context"]
    Context --> Planner
    Memory --> RunStore["Run Store"]
    Memory --> Ledger["Evidence Ledger"]
    Memory --> FailureStore["Failure Store"]
    Controller --> SkillExecutor["Skill Executor"]
    SkillRegistry --> SkillExecutor
    SkillExecutor --> Registry["Tool Registry"]
    WorkflowTemplates["Workflow Skill Templates"] --> Planner
    Registry --> RetrievalTool["retrieve_evidence"]
    RetrievalTool --> Bundle["EvidenceBundle"]
    Bundle --> EvidenceTool["assess_evidence"]
    EvidenceTool --> GateDecision["GateDecision"]
    Bundle --> AnswerTool["generate_grounded_answer"]
    GateDecision --> AnswerTool
    AnswerTool --> Artifact["AnswerArtifact"]
    Artifact --> AuditTool["audit_answer"]
    Bundle --> AuditTool
    Registry --> ScholarTool["search_scholarly_sources"]
    RetrievalTool --> Core["ResearchGuard Core"]
    EvidenceTool --> Core
    AnswerTool --> Generator["Existing AnswerGenerationPipeline"]
    AuditTool --> Core
    ScholarTool --> Candidates["Candidate Paper Metadata"]
    Generator --> Core
    SkillExecutor --> Trace["Agent Trace"]
    Controller --> Trace
    Trace --> Evaluation["Agent Evaluation"]
    Trace --> Demo["CLI / Streamlit Demo v3 / FastAPI"]
```

当前新增的统一契约位于 `researchguard/tools/contracts.py`：

- `ToolResult`：统一记录 `status`、`message/reason`、UTC timestamp、latency、tool name/version、trace ID、data 与结构化 error。
- `EvidenceRecord`：保留 canonical `chunk_id`、`doc_id`、`section`、`page`、content、source、score 与完整 provenance，不引入其他证据 ID 替代现有 chunk ID。
- `EvidenceBundle`：封装 query、不可重复的 EvidenceRecord、retrieval metadata、provenance 与基于规范内容计算的稳定 `bundle_id`。
- `GateDecision`：记录 `strong/partial/unsupported`、reason、supporting chunk IDs 和对应 `evidence_bundle_id`，防止 gate 与生成证据错配。
- `ToolError`：统一表达 invalid input、API failure、timeout、retrieval failure 与 execution failure。
- 所有契约带 schema version，并可直接序列化为 JSON。

Tool Facade 位于 `researchguard/tools/`：

| Tool | 包装的既有能力 | 安全边界 |
|---|---|---|
| `retrieve_evidence` | Query Rewrite、Hybrid Retrieval、Chroma、Reranker | 输出 canonical evidence 和稳定 EvidenceBundle，不复制召回或排序逻辑 |
| `assess_evidence` | Evidence Sufficiency | 只评估给定 EvidenceBundle，输出绑定同一 bundle ID 的 GateDecision |
| `generate_grounded_answer` | `AnswerGenerationPipeline` | 只接受 EvidenceBundle + strong GateDecision；不能调用 Retrieval、Evidence Judge 或统一 Pipeline |
| `audit_answer` | Citation Audit | 只审计 AnswerArtifact 与同源 EvidenceBundle；拒绝 bundle 外 citation 或裸答案字符串 |
| `search_scholarly_sources` | arXiv / OpenAlex metadata APIs | 只返回 `metadata_only=true` 的候选论文，不产生 EvidenceRecord |

`researchguard/tools/registry.py` 提供可枚举、可注册、可统一调用的 Tool Registry。默认 Registry 包含上述五个受控接口，依赖在首次调用时加载。Phase 2 Controller 只能通过该 Registry 调用能力，不直接 import Retrieval、Answer Generator 或 Citation Audit 的内部实现。

对应 synthetic tests 位于 `tests/tools/`，覆盖 schema/JSON 序列化、canonical provenance 往返、重复注册与未知工具错误、裸答案审计拒绝，以及 unsupported/partial evidence 无法触发 Answer Generator。

## Hybrid Planner Architecture

Phase 8 Round 1 将初始规划从单一规则分类升级为可回退的 Hybrid Planner，但执行权仍属于 Controller：

```text
Query
→ Hybrid LLM Planner（task understanding / plan proposal）
→ StructuredPlan strict schema
→ PlannerValidator（skill、tool、budget、evidence-first、audit）
→ Controller（execution / observation / bounded replanning）

任一 Planner API、JSON、schema 或 policy 校验失败
→ DeterministicPlanner
→ 同一 PlannerValidator
→ Controller
```

- `researchguard/agent/planner_schema.py` 定义版本化 `StructuredPlan`、`StructuredPlanStep` 与 `PlanBudget`。外部计划必须包含 `task_type`、`goal`、`skills`、`budget`、`reasoning_summary` 和 `planner_version`；每个 skill step 必须包含 `skill`、`purpose`、`expected_observation` 和 `max_retry`。未知字段、类型错误和未知 schema 均 fail closed。
- `researchguard/agent/hybrid_planner.py` 定义统一 `PlannerInterface.generate_plan(query)`、OpenAI strict JSON backend、`HybridPlanner` 与兼容原有规则的 `DeterministicPlanner`。配置位于 `configs/planner_v2.yaml`，默认 `temperature=0`、`max_steps=6`，并记录 model、prompt version、latency、token 与 API call count。
- `researchguard/agent/planner_validator.py` 独立验证 task type、skill/tool 是否存在、step/retry/tool-call/revision budget、Evidence Gate 顺序、受保护的 Answer Tool 和 Citation Audit。LLM 只提议计划，不能扩大 Controller policy。
- `researchguard/skills/` 提供动态 `SkillRegistry`、版本化 `SkillSpec/SkillResult` 和实际 `SkillExecutor`。默认注册 `search_papers`、兼容别名 `search_scholarly_sources`、`retrieve_evidence`、`assess_evidence`、`analyze_claims`、`compare_evidence`、`audit_claims` 与 `generate_report`。
- StructuredPlan、Planner metadata、skill history 与 skill outputs 同时进入 `ResearchAgentState` 和 `AgentTrace`。Evaluation 的 API call count 会分别统计 Planner 与底层 Tool 调用。

当前实现是 **bounded adaptive planner**，不是无限自治 Agent。`AgentPolicy` 仍限制 steps、Tool calls、retry、revision 和 timeout；`max_plan_revisions` 默认且最大为 `2`。LLM 不执行 Tool、不生成最终答案，也不能把 scholarly metadata 变成 canonical evidence。

## Dynamic Skill Composition Architecture

Phase 8 Round 2 将 Round 1 的“skill 目录”升级为真正可执行的组合层：

```text
StructuredPlan.skills
→ PlannerValidator
→ SkillRegistry
→ SkillExecutor
→ existing Tool Registry
→ ResearchGuard Evidence Engine
```

- `researchguard/skills/base.py` 定义 `SkillSpec`、`SkillResult`、`SkillContext` 与 `ResearchSkill`。`SkillSpec` 声明 name、description、input/output schema、required tools、risk level 和 version；`SkillResult` 统一返回 status、output、canonical evidence、trace ID、latency、reason 和 timestamp，支持稳定 JSON 序列化。
- `researchguard/skills/registry.py` 同时保存可供 Planner 枚举的 spec 与可供 Controller 执行的 runtime，不允许重复注册或执行未知 skill。
- `researchguard/skills/executor.py` 在运行前后验证 required input 与 output schema；异常、未知 skill 和 schema 错误均转换为结构化失败，不让一次 skill 错误击穿 Controller。
- `researchguard/skills/runtime.py` 只做输入适配、受控组合与结果整理。`retrieve_evidence`、`assess_evidence`、`search_papers` 等能力仍调用现有 Tool Facade，不重新实现 Retrieval、Reranker、Evidence Judge、Generation 或 Citation Audit。
- `generate_report` 是受保护的复合 skill：必须先取得或执行 Evidence Gate，仅在 strong evidence 下调用 grounded generation，并在返回前完成 Citation Audit；partial/unsupported 不能生成报告。`audit_claims` 同样要求 canonical evidence 和 gate，不能把无 provenance 文本当成可审计证据。
- `researchguard/skills/templates.py` 把 `literature_review`、`paper_comparison` 和 `claim_audit` 表达为有限 skill templates。原 `researchguard/workflows/` 实现仍保留用于 API 与迁移兼容，但默认 Planner/Controller 路径会把模板编译为动态 skill sequence。
- `BoundedReplanner` 可以针对 retrieval miss、retrieval failure 或 partial evidence 替换后续 skill sequence；修订次数仍严格限制为 `0..2`，重复 skill/cycle、越过 budget、gate bypass、audit 后再次生成等计划都会被 Validator 拒绝。
- 每次 skill 执行都记录 skill name、输入摘要、状态、latency 和 trace ID；skill 内部 Tool 调用继续保留原 Tool trace，因此可以同时看到“研究步骤”和“证据引擎调用”两层 provenance。

当前 composition 是线性、受模板与策略约束的单 Agent 执行，不支持任意 DAG、并行 skill、动态代码生成或开放式自治循环。

## Bounded Single-Agent Controller

Phase 2 位于 `researchguard/agent/`：

- `state.py`：定义带 schema version 的 `ResearchAgentState`，记录 query、task type、StructuredPlan、planner metadata、可执行 plan、plan revisions、workflow、EvidenceBundle、GateDecision、tool trace、observations、answer、audit result 和状态时间；支持 JSON 保存与恢复。
- `planner.py`：保留原确定性分类与计划生成，作为 Hybrid Planner 的 fallback 与兼容实现。
- `replanner.py`：根据 retrieval miss、partial evidence 和 retrieval failure 生成版本化 `PlanRevision`，记录 previous plan、observation、new plan 与 reason。
- `policy.py`：默认限制为 `max_steps=6`、`max_tool_calls=10`、`max_retry=2`、`max_plan_revisions=2`、`timeout=120s`。
- `controller.py`：只依赖 Planner Interface，执行 `validated skill plan → SkillExecutor → Tool Registry → observation → state update → bounded replan/stop decision`；所有修订仍受 step、call、revision 和 timeout budget 约束。旧的 tool plan 与 Workflow 执行入口继续保留用于兼容。

`qa` 的默认 skill plan 为：

```text
retrieve_evidence
→ generate_report
```

`generate_report` 内部强制执行或复用 `assess_evidence → generate_grounded_answer → audit_answer`，因此 Planner 无法通过省略 gate 或 audit 绕过安全边界。如果 gate 返回 unsupported，skill 立即拒绝且不会调用 Answer Tool；partial evidence 可触发扩大 `top_k/candidate_k` 的 retrieval，达到 revision 上限后仍非 strong 则拒绝。`audit` 任务要求调用方提供带 citation provenance 的 answer artifact；如果没有现成 evidence，可以先执行一次 retrieval。

每次 Skill 调用都会记录 `skill_name`、输入摘要、输出状态、latency 和 trace ID；每次底层 Tool 调用继续记录 `tool_name`、输入摘要、输出状态、latency、API call count、EvidenceBundle ID、timestamp 和 trace ID。Hybrid Planner 可调用 LLM 提议有限计划，但不进行开放式工具执行；项目也没有 Multi-Agent、LangGraph、autonomous browsing 或无限 ReAct loop。Research Memory 只向 Planner 提供受限的历史 run/paper/failure 摘要，不让历史答案改变当前答案。

## Adaptive Agent Loop

Agent Intelligence Upgrade v1 将普通 QA 从固定执行升级为有限状态的 adaptive loop：

```text
Observation
→ BoundedReplanner
→ PlanRevision(previous_plan, observation, new_plan, reason)
→ Next Tool Action
```

- Retrieval 返回 0 evidence、显式低 coverage 或 retrieval failure：第一次修订为 `rewrite/multi-query + expanded retrieval`。
- 扩展 retrieval 仍无 corpus evidence：第二次修订为 scholarly metadata discovery；metadata 不会转成证据，Agent 安全拒绝。
- Evidence Gate 返回 partial：增加 retrieval candidate 范围并重新判断；unsupported 不重规划，直接拒绝。
- Answer、Audit 或未知 Tool 输出失败：不允许绕过 gate 或 citation audit，执行 safe stop。

`PlanRevision` 会进入 `ResearchAgentState.plan_revisions` 和 `AgentTrace.plan_revisions`，Trace timeline 显示 `replan` 事件。该机制是受 `max_plan_revisions=2` 约束的规则式恢复，不是无限 ReAct，也不宣称开放式自主研究。

## Scholarly Discovery Tools

Phase 3 增加了独立的 Scholar Discovery Layer，用于发现可能值得纳入语料库的外部论文：

```text
External Search
→ Candidate Paper Metadata
→ optional user selection / ingestion
→ Parser
→ Index
→ ResearchGuard Retrieval
→ Grounded Answer
```

当前 Provider：

- **arXiv**：默认 Provider，调用公开 Atom API，无需 API key。
- **OpenAlex**：提供 works metadata、venue、citation count 与 open-access metadata；当前官方 API 要求 key，使用环境变量 `OPENALEX_API_KEY`，且只有显式选择该 Provider 时才会调用。

Provider abstraction 位于 `researchguard/tools/scholarly/`，统一实现 `search(query, limit)`。`search_scholarly_sources` 接受 query、可选 source preference 和 `1..50` 的 limit，按 DOI 或 title/year 去重，并返回统一 `ScholarPaperRecord`。每条记录包含 title、authors、year、venue、DOI、URL、abstract、source、paper ID、source type、provider metadata 和 retrieval timestamp。

所有 `ScholarPaperRecord` 强制带有：

```text
metadata_only = true
evidence_eligible = false
```

它们不是 `EvidenceRecord`，没有 canonical chunk ID、page、section 或 parser provenance，不能传给 Answer Tool。Planner 对 `literature_search` 只生成一个 `search_papers` skill；该 skill 调用兼容的 `search_scholarly_sources` Tool，Controller 将结果保存到 `candidate_papers` 后结束，不会继续生成回答。

搜索缓存位于 `data/cache/scholarly_search_v1/`，cache key 包含 query、provider、config version 和 limit；缓存只保存请求、metadata response 与 timestamp，并被 Git 忽略。

**External Discovery ≠ Evidence Verification。** 外部 metadata 必须经过用户选择、PDF ingestion、Parser、Index 和 ResearchGuard Retrieval 后，才可能成为可引用证据。

## Research Workflow Skills

Phase 4 在 `researchguard/workflows/` 增加科研任务编排层；Phase 8 Round 2 将这三个 Workflow 保留为兼容实现，并为每个 Workflow 增加 `skill_template`。默认 Planner 根据 task/workflow 选择模板，Controller 再逐项执行 skill；Tool 仍负责原子能力，Evidence Engine 继续负责检索、充分性判断、生成和引用核验。

统一抽象包括：

- `ResearchWorkflow`：声明 `workflow_name`、description、required tools、input/output schema、`skill_template` 和兼容的 `run(state)`。
- `WorkflowResult`：记录 version、status、message/reason、结构化 output、完整 trace、起止时间和 latency，可直接序列化为 JSON。
- `WorkflowLimits`：复用 Agent policy 的 step、tool call、retry 和 timeout 上限。
- `WorkflowRegistry`：保留 `literature_review`、`paper_comparison` 和 `claim_audit` 的旧调用入口；默认 Agent 路径使用对应的 skill template。

当前支持三个有限工作流：

| Workflow | 默认 Skill Template | 输出 |
|---|---|---|
| `literature_review` | `search_papers → retrieve_evidence → analyze_claims → generate_report` | candidate papers、canonical evidence、grounded summary、citations、audit、trace |
| `paper_comparison` | `search_papers → retrieve_evidence → compare_evidence → generate_report` | papers、dimensions、分离的 evidence table、summary、citations、audit、trace |
| `claim_audit` | `retrieve_evidence → analyze_claims → audit_claims` | claim、support level、canonical evidence、citations、audit、trace |

Literature Review 中的 scholarly records 始终保持 `metadata_only=true`，只作为候选列表；summary 只能引用当前本地 corpus 中带 canonical chunk/page/section provenance 的 `EvidenceRecord`。Paper Comparison 可由用户通过 workflow input 指定两篇论文的 name、`doc_id` 和比较维度；提供 `doc_id` 时两次 retrieval 分别使用 document filter，证据表不会混合。Claim Audit 不调用 Answer Generator，而是把用户 claim 与 Evidence Sufficiency 返回的 supporting chunk IDs 组成可审计 artifact，再交给现有 Citation Audit Tool。

`tests/workflows/` 使用 synthetic Tool Registry 验证 metadata/evidence 隔离、unsupported 提前拒绝、双论文 evidence 分离、citation provenance、JSON schema、Registry 选择和 Controller 集成，不修改原 benchmark。

## Research Memory and Evidence Ledger

Phase 5 在 `researchguard/memory/` 增加面向科研任务的持久化层。它记录一次 research run 的任务、workflow、候选论文、canonical evidence IDs、claims、失败原因和 Tool trace；它不是聊天历史、用户画像或偏好学习系统。

默认存储目录为：

```text
data/memory/research_runs/
  runs.jsonl
  evidence_ledger.jsonl
  failures.jsonl
```

该目录被 Git 忽略。第一版使用 append-only JSONL，不引入数据库或 embedding memory。运行开始时写入 `created` RunRecord 快照，结束时写入最终快照；查询时按 `run_id` 读取最新版本，因此保留运行生命周期且不需要覆盖历史文件。中断导致的损坏尾行会被读取层忽略。

核心 schema：

- `RunRecord`：保存 run ID、时间、query、workflow、status、plan、tool trace、papers、evidence IDs、claim IDs、answer summary、audit result、latency、reason 和 schema/version。
- `EvidenceRef`：保存 canonical `chunk_id`、`doc_id`、section、page、source 和 `sha256` content hash；Ledger 不会只保存脱离来源的文本。
- `LedgerRecord`：建立 claim 与一个或多个 EvidenceRef 的关系，并保存 verification status 与来源 workflow。
- `FailureRecord`：记录 no evidence、insufficient evidence、rejected answer、tool failure 或 execution failure。

Workflow 或普通 Agent 任务完成后，Controller 从 Citation Audit claims 及 citations 建立 Ledger；若 workflow 没有逐 claim audit 输出，则使用带 citations 的最终 claim/summary 作为受限 fallback。无法在当前 canonical evidence 中解析的 citation 不会被伪造成 provenance。Rejected/failed run 会额外写入 Failure Store。

Memory 持久化错误不会改变 Evidence Engine 或 Agent 的任务结论。Controller 在 `memory_status` 中记录 started、persisted、ledger count、failure status 和 errors；科研任务结果与存储健康状态保持分离。

Phase 6 增加 `ResearchMemory.search_context(query)`。它只通过 keyword overlap、可选 workflow、以前使用的论文 metadata 和失败记录查找最多 5 个相关 run，不使用 embedding。Controller 在新 run 落盘前读取该上下文，并把经过数量限制的摘要作为 Planner advisory；Planner 的 task classification、workflow selection 和 Tool 列表仍由当前 query 与显式参数确定，历史答案不会进入 Retrieval 或 Answer Generation。

现有 `researchguard.memory.memory_store.MemoryStore` 是早期 EvidenceClaw 单案例兼容层，仍供历史 audit 和迁移验证使用。Phase 5 没有删除或重写它。

## Agent Tracing and Evaluation

Phase 6 在 `researchguard/tracing/` 建立版本化 `AgentTrace`，统一记录 query、StructuredPlan、Planner mode/fallback/latency、可执行 plan、plan revisions、workflow、Tool calls、observations、canonical evidence、answer、citation audit、memory context、memory snapshot 和最终状态。Trace 可直接序列化为 JSON，也可由 formatter 转换为 CLI Markdown；它是只读观测层，不参与 Controller stop decision。

`researchguard/evaluation/` 提供两种评测方式：

- **Labeled evaluation**：`AgentEvaluationCase` 明确给出 expected task type、workflow、Tool sequence、forbidden tools、relevant evidence IDs 和 expected status，用于计算可判定的 planning/workflow accuracy。
- **Runtime evaluation**：只检查当前 run 的 Tool、provenance、效率与 memory 健康度；没有 ground truth 时，task classification 和 workflow accuracy 明确返回 `null`，不会把执行成功冒充准确率。

Agent Evaluation 现在覆盖六类指标：

| Category | Metrics |
|---|---|
| Planning | task classification accuracy、workflow selection accuracy、invalid Tool rate |
| Tool | Tool success rate、Tool call count、unnecessary Tool calls |
| Evidence | provenance validity、evidence coverage、unsupported claim rate |
| Efficiency | latency、step count、retry count、API call count |
| Memory | persistence success、ledger completeness、failure recording |
| Agent Intelligence | replanning rate、successful recovery rate、average plan revision、duplicate Tool call rate、evidence reuse rate |

`scripts/validate_agent_evaluation_v1.py` 保留三个旧 deterministic state-level cases，用于兼容 schema regression。正式 Agent 集成评测入口是 `scripts/validate_agent_benchmark_v2.py`：30 个 case 均从 query 开始执行 Planner、Controller、Skill/Workflow compatibility layer 和 Tool Registry，禁止直接构造最终 state。该 benchmark 仍使用 deterministic Tool doubles，因此验证的是 Agent 控制逻辑，不代表外部 LLM 或真实科研问题上的回答质量。

## Key Features

- **Layout-aware parsing**：使用 PyMuPDF 提取 span、字体与 bbox，恢复单双栏阅读顺序，识别 heading、paragraph、caption、table、equation 和 reference entry。
- **Section-aware chunking**：section 边界强制切分，heading 与正文绑定，保留特殊 block provenance，并限制最终 chunk 长度。
- **Hybrid retrieval**：Dense 与 BM25 独立召回后通过 RRF 融合，可切换 NumPy 或 Chroma dense backend。
- **Query rewrite**：保留实体的标准化改写、最多两条 expansion、多查询 RRF 融合、缓存和失败回退。
- **Evidence gate**：在生成前判定 `strong`、`partial` 或 `unsupported`；非 strong 证据不会进入回答生成。
- **Grounded generation**：生成器只能使用 gate 选定的 evidence，并输出可回溯 citation。
- **Citation audit**：将答案拆为 atomic claims，逐条核验支持程度和 citation provenance。
- **Bounded single agent**：Hybrid LLM plan proposal、确定性校验与 fallback、Registry-only tool calling、可恢复 state 与 step/call/retry/revision/timeout 约束。
- **Dynamic skill composition**：用版本化 SkillSpec/SkillResult 描述并执行八个注册项，通过 schema、预算、gate 与 audit 规则校验动态 skill sequence，再映射到现有受保护 Tool。
- **Scholarly discovery**：通过可扩展 Provider 发现 arXiv/OpenAlex 候选论文，同时隔离 metadata 与 answer evidence。
- **Research workflow skills**：提供 Literature Review、Paper Comparison 和 Claim Audit 三个可解释、可审计的有限科研工作流。
- **Research memory**：使用版本化 JSONL 保存 research run、证据 lineage 和失败记录，并支持关键词、workflow 与时间过滤。
- **Agent observability**：版本化 trace 串联 plan、workflow、Tool、evidence、answer、audit 与 memory，并提供带标注 benchmark 和无标注 runtime health metrics。
- **Streamlit Demo v2**：可切换 Evidence Pipeline 与 Research Workflow，展示 Agent Plan、Tool Trace、canonical evidence、Evidence Ledger、Memory Context、Evaluation Metrics 和最终结果。

## Project Structure

本阶段新增或升级的公开工程入口：

```text
api/                         FastAPI application; /health and /research
demo/                        Streamlit Demo v3 and display sanitization
researchguard/evaluation/    Agent Benchmark v3, Planner A/B, holdout metrics
data/eval/agent_benchmark_v3 50 labeled Agent tasks without reference answers
data/eval/holdout/           10-paper manifest and 25 blind questions
scripts/                     benchmark, holdout, API and regression CLIs
configs/                     portable development and isolated holdout configs
Makefile                     install, test, demo, api and evaluation commands
```

```text
researchguard/
  ingestion/       PDF layout、block、heading、section 与 chunking
  indexing/        embedding、BM25、NumPy/Chroma index 与 metadata
  retrieval/       retrieval、rewrite、rerank、evidence、answer 与 citation audit
  tools/           Evidence contracts、Tool Facade 与 Registry
    scholarly/     arXiv/OpenAlex Provider abstraction 与 metadata cache
  skills/          Skill contract、Registry、Executor、runtime 与 Workflow templates
  workflows/       兼容的 Workflow contract、Registry 与三个科研工作流
  memory/          Research Run Store、Evidence Ledger、Failure Store 与历史兼容 MemoryStore
  tracing/         版本化 Agent Trace、collector 与 CLI formatter
  evaluation/      Agent evaluation schema、adaptive metrics、evaluator 与 report renderer
  pipeline.py      v1 统一 Pipeline
  cli.py           命令行入口
  agent/           Hybrid/Deterministic Planner、Validator、Replanner、Policy、State 与 Controller
  audit/           历史规则式 audit，当前 claim audit 位于 retrieval/
configs/           各阶段 YAML 配置
  planner_v2.yaml  Hybrid Planner model、预算、prompt version 与 fallback 配置
scripts/           build、retrieve 与独立 validation 入口
demo/              Streamlit Demo v2
tests/agent/       Controller、Policy、State、bounded replanning 与 planner_v2 tests
tests/evaluation/  Agent benchmark 与 metric regression tests
tests/tracing/     完整 trace 和 JSON/Markdown serialization tests
data/eval/         纳入版本控制的阶段标注与 30-case Agent Benchmark v2
data/parsed/       本地解析结果，不纳入 Git
data/indexes/      本地索引，不纳入 Git
data/cache/        本地模型/API 缓存，不纳入 Git
data/memory/       本地 Research Memory JSONL，不纳入 Git
outputs/           本地验证报告，不纳入 Git
```

早期迁移留下的 `EvidenceClaw` 与 `rag_agent_harness` orphan Git links 已从公开仓库索引移除；本机目录通过 `.gitignore` 保留，不再导致 GitHub checkout 出现不可访问链接。历史分类见 [`PROJECT_CLEANUP_REPORT.md`](PROJECT_CLEANUP_REPORT.md)。

## Installation

Windows PowerShell：

```powershell
git clone https://github.com/Xzh1844963039/ResearchGuard-Agent.git
cd ResearchGuard-Agent
py -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
python -m pip install -e .
```

`pyproject.toml` 记录项目 metadata、Python 版本和直接运行依赖；`requirements.txt` 继续保留当前开发环境的完整 pinned snapshot。如需完全复现本机依赖，可继续执行 `python -m pip install -r requirements.txt`。

需要调用 OpenAI 的阶段从环境变量读取凭据：

```powershell
$env:OPENAI_API_KEY = "your-key"
```

PDF、解析结果、索引、缓存和验证输出不会随仓库发布。运行完整 Pipeline 前，需要准备本地 corpus，并按 `configs/` 中路径构建 parser/chunk/index/Chroma 资产。运行时配置使用项目相对路径，可从仓库根目录直接执行；Python 文件首行的绝对路径注释仅用于遵守本项目文档规则。

## Quick Start

检查 CLI：

```powershell
python -m researchguard.cli --help
python -m researchguard.cli status
```

运行统一 Pipeline：

```powershell
python -m researchguard.cli run `
  --query "How does CRAG reduce hallucination?" `
  --config configs/pipeline_v1.yaml
```

也可以把完整 JSON 写入本地输出目录：

```powershell
python -m researchguard.cli run `
  --query "What is the difference between RAG-Sequence and RAG-Token?" `
  --output outputs/pipeline_result.json
```

运行受约束的单 Agent Controller：

```powershell
python -m researchguard.cli agent-run `
  --query "How does CRAG reduce hallucination?" `
  --planner-config configs/planner_v2.yaml `
  --show-trace `
  --show-memory `
  --show-evaluation
```

发现外部候选论文：

```powershell
python -m researchguard.cli agent-run `
  --query "Find papers about corrective retrieval augmented generation"
```

运行 Research Workflow：

```powershell
python -m researchguard.cli agent-run `
  --query "Review RAG hallucination mitigation" `
  --task-type literature_review

python -m researchguard.cli agent-run `
  --query "Compare CRAG and Self-RAG" `
  --task-type paper_comparison

python -m researchguard.cli agent-run `
  --query "Verify this claim: CRAG uses a retrieval evaluator." `
  --task-type claim_audit
```

可使用 `--task-type qa|comparison|audit|literature_search|literature_review|paper_comparison|claim_audit` 显式指定任务类型，并用 `--max-steps`、`--max-tool-calls`、`--max-retry`、`--max-plan-revisions` 和 `--timeout` 收紧 policy。`--planner-config` 选择 Hybrid Planner 配置；缺少 API key、API 失败、JSON/schema 非法或 policy 校验失败时自动使用 deterministic fallback。`--workflow-input-json` 可以传入 paper name/`doc_id`、comparison dimensions、source preference 或 candidate limit。`audit` 任务必须通过 `--answer-json` 提供完整 answer artifact；可通过 `--evidence-json` 提供 canonical evidence。`--show-trace`、`--show-memory` 和 `--show-evaluation` 分别显示完整执行链、当前 run memory 与 runtime health metrics。`--output` 保存展示报告，`--state-output` 保存可恢复的完整 Agent state。

查看 Research Memory：

```powershell
python -m researchguard.cli memory-list

python -m researchguard.cli memory-show `
  --run-id "agent-..."

python -m researchguard.cli memory-search `
  --query "RAG hallucination" `
  --workflow literature_review
```

`memory-list` 支持 status/workflow/limit 过滤；`memory-search` 使用简单 keyword overlap，并支持 workflow、timestamp 和 limit。它不使用 embedding，也不会把历史答案自动注入新任务。

启动 Demo：

```powershell
python -m streamlit run demo/app.py
```

Demo v3 的 Research Agent 模式支持 Auto Plan、Literature Review、Paper Comparison 和 Claim Audit，并分别展示 Generated Plan、Skill Timeline、Tool Calls、Evidence Bundle、Grounded Answer、Citation Audit、Evidence Ledger、Memory Trace 与 Evaluation Metrics。Evidence Pipeline 模式继续展示 Retrieval evidence、Evidence Sufficiency、Grounded Answer、Citation Audit，以及各阶段 latency、model、config version 和 status。异常会转换为页面错误信息，API key、环境变量与内部路径不会展示。

运行 Agent Evaluation v1 synthetic benchmark：

```powershell
python scripts/validate_agent_evaluation_v1.py
```

报告写入 `outputs/agent_evaluation_v1/agent_evaluation_report.json` 和 `agent_evaluation_report.md`；`outputs/` 不纳入 Git。

运行 Agent Benchmark v2：

```powershell
python scripts/validate_agent_benchmark_v2.py
```

Benchmark v2 从 `data/eval/agent_benchmark_v2/cases.jsonl` 读取 30 条标注，真实执行 Planner、Controller、Workflow Registry 和 Tool Registry。它使用 deterministic Tool doubles，不调用外部 LLM，因此用于验证 routing、replanning、safety boundary 和数据流不变量，而不是衡量真实回答质量。报告写入 `outputs/agent_benchmark_v2/agent_benchmark_v2_report.json` 与 `agent_benchmark_v2_report.md`。

## API and Demo

Streamlit Demo v3 默认进入 Research Agent 模式，可选择 Auto Plan、
Literature Review、Paper Comparison 或 Claim Audit。页面分别显示 Generated
Plan、Skill Execution Timeline、Tool Calls、Evidence Bundle、Grounded Answer、
Citation Audit、Evidence Ledger、Memory Trace 与 runtime evaluation。Evidence
Pipeline 模式继续保留，便于单独查看六阶段 Evidence Engine。

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run demo/app.py
```

FastAPI 只包装 `BoundedResearchAgentController.run()`，不复制 Planner、
Skill Executor 或 Evidence Engine 逻辑：

```powershell
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

```http
GET /health

POST /research
Content-Type: application/json

{
  "query": "Compare RAPTOR and GraphRAG.",
  "workflow": "paper_comparison",
  "workflow_input": {}
}
```

`workflow` 可省略以使用 Hybrid Planner，也可显式选择
`literature_review`、`paper_comparison` 或 `claim_audit`。响应包含 plan、
planner metadata、skill timeline、Tool calls、EvidenceBundle、canonical
evidence、answer、audit、memory 和 plan revisions，不返回 API key 或环境变量。

项目根目录提供统一命令：

```powershell
make install
make test
make demo
make api
```

Windows 未安装 `make` 时，可直接执行对应的 `.venv` Python 命令。所有运行时
YAML 路径已改为项目相对路径；新机器 clone 后仍需自行准备 `.env`、PDF、解析
数据与 indexes，这些内容不会进入 Git。

## Data Preparation

### Independent Holdout Corpus

`data/eval/holdout/manifest.jsonl` 固定 10 篇与五篇开发语料完全分离的公开
arXiv 论文，`blind_questions.jsonl` 固定 25 条盲测问题。问题只标注
expected task type、skill pattern 和 document-level target，不包含答案、
relevant chunk ID 或人工生成结论。PDF、parser 输出、embedding、Chroma
collection 与评测报告均写入被 Git 忽略的目录。

```powershell
python scripts/prepare_holdout_v1.py --download --parse
python scripts/build_index_v1.py --config configs/holdout_indexing_v1.yaml
python scripts/build_chroma_v1.py --config configs/holdout_chroma_v1.yaml
python scripts/validate_holdout_v1.py
```

Holdout 使用独立的 `holdout_indexing_v1.yaml`、
`holdout_chroma_v1.yaml`、`holdout_retrieval_v1.yaml` 和
`holdout_pipeline_v1.yaml`，不会把开发集 index 或 retrieval 标注混入结果。

2026-07-27 完整 holdout 运行结果为 `PASS`：10 篇论文解析为 637 个
chunks，25 条 blind questions 全部执行完成。pass rate 从优化前
`0.4800` 提升到 `0.8800`，answer/refusal correctness 与 completion rate
从 `0.6800` 提升到 `0.9600`，skill sequence accuracy 从 `0.6800` 提升到
`0.9200`；provenance validity 为 `1.0000`，gate bypass 为 `0`。
document coverage@10 与 hit@10 为 `0.9200`，低于旧报告的 `1.0000`，原因是
验证入口不再把标注中的目标 `doc_id` 注入 comparison workflow，当前结果更接近
真实 blind execution。剩余 3 条 CHECK 包括两条 Planner 将
“distinguish/different abstraction levels” 误判为 paper comparison，以及
一条 Answer/Evidence backend `APIConnectionError`；它们没有被修改标注或
特殊分支掩盖。完整报告位于
`outputs/holdout_validation_v1/holdout_evaluation_report.md`，该目录不纳入
Git。

单篇 PDF parser 入口：

```powershell
python -m researchguard.ingestion.parse_pdf `
  --input data/raw_docs/parser_eval/paper.pdf `
  --out_dir data/parsed/parser_eval_v5/paper
```

索引与 Chroma 构建入口：

```powershell
python scripts/build_index_v1.py --config configs/indexing_v1.yaml
python scripts/build_chroma_v1.py --config configs/chroma_v1.yaml
```

独立 retrieval 调试：

```powershell
python scripts/retrieve_v1.py `
  --config configs/retrieval_v1.yaml `
  --query "How does CRAG reduce hallucination?" `
  --mode hybrid --rerank --multi-query --evidence-check `
  --generate-answer --citation-audit
```

## Evaluation

### Agent Benchmark v3 and Planner A/B

Agent Benchmark v3 位于 `data/eval/agent_benchmark_v3/cases.jsonl`，包含
50 条真实 Agent 任务：20 条 literature research、15 条 paper comparison
和 15 条 claim audit。每条任务从 query 开始真实运行
`Planner -> Controller -> Skill Executor -> Evidence Engine`，不直接构造最终
state。评测覆盖 task routing、skill sequence、plan validity、完成率、无效
skill 调用、canonical provenance、evidence coverage、unsupported claims、
latency、Tool/API call、token cost、gate bypass 与 refusal correctness。

```powershell
python scripts/validate_agent_benchmark_v3.py
python scripts/compare_planners_v1.py
```

Planner A/B 对同一组标注分别运行 Deterministic Planner 与 Hybrid LLM
Planner，输出 `outputs/planner_comparison_v1/planner_comparison_report.md`。
只有 Hybrid 非 fallback 运行产生可度量的 routing 或 skill-plan 改善时，报告
才会判定 `HYBRID_BETTER`；全部 fallback 时明确标记为
`INCONCLUSIVE_FALLBACK_ONLY`。

优化前的完整 Agent Benchmark v3 结论为 `FAIL`：50 条任务总通过率
`0.6000`，literature research `0.9500`、paper comparison `0.2667`、
claim audit `0.4667`，refusal correctness `0.6400`，并记录 6 次
gate-bypass hard check。根因包括 comparison 将目标 `doc_id` 作为运行输入但
仍混合 evidence、partial recovery 生成 7-step plan 超过 6-step policy、
dynamic claim analysis 错把 evidence 首句当作待核验 claim、LLM claim
extraction schema failure，以及审计失败后中间 answer 留在 state。

2026-07-27 的修复后完整复跑结论为 `PASS`：50/50 case 通过，三个类别通过率
均为 `1.0000`，task classification、skill selection、plan validity、
evidence coverage、provenance validity 与 refusal correctness 均为
`1.0000`；gate bypass、invalid plan、provenance failure、result schema
failure 和 execution exception 均为 `0`。successful completion 为
`0.4400`，因为安全拒绝也可满足 benchmark 的 `answer_or_refuse` /
`audit_or_refuse` 契约，不能把该值解释成总通过率。最终 cached rerun 的平均
latency 为 `3893.42 ms`、平均 Tool calls 为 `3.98`；首次生成新 explicit
claim cache 时延迟更高。状态分布为 literature `15 completed / 5
rejected`、paper comparison `1 completed / 14 rejected`、claim audit
`6 completed / 9 rejected`。因此 100% benchmark pass 主要证明 routing、
coverage、provenance 和 fail-closed behavior 满足当前契约，不证明 15 个
comparison 都成功生成了比较结论。

同一批 50 条任务的 Planner A/B 结论为
`INCONCLUSIVE_FALLBACK_ONLY`。Deterministic 与 Hybrid 的 routing、
skill sequence、plan completeness 和 plan validity 均为 `1.0000`，但
Hybrid 50/50 次全部 fallback，产生 50 次 API 调用、71,589 input tokens、
13,380 output tokens，平均规划 latency 为 `3890.8216 ms`，相比
Deterministic 的 `0.1781 ms` 没有可证明的质量提升。因此当前结果只能证明
fallback 路径稳定，不能证明 LLM Planner 优于确定性规划。

以下数字来自仓库开发阶段的固定 benchmark 与五篇论文语料，不是独立 blind hold-out，也不能外推为生产质量。

| Stage | Benchmark result | Interpretation |
|---|---:|---|
| Raw Hybrid Retrieval | Recall@10 `0.9444`, MRR@10 `0.6847`, nDCG@10 `0.6197` | Dense + BM25 RRF baseline |
| Hybrid + Reranker | Recall@10 `0.9722`, MRR@10 `0.7394`, nDCG@10 `0.6626` | Cross-Encoder 提升 Top-10 覆盖与前排排序 |
| Multi-query + Reranker | Recall@10 `0.9722`, MRR@10 `0.7394`, nDCG@10 `0.6468` | 保持 Recall/MRR，nDCG 比 raw rerank 略低 |
| Evidence Sufficiency | accuracy `0.8182`, precision `1.0000`, recall `0.7714` | 44 条评测中 0 false positive、8 false negative |
| No-answer gate | false positive rate `0.0000` | 仅指当前 4 条 unrelated no-answer；retrieval-only FPR 为 `1.0000` |
| Answer Generation | citation coverage `1.0000`, refusal accuracy `1.0000` | 22 条阶段评测；不等同于 citation entailment |
| Citation Audit | unsupported claim recall `1.0000`, citation precision/recall `0.9600/1.0000` | 12 个答案、26 个 atomic claims |

Parser v5 已通过 reading order、heading、block-level section 和 references 验收。Chunking v1 的长度、跨 section、heading-only、重复 block 与 special block coverage 硬性检查通过，并完成 overlap provenance 和 special block 边界修复。统一 Pipeline 的 strong 案例六阶段全部完成；unsupported 与 partial 案例在 Evidence Gate 后拒绝，Answer Generation 和 Citation Audit 跳过。Streamlit startup、strong render contract、unsupported render contract 与显示脱敏检查均通过。

Phase 4 workflow synthetic tests 覆盖三个 workflow 的成功与拒绝路径、metadata/evidence 边界、双论文 document filter、supporting chunk citation、Workflow Registry、Planner/Controller 委托和 Agent state JSON 往返。它们验证编排不变量，不替代真实 corpus 的质量评测。

Phase 5 memory tests 覆盖 RunRecord JSON roundtrip、跨实例持久化、canonical chunk/doc/section/page/hash provenance、failure memory、run isolation、keyword/workflow search、Controller 自动落盘，以及 memory failure 不影响 Agent 任务结论。

Phase 6 synthetic benchmark 的 3 个 case 全部通过：task classification accuracy `1.0000`、workflow selection accuracy `1.0000`、invalid Tool rate `0.0000`、Tool success rate `1.0000`、provenance validity `1.0000`、evidence coverage `1.0000`、unnecessary Tool calls `0`、unsupported claim rate `0.0000`。测试同时覆盖 completed run 的 Evidence Ledger 和 rejected run 的 Failure Store；这些数字只描述 deterministic synthetic cases。

Agent Benchmark v2 的 30 个 integration cases 全部通过：task/workflow accuracy `1.0000`、plan revision accuracy `1.0000`、invalid Tool rate `0.0000`、duplicate Tool call rate `0.0000`、provenance validity `1.0000`、evidence reuse rate `1.0000`。12/30 case 发生重规划，平均 `0.4667` 次 revision；需要恢复的 case 中 `successful_recovery_rate=0.6667`，其余恢复路径按设计安全拒绝。Benchmark 包含 10 个 task routing、10 个 replanning 和 10 个 safety boundary case。

Phase 8 Round 1 的 planner_v2 定向测试覆盖合法 strict JSON 与序列化、非法 JSON、Planner API 异常、未知 skill、超预算、Evidence Gate 绕过、审计后再次生成、Controller Planner Interface、deterministic fallback 与结构化 PlanRevision。Round 2 在 `tests/skills/` 增加 Registry、SkillResult JSON、真实 skill execution、未知 skill、budget/cycle 拒绝、gate bypass、动态 Controller 路径与 skill-level replanning 测试。Reliability 修复进一步覆盖 paper identity extraction、同文档双 identity、独立 filtered retrieval、comparison provenance、atomic claim decomposition、逐 claim verification、unsupported detection、matching GateDecision 和审计失败不释放中间 answer；当前全量 unittest 为 `127/127 PASS`。

最终回归还通过 `python -m compileall researchguard api scripts`、
`scripts/validate_pipeline_v1.py` 和 `scripts/validate_api_v1.py`。Pipeline
验证的 6 个 synthetic invariants 与 8 项 hard checks 全部通过，strong case
完成回答和 citation audit，partial/unsupported case 在 Evidence Gate 后
拒绝；API 的 health、Controller route 与 input validation hard checks 均为
0。单元测试与 smoke test 通过不改变上面的 Benchmark v3 `FAIL` 结论。

Round 2 回归同时通过 `validate_pipeline_v1.py`、`validate_retrieval_v1.py`、`validate_agent_evaluation_v1.py` 与 `validate_agent_benchmark_v2.py`。Agent Benchmark v2 仍为 30/30 PASS，task/workflow accuracy、plan revision accuracy、provenance validity 与 evidence reuse rate 均为 `1.0000`，duplicate Tool call rate 为 `0.0000`；这证明组合层没有破坏现有边界，不代表 LLM 对开放科研任务的规划质量。

`.github/workflows/test.yml` 在 push 和 pull request 上安装 `pyproject.toml` 依赖，并运行 `python -m compileall researchguard` 与 `python -m unittest discover -s tests`。

可重复运行的验证入口位于 `scripts/validate_*_v1.py` 和 `scripts/validate_parser_v5.py`。验证会读取本地 data/index/cache，并可能更新被 `.gitignore` 排除的 `outputs/`。

## Design Decisions

- **先恢复文档结构，再做检索**：section、heading 和 block provenance 是后续 evidence 展示与引用审计的基础。
- **召回与排序分离**：Hybrid/RRF 承担候选覆盖，Cross-Encoder 负责 query-chunk 相关性。
- **生成前 fail closed**：证据不是 strong 时拒绝回答；API、JSON、schema 或 provenance 异常不会绕过 gate。
- **生成后逐 claim 审计**：合法 chunk ID 不自动等于引用充分，claim verifier 单独判断支持关系。
- **论文身份先解析再比较**：Paper Comparison 从 query 或显式输入提取两个 identity，通过 corpus probe 解析 canonical `doc_id`，再为每个 identity 独立 filtered retrieval；不同论文不能折叠到同一文档，同一论文内的两个方法可以共享 `doc_id`，但 chunk 必须分组去重。
- **显式原子 claim 复用现有 verifier**：Dynamic Claim Audit 和 Agent report 对用户 statement/answer 做确定性、实体保留的 bounded decomposition，再让现有 verifier 逐 claim 核验；这避免二次 LLM extraction schema failure，但首次运行会增加逐 claim verification latency。
- **Skill 状态事务式提交**：skill 内部 ToolResult 只记录 trace，不提前写入全局 answer/audit state；只有 skill 成功才发布 answer，failed/rejected 路径清除中间 answer。
- **缓存键包含模型与配置版本**：rewrite、rerank、evidence、answer 和 audit cache 避免跨版本污染。
- **阶段输出保留 provenance**：chunk、document、section、page、query variant 和 stage metadata 可追踪。

## Limitations

- Agent Benchmark v3 与 holdout 提高了真实执行覆盖，但 document-level labels
  仍不能替代逐 claim 专家标注；报告会保留 miss、拒绝和 parser warning，不为
  追求 PASS 修改 benchmark。
- Agent Benchmark v3 当前为 `PASS` 且 hard checks 为 0，但 benchmark 只有
  50 条固定任务，而且 Paper Comparison 只有 1/15 completed；100% 不代表
  开放科研问题的生成质量，也不允许据此删除 fail-closed refusal。
- Holdout v1 当前 pass rate 为 `0.8800`，仍有两条 deterministic Planner
  routing mismatch 和一次外部 API connection failure。文档 hit/coverage 为
  `0.9200`，说明 query-level paper identity resolution 仍不能替代稳定的
  title/DOI-to-`doc_id` catalog。
- FastAPI v1 是单进程本地服务包装，没有认证、租户隔离、队列、rate limit、
  分布式 tracing 或生产部署保证；同步 Evidence Engine 调用由线程池承载，底层
  Tool timeout 仍是主要边界。

- Hybrid Planner 只负责一次受 schema 约束的 task understanding 与 plan proposal；当前没有多轮 LLM reflection。Planner A/B 的 50 次 Hybrid 运行全部回退到确定性规则，因此尚无有效的非 fallback 质量证据；任务路由和 skill sequence 标签也不等同于专家对 reasoning quality 的人工评审。
- Dynamic Skill Composition 当前只支持经过 Registry 和 Validator 认可的线性 skill sequence；不支持任意 DAG、并行执行、运行时定义新 skill 或动态代码生成。
- `analyze_claims` 当前是确定性的轻量 claim candidate 提取，用于提供可追踪中间产物，不等同于独立的语义 claim verification 模型。
- `generate_report` 会在一个受保护 skill 内顺序执行 gate、generation 与 audit；当前没有事务式回滚，底层同步 API 仍依赖各 Tool 自身 timeout。
- Replanner 根据 retrieval miss、partial evidence、retrieval failure 等已定义 observation 选择受控恢复路径，最多修订两次，不能为任意科研任务创造新策略。
- `generate_grounded_answer` 已消除内部重复 retrieval/judge/audit，但同步 Answer Generator 仍依赖外部 API timeout，Controller 无法抢占正在执行的同步调用。
- Literature Review 的外部候选 metadata 不会自动下载、解析或进入本地 corpus，因此只列出 discovery candidates；它不会扩大当前可回答证据范围。
- Paper Comparison 只有在 workflow input 提供 canonical `doc_id` 时才能严格隔离每篇论文的 retrieval filter；仅从自然语言识别 title 时，名称查询仍可能召回同一 corpus 中的其他文档。
- EvidenceBundle ID 保证 Agent 各阶段使用同一规范内容，但当前仅在进程内传递，没有签名、外部存储事务或跨服务防篡改机制。
- Workflow state 可以序列化，但当前不支持从 workflow 内部某一步断点续跑；重试恢复会从整个 workflow 起点重新执行，依赖各 Tool cache 控制外部调用成本。
- 独立 `audit_answer` 要求带 citation 与 generation evidence IDs 的完整 answer artifact，不能用来审计无 provenance 的任意文本。
- Controller 的 wall-clock timeout 会在同步 Tool 返回后立即生效，但不能抢占正在执行的同步 Tool；底层 API 调用仍依赖各 Tool 自身 timeout。
- Research Memory 只支持结构化 run、evidence 和 failure 查询；当前跨任务上下文是最多 5 个 keyword/workflow 匹配的受限 Planner advisory，没有聊天历史、用户画像、偏好学习或 embedding retrieval。
- JSONL storage 使用进程内写锁，适合本地单进程 Demo 和开发验证；尚未提供多进程并发事务、压缩、归档或 retention policy。
- Evidence Ledger 的 claim 粒度优先来自 Citation Audit；如果 audit 没有逐 claim 输出，只能把带 citations 的 workflow summary/claim 作为单条 fallback 记录。
- Memory 写入是 fail-open 的旁路记录：持久化失败会出现在 `memory_status.errors`，但不会把已经完成的科研任务改判失败。
- 当前没有跨任务 session、Multi-Agent、autonomous browsing、无限 reflection 或 autonomous workflow。
- Agent Benchmark v2 仍主要验证 deterministic Tool doubles；Benchmark v3 改为真实 Evidence Engine，holdout v1 改用独立 10 篇论文，但两者仍缺少逐 claim 专家参考答案与系统化 API failure injection。
- Scholarly Discovery 只返回第三方 metadata，未实现 PDF 自动下载、license 判断、用户选择工作流或自动 corpus ingestion。
- arXiv/OpenAlex 的 metadata 完整性、去重与来源类型取决于第三方记录；abstract 不能替代解析后 chunk evidence。
- OpenAlex 需要单独的 `OPENALEX_API_KEY`，其额度、价格与 API 行为由第三方服务控制。

- OCR fallback 尚未接入当前 Parser 主流程；扫描件质量依赖原始 PDF 文本层。
- 复杂跨栏表格和视觉结构仍有限；caption/table/equation 主要按同 section、同页、y 距离和文档顺序绑定，不是完整视觉语义理解。
- 主开发语料仍只有五篇论文；新增 10 篇独立 holdout 和 25 条 blind questions 用于隔离评测，但规模仍不足以外推到开放领域科研质量。
- Retrieval 会为 no-answer query 返回最近邻；Evidence Gate 只判断传入 Top-k，不能证明整个 corpus 中不存在答案。
- Evidence Judge 当前偏保守，44 条评测仍有 8 个 false negatives。
- LLM 阶段依赖外部 API、模型版本和网络；缓存结果不代表重新调用时的长期稳定性。
- 运行时 YAML 与脚本入口已迁移为项目相对路径；Python 首行保留的绝对路径仅是本项目开发文档规则，不参与路径解析。
- 历史 Agent、旧 `MemoryStore` 与旧 Audit 仍作为兼容代码保留；两个 orphan Git links 已从公开仓库索引移除，但对应本机迁移仓库仍保留且被忽略。
- 当前提供本地 Streamlit Demo v3 与 FastAPI v1，但没有认证、并发隔离、容器化、持久任务队列或线上部署方案。

## Future Work

1. 增加 OCR fallback、复杂表格结构恢复与视觉语义绑定。
2. 扩大独立 holdout corpus，并增加人工逐 claim answerability 与 citation entailment 标注。
3. 校准 Evidence Sufficiency，降低 false negatives，同时保持 fail-closed 行为。
4. 在 Linux、macOS 和全新 Windows clone 上验证安装、数据准备与 Makefile/直接命令的一致性。
5. 扩展 Hybrid Planner、真实 Retrieval、LLM Judge、Generation 与 Audit 的成本、失败注入和人工质量评测。
6. 为候选论文增加显式用户选择与受控 ingestion manifest，并建立 title/DOI 到 canonical `doc_id` 的映射，但仍禁止 metadata 直接进入回答生成。
7. 根据真实 hold-out 的失败分布校准 bounded replanning 触发条件，避免无收益的重复 retrieval，同时维持最大两次 revision。
8. 为 JSONL memory 增加归档、retention、完整性审计和多进程安全方案，再评估是否迁移 SQLite。
9. 评估是否需要比 keyword advisory 更严格的短期 session context；在有 benchmark 前不引入用户画像、Multi-Agent 或开放式自治循环。
10. 基于真实失败样本评估条件分支或有限并行是否有明确收益；在 provenance、预算和可重复性验证完成前，不把当前线性 Skill Executor 扩展为任意工作流引擎。
11. 为 FastAPI 增加认证、并发隔离、请求队列、服务级指标与受控部署方案。

## Development Documentation Rule

每次新增或修改功能、调整目录、增加脚本、改变运行命令、更新验证结果或推进项目阶段时，必须同步更新 `README.md`。更新至少覆盖当前状态、数据流、核心文件、方法说明、运行命令、输出目录、验证结果、已知限制和下一步计划。若任务明确禁止修改 README，应先说明 README 尚未同步。
