# ResearchGuard-Agent Capability Audit

## Final Review 结论（2026-09-08）

ResearchGuard-Agent 当前可以准确定位为 **Evidence-Grounded Adaptive Research Agent**：它能把部分 QA / literature-review 开放目标拆成 2-4 个有独立 query、objective 和 success criterion 的 subgoal；每个 subgoal 独立检索 canonical evidence，并通过局部 Evidence Gate 判断目标是否真实完成。局部 partial/unsupported observation 会替换对应 subgoal 的检索 query；当全部局部目标 strong 但最终整题 Gate 仍发现缺口时，Replanner 可插入一个有界 runtime gap subgoal。所有回答仍必须通过最终 aggregate Gate、Guarded Answer 和 Citation Audit。

这比上一轮的 targeted retry 更接近 plan adaptation，但仍是**线性、有界、确定性约束内的 adaptation**，不是任意计划图、无限 ReAct 或 fully autonomous researcher。默认预算为 6 steps、14 Tool calls、2 revisions，Controller 在每次 Tool invoke 前执行硬检查。

### 本轮代码级发现与修复

- 旧状态把 retrieval success / evidence count 当作 subgoal satisfied，`success_criterion` 只存储、不参与决策。
- Aggregate Gate 能拒绝整题，但无法稳定指出哪个 subgoal 缺证，因而会触发 whole-query expanded retrieval。
- Hybrid Planner 对简单问题偶尔生成单 subgoal，Validator 正确拒绝，但造成不必要 fallback。
- Skill 内部连续 Tool 调用原来只在 Skill 开始前检查预算，理论上可能晚一个调用发现越界。

修复后，subgoal assessment bundle 使用原 canonical EvidenceRecord，并把 question、objective、success criterion 作为判断问题；GateDecision 与 assessment bundle ID 绑定。`partial` 不得被 aggregate answer 隐藏。Replanner 记录 `revise_query` 或 `add` subgoal change，Trace v4 则显式输出 completed/missing objectives、Gate status、observation、plan change 与最终 bundle/status。

### 候选方向取舍

- **已做：Subgoal-aware Evidence Sufficiency。** 这是 targeted replanning 的直接瓶颈，且可在 Agent 层复用现有 Evidence Tool，无需重写 Evidence Engine。
- **已做：Bounded Structural Replanning。** 支持局部 query replacement 和最多一个 aggregate-gap subgoal insertion；没有开放任意 remove/reorder/DAG。
- **已做：Planner A/B v2。** 5 条 live paired goal 均执行完整 Agent 链路；Hybrid fallback 0、decomposition quality 0.80 对 0.40，但 completion 都是 0.80，因此结论是 `HYBRID_STRUCTURAL_GAIN_ONLY`。
- **未做：On-demand Corpus Expansion。** 当前缺少开放许可校验、下载安全、临时 Parser/Chunk/Index 事务、session index isolation 与 rollback；metadata 继续保持 `evidence_eligible=false`。
- **未做：Long-running checkpoint/resume。** 当前最多 6 steps、2 revisions、120 秒控制目标，不值得引入第二套持久化执行系统。
- **未做：新 task family / Multi-Agent / LangGraph。** 真实瓶颈是 observation 语义与 Planner 稳定性，不是执行框架数量。

### 最终验证

- Full unittest：145/145 PASS。
- Pipeline：PASS；Evidence Gate bypass 0。
- Retrieval：PASS；Hybrid Recall@10 0.9444。
- Agent Benchmark v3：50/50 PASS；provenance 1.0；gate bypass 0；successful completion 0.46。
- Holdout v1：22/25 PASS；answer/refusal correctness 0.96；provenance 1.0；gate bypass 0。
- Capability v2：14/14 PASS；partial hidden、metadata leak、gate/audit bypass、budget nondeterminism hard checks 0。
- Planner A/B v2：5/5 Hybrid non-fallback，结构质量提升但无 completion lift；平均 Tool calls 4.0 → 5.4，观测 latency 12.05s → 23.53s。

因此建议将当前版本冻结为 bounded adaptive Agent baseline。进一步扩展应优先增加独立开放目标样本和无缓存成本评测；corpus expansion 必须作为单独安全阶段，不应混入本轮。

---

## Historical Audit Baseline（Goal Decomposition 实现前）

以下内容保留上一轮审计时的代码缺口，作为本次升级的前置基线；其中“尚不能分解 subgoal”等描述不代表当前状态。

## 已经做得很好的部分

- **执行链真实存在**：Planner 输出计划，Validator 在执行前验证，Controller 逐步调用 Skill，Skill 再通过 Tool Registry 调用 Evidence Engine；并非把整条 pipeline 包装成一次 Agent 调用。
- **安全边界清楚**：EvidenceBundle 是回答路径的 canonical evidence；Evidence Gate 在生成前执行；Citation Audit 在答案释放前执行；失败默认拒绝，不允许裸生成绕过门控。
- **执行有界**：`max_steps`、`max_tool_calls`、retry、plan revision 和 timeout 均有显式上限，Replanner 也有 revision budget。
- **可观测性完整**：State 保存 plan、skill/tool history、observations、revisions、evidence、gate、answer 和 audit；Memory/Trace 能记录一次运行的决策与证据链。
- **科研任务专用能力已经落地**：paper comparison 已有论文身份提取、按论文隔离检索与 provenance；claim audit 已有 atomic claim decomposition 和逐 claim verification。
- **可靠性基线有真实验证**：Agent Benchmark v3 为 50/50 PASS，gate bypass 为 0，provenance validity 为 1.0；独立 holdout 为 22/25 PASS，answer/refusal correctness 为 0.96，gate bypass 为 0。

## 当前真实 Agent 自主性边界

### Planner 更接近受限路由器，而不是通用任务分解器

`StructuredPlan` 只表达 `task_type + goal + flat steps + budget`。每个 step 只有 Skill、purpose、expected observation 和 retry，没有 subgoal、step-local research question、依赖关系或完成条件。Deterministic Planner 直接将七种 task type 映射到固定 Skill 模板。

Hybrid Planner 理论上可选择 Skill，但 Planner A/B 的 50 个 Hybrid case 全部走 deterministic fallback；报告结论为 `INCONCLUSIVE_FALLBACK_ONLY`，同时产生 50 次 API 调用、71,589 input tokens 和 13,380 output tokens。现有评测没有证明 LLM Planner 比固定模板提供了有效分解能力。

### Dynamic Skill Composition 是“动态选择已有 Skill 序列”

Skill Registry、strict schema 和运行时适配器是真实的动态执行基础，但 Validator 禁止任何重复 Skill。一个开放目标即便包含三个独立证据需求，也不能合法规划三次不同 query 的 `retrieve_evidence`。Controller 构造检索输入时默认始终使用原始 `state.query`，进一步限制了子问题级检索。

### Replanning 主要是 retry / fallback

当前 Replanner 会读取 evidence count、coverage、support level 和错误状态，并在检索为空或覆盖不足时加入 expanded retrieval；第二次失败后转为 scholarly metadata fallback 并安全拒绝。这是有用且真实的环境反馈，但它根据 task type 重建固定恢复序列，不表达“缺少哪篇论文的哪类实验信息”，也不能只修改对应子目标而保留其余计划。

### Scholarly discovery 只产生候选元数据

OpenAlex/arXiv 搜索结果是 discovery candidate，不是 canonical evidence。系统目前没有把外部论文执行合法获取、PDF 校验、Parser、Chunking、Indexing、事务提交和回滚的完整路径。这个边界是正确的：外部 title/abstract/metadata 不会进入 grounded answer。

### 未预定义复杂任务的实际行为

规则 Planner 会把无法匹配的请求归入默认 `qa`，通常执行 `retrieve_evidence -> generate_report`；命中特定表达时才进入 literature review、comparison 或 audit 模板。系统能安全完成或拒绝，但不会可靠地产生跨多个信息需求的研究计划，也不会保留每个子目标独立的证据完成状态。

## 真正存在的能力缺口

1. 缺少版本化的 Research Goal/Subgoal schema，以及每个 subgoal 的问题和成功条件。
2. Plan step 缺少绑定 subgoal 的 query/parameters，无法让同一 Skill 为不同信息需求独立执行。
3. Validator 把所有重复 Skill 当作 cycle，无法区分“有意义的多子目标检索”和“无意义重复”。
4. State 不能表达 subgoal 的 pending/satisfied/missing 状态，重复 Skill 输出还会按 Skill 名覆盖。
5. 多次检索得到的 canonical EvidenceBundle 没有子目标级累积与 provenance-preserving 聚合契约。
6. Replanner 只知道检索弱，不知道哪个 subgoal 缺失，因此无法进行局部计划修补。
7. Benchmark v3 主要验证固定路由、固定 Skill 序列与安全性；尚无开放目标、分解质量和 observation-driven plan patch 的专项评测。

## 看似缺失但实际上已有

- 不缺 Tool Calling、Skill Registry 或 Controller；重复造一套不会增加自主性。
- 不缺 observation 和 replanning 基础设施；需要扩展 observation 的语义粒度，而不是另建 ReAct loop。
- 不缺 Evidence Memory、Evidence Ledger 和 Trace；需要把 subgoal 状态接入现有结构，而不是增加第二套 Memory。
- 不缺论文发现能力；缺的是候选论文进入 canonical corpus 的受控 ingestion 事务。
- 不缺 Evidence Safety；它是升级时必须保持的系统边界。

## 值得实施的升级

### 1. Bounded Goal Decomposition

为 Structured Plan 增加最多 4 个 subgoal。每个 subgoal 包含稳定 ID、独立 research question、objective 和 success criterion；step 可以引用 subgoal，并携带经过白名单验证的参数。只在目标确有多个独立证据需求时分解，简单 QA 保持原路径。

### 2. Subgoal-aware Evidence Accumulation

每个子目标独立调用现有 Retrieval Tool，保存其 canonical EvidenceBundle；Controller 按 `chunk_id` 去重聚合，聚合 bundle 明确记录来源 bundle、subgoal 和原始 provenance。metadata candidate 仍不得进入该聚合。

### 3. Targeted Observation-driven Replanning

Observation 显式记录 subgoal、query、evidence count、coverage 和完成状态。某个子目标证据不足时，只对该子目标追加一次 expanded retrieval，并保留已经满足的子目标与未受影响的未来步骤；再次失败则走 metadata-only discovery 后拒绝，不重跑整个 workflow。

### 4. Open-goal Capability Evaluation

新增专门评测，验证多子目标分解、step-local query、无多余步骤、局部 replan、canonical provenance、metadata isolation、gate/audit、loop/budget，以及相同 observation 下的确定性。该评测不能替代 Benchmark v3、holdout 和 pipeline regression。

## 本轮不值得实施的复杂度

- **Multi-Agent、LangGraph、CrewAI、AutoGPT 或无限 ReAct**：当前瓶颈是计划契约与状态粒度，不是缺少调度框架。
- **增加更多 task label 或 Skill**：会扩大固定模板表面，不能解决开放目标分解。
- **重写 Retrieval、Evidence Gate、Citation Audit 或 Memory**：现有核心已经成熟，重写会扩大可靠性风险。
- **本轮实现 on-demand corpus ingestion**：合法来源判断、下载校验、Parser/Chunking/Indexing 的事务一致性、失败回滚与并发隔离尚无契约；仓促接入会破坏 canonical evidence 边界。本轮只保留 metadata-only discovery 和 fail-closed。
- **无边界长程自主循环**：当前项目优势是可解释、可评测、可停止；长程能力应由有界 subgoal 和局部 plan patch 逐步获得。

## 推荐升级边界与验收

升级应只扩展现有 Planner Schema、Validator、Controller、State、Replanner 和 Trace 的数据表达，不新增第二套执行引擎。Deterministic Planner 与所有旧计划必须向后兼容，简单任务的 Skill 序列不变。

验收必须同时满足：

- 开放目标可形成 2-4 个有意义且不重复的 subgoal；
- 不同 subgoal 使用不同、可追踪的检索 query；
- 一个 subgoal 的失败只触发该 subgoal 的恢复动作；
- 已满足 subgoal 不重复执行，未受影响的后续计划被保留；
- 聚合 evidence 全部来自 canonical EvidenceBundle，provenance 不丢失；
- scholarly metadata 不能成为回答 evidence；
- Evidence Gate、Citation Audit、step/tool/revision budget 与 fail-closed 全部保持；
- Agent Benchmark v3、holdout、pipeline regression 不退化。

完成这些条件后，可以准确宣称 ResearchGuard-Agent 从“固定任务模板下的受限编排”升级为“能够在安全边界内分解开放目标并针对缺失子目标调整计划的 bounded research agent”。仍不能宣称它具备自动扩展 canonical corpus 或无限长程自主研究能力。
