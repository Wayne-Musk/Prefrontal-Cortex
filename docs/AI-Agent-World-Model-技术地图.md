# 从 Agent Skill 到 World Model：技术地图与可攻克问题

> 目标：把「能调用工具的 LLM」逐步升级为「会先预测后行动、能在不确定性下规划、并能从预测误差中收敛」的 Agent。本文只链接原始论文、规范、项目官方文档或源码库；“机会判断”是基于这些资料和当前仓库实现作出的推论。

## 1. 一张全景图

```mermaid
flowchart LR
  U[用户目标] --> A[Agent runtime\n会话、权限、审计]
  A --> S[Skill\n任务知识 + 工具使用说明]
  S --> P[Proposal\n生成候选动作]
  P --> R[RAG\n检索外部事实]
  R --> L[LLM backbone]
  L --> X[Tool / Environment\n真实动作与观测]
  X --> B[Belief\n状态与不确定性]
  B --> W[World Model\nP(s'|s,a), P(o|s')]
  W --> M[Planner\n反事实 rollout / 搜索]
  M --> P
  X --> E[Prediction error / surprise]
  E --> W
  T[Transformer 或 Mamba] --> L
  T --> W
```

关键分层：Skill 规定“可怎样做”；RAG 提供“已知资料”；LLM 负责“表达、归纳、候选生成”；World Model 则试图回答“做了以后世界会怎样”。后三者不能互相替代。

## 2. 各组件：定义、作用和边界

| 组件 | 最小定义 | 擅长 | 关键边界 |
|---|---|---|---|
| **Agent Skill** | 带元数据的 `SKILL.md` 指令包，描述触发条件、流程与工具用法。| 把稳定工作流、领域约束、工具协议复用给 Agent。| 它不是状态转移模型；不能仅靠 prompt 保证动作后果正确。Agent Skills 规范是跨工具格式，OpenClaw 明确按该规范加载。[[规范](https://agentskills.io/specification)] [[OpenClaw 文档](https://github.com/openclaw/openclaw/blob/main/docs/tools/skills.md)] |
| **OpenClaw** | 开源、本地优先的 Agent runtime：Gateway 管会话、工具、事件和频道连接；模型/agent harness 可替换。| 把 Skill、工具、消息渠道、凭证和运行环境接起来。| 应视作执行与治理层，不是 RAG、规划器或世界模型本身。[[官方仓库](https://github.com/openclaw/openclaw)] |
| **RAG** | 检索外部语料，再将检索结果与输入共同条件化生成。原始 RAG 将参数化记忆与非参数化检索记忆结合。| 降低知识过期、给答案提供可追溯证据、让 Agent 读私有文档。| 检索到文本不等于环境会如何变化：RAG 没有显式 `P(s'|s,a)`，也不天然处理多步副作用。[[Lewis et al., 2020](https://arxiv.org/abs/2005.11401)] |
| **Transformer** | 以注意力为核心的序列到序列架构。| 通用表示、语言建模、跨 token 关联、作为 Agent 的语言/多模态骨干。| 标准全注意力的序列计算/内存随长度呈二次增长；长时程轨迹会昂贵。[[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)] |
| **Mamba** | 选择性状态空间模型（selective SSM）：让 SSM 参数依赖输入，并使用硬件感知并行算法。| 长序列的线性扩展、流式状态维护；适合作为轨迹、日志、视频的时序骨干候选。| 不是“自动拥有世界模型”；仍须定义状态、训练数据、预测目标、校准和规划闭环。[[Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)] [[官方实现](https://github.com/state-spaces/mamba)] |
| **World Model** | 学得或写出的生成式动力学：给定状态和动作，预测后续状态/观测，并在模型内评估动作序列。| 在真实行动前反事实试演、主动诊断、风险敏感规划、用预测失败识别未知。| 误差会沿 rollout 放大；开放世界中“状态、动作和目标”都不封闭。早期 World Models 与 Dreamer 系列展示了在潜空间中训练/规划的路线。[[Ha & Schmidhuber, 2018](https://arxiv.org/abs/1803.10122)] [[DreamerV3, 2023](https://arxiv.org/abs/2301.04104)] |

### Transformer 与 Mamba：该怎样选？

不是二选一的“智能高低”。在本题中，一个务实组合是：Transformer/LLM 负责把自然语言、代码、文档压缩为候选假设与动作；Mamba/SSM 用于长时程观测流的紧凑 belief 更新；显式或潜变量 World Model 负责 `action → consequence`；搜索器负责选择。是否有效必须在同一任务、同一 token/延迟预算和同一安全约束下评测。

## 3. 从 Skill Agent 到 World-Model Agent 的数据流

```text
Skill + RAG + LLM
  用户意图 + 文档 ─→ 候选动作 a
                           │
                           ▼
World Model            预测 s'、观测 o'、不确定性、风险
                           │     （仅在内部 rollout）
                           ▼
Planner                比较多条 a1...aH 的回报/约束/信息增益
                           │
                           ▼
Runtime (OpenClaw)     权限检查 ─→ 真正调工具 ─→ 取得观测 o
                           │                         │
                           └──── 审计、回滚、批准 ────┘
                                                     │
                                  prediction error ◀─┘ → 更新模型可信度/请求澄清/停止
```

这里的关键设计原则是：**Skill 是 proposal，不是最终裁决器**；高影响动作必须先过模型预测、风险约束与 runtime 权限闸门。RAG 检索结果则应成为带出处和时效的观测/证据，而不是被悄悄写入“世界真相”。

## 4. World Model 仍未解决的痛点（及可做方向）

| 痛点 | 为什么难 | 可验证的解决切口 |
|---|---|---|
| **开放世界状态表示** | 论文中的控制环境状态/动作相对封闭；真实代码库、浏览器、组织流程会出现新实体和新因果边。当前仓库也明确采用可枚举离散先验。| 做“可增长的对象—关系—事件”状态图；新实体出现时生成 hypothesis，而非硬塞进旧标签。指标：新对象后的预测对数似然、任务成功率和拒绝/求证质量。|
| **长时程模型误差（compounding error）** | 单步预测正确不代表 H 步 rollout 可靠；Dreamer 也采用 imagined trajectories，说明这正是核心依赖。| 学习 rollout-horizon 校准曲线；以 ensemble/后验分歧限制搜索深度；不确定时主动探测或交给人。指标：按预测 horizon 的 Brier/NLL、真实回报 regret。|
| **因果而非相关** | 观测日志常让模型学到“同时发生”，不是“操作导致”；干预动作少且昂贵。| 记录 action provenance、前提条件、反事实/回滚结果；用安全沙箱做受控干预。指标：对干预的 effect 预测，而非仅 next-token/next-observation。|
| **部分可观测与观测噪声** | Agent 看见的是日志、网页和工具回包，不是完整系统状态；错误观测会让错误信念看似自洽。| 显式 belief 分布、观测源可靠度和信息价值（VOI）；为诊断动作单列预算。指标：校准误差、误修率、连带损伤。|
| **动作语义与 grounding** | API/schema 会变，语言中的“部署”“删除”与实际副作用不等价。| 将每个 Skill 编译为带 precondition/effect/成本/可逆性的 operator；持续用执行轨迹校验。指标：operator coverage、effect fidelity、未知动作拦截率。|
| **安全与目标错配** | “预计成功”不等于“允许执行”；模型也可能对罕见灾难过度自信。| 将硬约束置于规划回报之外，由 runtime 独立执行最小权限、审批、审计和回滚。指标：约束违例率、误阻率、人工升级率。OpenClaw 的 Gateway/工具与 Skill gating 可承接此层。[[官方工具文档](https://github.com/openclaw/openclaw/blob/main/docs/tools/index.md)] |
| **计算成本与实时性** | 搜索需要多分支、多步 rollout；高保真生成模型又贵。| 分层：廉价模型筛选 → 高保真复核 → 只对不可逆/高成本动作深搜；缓存和 temporal abstraction。指标：每次决策延迟、token/算力、相对安全收益。|
| **评测缺口** | 仅以最终成功率会掩盖“靠试错撞对”和灾难性副作用。| POMDP 基准中同时报成功、回报、误动作、损伤、预测校准、OOD、人工干预；固定训练/测试分布，禁用测试集调参。|

对通用交互世界模型，Google DeepMind 的 Genie 3 官方资料也明确列出长程环境一致性、有限 action space、多个独立 agent 的准确交互和真实地点保真度等限制；它们是软件 Agent 做“可验证任务世界模型”时应当提前量化的同构问题。[[Genie 3：长程一致性](https://deepmind.google/models/genie/#enabling-environmental-consistency-over-a-long-horizon)] [[Genie 3：限制](https://deepmind.google/models/genie/#limitations)]

上表关于“误差放大、潜空间 imagined rollout、部分可观测”的问题，是从 World Models 与 Dreamer 的方法前提推导出的工程风险，而非声称两篇论文逐项证明所有真实世界场景。最先要攻克的通常不是换 Transformer/Mamba，而是让**状态、动作效果、失败信号和评测**可被定义与观测。

## 5. 针对本仓库的最小路线图

现有 `worldmodel/` 已具备 belief、动力学、MCTS、信息增益、surprise monitor 和 Skill grounding 的骨架。README 也诚实指出：离散枚举先验、人工动力学、toy-only grounding、无真实 LLM、无时间抽象。

建议按风险最小的顺序推进：

1. **把实验变成证据闭环**：新增一个“schema/API 随时间变化”的 POMDP；分别报告单步与多步 NLL/校准、成功、误修和损伤。先证明 monitor 何时能预警，何时不能。
2. **Operator 轨迹学习**：统一日志为 `(belief/observation, action, outcome, cost, reversibility, source)`；从执行轨迹估计 precondition/effect 和置信区间，未知或低支持 operator 默认不可执行或只允许沙箱。
3. **开放状态的 hypothesis generation**：LLM/RAG 只提出新实体、关系、候选动作和证据；确定性解析器将其写为可审计假设，交由探测动作验证，不能直接成为事实。
4. **接入真实 runtime 的影子模式**：先只读取 OpenClaw/工具审计日志并做预测，不行动；通过离线 replay 和审批队列验证 calibration，再逐类开放低风险、可逆动作。
5. **长时程与效率实验**：将 Transformer 和 Mamba 作为同一 belief/dynamics 接口的可替换编码器，在相同数据、参数/延迟预算下对比；若 Mamba 只提高吞吐却损失不确定性校准，就不应替换安全关键路径。

## 6. 建议的首个“可解决”问题

不要从“通用世界模型”起步。一个更可检验的题目是：

> **面向运维/代码变更的、可校准的 action-effect world model：在 API 漂移与部分观测下，主动选择诊断，减少不可逆误操作。**

它与仓库当前的 ops POMDP 一致，能量化收益（损伤、误修、成功、延迟、校准），并可安全地从影子模式走到有限执行。只有当它在 OOD 和模型错配下仍优于 Skill-only/RAG-only 基线，才值得扩大到浏览器或通用办公 Agent。

## 原始资料索引

- [Attention Is All You Need — arXiv](https://arxiv.org/abs/1706.03762)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces — arXiv](https://arxiv.org/abs/2312.00752)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — arXiv](https://arxiv.org/abs/2005.11401)
- [World Models — arXiv](https://arxiv.org/abs/1803.10122)
- [Mastering Diverse Domains through World Models (DreamerV3) — arXiv](https://arxiv.org/abs/2301.04104)
- [Google DeepMind Genie 3 — 官方模型页及限制](https://deepmind.google/models/genie/)
- [Agent Skills specification](https://agentskills.io/specification)
- [OpenClaw official repository](https://github.com/openclaw/openclaw)；[Skill loading and policy](https://github.com/openclaw/openclaw/blob/main/docs/tools/skills.md)
