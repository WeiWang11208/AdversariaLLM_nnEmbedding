# Natural Suffix Embedding Attack（Idea 2.5）论文实验设计草案

目标：把 `ideas/01_natural_suffix_embedding_attack.md` 的想法扩展成可投稿/可复现的一套实验与叙事：**优势是什么、意义是什么、贡献是什么、怎么对比、怎么消融、怎么展示核心特点**。

---

## 1. 论文定位与核心主张（要能用实验支撑）

### 1.1 Threat Model（威胁模型）
- **白盒**：攻击者可访问目标模型（或其等价 surrogate）的 logits / gradients / embedding 层。
- **输入侧限制**：最终对用户可见文本必须是“自然语言后缀”（非乱码），并且**后缀的离散 tokens/文本保持不变**（解码一致性）。
- **目标**：在不改变可见后缀文本的前提下，提高越狱/有害响应生成概率（ASR↑），同时保持可检测性低（PPL≈自然语言、文本外观不变）。

### 1.2 核心贡献（建议写成 3–4 条）
1. **新攻击面**：提出“同一段自然语言后缀（同一 token 序列）在 embedding cell 内的微扰动可以显著改变模型行为”，揭示“语义-行为分离”现象。
2. **可部署隐蔽性**：最终输出仍是自然语言后缀（非乱码），可绕过基于 PPL/规则的异常后缀检测；同时比纯 prompt engineering 更强。
3. **工程可行实现**：两阶段（离散选后缀 + embedding 微调）+ 一致性约束（Logit Barrier / logits margin check），提供可复现实验代码与配置。
4. **系统性分析**：给出 stealth-attack tradeoff 曲线、消融、跨模型/跨数据分析，说明何时有效/何时失败。

### 1.3 需要被实验验证的“关键主张”
- **H1（有效性）**：在相同可见后缀文本下，embedding 微扰能提升 ASR（vs 纯自然后缀）。
- **H2（隐蔽性）**：后缀文本完全不变、PPL 与自然语言同量级（vs GCG/AutoDAN 的乱码后缀）。
- **H3（可行域/约束作用）**：Logit Barrier（margin）越强，解码一致性越好但 ASR 降低，存在可观的 Pareto 区间。
- **H4（机制展示）**：即便 token 不变，扰动会系统性影响早期 logits / attention / refusal direction，从而改变生成轨迹。

---

## 2. 实验设置（最少要写清楚，保证可复现）

### 2.1 数据集
建议至少包含：
- **HarmBench Behaviors**（项目已有：`dataset=adv_behaviors`）：覆盖多类 harmful intent。
- （可选增强）**AdvBench** / **JBB behaviors**：对齐不同基准，证明不是 dataset-specific。

输出建议：
- 每个类别分别报告 ASR；再报告总体宏平均/加权平均。

### 2.2 模型
建议分 2 档：
- **主结果**：1 个强指令模型（例如 Qwen3-8B / Llama3-8B / Mistral-7B）。
- **泛化**：再加 1–2 个不同家族/不同对齐策略的模型（证明方法普适）。

报告中明确：
- dtype（bf16/fp16）、温度、max_new_tokens、stop tokens、chat template。

### 2.3 成功判定（ASR 定义）
建议两条线并行：
- **Judge/Classifier**：项目已有 `strong_reject`（refusal classifier）。
- **规则化辅助**：检测是否出现典型 refusal 语句；或是否包含目标危害信息（简单关键词/模板仅做 sanity，不作为唯一指标）。

### 2.4 Stealth 指标（隐蔽性/可检测性）
最少 3 个：
- **文本不变性**：后缀字符串相同（exact match），或 token 序列一致（exact token ids）。
- **PPL/LM score**：后缀的困惑度接近自然语言（对比 GCG/AutoDAN）。
- **检测器通过率**：基于 perplexity 阈值、基于“非 ASCII/重复标点/长度异常”等规则的检测（可实现简单 detector）。

（可选更强）：
- 人类评审：后缀“自然程度”主观评分（小规模即可）。

### 2.5 计算与资源
必须报告：
- **VRAM 峰值**、每样本 wall-time、总步数。
- 提示：Phase2 是反传循环，复杂度与 `phase2_num_steps × seq_len × model_size` 成正比。

---

## 3. Baselines（对比实验）

至少要有这几类对比，才能体现“我们的独特优势”：

### 3.1 纯自然语言后缀（Prompt-only）
- 直接用 `s_nat`（Phase1 选出的最优后缀）但**不做 embedding 微扰**。
- 这是最关键 baseline：证明“同样的文本后缀，我们更强”。

### 3.2 离散乱码后缀攻击（有效但不隐蔽）
- **GCG / AutoDAN**（项目已有）。
- 对比点：ASR（可能更高） vs Stealth（PPL、可见异常）。

### 3.3 连续 embedding 攻击（强但不可部署）
- **PGD / RandomRestart**（项目已有）。
- 对比点：ASR vs **可解码性**（我们保持 token 不变，连续攻击通常无法稳定解码/输出乱码）。

### 3.4 其他 jailbreak 方法（可选）
- PAIR / Crescendo / Human jailbreaks（若项目已有且开销可控）。

---

## 3.5 公平性与“可行域”对齐（重要：应对审稿质疑）

在论文写作中，GCG/PGD/BEAST 与本方法的优化空间并不一致：
- GCG/BEAST：在 **离散 token 空间**直接优化，往往产生“乱码/高困惑度”后缀，属于 **隐蔽性约束放宽** 的可行域。
- PGD/RR：在 **连续 embedding 空间**优化，通常无法保持可解码性（或不保持 token 不变），属于 **不可部署/不可见一致** 的可行域。
- 本方法：在连续空间优化但引入 **“后缀文本（或 token）保持不变”** 的强约束，属于更严格的可行域。

因此建议将对比拆成两条主线，并配套补充实验以保证公平：

### 主线 A：上限对比（Unconstrained Upper Bound）
目的：说明“如果不考虑隐蔽性约束，传统白盒攻击能达到多高 ASR”，并作为上限参考。
- 对比：GCG / AutoDAN / BEAST / PGD / RR
- 注意写法：明确这些方法不一定满足自然语言/可部署约束，因此不是同一可行域下的直接竞争者。

### 主线 B：隐蔽性约束下的对比（Constraint-Matched）
目的：验证本文核心主张——**同样的自然语言后缀文本**下，embedding 微扰能显著提升 ASR。
- 最关键 baseline：Prompt-only natural suffix（同一 `s_nat` 文本，不做扰动）
- 额外建议 baseline（桥接）：在 GCG/PGD 上加入自然性/可解码约束，使其回到相近可行域（见下文）。

### 3.5.1 对齐“后缀 token 长度”的实验（Length-matched）
审稿人常见质疑：你是否通过更长的后缀获得优势？
- 固定后缀 token 数 `L ∈ {20, 40, 80}`（或按模型上下文预算选）
- 让 GCG/PGD/BEAST/我们都只作用于同样数量的 suffix tokens
- 报告：ASR vs L 曲线；同时报告 PPL/检测器得分随 L 的变化

### 3.5.2 对齐“计算预算”的实验（Compute-matched）
审稿人常见质疑：你是否用了更多反传/更多步数？
- 对齐指标之一（任选一个作为主标准，其他作为补充）：
  - 总优化步数（steps）
  - forward/backward 调用次数
  - 总 wall-time 或 GPU-hours
- 报告：ASR vs compute 曲线；并注明每方法的平均耗时/显存

### 3.5.3 对齐“起始点”的实验（Initialization/Seed-matched）
审稿人常见质疑：初始化差异会影响收敛与最优值。
- 对每种方法使用多个初始化/随机种子（例如 N=5）
- 同时报告：
  - Single-run（固定 seed）的 ASR（稳定性/可复现）
  - Best-of-N（oracle over seeds/steps）的 ASR（存在性/上限）

### 3.5.4 桥接 baseline：把 GCG/PGD 拉回“自然/可部署”可行域
目的是消除“你在更严格约束下却和无约束攻击直接比”的争议，让结论更有说服力。

- **Constrained-GCG（自然词表/ASCII/高频词）**：
  - 将 GCG 候选 token 限制为可打印 ASCII / 高频词 / curated phrase bank
  - 预期：ASR 会下降，但可用于展示“自然性约束下我们的优势”

- **PGD-Realizable / PGD-Projected（可解码约束）**：
  - 为 PGD 增加“投影到可解码 manifold / nearest-token / logits margin”约束
  - 预期：也会下降，但可作为“连续攻击在可部署约束下的基线”

### 3.5.5 推荐的论文表述模板
- “We compare against GCG/PGD/BEAST as strong white-box baselines. These methods optimize in a different feasibility region (unconstrained discrete tokens or unconstrained continuous embeddings) and therefore provide an upper bound when stealth constraints are relaxed.”
- “For fair comparison under our deployment constraint, we report a prompt-only natural suffix baseline that uses exactly the same visible suffix text, and we additionally evaluate length-/compute-matched baselines.”

---

## 4. 主要结果（主表 + 主图）

### 4.1 主表（建议结构）
| Model | Dataset | Method | ASR↑ | Refusal↓ | PPL(suffix)↓ | Suffix unchanged↑ | Time/sample↓ |
|---|---|---:|---:|---:|---:|---:|---:|

必须包含：
- Prompt-only natural suffix（我们的 Phase1 结果）
- NaturalSuffixEmbedding（我们）
- GCG/AutoDAN
- RandomRestart/PGD

### 4.2 主图：Stealth–Effectiveness Pareto
横轴：PPL 或 detector score（越小越隐蔽）  
纵轴：ASR（越高越强）
- 我们的方法应落在“自然语言区域”但 ASR 明显优于 prompt-only。
- GCG 在 ASR 高但 PPL 高（更可疑）。
- PGD/RR 在 ASR 高但“suffix 不可部署”（可用不同 marker 表示不可解码）。

---

## 5. 消融实验（Ablations）

消融的目标：证明每个设计点都有价值，并解释 tradeoff。

### 5.1 Phase1 的作用
对比：
- (A) 固定一个常见后缀（手工）
- (B) Phase1 在候选池中选
- (C) Phase1+Phase2（完整）
观察：Phase1 是否降低 Phase2 需要的步数/提高成功率/稳定性。

### 5.2 一致性约束强度（核心）
扫参数：
- `logit_margin ∈ {0, 1, 3, 5, 8}`
- `lambda_consist ∈ {0.1, 1, 3}`
报告：
- token 不变率（Decode pass rate）
- ASR
- 平均 margin（安全余量）
预期：margin 越大 → decode 更稳 → ASR 下降，但存在可用区间。

### 5.3 扰动预算与步数
扫参数：
- `epsilon`（L2 per token）与 `phase2_num_steps`
产物：
- ASR vs epsilon
- ASR vs steps（收敛曲线）
- decode pass rate vs epsilon

### 5.4 “哪里加扰动”更有效
对比（同预算）：
- 只扰动 suffix tokens
- 扰动 prefix+suffix（若代码支持）
结论：最小可见改动下，哪部分贡献最大。

### 5.5 一致性检查方式（工程消融）
对比：
- logits_margin check（快）
- nearest-cosine check（更严格但慢）
报告：峰值显存、速度、decode pass rate、ASR 差异。

---

## 6. 展示“核心特点”的分析实验（让读者信服）

### 6.1 “同一句话，不同行为”的定性案例
挑 5–10 个例子，每个例子展示：
- 相同的后缀文本（完全一致）
- Prompt-only 生成：拒绝/安全回答
- 我们方法生成：更容易越狱（或更接近目标有害内容）

### 6.2 轨迹/机制：早期 logits 或 refusal score 的变化
建议展示之一：
- 画出生成前若干步（例如前 20 token）的
  - `P(refusal)` 或 refusal classifier score
  - “危险 token / 目标 token”概率
对比：prompt-only vs 我们（同 token 文本）。

### 6.3 几何视角：margin 分布 / cell 内移动
展示：
- 攻击 token 上的 `correct_logit - second_best` 分布（优化前/后）。
- 扰动范数分布（每 token δ 的 L2）。

### 6.4 检测对抗（Stealth）
实现一个简单 detector（论文里写清楚即可）：
- 规则：非 ASCII 比例、重复标点比例、平均 token 罕见度、PPL 阈值
展示：
- GCG/AutoDAN 高检出率
- 我们与 prompt-only 检出率低且接近

---

## 7. 失败模式与局限（也需要实验或至少统计）

建议至少记录并报告：
- 解码一致性无法维持（margin 太高 / epsilon 太小）
- 某些模型（强对齐或特定 template）对该攻击更鲁棒
- 长输出（max_new_tokens 大）时越狱成功可能不稳定（前期绕过但后期自我修正）

---

## 8. 最小可复现实验清单（建议论文/附录里给）

### 8.1 主结果（每模型×每数据集）
- 固定 seed
- 固定生成参数（temperature=0, top_p=1）
- `phase1_num_steps`（候选数）、`phase2_num_steps`、`epsilon`、`logit_margin`

### 8.2 必要的日志项（便于复现与分析）
- token 不变率（逐 step）
- margin 统计（逐 step 或最终）
- 运行时间/显存（峰值）
- 生成文本（用于 human/judge）

---

## 9. 建议的论文叙事结构（写作提纲）

1. 背景：乱码后缀易检测、纯 prompt 工程弱、纯 embedding 攻击不可部署  
2. 方法：自然后缀 + embedding 微调 + 解码一致性约束（Logit Barrier）  
3. 结果：在自然语言隐蔽区间显著提升 ASR（主表+主图）  
4. 机制分析：同 token 不同行为、logits/margin 轨迹、检测实验  
5. 消融：phase1、margin/epsilon/steps、检查方式、扰动位置  
6. 局限与伦理：白盒假设、潜在防御方向（margin 强化/鲁棒训练）  
