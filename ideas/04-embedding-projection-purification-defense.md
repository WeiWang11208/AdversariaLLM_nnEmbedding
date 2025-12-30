# Idea 4：Embedding 投影/净化（Projection & Purification）防御：把连续输入压回可控集合

## 目标（你要得到什么）
在允许 embedding 输入（或无法完全禁止 `inputs_embeds`）的前提下，设计一个推理时“净化层”，实现：
- **显著降低 embedding-space 越狱成功率**（对 PGD/软提示注入有效）
- **尽量不损伤 benign 输入的可用性**（正常回答质量尽量不变）
- 能在 **自适应攻击**（攻击者知道你的净化规则）下仍保持优势，或至少量化“绕过代价”

> 这条线的价值在于：不是“识别并报警”，而是直接改变输入可行域，很多检测器的阈值问题与绕过问题会弱化。

---

## 防御策略族谱（建议做 3 个强基线 + 1 个你自己的创新点）

### Baseline 0：接口完整性（最强、最工程，但要明确定义威胁模型）
- 规则：除可信模块外 **禁止 `inputs_embeds`**，只接受文本 → tokenizer → ids → embedding 的路径。
- 研究价值：把“embedding threat model 下文本检测失效”这个问题讲清楚。
- 局限：如果你的研究设定就是“必须允许 embedding 输入”，那它只是上界基线。

### Baseline 1：逐 token 最近邻投影（NN Projection）
对每个位置 embedding \(e_t\)，投影为最近词表向量 \(W_{i^\*}\)：
- \(e'_t = W_{\arg\min_i \|e_t-W_i\|}\)

优点：实现最简单，效果通常很猛。  
缺点：会把连续输入强行离散化，可能伤害一些 benign 场景（例如真·软提示、或上游模块输出不是词表 embedding）。

### Baseline 2：Top-K 加权投影（Soft Projection）
取 Top-K 近邻 token，按距离 softmax 得权重：
- \(e'_t=\sum_{i\in \text{TopK}} \alpha_i W_i\)
- \(\alpha_i \propto \exp(-\|e_t-W_i\|^2/\tau)\)

优点：更平滑、可调节（\(\tau\)、K）。  
缺点：对强攻击可能仍可绕过，需要自适应评测。

### Baseline 3：码本/子空间投影（Codebook / Subspace）
构建一个码本 \(C\)（例如对大量真实 token embedding 做 k-means），投影到码本中心或其局部子空间：
- \(e'_t = \Pi_{\text{span}(C_{nn})}(e_t)\) 或 \(e'_t=C_{nn}\)

优点：比词表投影更“宽”，对 benign 更友好；对 off-manifold 攻击仍有约束。  
缺点：需要额外训练/预处理码本。

### 你的创新点（建议你围绕 Idea 3 的 realizability 做“自适应强净化”）
**Realizability-aware Purification**：
- 对每个位置先计算 realizability 分数（Idea 3）
- 只对“可实现性异常”的位置执行强投影；其余位置保持原样或轻投影  
这能显著减少对 benign 的损伤，同时保留对 embedding 攻击的抑制。

---

## 实验步骤（端到端）
### Phase A：定义评测协议（一定要写清楚）
你至少需要三类输入：
1. **Benign**：正常任务
2. **Token attacks**：GCG/BEAST（应尽量不被你的净化“误杀”，否则说明你在防别的 threat model）
3. **Embedding attacks**：PGD（核心）

并定义两个输出指标：
- **安全指标**：越狱成功率（judge 评分）
- **效用指标**：benign 的任务质量（可用同一 judge 或单独的 helpfulness/accuracy；至少做“回答长度/是否拒绝”的 proxy）

### Phase B：实现净化层（建议以最小侵入方式插入）
最省事的插入点是所有攻击/采样调用最终都走的生成函数（你仓库里是 `src/lm_utils/generation.py` + `generate_ragged_batched`）。

实现策略（建议你后续做成可配置的 wrapper）：
- 新增一个可选参数 `inputs_embeds_preprocessor: Callable[[Tensor], Tensor] | None`
- 在送入 `model(inputs_embeds=...)` 前调用它

> 如果你暂时不想改公共生成函数，也可以先在 `pgd.py` 的 `model(inputs_embeds=...)` 前做净化，先证明对 embedding 攻击有效。

### Phase C：跑基线与消融

建议你至少跑下面这些组合（每一组都要包含 benign + token-attacks + embedding-attacks）：

1. **No defense**（原始）  
2. **NN Projection**（Baseline 1）  
3. **Soft Projection**（Baseline 2：K∈{8,32}，\(\tau\)∈{0.01,0.1,1.0}）  
4. **Codebook/Subspace**（Baseline 3：码本大小 M∈{256,1024}）  
5. **Realizability-aware**（你的方法：阈值用 benign P99 控制 FPR）

每组都记录：
- **Embedding-attack ASR**（越低越好）
- **Token-attack ASR**（不一定要降低；重点是别把 token threat model 评测混掉）
- **Benign utility**（拒绝率、回答长度、任务 judge 分数）

---

## 自适应攻击评测（这部分决定“论文硬度”）
你必须假设攻击者知道你的净化函数 \(P(\cdot)\)，于是攻击目标变为：
\[
\min_{\Delta} \text{Loss}(f(P(E+\Delta))) \quad s.t.\ \|\Delta\|\le \epsilon
\]

实现上你可以做两种近似：
1. **EOT（Expectation over Transformations）**  
   - 如果你的净化带随机性（比如 soft projection 采样、加噪声），PGD 需要对随机性取期望。
2. **STE（Straight-Through Estimator）**  
   - 对 NN Projection 这种不可导算子，用 STE 近似梯度，让对手“尽力而为”。

你报告里至少给两条曲线：
- 固定 \(\epsilon\)：no-defense vs defense vs adaptive-attack-under-defense 的 ASR
- 固定 ASR：达到同等成功率所需的 \(\epsilon\)（与 Idea 2 的半径指标打通）

---

## 实现细节建议（避免把工程坑变成研究结论）
### 1) Gemma/部分模型的 embedding scale
有些模型 embedding layer 有 `embed_scale`（你仓库里已经有判断）。  
净化距离与投影要在 **同一尺度** 做，否则阈值不可迁移。

### 2) 只净化“可疑位置”可以显著减少效用损失
如果你发现 NN Projection 把 benign 也伤了：
- 用 Idea 3 的 realizability score 做 gating：只对异常位置投影  
这常常能把 utility drop 降到很低，同时保留对 PGD 的强抑制。

### 3) 兼容文本输入与 embedding 输入
你最终最好把净化放在统一入口：
- 文本：`input_ids -> embedding -> (可选净化) -> model`
- embedding：`inputs_embeds -> (净化) -> model`

---

## 你应该如何写“贡献点”（避免被审稿人说只是工程）
1. 我们系统比较了 embedding threat model 下的多种投影/净化策略，并提出 realizability-aware 的自适应净化机制，在低误报（或低效用损失）约束下显著降低 embedding-space 越狱。
2. 我们在自适应攻击（EOT/STE）设置下评测防御，量化了绕过代价，并用安全边界距离（Idea 2）统一刻画防御提升的几何意义。

