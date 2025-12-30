# Idea 3：Embedding 可实现性（Realizability）/ Off-manifold 检测：检测“是否来自合法 token 序列”

## 为什么这条线更“本质”
当攻击从 token-space 转为 embedding-space 时，很多文本层检测天然失明。你真正可以抓住的“硬差异”是：
- **正常输入**的 embedding 在一个非常窄的“可实现集合”里：它必须等于 `EmbeddingTable[token_id]`（再叠加位置、上下文处理），从宏观上看是一个强结构的离散/近离散流形。
- **连续攻击**为了强，往往要离开这个集合（off-manifold），否则等价于又回到离散 token 优化。

因此，与其做“表征分类器”，不如做更可解释、更接近接口安全的检测：**判断输入 embedding 是否“能被某个 token 序列解释”。**

---

## 你要实现/验证的核心假设
1. 攻击 embedding（PGD/软提示/向量注入）的分布，在“可实现性分数”上显著异常。
2. 攻击者如果强行把 embedding 投影回可实现集合，会显著损失攻击成功率（这可以自然连接到 Idea 4 的净化/投影防御）。
3. 该检测对“语义型攻击”（PAIR）不一定有效——这是好事：说明它专门针对 embedding threat model，边界清晰。

---

## 可实现性分数（建议从易到难做三档）
### 档 1：逐 token 最近邻重构误差（最简单基线）
对每个位置的输入 embedding \(e_t\)，在词表 embedding 矩阵 \(W\in\mathbb{R}^{V\times d}\) 中找最近邻：
- \(i^\*(t)=\arg\min_i \|e_t - W_i\|_2\)
- 误差：\(\delta_t=\|e_t-W_{i^\*(t)}\|_2\)
聚合：
- `mean(delta_t)` / `max(delta_t)` / 分位数（P90）

优点：实现超快，工程上最容易落地。  
缺点：攻击者可把每个点都贴近某个 token（“点对点伪装”），需要更强的序列级约束（见档 2/3）。

### 档 2：序列级全局解释（Viterbi/CRF 风格）
目标：不是每个位置独立选 token，而是找一条“整体最像自然 token 序列”的解释路径。

做法（一个可行的实现）：
1. 每个位置取 Top-K 近邻 token（K=10~50）
2. 定义路径代价：
   - 发射代价：\(\|e_t - W_{i_t}\|^2\)
   - 转移代价：使用一个轻量语言模型先验（例如目标模型自己的 token bigram 统计，或直接用目标模型在该位置对 token 的 logit 作为先验）  
3. 用 Viterbi 找总代价最小的 token 序列解释 \((i_1,\dots,i_T)\)
4. 以最小总代价作为 **sequence realizability score**

直觉：攻击者可以伪装单点，但很难让“整段 token 解释路径”同时合理。

### 档 3：可实现性 = “tokenizer+embedding 的可逆一致性”
如果你的 threat model 是从文本进入系统（例如 RAG 文本→编码→拼接），那正常路径满足：
`text -> tokenizer -> ids -> embedding -> (某种可逆性统计)`

你可以测两种一致性：
- **重构一致性**：把输入 embedding 先解码成 ids（用档 2 的全局解释），再用 tokenizer decode 成文本，再 tokenize 回 ids，检查是否稳定（避免 token merge 造成误判）。
- **模板一致性**：在 `prepare_conversation` 之后，对关键分隔 token（special tokens / role tokens）的 embedding 必须“极近似”，否则判为注入。

---

## 实验步骤（强烈建议按这个顺序做）
### Phase A：收集四类数据（只要能生成就行）
1. **Benign**：正常问答（可用你数据集里非敏感任务，或 `run_sampling.py`）。
2. **Refusal**：直接问敏感（模型拒绝），用于区分“拒绝态 ≠ 攻击态”。
3. **Token-space attacks**：GCG/BEAST/RandomSearch（文本层攻击）。
4. **Embedding-space attacks**：PGD（你仓库已有），并记录攻击时实际喂给模型的 `inputs_embeds`（至少记录被扰动位置）。

> 注意：为了对比公平，建议所有类别都用相同的 prompt 模板与长度分布；否则 realizability 会被长度/特殊 token 位置影响。

### Phase B：实现特征提取（无需训练就能先看分布）
对每条样本，提取：
- `mean_nn_l2` / `p90_nn_l2` / `max_nn_l2`（档 1）
- `seq_realizability_cost`（档 2）
- `special_token_consistency`（档 3，可选）

### Phase C：做最小可行检测器
你可以先不训练复杂模型，直接做阈值：
- 取 benign 的 `p90_nn_l2` 分布，设阈值为 P99（控制假阳性）
- 测 PGD 的检出率、token-space attacks 的误报率

如果要更严谨：
- 用 Logistic Regression/LightGBM 输入上述 5~10 个 scalar 特征，训练二分类（benign vs embedding-attack）。

### Phase D：关键消融（这才是论文味道）
1. **攻击强度 vs 可实现性**  
   - 横轴：PGD 的 \(\epsilon\) 或 loss  
   - 纵轴：realizability score  
   - 看是否存在明显 trade-off（越强越 off-manifold）

2. **对抗适应性测试（重要）**  
   - 修改 PGD 目标：在 loss 中加入 “贴近词表 embedding” 正则项（例如 \(\sum_t \min_i \|e_t-W_i\|^2\) 的近似）  
   - 观察：攻击成功率是否下降？realizability 是否上升？  
   - 这一步能回答“检测能否被轻易绕过”

3. **与 Idea 4 的联动**  
   - 把输入 embedding 做投影净化（见 Idea 4），再测 PGD 成功率  
   - 如果净化显著降低攻击且不伤 benign，你就有很强的闭环故事。

---

## 评测指标（建议你后续报告统一用这套）
- **ROC-AUC / PR-AUC**：检测 embedding 注入（benign vs embed-attack）
- **FPR@TPR=95%**：安全场景更关心低误报
- **Attack adaptivity gap**：自适应攻击后检出率下降多少
- **Utility drop**：对 benign 的生成质量/困惑度/任务成功率下降多少（与 Idea 4 一起评）

---

## 你最可能踩的坑
- **模型 embedding 有 `embed_scale`（如 gemma）**：距离计算要用同尺度 embedding，否则误判。
- **special tokens**：有些 tokenizer 会引入无 embedding 的 token（你 `gcg.py` 里已经处理过），realizability 也要过滤到真实 embedding 范围。
- **位置与 role 模板**：模板 token 的 embedding 很稳定，反而是检测的强信号；但要避免把“不同模板”当成攻击。

---

## 一句话贡献点（写论文用）
我们提出了针对 embedding 注入 threat model 的 **可实现性检测**：通过估计输入 embedding 是否能被合法 token 序列全局解释，从而在不依赖文本表面特征的情况下检测连续通道越狱，并系统评估了自适应攻击下的可绕过性与代价。

