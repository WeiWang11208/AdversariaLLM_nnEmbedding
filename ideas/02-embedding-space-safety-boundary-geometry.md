# Idea 2：Embedding Space 的“安全边界几何”度量（Safety Boundary Geometry）

## 目标（你要得到什么）
- **把“越狱强不强”的比较，从 ASR（attack success rate）升级为几何量**：对每条样本，估计“从安全/拒绝态到违规态的最小 embedding 扰动距离”。
- 输出可复现的指标：
  - **最小扰动半径分布**（per-sample 的 \(r^\*\)，再汇总均值/分位数）
  - **边界敏感性曲线**：扰动预算 \(\epsilon\) vs 违规概率/拒绝概率
  - **跨模型可比性**：同一数据集在不同开源模型上的边界距离对比

> 直觉：ASR 很依赖攻击实现与 tokenization；而 “到安全边界的距离”更像模型的几何性质，更稳定、更可解释。

---

## 威胁模型与定义（建议写进你后续论文/报告的核心）
### 1) 输入通道
我们假定攻击者可以通过 `inputs_embeds` 或等价路径注入连续 embedding（比如软提示、RAG 向量拼接、前缀缓存注入等）。

### 2) 安全属性（判定函数）
你需要一个二值判定函数 \(J(x)\in\{0,1\}\)：
- \(J(x)=1\)：模型输出“违规/完成目标 harmful 行为”
- \(J(x)=0\)：模型拒绝或安全回答

在本仓库里最自然的做法是复用 `run_judges.py` 的 classifier 流程，把 `run.json` 的 completions 打分后得到 \(J\)。

### 3) 安全边界距离（核心度量）
对每条样本 prompt 的 embedding 序列 \(E\)，定义最小扰动：
\[
r^\* = \min_{\Delta} \|\Delta\| \quad s.t.\quad J(E+\Delta)=1,\ \Delta \text{ 只作用于允许的位置}
\]
其中“允许的位置”你需要明确：只允许改 attack suffix 部分，或允许改整段 prefix（取决于你的 threat model）。

---

## 实验总流程（端到端）
你将跑 4 个阶段：数据→判定器→半径估计→分析。

### Phase A：准备数据集与“安全/目标任务”
1. 选一个数据集（例：你正在用的 `adv_behaviors`）。
2. 固定评测子集（很重要，保证可复现）：
   - 例：`datasets.adv_behaviors.idx="range(0,100)"`
3. 确定你是测“诱导输出目标字符串”（targeted）还是“任意违规输出”（untargeted）。
   - 若是 targeted，你可沿用现有数据集里的 `conversation[1]["content"]` 作为目标前缀。
   - 若是 untargeted，需要选择一个 judge（如“harmfulness / policy violation”）。

### Phase B：配置 judge（判定函数 \(J\)）
1. 先跑一小批生成（不攻击）作为 sanity check：
   - 用 `run_sampling.py`（若你已有 sampling 配置）或写一个最简单的 sampling run。
2. 跑 `run_judges.py`，确保 `run.json` 被回写 `scores[<classifier>]`。
3. 定义你的最终判定：
   - 二值化规则建议写死：例如 `score>=9` 视为 jailbroken；或针对二分类 judge 直接用 0/1。

### Phase C：半径估计（核心）
你有两条路线：**(C1) 基于 PGD 的最小 epsilon 搜索**（推荐最先做）与 **(C2) 直接求近似边界（线搜索/二分）**。

#### (C1) PGD + 二分 epsilon（最实用）
1. 固定 PGD 迭代步数 `num_steps`（例如 50 或 100），固定学习率策略。
2. 对每条样本，做 epsilon 的二分搜索：
   - 设定 `eps_low=0`，`eps_high=E_max`（比如 0.5、1、2，视 embedding norm 尺度）
   - 对每个 `eps_mid`，跑一次 PGD（只要攻击成功即可，不追求最优 loss）
   - 如果成功：`eps_high=eps_mid`；否则 `eps_low=eps_mid`
   - 迭代 10~15 次得到近似 \(r^\*\approx eps_high\)
3. 输出每条样本的 \(r^\*\)，并保存攻击成功时的 completion 与攻击 embedding（用于复核）。

> 注意：你现有的 `src/attacks/pgd.py` 当前会在每个 step 生成 completion，极易 OOM。做半径估计时建议你先把 PGD 改成“只在最终 step 生成/或者只生成少量 step”。如果你不想立刻改代码，可以先把 `num_steps` 降到 10~20 做 prototype，再逐步完善。

#### (C2) 线搜索（更像几何学，但实现更费）
1. 选择一个“方向”：
   - 可用 PGD 的最终梯度方向
   - 或用随机方向（做多次平均得到方向不确定性）
2. 沿方向做线搜索找到最小步长触发 \(J=1\)。
3. 这种方法更能分析“边界形状/曲率”，但需要更多 forward/judge 调用。

### Phase D：分析与可视化（你要拿到的结论）

你至少应该产出以下 6 组图/表（建议每组都做跨模型对比）：

1. **\(r^\*\) 的分布**  
   - 直方图 + 分位数表（P10/P50/P90）  
   - 按“任务类别/样本长度/目标类型”分组分析

2. **\(\epsilon\)–成功率曲线**  
   - 横轴：\(\epsilon\)（扰动预算）  
   - 纵轴：越狱成功率（judge 判定为 1 的比例）

3. **同一模型的“安全边界稳定性”**  
   - 同一 prompt 重复多次（固定 seed/不同 seed），看 \(r^\*\) 方差  
   - 这能回答“边界是尖锐的还是平滑的”

4. **跨模型对比**  
   - 同一数据集：模型 A/B/C 的 \(r^\*\) 分布对比  
   - 你会得到“哪个模型在 embedding threat model 下更脆”的量化结论

5. **扰动位置消融**（非常关键）  
   - 只允许改 suffix attack tokens 对应的 embedding  
   - 允许改整个 user prompt embedding  
   - 允许改 system+user（如果你的模板里有 system）  
   - 结论将直接对应不同工程攻击面（RAG prefix、软提示、输入中间件）

6. **扰动范数消融**  
   - \(L_2\) vs \(L_\infty\)（你 `pgd.py` 里已有 `projection=l2/l1`）  
   - 对比哪些约束更接近“现实可实施”的注入方式

---

## 建议的实现/跑法（尽量不改太多代码）
### 1) 先做一个“小规模可跑通”的版本
建议你先把规模压到可 debug：
- `datasets.adv_behaviors.idx="range(0,30)"`
- `attacks.pgd.num_steps=20`
- `generation_config.num_return_sequences=1`

然后只关注：能否顺利产出 `outputs/**/run.json` + `run_judges.py` 能否回写 `scores`。

### 2) 改造 PGD 以支持“半径估计而非全量 step 生成”（强烈建议）
你现在的 `src/attacks/pgd.py` 的主要 OOM 根源是：把 **每个 step** 的 embedding 都拿去 `generate_ragged_batched`。  
为了做 \(r^\*\) 的估计，你只需要：
- **每次 epsilon 试探只生成 1 次**（最终 step 或最优 step）
- 或者只生成极少 step（例如每 10 step 采样一次用于观测）

最低侵入的做法是新增配置项（建议你后续实现）：
- `generate_every: int = 0`（0 表示只最终 step 生成；>0 表示间隔生成）
- `keep_topk_steps: int = 1`（保留 loss 最低的 k 个 step 用于生成）

### 3) 二分搜索 epsilon 的落地脚本（建议新写一个小脚本）
建议你新增 `scripts/estimate_radius.py`（或 notebook），核心循环：
- for each prompt:
  - for t in range(T_bisect):
    - set cfg.attacks.pgd.epsilon = eps_mid
    - run attack (单样本/小批)
    - run judge (或直接在内存里打分)
    - update eps_low/eps_high
  - save r*

> 你可以先用“离线 judge”：先生成 outputs，再用 `run_judges.py` 批量回写；之后脚本解析 run.json 得到成功与否。这样实现最简单。

---

## 关键对照实验（决定你论文是否“硬”）
1. **token-space baseline vs embedding-space**  
   - baseline：`gcg` / `beast` / `random_search`（同样预算）  
   - 指标：ASR vs \(r^\*\) 分布（证明几何指标能更稳定地区分模型）

2. **自适应评测意识**  
   - 如果你后续引入防御（Idea 4），请务必让攻击知道你的防御，并尝试 EOT/STE（否则容易被审稿人认为是“安全幻觉”）

3. **判定器鲁棒性**  
   - 用两个不同 judge（或不同阈值）重复，验证结论不依赖某个 judge 的偶然性

---

## 常见坑与规避
- **生成随机性**：为了估计边界距离，建议 `temperature=0` 或固定 seed，先把随机性压到最低。
- **embedding 尺度**：不同模型 embedding norm 不同，建议把 \(\epsilon\) 归一化到 `embedding_scale`（你 `pgd.py` 已经在算）。
- **padding 木桶效应**：半径估计建议先单样本跑，避免 batch padding 把成本/显存搞爆。

---

## 你能写成论文贡献点的句子（给你一个“落笔方式”）
- 我们提出了在 embedding threat model 下衡量开源 LLM 对越狱的 **安全边界距离 \(r^\*\)**，并展示该指标比传统 ASR 更稳定、可比且更能解释模型差异。
- 我们发现不同模型的安全边界几何差异显著，且该差异在不同 judge/不同扰动约束下保持一致。

