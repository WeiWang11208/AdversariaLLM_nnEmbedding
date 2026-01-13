# PGD 嵌入空间攻击算法详解

这是一个针对大语言模型（LLM）的**连续优化攻击**，基于论文 *"Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space"* 实现。下面从算法层面进行详细介绍。

---

## 1. 问题形式化

### 1.1 攻击场景

给定一个安全对齐后的LLM $f_\theta$，攻击者希望找到一个**对抗性输入**，使模型生成预定的有害目标输出 $y^*$。

传统的离散 token 攻击（如 GCG）在离散空间搜索，优化困难。本方法的核心思想是：

> **绕过离散 token 空间，直接在连续的嵌入空间中优化攻击向量。**

### 1.2 优化目标

设输入序列的嵌入为 $E = [e_1, e_2, ..., e_n]$，其中部分位置 $\mathcal{A}$（attack mask）是可优化的"攻击嵌入"。

优化问题可以表述为：

$$
\min_{\delta} \mathcal{L}\big(f_\theta(E + \delta \cdot M_\mathcal{A}), y^*\big)
$$

其中：
- $\delta$ 是嵌入空间中的扰动
- $M_\mathcal{A}$ 是攻击掩码，限制扰动只作用于攻击位置
- $\mathcal{L}$ 是损失函数（通常是交叉熵）
- $y^*$ 是目标输出序列

---

## 2. 两种攻击空间

### 2.1 嵌入空间攻击 (Embedding Space)

**直接优化嵌入向量本身：**

$$
\tilde{E}_\mathcal{A} = E_\mathcal{A} + \delta, \quad \|\delta\|_p \leq \epsilon
$$

这是最强的攻击形式，因为嵌入空间是连续可微的，梯度信号清晰。

### 2.2 连续松弛 One-Hot 攻击

将离散的 token 选择松弛为连续分布：

$$
\tilde{e} = P \cdot W_{\text{embed}}
$$

其中：
- $P \in \mathbb{R}^{|V|}$ 是词表上的概率分布（软分配）
- $W_{\text{embed}}$ 是嵌入矩阵

约束 $P$ 位于单纯形（simplex）上：$P_i \geq 0, \sum_i P_i = 1$

这种攻击更"可实现"——理论上可以找到离散 token 近似，但攻击效果较弱。

---

## 3. 损失函数

### 3.1 交叉熵损失（主要）

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{|T|} \sum_{t \in T} \log p_\theta(y^*_t | \tilde{E}, y^*_{<t})
$$

其中 $T$ 是目标 token 的位置集合（由 target mask 指定）。

### 3.2 可选的正则化损失

**Logit Tying（逻辑层绑定）：**

$$
\mathcal{L}_{\text{tie-logits}} = D_{\text{KL}}\big(\text{softmax}(f_\theta(\tilde{E})) \| \text{softmax}(f_{\theta_0}(\tilde{E}))\big)
$$

其中 $f_{\theta_0}$ 是原始未对齐模型。这确保攻击后的行为不偏离原始模型太远。

**Feature Tying（特征层绑定）：**

$$
\mathcal{L}_{\text{tie-features}} = 1 - \frac{1}{L}\sum_{\ell=1}^L \cos(h_\ell, h^0_\ell)
$$

约束中间层隐状态与原始模型保持相似。

### 3.3 熵损失（探索性）

用于寻找更多样化的攻击点，而非特定目标：
- **Entropy First Token**：最大化第一个输出 token 的熵
- **Entropy Allowed**：在允许的 token 集上最大化均匀分布

---

## 4. 优化算法

### 4.1 梯度更新

使用 **Adam** 或 **FGSM** 进行梯度下降：

**Adam 更新：**

$$
\delta^{(t+1)} = \delta^{(t)} - \eta \cdot \text{Adam}(\nabla_\delta \mathcal{L})
$$

**FGSM 更新（更激进）：**

$$
\delta^{(t+1)} = \delta^{(t)} - \eta \cdot \text{sign}(\nabla_\delta \mathcal{L})
$$

### 4.2 梯度过滤

在计算梯度后进行过滤：
1. **位置过滤**：只保留攻击位置的梯度，其余置零
2. **Token 过滤**（one-hot 模式）：禁止某些 token（如非 ASCII 字符、特殊标记）

---

## 5. 投影操作（约束满足）

每步更新后，需要将扰动投影回可行域。

### 5.1 L2 球投影

$$
\text{Proj}_{L_2}(\delta) = \begin{cases}
\delta & \text{if } \|\delta\|_2 \leq \epsilon \\
\epsilon \cdot \frac{\delta}{\|\delta\|_2} & \text{otherwise}
\end{cases}
$$

### 5.2 L1 球投影

使用 Duchi et al. (2008) 的高效算法，将每个 token 的扰动投影到 L1 球内。

### 5.3 单纯形投影（one-hot 模式）

确保概率分布有效：

$$
\text{Proj}_{\Delta}(P) = \arg\min_{Q \in \Delta} \|Q - P\|_2^2
$$

其中 $\Delta = \{Q | Q_i \geq 0, \sum_i Q_i = 1\}$

---

## 6. 自适应学习率

学习率根据嵌入空间的几何特性自适应调整：

$$
\eta = \alpha \cdot \bar{\|e\|}
$$

其中：
- $\alpha$ 是基础学习率（配置参数）
- $\bar{\|e\|}$ 是嵌入向量的平均范数

这确保步长与嵌入空间的尺度匹配。

---

## 7. 随机重启

为避免陷入局部最优，可选择性地进行随机重启：

$$
\delta^{(t)} = \mathcal{N}(0, \epsilon_{\text{restart}}^2 I) \quad \text{if } t \mod K = 0
$$

---

## 8. 算法流程总结

```
输入: 模型 f_θ, 对话模板, 目标输出 y*, 攻击参数
输出: 优化后的对抗嵌入

1. 初始化攻击嵌入 δ = 0（或随机）
2. 计算嵌入尺度，设置自适应学习率
3. FOR step = 1 to num_steps:
   a. 前向传播: logits = f_θ(E + δ·M_A)
   b. 计算损失: L = L_CE + λ₁·L_tie-logits + λ₂·L_tie-features
   c. 反向传播: g = ∇_δ L
   d. 梯度过滤: g ← g ⊙ M_A
   e. 优化器更新: δ ← δ - η·Update(g)
   f. 投影: δ ← Proj(δ)
   g. (可选) 随机重启
4. 使用优化后的嵌入生成模型输出
```

---

## 9. 与离散攻击的对比

| 特性 | PGD 嵌入攻击 | GCG 离散攻击 |
|------|-------------|-------------|
| 优化空间 | 连续（嵌入） | 离散（token） |
| 梯度利用 | 直接梯度下降 | 梯度引导采样 |
| 攻击强度 | 更强 | 较弱 |
| 可实现性 | 需要嵌入访问 | 可用真实 token |
| 可转移性 | 较差 | 较好 |

---

## 10. 核心洞察

1. **连续放松**：将离散优化问题放松为连续优化，使梯度下降有效
2. **嵌入空间的脆弱性**：安全对齐在 token 空间有效，但在嵌入空间存在漏洞
3. **约束重要性**：投影操作确保扰动不会偏离太远，保持语义合理性
4. **多目标权衡**：通过正则化项平衡攻击效果与行为保持

---

## 参考文献

```bibtex
@article{schwinn2024soft,
  title={Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space},
  author={Schwinn, Leo and Dobre, David and Xhonneux, Sophie and Gidel, Gauthier and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:2402.09063},
  year={2024}
}
```
