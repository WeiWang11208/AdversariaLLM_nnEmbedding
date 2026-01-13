# GCG (Greedy Coordinate Gradient) 攻击算法详解

这是一个针对大语言模型（LLM）的**离散优化攻击**，基于论文 *"Universal and Transferable Adversarial Attacks on Aligned Language Models"* 实现。下面从算法层面进行详细介绍。

---

## 1. 问题形式化

### 1.1 攻击场景

给定一个安全对齐后的 LLM $f_\theta$ 和一个有害请求 $x$，攻击者希望找到一个**对抗性后缀** $s$，使得模型生成预定的有害目标输出 $y^*$。

### 1.2 优化目标

$$
\min_{s \in \mathcal{V}^L} \mathcal{L}\big(f_\theta(x \oplus s), y^*\big)
$$

其中：
- $s = [s_1, s_2, ..., s_L]$ 是长度为 $L$ 的对抗性后缀（离散 token 序列）
- $\mathcal{V}$ 是词表
- $\oplus$ 表示字符串拼接
- $\mathcal{L}$ 是损失函数（通常是交叉熵）

### 1.3 核心挑战

离散优化问题的搜索空间巨大：$|\mathcal{V}|^L$（例如 $32000^{20} \approx 10^{90}$）

GCG 的核心思想是：

> **利用梯度信息在离散空间中进行高效的贪心坐标搜索。**

---

## 2. 梯度引导的离散优化

### 2.1 可微松弛

虽然 token 是离散的，但我们可以通过 **one-hot 编码** 建立与嵌入空间的联系：

$$
e_i = \text{one-hot}(s_i) \cdot W_{\text{embed}}
$$

其中 $\text{one-hot}(s_i) \in \{0,1\}^{|\mathcal{V}|}$ 是 token $s_i$ 的 one-hot 编码。

### 2.2 梯度计算

对 one-hot 编码计算损失的梯度：

$$
G = \nabla_{\text{one-hot}(s)} \mathcal{L}(f_\theta(x \oplus s), y^*)
$$

其中 $G \in \mathbb{R}^{L \times |\mathcal{V}|}$，$G_{i,j}$ 表示将位置 $i$ 替换为 token $j$ 的梯度值。

**关键洞察**：梯度越负（损失下降越快），该替换越有希望。

---

## 3. 候选生成策略

### 3.1 默认策略：Top-K 采样

对于每个位置 $i$，选择梯度最负的 $K$ 个 token 作为候选：

$$
\mathcal{C}_i = \text{argtopk}_{j \in \mathcal{V}}(-G_{i,j}, K)
$$

然后随机采样生成 $B$ 个候选序列：
1. 随机选择 $n_{\text{replace}}$ 个位置
2. 从对应位置的 Top-K 候选中随机选择一个 token
3. 重复 $B$ 次得到候选集

### 3.2 随机策略

**Random Overall**：完全随机选择位置和 token，不使用梯度信息

**Random Per Position**：每个位置均匀采样，确保位置覆盖均衡

### 3.3 最小梯度策略

选择梯度幅度最小的替换（用于探索性搜索）

---

## 4. 损失函数

### 4.1 交叉熵损失（默认）

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{|T|} \sum_{t \in T} \log p_\theta(y^*_t | x \oplus s, y^*_{<t})
$$

### 4.2 Mellowmax 损失

交叉熵的平滑替代，对困难样本更鲁棒：

$$
\mathcal{L}_{\text{mellowmax}} = \frac{1}{\alpha} \log \frac{1}{|T|} \sum_{t \in T} \exp(-\alpha \cdot \log p_\theta(y^*_t))
$$

### 4.3 Carlini-Wagner (CW) 损失

$$
\mathcal{L}_{\text{CW}} = \frac{1}{|T|} \sum_{t \in T} \max(0, \max_{j \neq y^*_t} z_j - z_{y^*_t})
$$

其中 $z$ 是 logits。这个损失鼓励目标 token 的 logit 成为最大值。

### 4.4 熵相关损失（无目标攻击）

**Entropy**：最大化输出熵，使模型"困惑"

$$
\mathcal{L}_{\text{entropy}} = -H(p_\theta) = \sum_j p_j \log p_j
$$

**Entropy First Token**：只最大化第一个 token 的熵

**KL to Uniform**：使输出分布接近均匀分布

$$
\mathcal{L}_{\text{KL}} = D_{\text{KL}}(U_{\text{allowed}} \| p_\theta)
$$

其中 $U_{\text{allowed}}$ 是在允许 token 上的均匀分布。

---

## 5. 优化算法流程

### 5.1 贪心坐标下降

```
输入: 模型 f_θ, 对话模板, 目标输出 y*, 初始后缀 s⁰
输出: 优化后的对抗后缀 s*

1. 初始化后缀 s = s⁰
2. 初始化缓冲区 Buffer
3. FOR step = 1 to num_steps:
   a. 计算 one-hot 梯度: G = ∇ L(f_θ(x⊕s), y*)
   b. 对每个位置选择 Top-K 候选
   c. 随机采样 B 个候选序列 {s̃₁, ..., s̃_B}
   d. 过滤不合法候选（tokenization 一致性检查）
   e. 评估所有候选: loss_i = L(f_θ(x⊕s̃_i), y*)
   f. 选择最优: s = argmin loss_i
   g. 更新 Buffer
   h. (可选) 早停：如果找到完美匹配
4. 返回 Buffer 中的最佳后缀
```

### 5.2 Token 过滤

为确保攻击的可执行性，需要过滤掉以下 token：
- **非 ASCII 字符**：避免不可打印字符
- **特殊标记**：如 `<eos>`, `<pad>` 等
- **导致 tokenization 不一致的 token**：确保解码-重编码后序列不变

---

## 6. 优化技巧

### 6.1 KV Cache 缓存

对于不变的前缀部分（用户消息模板），预计算 Key-Value 缓存：

$$
\text{KVCache} = f_\theta^{\text{layers}}(x_{\text{prefix}})
$$

后续只需计算攻击后缀和目标的前向传播，显著减少计算量。

### 6.2 Attack Buffer

维护一个固定大小的缓冲区，保存历史最佳的 $k$ 个攻击后缀：

$$
\text{Buffer} = \{(s_1, l_1), ..., (s_k, l_k)\} \quad \text{sorted by } l_i
$$

好处：
- 避免丢失好的候选
- 提供多样性
- 支持集成多个攻击

### 6.3 梯度平滑

通过在 $N$ 个随机扰动邻域上平均梯度，减少噪声：

$$
\bar{G} = \frac{1}{N} \sum_{n=1}^N \nabla \mathcal{L}(f_\theta(x \oplus s^{(n)}), y^*)
$$

其中 $s^{(n)}$ 是 $s$ 的随机单 token 扰动。

### 6.4 梯度动量

引入 momentum 加速收敛：

$$
G^{(t)} = \beta \cdot G^{(t-1)} + (1-\beta) \cdot \nabla \mathcal{L}^{(t)}
$$

---

## 7. 算法复杂度分析

### 7.1 时间复杂度

每步的主要计算：
- **梯度计算**：$O(1)$ 次前向+反向传播
- **候选评估**：$O(B)$ 次前向传播（可批处理）

总复杂度：$O(T \cdot B \cdot C_{\text{forward}})$

其中 $T$ 是步数，$B$ 是 search width，$C_{\text{forward}}$ 是单次前向传播成本。

### 7.2 KV Cache 加速

使用前缀缓存后，每次前向传播只需处理后缀部分，加速比约为：

$$
\text{Speedup} \approx \frac{L_{\text{prefix}} + L_{\text{suffix}}}{L_{\text{suffix}}}
$$

---

## 8. 与连续攻击的对比

| 特性 | GCG 离散攻击 | PGD 嵌入攻击 |
|------|-------------|-------------|
| 优化空间 | 离散（token） | 连续（嵌入） |
| 梯度利用 | 梯度引导采样 | 直接梯度下降 |
| 攻击强度 | 较弱 | 更强 |
| 可实现性 | ✓ 可用真实 token | ✗ 需要嵌入访问 |
| 可转移性 | ✓ 较好 | ✗ 较差 |
| 通用性 | ✓ 可构造通用攻击 | ✗ 每次需重新优化 |

---

## 9. 关键超参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `num_steps` | 优化步数 | 250-500 |
| `search_width` | 每步候选数量 | 512 |
| `topk` | 每位置 Top-K 候选 | 256 |
| `n_replace` | 每步替换位置数 | 1 |
| `buffer_size` | 缓冲区大小 | 0-10 |
| `optim_str_init` | 初始后缀 | "x x x x..." |

---

## 10. 核心洞察

1. **梯度作为启发式**：虽然是离散优化，但梯度提供了有价值的搜索方向
2. **贪心 + 随机**：结合贪心选择和随机采样，平衡探索与利用
3. **可转移性**：发现的攻击后缀可以转移到其他模型
4. **通用攻击**：可以找到对多个 prompt 都有效的通用后缀
5. **Token 空间的脆弱性**：即使离散搜索，也能找到有效攻击点

---

## 参考文献

```bibtex
@article{zou2023universal,
  title={Universal and Transferable Adversarial Attacks on Aligned Language Models},
  author={Zou, Andy and Wang, Zifan and Carlini, Nicholas and Nasr, Milad and Kolter, J Zico and Fredrikson, Matt},
  journal={arXiv preprint arXiv:2307.15043},
  year={2023}
}
```
