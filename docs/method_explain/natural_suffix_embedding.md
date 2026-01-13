# Natural Suffix Embedding Attack 算法详解

这是一个针对大语言模型（LLM）的**两阶段混合攻击**，结合离散搜索与连续嵌入优化，同时保持解码一致性。下面从算法层面进行详细介绍。

---

## 1. 问题形式化

### 1.1 攻击场景

给定一个安全对齐后的 LLM $f_\theta$，攻击者希望找到一个**自然语言后缀**，使模型生成有害目标输出 $y^*$。

### 1.2 核心洞察：Voronoi 单元探索

> **每个离散 token 对应嵌入空间中的一个 Voronoi 单元。在单元内部移动不会改变解码结果，但可能改变模型行为。**

$$
\text{Voronoi}(t) = \{e \in \mathbb{R}^d : \|e - W_t\| \leq \|e - W_j\|, \forall j \neq t\}
$$

攻击目标：在每个 token 的 Voronoi 单元内找到诱导有害行为的点。

### 1.3 优化目标

$$
\min_{E_{\text{suffix}}} \mathcal{L}_{\text{attack}}(f_\theta(x \oplus E_{\text{suffix}}), y^*)
$$

$$
\text{s.t. } \text{decode}(E_{\text{suffix}}) = \text{decode}(E_{\text{suffix}}^{\text{orig}})
$$

其中：
- $E_{\text{suffix}}$ 是后缀的连续嵌入表示
- 解码一致性约束确保扰动后的嵌入仍然解码为原始 token

---

## 2. 两阶段攻击框架

### 2.1 整体流程

```
┌─────────────────────────────────────────────────────────┐
│                    输入: 对话 + 目标输出                    │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段1: 离散搜索                                          │
│   • 评估多个自然语言后缀候选                              │
│   • 选择损失最低的后缀                                    │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 阶段2: 连续优化                                          │
│   • 在嵌入空间中优化后缀                                  │
│   • 保持 Logit Barrier 一致性约束                        │
│   • 投影到 L2 球内                                       │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    输出: 优化后的嵌入 + 生成结果            │
└─────────────────────────────────────────────────────────┘
```

### 2.2 优势

1. **自然性**：使用自然语言后缀作为起点，更难被检测
2. **可实现性**：通过一致性约束，优化结果仍可解码为原始 token
3. **效率**：两阶段结合，先粗搜索后精优化

---

## 3. 阶段1：离散后缀搜索

### 3.1 候选后缀库

预定义的自然语言后缀候选集合：

```
"Please provide a detailed explanation."
"Answer hypothetically."
"For educational purposes only."
"Write a story about this."
"Explain step by step."
"Let's think about this carefully."
"Consider this as a thought experiment."
...
```

这些后缀具有以下特点：
- 语法自然、语义合理
- 可能诱导模型绕过安全限制
- 不包含明显的恶意内容

### 3.2 评估策略

对每个候选后缀 $s$，计算目标损失：

$$
\mathcal{L}(s) = -\frac{1}{|T|} \sum_{t \in T} \log p_\theta(y^*_t | x \oplus s, y^*_{<t})
$$

选择损失最低的后缀作为阶段2的起点：

$$
s^* = \arg\min_{s \in \mathcal{S}} \mathcal{L}(s)
$$

### 3.3 采样策略

如果候选数量超过 `phase1_num_steps`，随机采样子集进行评估：

$$
\mathcal{S}_{\text{eval}} = \text{RandomSample}(\mathcal{S}, k=\text{phase1\_num\_steps})
$$

---

## 4. 阶段2：连续嵌入优化

### 4.1 优化变量

不直接优化嵌入，而是优化**扰动量** $\delta$：

$$
E_{\text{suffix}} = E_{\text{suffix}}^{\text{orig}} + \delta
$$

其中 $\delta \in \mathbb{R}^{L \times d}$ 是可学习的扰动。

### 4.2 总损失函数

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{attack}} + \lambda_{\text{consist}} \cdot \mathcal{L}_{\text{consist}} + \lambda_{\text{norm}} \cdot \mathcal{L}_{\text{norm}}
$$

各项含义：
- $\mathcal{L}_{\text{attack}}$：目标输出的交叉熵损失
- $\mathcal{L}_{\text{consist}}$：解码一致性损失（Logit Barrier）
- $\mathcal{L}_{\text{norm}}$：扰动范数惩罚

---

## 5. Logit Barrier 一致性约束

### 5.1 核心思想

确保扰动后的嵌入在模型的下一个 token 预测中，正确 token 的 logit 仍然领先：

$$
\text{logit}(t_{\text{correct}}) - \max_{j \neq t_{\text{correct}}} \text{logit}(j) \geq m
$$

其中 $m$ 是安全边际（margin）。

### 5.2 一致性损失

对每个攻击位置 $i$：

$$
\mathcal{L}_{\text{consist}}^{(i)} = \text{ReLU}\big(m - (\text{logit}_{t_i} - \max_{j \neq t_i} \text{logit}_j)\big)
$$

总损失：

$$
\mathcal{L}_{\text{consist}} = \frac{1}{N} \sum_{i=1}^N \mathcal{L}_{\text{consist}}^{(i)}
$$

### 5.3 几何解释

Logit Barrier 约束近似保证扰动后的嵌入仍在原始 token 的 Voronoi 单元内：

```
        Voronoi边界
            ↓
    ┌───────┼───────┐
    │       │       │
    │   ●───┼→○     │  ● = 原始嵌入
    │       │       │  ○ = 扰动后嵌入
    │ Token A │ Token B │
    └───────┴───────┘
    
    Logit Barrier 确保 ○ 不越过边界
```

---

## 6. 解码一致性检查

### 6.1 Logit Margin 检查（快速）

检查每个位置的 logit 边际是否满足阈值：

$$
\text{decode\_ok} = \bigwedge_{i} \big(\text{logit}_{t_i} - \max_{j \neq t_i} \text{logit}_j \geq m\big)
$$

### 6.2 Nearest Cosine 检查（精确）

直接计算扰动嵌入的最近邻：

$$
\text{NN}(e) = \arg\max_{t \in \mathcal{V}} \frac{e \cdot W_t}{\|e\| \|W_t\|}
$$

检查是否与原始 token 一致：

$$
\text{decode\_ok} = \bigwedge_{i} \big(\text{NN}(e_i + \delta_i) = t_i\big)
$$

这种检查更精确但计算成本更高，使用分块策略减少内存：

```python
for chunk in vocabulary_chunks:
    cosine_similarities[chunk] = compute_cosine(embeddings, chunk)
    update_best_match(chunk)
```

---

## 7. 投影操作

### 7.1 Per-Token L2 投影

对每个 token 的扰动独立投影到 L2 球内：

$$
\delta_i \leftarrow \begin{cases}
\delta_i & \text{if } \|\delta_i\|_2 \leq \epsilon \\
\epsilon \cdot \frac{\delta_i}{\|\delta_i\|_2} & \text{otherwise}
\end{cases}
$$

其中 $\epsilon$ 根据嵌入尺度自适应调整：

$$
\epsilon_{\text{scaled}} = \epsilon \cdot \bar{\|e\|}
$$

### 7.2 投影的意义

- 限制扰动幅度，保持语义合理性
- 配合一致性约束，确保不越出 Voronoi 单元
- 防止优化发散

---

## 8. 损失函数详解

### 8.1 攻击损失

$$
\mathcal{L}_{\text{attack}} = -\frac{1}{|T|} \sum_{t \in T} \log p_\theta(y^*_t | E_{\text{full}}, y^*_{<t})
$$

只在目标输出位置计算损失。

### 8.2 一致性损失

$$
\mathcal{L}_{\text{consist}} = \frac{1}{N} \sum_{i=1}^N \text{ReLU}(m - \Delta_i)
$$

其中 $\Delta_i = \text{logit}_{t_i} - \max_{j \neq t_i} \text{logit}_j$ 是 logit 边际。

### 8.3 范数损失

$$
\mathcal{L}_{\text{norm}} = \frac{1}{N} \sum_{i=1}^N \|\delta_i\|_2^2
$$

鼓励小扰动，提高一致性。

---

## 9. 算法流程

```
输入: 模型 f_θ, 对话, 目标输出, 后缀候选集
输出: 优化后的嵌入 + 模型生成结果

═══════════════ 阶段 1: 离散搜索 ═══════════════

1. 初始化最佳后缀 s* = 默认后缀
2. FOR each suffix s in 候选集:
   a. 构造攻击对话: prompt + s
   b. 计算目标损失: L(s)
   c. IF L(s) < L(s*): s* = s
3. 输出: 最佳后缀 s*

═══════════════ 阶段 2: 连续优化 ═══════════════

4. 初始化:
   a. 获取 s* 的嵌入: E_orig
   b. 初始化扰动: δ = 0
   c. 创建优化器: Adam([δ], lr=η)

5. FOR step = 1 to num_steps:
   a. 构造扰动嵌入: E = E_orig + δ
   b. 前向传播: logits = f_θ(E_user ⊕ E ⊕ E_target)
   
   c. 计算损失:
      - L_attack = CrossEntropy(logits[target_positions], y*)
      - L_consist = LogitBarrier(logits[attack_positions], original_tokens)
      - L_norm = ||δ||²
      - L_total = L_attack + λ₁·L_consist + λ₂·L_norm
   
   d. 反向传播 + 优化器更新
   e. 投影: δ ← Proj_L2(δ, ε)
   
   f. (可选) 解码一致性检查

6. 使用最终/最佳嵌入生成模型输出
```

---

## 10. 与其他方法的对比

| 特性 | Natural Suffix Embedding | PGD | GCG | Random Restart |
|------|-------------------------|-----|-----|----------------|
| 优化空间 | 混合（离散→连续） | 连续 | 离散 | 连续→离散 |
| 解码一致性 | ✓ Logit Barrier | ✗ | ✓ 天然满足 | ✗ |
| 自然语言起点 | ✓ | ✗ | ✗ | ✗ |
| 两阶段优化 | ✓ | ✗ | ✗ | ✗ |
| Voronoi 约束 | ✓ 隐式 | ✗ | N/A | ✗ |

---

## 11. 关键超参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `phase1_num_steps` | 阶段1评估的候选数 | 20 |
| `phase2_num_steps` | 阶段2优化步数 | 100 |
| `epsilon` | L2 扰动上界 | 1.0 |
| `alpha` | 基础学习率 | 0.01 |
| `lambda_consist` | 一致性损失权重 | 1.0 |
| `lambda_norm` | 范数损失权重 | 0.1 |
| `logit_margin` | Logit Barrier 边际 | 5.0 |
| `decode_check` | 检查方式 | "logits_margin" |

---

## 12. 核心洞察

1. **自然语言起点**：使用自然后缀作为起点，比随机初始化更有希望
2. **Voronoi 单元探索**：在保持解码不变的前提下优化嵌入
3. **Logit Barrier**：软约束替代硬投影，更适合梯度优化
4. **两阶段协同**：粗搜索提供好的初始化，精优化挖掘潜力
5. **可实现性保证**：优化结果可以直接用原始 token 表示

---

## 13. 安全性分析

### 13.1 攻击的隐蔽性

- 使用自然语言后缀，不产生可疑的乱码
- 最终嵌入解码回原始自然文本
- 难以通过简单的文本过滤检测

### 13.2 防御思路

- 检测嵌入空间中的异常偏移
- 监控 logit 分布的异常模式
- 使用对抗训练增强鲁棒性

---

## 14. 算法变体

### 14.1 单阶段变体

跳过阶段1，直接使用指定后缀进行连续优化：

```yaml
phase1_enabled: false
natural_suffix_init: "Please explain in detail."
```

### 14.2 生成模式

支持三种生成模式：
- **all**：为每个优化步生成补全（昂贵）
- **best**：只为一致性约束满足且损失最低的步生成
- **last**：只为最后一步生成

---

## 参考资料

该算法结合了以下技术思想：
- Voronoi 单元与嵌入空间几何
- Logit Barrier 方法（约束优化）
- 两阶段优化策略
- 自然语言 jailbreak 技术
