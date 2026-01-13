# Random Restart (RR) 攻击算法详解

这是一个针对大语言模型（LLM）的**嵌入空间连续优化攻击**，核心思想是在连续空间优化后投影回离散 token。下面从算法层面进行详细介绍。

---

## 1. 问题形式化

### 1.1 攻击场景

给定一个安全对齐后的 LLM $f_\theta$，攻击者希望找到一个**对抗性后缀**，使模型生成预定的有害目标输出 $y^*$。

### 1.2 优化目标

与 PGD 类似，Random Restart 在连续嵌入空间中优化：

$$
\min_{E_{\text{adv}}} \mathcal{L}\big(f_\theta(E_{\text{user}} \oplus E_{\text{adv}} \oplus E_{\text{target}}), y^*\big)
$$

其中：
- $E_{\text{user}}$ 是用户提示的嵌入
- $E_{\text{adv}}$ 是可优化的对抗嵌入
- $E_{\text{target}}$ 是目标输出的嵌入
- $\mathcal{L}$ 是交叉熵损失

### 1.3 核心特点

> **连续优化 + 离散投影 + 检查点保存机制**

---

## 2. 算法框架

### 2.1 双轨评估

Random Restart 同时跟踪两个损失值：

1. **连续损失** $\mathcal{L}_{\text{cont}}$：直接使用优化中的连续嵌入计算
2. **离散损失** $\mathcal{L}_{\text{disc}}$：将连续嵌入投影到最近的离散 token 后计算

$$
\mathcal{L}_{\text{disc}} = \mathcal{L}\big(f_\theta(E_{\text{user}} \oplus \text{Proj}(E_{\text{adv}}) \oplus E_{\text{target}}), y^*\big)
$$

### 2.2 投影操作

将连续嵌入投影到最近的离散 token：

$$
\text{Proj}(e) = \arg\min_{v \in \mathcal{V}} \|e - W_{\text{embed}}[v]\|_2
$$

实际实现中使用归一化的余弦距离：

$$
\text{Proj}(e) = \arg\min_{v \in \mathcal{V}} \left\|\frac{e}{\|e\|} - \frac{W_v}{\|W_v\|}\right\|_2
$$

---

## 3. 检查点机制

### 3.1 多阈值保存

定义一系列损失阈值 $[\tau_1, \tau_2, ..., \tau_K]$（如 $[10.0, 5.0, 2.0, 1.0, 0.5]$）。

当连续损失首次低于阈值 $\tau_i$ 时，保存当前状态：
- 离散损失值
- 投影后的 token 序列
- 连续嵌入到离散嵌入的距离
- （可选）完整的连续嵌入用于可实现性分析

### 3.2 早停策略

当所有检查点都被填充时，提前终止优化：

$$
\text{Stop if } \forall i: \mathcal{L}_{\text{cont}} < \tau_i \text{ has been satisfied}
$$

---

## 4. 优化细节

### 4.1 优化器

使用 **AdamW** 优化器：

$$
E_{\text{adv}}^{(t+1)} = E_{\text{adv}}^{(t)} - \eta_t \cdot \text{AdamW}(\nabla \mathcal{L})
$$

### 4.2 学习率调度

指数衰减学习率：

$$
\eta_t = \eta_0 \cdot \gamma^t
$$

其中：
- $\eta_0$ 是初始学习率（默认 0.1）
- $\gamma$ 是衰减率（默认 0.99）

### 4.3 梯度裁剪

防止梯度爆炸：

$$
\nabla \mathcal{L} \leftarrow \text{clip}(\nabla \mathcal{L}, \text{max\_norm})
$$

### 4.4 初始化噪声

在初始嵌入上添加高斯噪声，增加探索性：

$$
E_{\text{adv}}^{(0)} = W_{\text{embed}}[s_{\text{init}}] + \mathcal{N}(0, \sigma^2 I)
$$

其中 $\sigma$ 是噪声标准差（默认 0.1）。

---

## 5. 损失函数

### 5.1 交叉熵损失

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{|T|} \sum_{t \in T} \log p_\theta(y^*_t | E_{\text{full}}, y^*_{<t})
$$

其中 $T$ 是目标 token 的位置集合。

### 5.2 损失计算位置

损失只在目标输出的位置计算，输入拼接方式为：

$$
E_{\text{full}} = [E_{\text{user}}, E_{\text{adv}}, E_{\text{target}}]
$$

损失起始位置：

$$
\text{start} = |E_{\text{user}}| + |E_{\text{adv}}|
$$

---

## 6. Token 过滤

### 6.1 非 ASCII 过滤

投影时可以选择排除非 ASCII token：

$$
\mathcal{V}_{\text{allowed}} = \{v \in \mathcal{V} : \text{decode}(v) \text{ is ASCII and printable}\}
$$

### 6.2 特殊标记过滤

排除特殊控制标记：
- BOS (Beginning of Sequence)
- EOS (End of Sequence)
- PAD (Padding)
- UNK (Unknown)

---

## 7. 算法流程

```
输入: 模型 f_θ, 用户提示, 目标输出, 初始后缀, 检查点阈值列表
输出: 各检查点的最佳攻击后缀和模型生成结果

1. 初始化:
   a. 获取嵌入矩阵 W_embed
   b. 编码用户提示、初始后缀、目标输出为嵌入
   c. 对后缀嵌入添加高斯噪声
   d. 初始化 AdamW 优化器

2. 优化循环 (t = 1 to num_steps):
   a. 计算连续损失: L_cont = CE(f_θ(E_user ⊕ E_adv ⊕ E_target), y*)
   b. 反向传播: ∇L_cont
   
   c. 投影到离散空间:
      - 找到最近的离散嵌入: E_disc = Proj(E_adv)
      - 计算离散损失: L_disc
      - 解码得到当前后缀字符串
   
   d. 检查点保存:
      FOR each threshold τ_i:
         IF L_cont < τ_i AND checkpoint_i is empty:
            保存 (L_disc, suffix, distance, embeddings)
   
   e. 早停检查: 如果所有检查点已填充则停止
   
   f. 梯度裁剪和优化器更新
   g. 学习率衰减

3. 填充未达到的检查点（使用最终状态）

4. 对所有检查点的后缀生成模型补全

5. 返回结果
```

---

## 8. 与其他方法的对比

| 特性 | Random Restart | PGD | GCG |
|------|---------------|-----|-----|
| 优化空间 | 连续 → 离散投影 | 纯连续 | 纯离散 |
| 投影策略 | 每步投影评估 | 无投影/可选投影 | N/A |
| 检查点机制 | ✓ 多阈值保存 | ✗ | ✗ |
| 学习率 | 指数衰减 | 固定 | N/A |
| 初始化 | 噪声扰动 | 可选随机重启 | 固定初始化 |
| 可实现性 | 关注离散损失 | 关注连续损失 | 天然可实现 |

---

## 9. 关键超参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `num_steps` | 优化步数 | 500 |
| `initial_lr` | 初始学习率 | 0.1 |
| `decay_rate` | 学习率衰减率 | 0.99 |
| `weight_decay` | 权重衰减 | 0.0 |
| `max_grad_norm` | 梯度裁剪阈值 | 1.0 |
| `init_noise_std` | 初始化噪声标准差 | 0.1 |
| `checkpoints` | 损失阈值列表 | [10.0, 5.0, 2.0, 1.0, 0.5] |

---

## 10. 核心洞察

1. **双轨评估**：同时跟踪连续和离散损失，真实反映攻击效果
2. **检查点机制**：保存不同阶段的最佳结果，避免过拟合
3. **投影即评估**：每步都评估离散化后的效果，确保可实现性
4. **噪声初始化**：增加探索性，避免陷入初始点附近的局部最优
5. **衰减学习率**：前期大步探索，后期细粒度优化

---

## 11. 可实现性分析

Random Restart 的一个重要特点是关注**可实现性（Realizability）**——即连续嵌入能否被真实 token 近似。

通过同时计算：
- 连续损失（优化目标）
- 离散损失（实际效果）
- 投影距离（可实现性度量）

可以分析攻击在嵌入流形上的行为，为理解嵌入空间攻击的局限性提供依据。

---

## 12. 算法变体

### 12.1 多次随机重启

可以多次运行整个优化过程，使用不同的随机初始化：

$$
E_{\text{adv}}^{(0)} \sim W_{\text{embed}}[s_{\text{init}}] + \mathcal{N}(0, \sigma^2 I)
$$

取多次运行中离散损失最低的结果。

### 12.2 与 PGD 的关系

Random Restart 可以看作是 PGD 的一个变体：
- 使用 AdamW 代替 SGD
- 添加检查点机制
- 更关注离散投影的效果
- 使用指数衰减学习率

---

## 参考资料

该算法结合了以下技术思想：
- 嵌入空间攻击（Soft Prompt Threats）
- 投影梯度下降（Projected Gradient Descent）
- 检查点保存机制
- 余弦相似度最近邻搜索
