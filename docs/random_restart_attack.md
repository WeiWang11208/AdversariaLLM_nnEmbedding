# Random Restart Attack 实现文档

## 概述

Random Restart (RR) 是一种基于连续embedding优化的对抗攻击方法。该攻击在embedding空间中使用梯度优化adversarial suffix，然后将其投影回离散tokens。

## 方法特点

1. **连续优化**: 在embedding空间中进行梯度下降优化
2. **离散投影**: 通过最近邻搜索将连续embedding映射回离散tokens
3. **Checkpoint机制**: 在不同loss阈值处保存最佳结果
4. **学习率衰减**: 使用指数衰减调整学习率
5. **Token过滤**: 支持过滤非ASCII字符

## 文件结构

```
src/attacks/
└── random_restart.py          # 主实现文件
    ├── RandomRestartConfig    # 配置类
    └── RandomRestartAttack    # 攻击类

conf/attacks/
└── attacks.yaml               # 配置文件(新增random_restart部分)
```

## 配置参数说明

### 基本参数

- `name`: 攻击名称 (固定为 "random_restart")
- `type`: 攻击类型 (continuous - 连续优化)
- `version`: 版本号
- `placement`: 放置位置 (suffix - 在prompt后添加)
- `seed`: 随机种子

### 优化参数

- `num_steps`: 优化迭代次数 (默认: 500)
- `initial_lr`: 初始学习率 (默认: 0.1)
- `weight_decay`: 权重衰减 (默认: 0.0)
- `decay_rate`: 学习率衰减率 (默认: 0.99)
- `optim_str_init`: 初始化字符串 (默认: 20个"x")

### Checkpoint参数

- `checkpoints`: Loss阈值列表 (默认: [10.0, 5.0, 2.0, 1.0, 0.5])
  - 当loss低于某个阈值时，保存当前最佳结果
  - 所有checkpoint都达到后会提前停止

### 过滤参数

- `allow_non_ascii`: 是否允许非ASCII字符 (默认: False)
- `allow_special`: 是否允许特殊token (默认: False)

### 其他参数

- `max_grad_norm`: 梯度裁剪阈值 (默认: 1.0)
- `init_noise_std`: 初始化噪声标准差 (默认: 0.1)
- `generation_config`: 生成配置(温度、top_p等)

## 使用方法

### 1. 基本使用

```bash
# 在单个模型和数据集上运行
python run_attacks.py \
    attack=random_restart \
    dataset=adv_behaviors \
    model=google/gemma-2-2b-it
```

### 2. 覆盖默认参数

```bash
# 调整优化步数和学习率
python run_attacks.py \
    attack=random_restart \
    attacks.random_restart.num_steps=1000 \
    attacks.random_restart.initial_lr=0.05
```

### 3. 修改Checkpoint阈值

```bash
# 使用更严格的checkpoint
python run_attacks.py \
    attack=random_restart \
    attacks.random_restart.checkpoints=[5.0,2.0,1.0,0.5,0.1]
```

### 4. 在数据集子集上测试

```bash
# 仅在前10个样本上测试
python run_attacks.py \
    attack=random_restart \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="range(0,10)"'
```

### 5. 允许非ASCII字符

```bash
# 允许使用非ASCII字符(可能提高攻击成功率)
python run_attacks.py \
    attack=random_restart \
    attacks.random_restart.allow_non_ascii=True
```

### 6. 批量运行多个模型

```bash
# 在多个模型上运行
python run_attacks.py -m \
    attack=random_restart \
    dataset=adv_behaviors \
    model=google/gemma-2-2b-it,meta-llama/Meta-Llama-3.1-8B-Instruct
```

## 算法流程

### 1. 初始化阶段

```python
# 1. 获取embedding矩阵
embed_weights = model.get_embedding_matrix()

# 2. 将prompt、adversarial suffix、target转为embeddings
embeddings_user = embed_weights[user_prompt_ids]
embeddings_adv = embed_weights[adv_init_ids] + noise
embeddings_target = embed_weights[target_ids]

# 3. 设置adversarial embeddings为可优化参数
embeddings_adv.requires_grad = True
```

### 2. 优化循环

```python
for iteration in range(num_steps):
    # (1) 计算连续loss
    loss = calc_ce_loss(embeddings_user, embeddings_adv, embeddings_target)

    # (2) 反向传播
    loss.backward()

    # (3) 投影到最近的离散tokens
    closest_embeddings = find_closest_embeddings(embeddings_adv)

    # (4) 计算离散loss
    discrete_loss = calc_ce_loss(embeddings_user, closest_embeddings, embeddings_target)

    # (5) 保存checkpoint
    if loss < checkpoint_threshold:
        save_checkpoint(discrete_loss, closest_tokens)

    # (6) 梯度裁剪 + 优化更新
    clip_gradients(embeddings_adv)
    optimizer.step()

    # (7) 调整学习率
    lr = initial_lr * (decay_rate ** iteration)
```

### 3. 生成阶段

```python
# 对每个checkpoint的最佳suffix生成模型回复
for best_suffix in checkpoint_suffixes:
    attack_prompt = original_prompt + best_suffix
    completions = model.generate(attack_prompt)
    save_result(completions, discrete_loss)
```

## 核心函数说明

### `_find_closest_embeddings()`

将连续embedding投影到最近的离散token embedding：

```python
def _find_closest_embeddings(embeddings_adv, embed_weights):
    # 归一化
    embeddings_adv_norm = normalize(embeddings_adv)
    embed_weights_norm = normalize(embed_weights)

    # 计算L2距离
    distances = torch.cdist(embeddings_adv_norm, embed_weights_norm, p=2)

    # 过滤非ASCII tokens (可选)
    if not allow_non_ascii:
        distances[:, non_ascii_toks] = float("inf")

    # 找到最近的token
    closest_indices = distances.argmin(dim=-1)
    closest_embeddings = embed_weights[closest_indices]

    return closest_embeddings, closest_indices
```

### `_calc_ce_loss()`

计算交叉熵损失：

```python
def _calc_ce_loss(embeddings_user, embeddings_adv, embeddings_target, target_ids):
    # 拼接所有embeddings
    full_embeddings = torch.cat([
        embeddings_user,
        embeddings_adv,
        embeddings_target
    ], dim=1)

    # 前向传播
    logits = model(inputs_embeds=full_embeddings).logits

    # 计算target部分的loss
    loss_start = len(embeddings_user) + len(embeddings_adv)
    loss = CrossEntropyLoss()(
        logits[:, loss_start-1:-1, :],
        target_ids
    )

    return loss
```

## 输出结果

攻击结果保存在 `${root_dir}/outputs/YYYY-MM-DD/HH-MM-SS/` 目录下，格式为JSON：

```json
{
    "runs": [
        {
            "original_prompt": [...],
            "steps": [
                {
                    "step": 0,  // Checkpoint 0: loss < 10.0
                    "model_completions": ["..."],
                    "loss": 9.5,
                    "model_input": [
                        {"role": "user", "content": "original + suffix1"},
                        {"role": "assistant", "content": ""}
                    ],
                    "model_input_tokens": [1, 2, 3, ...]
                },
                {
                    "step": 1,  // Checkpoint 1: loss < 5.0
                    "model_completions": ["..."],
                    "loss": 4.8,
                    ...
                },
                // ... 更多checkpoints
            ],
            "total_time": 120.5
        }
    ]
}
```

## 与原始代码的对比

### 主要改动

1. **结构化配置**
   - 原代码: 使用单独的config模块
   - 新代码: 使用dataclass和Hydra配置系统

2. **数据格式**
   - 原代码: messages (str), target (str)
   - 新代码: Conversation (list[dict])

3. **结果格式**
   - 原代码: 自定义RRResult类
   - 新代码: 统一的AttackResult/SingleAttackRunResult/AttackStepResult

4. **批处理支持**
   - 原代码: 单样本处理
   - 新代码: 支持数据集批处理和分批运行

5. **工具函数集成**
   - 原代码: 独立的helper.py
   - 新代码: 集成到Attack类的方法中

### 保留的核心逻辑

✅ Embedding优化算法
✅ 最近邻投影机制
✅ Checkpoint保存策略
✅ 学习率衰减
✅ 梯度裁剪
✅ Token过滤

## 调试和日志

### 查看详细日志

攻击过程会输出详细日志：

```
2026-01-05 10:30:15 [random_restart.py:123] Iteration 0: loss=12.5, discrete_loss=13.2
2026-01-05 10:30:20 [random_restart.py:145] Checkpoint 0: loss=9.8, discrete_loss=10.1, suffix=!!! describe ways...
2026-01-05 10:30:25 [random_restart.py:123] Iteration 50: loss=5.2, discrete_loss=5.5
2026-01-05 10:30:30 [random_restart.py:145] Checkpoint 1: loss=4.9, discrete_loss=5.1, suffix=certainly here ways...
```

### 常见问题

#### Q1: 攻击运行很慢

A: 尝试减少优化步数或使用更小的模型：
```bash
python run_attacks.py attack=random_restart attacks.random_restart.num_steps=100
```

#### Q2: Loss不降低

A: 尝试增加学习率或调整初始化：
```bash
python run_attacks.py attack=random_restart \
    attacks.random_restart.initial_lr=0.2 \
    attacks.random_restart.init_noise_std=0.2
```

#### Q3: 找不到合适的token

A: 允许非ASCII字符或调整梯度裁剪：
```bash
python run_attacks.py attack=random_restart \
    attacks.random_restart.allow_non_ascii=True \
    attacks.random_restart.max_grad_norm=2.0
```

#### Q4: 内存不足 (OOM)

A: 使用batch_size分批处理：
```bash
python run_attacks.py attack=random_restart batch_size=8
```

## 性能建议

1. **快速测试**: 使用较少的步数和宽松的checkpoints
   ```bash
   python run_attacks.py attack=random_restart \
       attacks.random_restart.num_steps=100 \
       attacks.random_restart.checkpoints=[5.0,1.0]
   ```

2. **高质量攻击**: 使用更多步数和严格的checkpoints
   ```bash
   python run_attacks.py attack=random_restart \
       attacks.random_restart.num_steps=1000 \
       attacks.random_restart.checkpoints=[2.0,1.0,0.5,0.1,0.01]
   ```

3. **内存优化**: 启用batch processing
   ```bash
   python run_attacks.py attack=random_restart batch_size=16
   ```

## 实现细节

### 支持的模型类型

- ✅ LlamaForCausalLM (Llama, Vicuna等)
- ✅ FalconForCausalLM
- ✅ MistralForCausalLM
- ✅ 其他标准HuggingFace causal LM模型

### 不支持的特性

- ❌ API模型 (需要白盒访问embedding)
- ❌ 多轮对话 (当前仅支持单轮)
- ❌ Multi-modal输入

## 扩展和改进建议

1. **自适应学习率**: 根据loss变化动态调整学习率
2. **多起点重启**: 从多个随机初始化开始优化
3. **温度退火**: 在投影时使用温度参数
4. **梯度正则化**: 添加额外的正则化项
5. **早停策略**: 根据discrete loss的改善情况早停

## 参考文献

本实现基于embedding-based optimization攻击方法，结合了：
- GCG的token搜索策略
- PGD的连续优化思想
- Checkpoint机制保证鲁棒性

---

**实现日期**: 2026-01-05
**版本**: 0.0.1
**维护者**: AdversariaLLM Team

---

## 实验运行脚本

### 基础实验 - 收集数据

```bash
# 在adv_behaviors数据集上运行Random Restart攻击
# 使用Qwen3-8B模型，前2个样本
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="list(range(0,2))"' \
    attack=random_restart

# 使用默认参数在完整数据集上运行
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    attack=random_restart
```

### 批量实验脚本

```bash
#!/bin/bash
# test_random_restart.sh - Random Restart攻击批量测试脚本

# 设置模型列表
MODELS=(
    "Qwen/Qwen3-8B"
    "google/gemma-2-2b-it"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

# 设置数据集范围
SAMPLE_RANGE="list(range(0,50))"

for model in "${MODELS[@]}"; do
    echo "Running Random Restart on $model..."
    python run_attacks.py \
        model="$model" \
        dataset=adv_behaviors \
        "datasets.adv_behaviors.idx=\"$SAMPLE_RANGE\"" \
        attack=random_restart \
        attacks.random_restart.num_steps=500
done
```

### 参数调优实验

```bash
# 实验1: 测试不同学习率
for lr in 0.05 0.1 0.2; do
    python run_attacks.py \
        model=Qwen/Qwen3-8B \
        dataset=adv_behaviors \
        'datasets.adv_behaviors.idx="list(range(0,10))"' \
        attack=random_restart \
        attacks.random_restart.initial_lr=$lr
done

# 实验2: 测试不同checkpoint阈值
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    attack=random_restart \
    'attacks.random_restart.checkpoints=[5.0,2.0,1.0,0.5,0.1]'

# 实验3: 更多优化步数
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    attack=random_restart \
    attacks.random_restart.num_steps=1000
```

---

## 与PGD方法的数据保存对比

### 输出数据结构

两种方法都使用统一的`AttackResult`结构保存结果：

| 字段 | PGD | Random Restart | 说明 |
|------|-----|----------------|------|
| `model_completions` | ✅ | ✅ | 模型生成的completions |
| `loss` | ✅ 每步 | ✅ 每个checkpoint | loss值 |
| `time_taken` | ✅ 精确 | ⚠️ 设为0 | 每步时间 |
| `model_input` | ✅ | ✅ | 输入对话 |
| `model_input_tokens` | ❌ | ✅ | token列表 |
| `model_input_embeddings` | ✅ 可选 | ❌ | embedding数据 |
| `scores` | ✅ 后处理 | ✅ 后处理 | judge评分 |

### PGD特有功能

PGD通过`generate_completions`参数控制保存哪些步骤的结果：
- `"all"`: 保存所有步骤的结果
- `"best"`: 只保存loss最低的步骤
- `"last"`: 只保存最后一步

### Random Restart特有功能

Random Restart使用checkpoint机制：
- 在loss首次低于各个阈值时保存结果
- 默认阈值: `[10.0, 5.0, 2.0, 1.0, 0.5]`
- 每个checkpoint保存:
  - discrete loss值
  - 对应的suffix字符串
  - 模型生成的completion

### 后续分析所需数据

当前Random Restart实现已保存：
- ✅ 每个checkpoint的discrete loss
- ✅ 每个checkpoint的suffix字符串
- ✅ 每个checkpoint的model completion
- ✅ model_input_tokens (用于重现实验)
- ✅ total_time (整体运行时间)

如需额外分析（如loss曲线），可修改代码添加：
- `all_losses`: 所有步骤的loss历史
- `all_suffixes`: 所有步骤的suffix历史
- `distances`: embedding到token的投影距离

---

## 快速验证命令

```bash
# 快速测试 - 2个样本，100步
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="list(range(0,2))"' \
    attack=random_restart \
    attacks.random_restart.num_steps=100

# 查看输出结果
ls -la outputs/*/
```
