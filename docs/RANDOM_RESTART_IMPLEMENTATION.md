# Random Restart Attack 实现总结

本文档总结了Random Restart (RR) 攻击方法在AdversariaLLM框架中的集成工作。

## 实现概述

Random Restart是一种基于连续embedding优化的对抗攻击方法，通过在embedding空间中进行梯度下降，然后投影回离散tokens来生成adversarial suffixes。

### 关键特性

- ✅ **连续优化**: 在embedding空间使用AdamW优化器
- ✅ **离散投影**: L2距离最近邻搜索
- ✅ **Checkpoint机制**: 多阈值保存最佳结果
- ✅ **学习率衰减**: 指数衰减策略
- ✅ **Token过滤**: 支持过滤非ASCII和特殊tokens
- ✅ **框架集成**: 完全兼容AdversariaLLM框架

## 文件清单

### 核心实现
```
src/attacks/random_restart.py
├── RandomRestartConfig (dataclass)
│   ├── 优化参数 (num_steps, initial_lr, decay_rate等)
│   ├── Checkpoint配置 (checkpoints列表)
│   ├── Token过滤选项 (allow_non_ascii, allow_special)
│   └── 其他参数 (max_grad_norm, init_noise_std)
└── RandomRestartAttack (class)
    ├── run() - 主入口方法
    ├── _attack_single_conversation() - 单样本攻击
    ├── _generate_completions() - 生成模型回复
    ├── _get_embedding_matrix() - 获取embedding矩阵
    ├── _get_nonascii_toks() - 获取非ASCII tokens
    ├── _find_closest_embeddings() - 最近邻投影
    ├── _calc_ce_loss() - 计算交叉熵loss
    └── _adjust_learning_rate() - 学习率调整
```

### 配置文件
```
conf/attacks/attacks.yaml
└── random_restart:
    ├── 基本配置 (name, type, version, seed)
    ├── 优化参数 (num_steps, lr, weight_decay, decay_rate)
    ├── Checkpoint阈值 (checkpoints: [10.0, 5.0, 2.0, 1.0, 0.5])
    ├── Token过滤 (allow_non_ascii, allow_special)
    └── 其他参数 (max_grad_norm, init_noise_std)
```

### 注册
```
src/attacks/attack.py
└── Attack.from_name()
    └── case "random_restart": return RandomRestartAttack
```

### 文档
```
docs/
├── attack_template_structure.md      # 框架整体结构文档
└── random_restart_attack.md          # RR攻击详细文档
```

### 测试
```
test_random_restart.sh                 # 自动化测试脚本
```

## 快速开始

### 1. 基本运行

```bash
# 在完整数据集上运行
python run_attacks.py \
    attack=random_restart \
    dataset=adv_behaviors \
    model=google/gemma-2-2b-it
```

### 2. 快速测试 (推荐首次使用)

```bash
# 在前5个样本上快速测试
python run_attacks.py \
    attack=random_restart \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="range(0,5)"' \
    attacks.random_restart.num_steps=100
```

### 3. 运行自动化测试

```bash
# 运行所有测试用例
./test_random_restart.sh
```

### 4. 查看结果

```bash
# 结果保存在outputs目录
ls -lh outputs/

# 查看最新结果
cat outputs/$(ls -t outputs/ | head -1)/*.json | jq
```

## 与原始实现的对比

### 保持一致的部分

| 组件 | 原始实现 | 框架实现 | 状态 |
|------|---------|---------|------|
| 优化算法 | AdamW + 梯度裁剪 | AdamW + 梯度裁剪 | ✅ 一致 |
| 最近邻投影 | L2距离 + 归一化 | L2距离 + 归一化 | ✅ 一致 |
| Checkpoint机制 | 多阈值保存 | 多阈值保存 | ✅ 一致 |
| 学习率衰减 | 指数衰减 | 指数衰减 | ✅ 一致 |
| Token过滤 | allow_non_ascii | allow_non_ascii | ✅ 一致 |
| Loss计算 | CrossEntropyLoss | CrossEntropyLoss | ✅ 一致 |

### 改进和增强

| 功能 | 原始实现 | 框架实现 | 改进说明 |
|------|---------|---------|---------|
| 配置管理 | 单独config.py | Hydra配置 | 更灵活的配置组合 |
| 数据格式 | str输入 | Conversation对象 | 统一的数据接口 |
| 结果格式 | 自定义RRResult | AttackResult体系 | 标准化结果格式 |
| 批处理 | 不支持 | batch_size参数 | 支持大数据集 |
| 模型支持 | 手动配置路径 | Hydra模型管理 | 自动加载模型 |
| 日志系统 | 简单print | logging模块 | 结构化日志 |
| 工具函数 | 独立helper.py | 类方法 | 更好的封装性 |

## 技术细节

### 算法流程

```
1. 初始化
   ├── 加载embedding矩阵
   ├── 准备user/adversarial/target embeddings
   ├── 添加初始化噪声
   └── 创建优化器

2. 优化循环 (num_steps次)
   ├── 计算连续loss (embedding空间)
   ├── 反向传播
   ├── 投影到最近的离散tokens
   ├── 计算离散loss
   ├── 检查并保存checkpoints
   ├── 梯度裁剪
   ├── 优化器更新
   └── 调整学习率

3. 生成阶段
   ├── 对每个checkpoint的suffix
   ├── 构造attack prompt
   ├── 生成模型回复
   └── 保存结果
```

### Checkpoint机制

```python
checkpoints = [10.0, 5.0, 2.0, 1.0, 0.5]

# 当loss首次低于某个阈值时保存
if loss < 10.0 and best_loss[0] == inf:
    best_loss[0] = discrete_loss
    best_suffix[0] = current_suffix

# 所有checkpoint都达到后早停
if all(loss != inf for loss in best_loss):
    break
```

### 投影机制

```python
# 1. 归一化
adv_norm = embeddings_adv / ||embeddings_adv||
weight_norm = embed_weights / ||embed_weights||

# 2. 计算L2距离
distances = ||adv_norm - weight_norm||₂

# 3. 过滤非ASCII (可选)
distances[non_ascii_toks] = ∞

# 4. 选择最近的token
closest_idx = argmin(distances)
closest_emb = embed_weights[closest_idx]
```

## 性能考虑

### 时间复杂度

- **每次迭代**: O(N × V × D)
  - N: adversarial suffix长度
  - V: 词表大小
  - D: embedding维度

- **完整攻击**: O(num_steps × N × V × D)

### 内存占用

- **主要开销**:
  - Embedding矩阵: V × D
  - 优化器状态: N × D
  - 梯度: N × D
  - 距离矩阵: N × V

- **优化建议**:
  - 使用较小的suffix长度
  - 使用batch processing分批运行
  - 及时清理不需要的中间结果

### 速度优化建议

1. **减少迭代次数**: `num_steps=100-200` 用于快速测试
2. **简化checkpoints**: 只使用2-3个阈值
3. **允许非ASCII**: 增大搜索空间，可能更快收敛
4. **调整学习率**: 更大的学习率可能更快收敛(但不稳定)

## 使用示例

### 场景1: 快速原型测试

```bash
python run_attacks.py \
    attack=random_restart \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="range(0,10)"' \
    attacks.random_restart.num_steps=50 \
    attacks.random_restart.checkpoints=[5.0,1.0]
```

### 场景2: 高质量攻击

```bash
python run_attacks.py \
    attack=random_restart \
    dataset=adv_behaviors \
    attacks.random_restart.num_steps=1000 \
    attacks.random_restart.checkpoints=[10.0,5.0,2.0,1.0,0.5,0.1] \
    attacks.random_restart.initial_lr=0.05
```

### 场景3: 多模型评估

```bash
python run_attacks.py -m \
    attack=random_restart \
    dataset=adv_behaviors \
    model=google/gemma-2-2b-it,meta-llama/Meta-Llama-3.1-8B-Instruct \
    attacks.random_restart.num_steps=500
```

### 场景4: 参数网格搜索

```bash
# 搜索最佳学习率
for lr in 0.05 0.1 0.2; do
    python run_attacks.py \
        attack=random_restart \
        attacks.random_restart.initial_lr=$lr \
        'datasets.adv_behaviors.idx="range(0,5)"'
done
```

## 故障排除

### 问题1: Loss不下降

**症状**: Loss一直维持在高值
**可能原因**:
- 学习率太小
- 初始化不好
- Token过滤太严格

**解决方案**:
```bash
python run_attacks.py \
    attack=random_restart \
    attacks.random_restart.initial_lr=0.2 \
    attacks.random_restart.init_noise_std=0.2 \
    attacks.random_restart.allow_non_ascii=True
```

### 问题2: 内存溢出 (OOM)

**症状**: CUDA out of memory错误
**可能原因**:
- 模型太大
- Suffix太长
- 批处理太大

**解决方案**:
```bash
# 使用分批处理
python run_attacks.py \
    attack=random_restart \
    batch_size=8
```

### 问题3: 运行时间太长

**症状**: 单个样本需要很长时间
**可能原因**:
- num_steps太大
- 模型推理慢

**解决方案**:
```bash
# 减少迭代次数
python run_attacks.py \
    attack=random_restart \
    attacks.random_restart.num_steps=100
```

### 问题4: 找不到合适的tokens

**症状**: 投影后的tokens很奇怪
**可能原因**:
- 过滤太严格
- Embedding空间中没有好的解

**解决方案**:
```bash
# 放宽token过滤
python run_attacks.py \
    attack=random_restart \
    attacks.random_restart.allow_non_ascii=True \
    attacks.random_restart.allow_special=True
```

## 验证清单

在提交代码前，请确保：

- [x] 代码实现完成 (`src/attacks/random_restart.py`)
- [x] 配置文件更新 (`conf/attacks/attacks.yaml`)
- [x] Attack注册完成 (`src/attacks/attack.py`)
- [x] 文档编写完成 (`docs/random_restart_attack.md`)
- [x] 测试脚本创建 (`test_random_restart.sh`)
- [ ] 测试通过 (运行 `./test_random_restart.sh`)
- [ ] 在真实数据集上验证
- [ ] 性能profiling完成

## 后续工作

### 短期改进
1. [ ] 添加单元测试
2. [ ] 优化内存使用
3. [ ] 支持多轮对话
4. [ ] 添加更多checkpoint策略

### 长期扩展
1. [ ] 自适应学习率调整
2. [ ] 多起点random restart
3. [ ] 温度退火策略
4. [ ] 与GCG/PGD的hybrid方法

## 相关资源

- **框架文档**: `docs/attack_template_structure.md`
- **攻击文档**: `docs/random_restart_attack.md`
- **配置文件**: `conf/attacks/attacks.yaml`
- **测试脚本**: `test_random_restart.sh`
- **源代码**: `src/attacks/random_restart.py`

## 联系方式

如有问题或建议，请：
1. 查看文档: `docs/random_restart_attack.md`
2. 运行测试: `./test_random_restart.sh`
3. 查看日志输出中的错误信息
4. 提交Issue到项目仓库

---

**实现完成日期**: 2026-01-05
**版本**: v0.0.1
**状态**: ✅ 已完成集成
**测试状态**: ⏳ 待运行测试
