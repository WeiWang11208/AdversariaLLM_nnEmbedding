# 批处理攻击脚本使用指南

## 问题背景

在运行 embedding-space 攻击（如 PGD）时，如果一次性处理整个数据集可能会导致 GPU 显存溢出（OOM）。为了解决这个问题，我们提供了两个批处理脚本，可以将数据集分批次处理。

## 新功能：自动数据集大小检测 🎉

Python 脚本现在支持自动检测数据集大小！你不再需要手动查看每个数据集有多少条数据，只需省略 `--total-samples` 参数，脚本会自动为你检测。

```bash
# 自动检测数据集大小（推荐）
python scripts/batch_run_attacks.py \
    --model Qwen/Qwen3-8B \
    --dataset adv_behaviors \
    --attack pgd \
    --batch-size 10 \
    --extra-args "attacks.pgd.attack_space=embedding attacks.pgd.num_steps=100"

# 手动指定（可选）
python scripts/batch_run_attacks.py \
    --model Qwen/Qwen3-8B \
    --dataset adv_behaviors \
    --attack pgd \
    --batch-size 10 \
    --total-samples 300 \
    --extra-args "attacks.pgd.attack_space=embedding attacks.pgd.num_steps=100"
```

## 脚本概述

### 1. `batch_run_attacks.sh` (Bash 脚本)
- 简单快速，适合快速使用
- 基本的批处理功能
- 轻量级，无额外依赖

### 2. `batch_run_attacks.py` (Python 脚本)
- 功能更强大，推荐使用
- 支持错误重试
- 支持断点续传
- 更详细的日志和进度报告
- 更好的错误处理

## 使用方法

### Bash 脚本

#### 基本用法
```bash
bash scripts/batch_run_attacks.sh \
    <model> \
    <dataset> \
    <attack> \
    <batch_size> \
    [total_samples] \
    [start_idx] \
    [extra_args...]
```

#### 示例 1: PGD embedding 攻击，每批 10 个样本
```bash
bash scripts/batch_run_attacks.sh \
    Qwen/Qwen3-8B \
    adv_behaviors \
    pgd \
    10 \
    300 \
    0 \
    attacks.pgd.attack_space=embedding \
    attacks.pgd.num_steps=100
```

#### 示例 2: 从第 100 个样本继续运行
```bash
bash scripts/batch_run_attacks.sh \
    Qwen/Qwen3-8B \
    adv_behaviors \
    pgd \
    10 \
    300 \
    100 \
    attacks.pgd.attack_space=embedding \
    attacks.pgd.num_steps=100
```

### Python 脚本（推荐）

#### 基本用法
```bash
python scripts/batch_run_attacks.py \
    --model <model_name> \
    --dataset <dataset_name> \
    --attack <attack_name> \
    --batch-size <batch_size> \
    [--total-samples <total>] \
    [--start-idx <start>] \
    [--extra-args "<additional_args>"]
```

#### 示例 1: PGD embedding 攻击，自动检测数据集大小（推荐）
```bash
python scripts/batch_run_attacks.py \
    --model Qwen/Qwen3-8B \
    --dataset adv_behaviors \
    --attack pgd \
    --batch-size 10 \
    --extra-args "attacks.pgd.attack_space=embedding attacks.pgd.num_steps=100"
```

#### 示例 2: 手动指定数据集大小
```bash
python scripts/batch_run_attacks.py \
    --model Qwen/Qwen3-8B \
    --dataset adv_behaviors \
    --attack pgd \
    --batch-size 10 \
    --total-samples 300 \
    --extra-args "attacks.pgd.attack_space=embedding attacks.pgd.num_steps=100"
```

#### 示例 3: 从第 100 个样本继续运行（断点续传）
```bash
python scripts/batch_run_attacks.py \
    --model Qwen/Qwen3-8B \
    --dataset adv_behaviors \
    --attack pgd \
    --batch-size 10 \
    --start-idx 100 \
    --extra-args "attacks.pgd.attack_space=embedding attacks.pgd.num_steps=100"
```

#### 示例 4: 自动重试 + 即使失败也继续
```bash
python scripts/batch_run_attacks.py \
    --model Qwen/Qwen3-8B \
    --dataset adv_behaviors \
    --attack pgd \
    --batch-size 10 \
    --retry 3 \
    --continue-on-error \
    --extra-args "attacks.pgd.attack_space=embedding attacks.pgd.num_steps=100"
```

## 参数说明

### Bash 脚本参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| model | 模型名称 | 必需 |
| dataset | 数据集名称 | 必需 |
| attack | 攻击类型 | 必需 |
| batch_size | 每批样本数 | 必需 |
| total_samples | 总样本数 | 300 (adv_behaviors) |
| start_idx | 起始索引 | 0 |
| extra_args | 额外的配置参数 | - |

### Python 脚本参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model | 模型名称 | 必需 |
| --dataset | 数据集名称 | 必需 |
| --attack | 攻击类型 | 必需 |
| --batch-size | 每批样本数 | 必需 |
| --total-samples | 总样本数 | 自动检测 |
| --start-idx | 起始索引 | 0 |
| --retry | 每批重试次数 | 2 |
| --delay | 批次间延迟（秒） | 2.0 |
| --extra-args | 额外的配置参数（字符串） | "" |
| --continue-on-error | 出错后继续执行 | False |

**注意**: `--total-samples` 现在默认为自动检测。脚本会读取数据集配置并自动计算大小。

## 批次大小选择建议

根据您的 GPU 显存，选择合适的批次大小：

| GPU 显存 | 推荐批次大小 |
|----------|--------------|
| 8GB | 5-10 |
| 16GB | 10-20 |
| 24GB | 20-30 |
| 40GB+ | 30-50 |

**注意**: 这些只是参考值，实际批次大小还取决于：
- 模型大小
- 攻击复杂度（如 PGD 的 num_steps）
- 序列长度

## 高级用法

### 1. 多个 epsilon 值的消融实验

使用 Bash 循环：
```bash
for epsilon in 0.5 1.0 2.0 5.0; do
    python scripts/batch_run_attacks.py \
        --model Qwen/Qwen3-8B \
        --dataset adv_behaviors \
        --attack pgd \
        --batch-size 5 \
        --total-samples 50 \
        --extra-args "attacks.pgd.attack_space=embedding attacks.pgd.epsilon=${epsilon}"
done
```

### 2. 并行运行多个批次（如果有多个 GPU）

```bash
# GPU 0: 处理样本 0-250
CUDA_VISIBLE_DEVICES=0 python scripts/batch_run_attacks.py \
    --model Qwen/Qwen3-8B \
    --dataset adv_behaviors \
    --attack pgd \
    --batch-size 10 \
    --total-samples 250 \
    --start-idx 0 \
    --extra-args "attacks.pgd.attack_space=embedding attacks.pgd.num_steps=100" &

# GPU 1: 处理样本 250-500
CUDA_VISIBLE_DEVICES=1 python scripts/batch_run_attacks.py \
    --model Qwen/Qwen3-8B \
    --dataset adv_behaviors \
    --attack pgd \
    --batch-size 10 \
    --total-samples 500 \
    --start-idx 250 \
    --extra-args "attacks.pgd.attack_space=embedding attacks.pgd.num_steps=100" &

wait
```

### 3. 使用 nohup 在后台运行

```bash
nohup python scripts/batch_run_attacks.py \
    --model Qwen/Qwen3-8B \
    --dataset adv_behaviors \
    --attack pgd \
    --batch-size 10 \
    --extra-args "attacks.pgd.attack_space=embedding attacks.pgd.num_steps=100" \
    > pgd_batch_run.log 2>&1 &

# 查看日志
tail -f pgd_batch_run.log
```

## 常见问题

### Q1: 如果中途失败了怎么办？
A: 使用 `--start-idx` 参数从失败的位置继续运行。例如，如果在处理第 150 个样本时失败：
```bash
python scripts/batch_run_attacks.py \
    --model Qwen/Qwen3-8B \
    --dataset adv_behaviors \
    --attack pgd \
    --batch-size 10 \
    --start-idx 150 \
    --extra-args "attacks.pgd.attack_space=embedding attacks.pgd.num_steps=100"
```

### Q2: 如何知道总共有多少样本？
A: 对于常见数据集：
- `adv_behaviors`: 520 样本
- 其他数据集可以先运行一次看日志，或者查看数据集文件

### Q3: 批次之间需要间隔吗？
A: Python 脚本默认在批次之间有 2 秒延迟，可以用 `--delay` 调整。这有助于确保 GPU 显存完全释放。

### Q4: 可以用于其他攻击吗？
A: 可以！这些脚本适用于所有攻击类型，只需修改 attack 参数和 extra_args。例如 GCG：
```bash
python scripts/batch_run_attacks.py \
    --model Qwen/Qwen3-8B \
    --dataset adv_behaviors \
    --attack gcg \
    --batch-size 20
```

## 故障排查

### 问题: 仍然 OOM
解决方案:
1. 减小批次大小（如从 10 降到 5）
2. 减少攻击步数（如 `attacks.pgd.num_steps=50`）
3. 使用更小的模型

### 问题: 脚本权限错误
解决方案:
```bash
chmod +x scripts/batch_run_attacks.sh
chmod +x scripts/batch_run_attacks.py
```

### 问题: 找不到 run_attacks.py
解决方案: 确保在项目根目录运行脚本：
```bash
cd /path/to/AdversariaLLM-main
python scripts/batch_run_attacks.py ...
```

## 性能优化建议

1. **选择合适的批次大小**: 尽量使用较大的批次（在不 OOM 的前提下），以减少模型加载次数
2. **使用 SSD**: 确保数据和模型存储在 SSD 上以加快 I/O
3. **多 GPU 并行**: 如果有多个 GPU，可以并行处理不同的样本范围
4. **监控显存**: 使用 `nvidia-smi -l 1` 监控显存使用情况，找到最优批次大小
