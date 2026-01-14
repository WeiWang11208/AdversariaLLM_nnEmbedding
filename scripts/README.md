# Scripts 使用说明

本目录包含用于评估攻击结果的脚本。

## 脚本概览

| 脚本 | 功能 |
|------|------|
| `compute_suffix_metrics.py` | 计算后缀相关指标（PPL、解码一致性等） |
| `eval_attack_results.py` | 汇总评估攻击结果（ASR、PPL、时间等） |

---

## compute_suffix_metrics.py

计算后缀相关指标并写入 `run.json`。**自动从 run.json 的 config 中读取目标模型进行 PPL 计算和 NN decode。**

### 计算的指标

| 指标 | 说明 | 适用方法 |
|------|------|----------|
| `suffix_text` | 后缀文本 | 全部 |
| `suffix_tokens` | 后缀 token 数量 | 全部 |
| `suffix_ppl` | 后缀困惑度（使用目标模型） | 全部 |
| `decode_text` | 从 embedding NN decode 的文本 | PGD, Ours |
| `decode_ppl` | decode 后文本的困惑度 | PGD, Ours |
| `decode_exact_text_match` | 解码文本是否与原始一致 | PGD, Ours |
| `decode_token_match_rate` | token 匹配率 | PGD, Ours |

### 使用方法

```bash
# 基础用法：自动从 run.json 读取目标模型
python scripts/compute_suffix_metrics.py \
    --results_dir /mnt/public/share/users/wangwei/202512/AdversariaLLM-main/outputs/Qwen3_8b/natural_suffix_embedding/2026-01-12/23-51-00/ \
    --recursive

# 处理单个文件
python scripts/compute_suffix_metrics.py \
    --results_file outputs/2026-01-13/12-00-00/0/run.json

# 可选：手动指定模型（覆盖 run.json 中的配置）
python scripts/compute_suffix_metrics.py \
    --results_dir outputs/ \
    --recursive \
    --model Qwen/Qwen3-8B

# 强制重新计算（即使已有 suffix_metrics）
python scripts/compute_suffix_metrics.py \
    --results_dir outputs/ \
    --recursive \
    --force
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--results_dir` | 结果目录 | - |
| `--results_file` | 单个结果文件 | - |
| `--recursive` | 递归搜索 | False |
| `--model` | 覆盖模型（可选） | 自动从 run.json 读取 |
| `--device` | 计算设备 | 自动选择 |
| `--force` | 强制重新计算 | False |

---

## eval_attack_results.py

汇总评估攻击结果，输出对比表格。

### 输出指标

| 指标 | 说明 |
|------|------|
| ASR | 攻击成功率 |
| PPL | 后缀困惑度（低=自然语言） |
| Tokens | 后缀 token 数量 |
| DecMatch | 解码文本匹配率（embedding 攻击） |
| DecPPL | 解码后文本的困惑度 |

### 使用方法

```bash
# 基础用法
python scripts/eval_attack_results.py \
    --results_dir outputs/ \
    --recursive

# 指定 Judge
python scripts/eval_attack_results.py \
    --results_dir outputs/ \
    --recursive \
    --judge strong_reject \
    --threshold 0.5

# 输出 JSON 格式
python scripts/eval_attack_results.py \
    --results_dir outputs/ \
    --recursive \
    --output_format json

# 保存结果到文件
python scripts/eval_attack_results.py \
    --results_dir outputs/ \
    --recursive \
    --output_file results.json
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--results_dir` | 结果目录 | - |
| `--results_file` | 单个结果文件 | - |
| `--recursive` | 递归搜索 | False |
| `--judge` | Judge 名称 | `harmbench` |
| `--threshold` | ASR 阈值 | `0.5` |
| `--output_format` | 输出格式 (`table`/`json`) | `table` |
| `--output_file` | 保存结果到文件 | - |

---

## 完整工作流

```bash
# 1. 运行攻击
python run_attacks.py attack=natural_suffix_embedding model=Qwen/Qwen3-8B dataset=adv_behaviors

# 2. 运行 Judge（计算 ASR）
python run_judges.py classifier=strong_reject suffixes=...

# 3. 计算后缀指标
python scripts/compute_suffix_metrics.py --results_dir outputs/ --recursive

# 4. 汇总评估
python scripts/eval_attack_results.py --results_dir outputs/ --recursive
```

---

## 输出示例

```
====================================================================================================
COMPARISON TABLE
====================================================================================================
Method                              ASR       PPL    Tokens   DecMatch     DecPPL
----------------------------------------------------------------------------------------------------
natural_suffix_embedding           65.0%      15.2       20      98.0%       15.2
gcg                                55.0%    1523.4       20        N/A        N/A
pgd                                80.0%      12.3       20       0.0%     2341.5
====================================================================================================
```

**指标解读**：
- **Ours (natural_suffix_embedding)**: 低 PPL（自然语言）+ 高 DecMatch（解码一致）
- **GCG**: 高 PPL（乱码后缀）
- **PGD**: DecMatch=0% 表示 embedding 已偏离原始 token，DecPPL 高表示解码后是乱码
