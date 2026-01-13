# Idea 3 实现计划：Embedding 可实现性检测

## 核心目标

检测 **embedding-space 攻击**（如 PGD、Random Restart）：通过判断输入 embedding 是否"可以被合法 token 序列解释"来识别 off-manifold 攻击。

## 核心假设

1. **H1**: Embedding-space 攻击的 embedding 在可实现性分数上显著异于正常输入
2. **H2**: 攻击者如果强行投影回可实现集合，会显著损失攻击成功率
3. **H3**: 该检测对 token-space 攻击（GCG）误报率低

---

## 方法设计

### 三档可实现性检测

**Level 1: 逐 token 最近邻重构误差**
- 对每个位置的 embedding $e_t$，在词表中找最近邻：$\delta_t = \|e_t - W_{i^*}\|_2$
- 聚合指标：`mean(δ)`, `max(δ)`, `p90(δ)`

**Level 2: 序列级全局解释 (Viterbi)**
- Top-K 近邻 + 动态规划
- 发射代价：$\|e_t - W_i\|^2$
- 转移代价：语言模型先验（可选）
- 输出：最优路径的总代价

**Level 3: 可逆一致性**
- 检查 special tokens 的 embedding 精确匹配
- Token decode-encode 循环一致性

---

## 实验流程

### Phase A: 数据收集

#### A1. Benign (正常查询)

```bash
# 使用 alpaca 数据集收集正常查询
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=alpaca \
    attack=direct
```

#### A2. Refusal (直接问有害问题 - 模型拒绝)

```bash
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    attack=direct
```

#### A3. Token-space attacks (离散攻击 - 用于检验误报率)

```bash
# GCG 攻击
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    attack=gcg \
    attacks.gcg.num_steps=50

# BEAST 攻击
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    attack=beast

# Random Search
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    attack=random_search
```

#### A4. Embedding-space attacks (连续攻击 - 检测目标)

**重要**: 必须设置 `log_embeddings=true` 来保存原始 embedding 向量！

```bash
# PGD 攻击 (默认 epsilon)
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="list(range(0,50))"' \
    attack=pgd \
    attacks.pgd.num_steps=100 \
    attacks.pgd.log_embeddings=true \
    generation_config.generate_completions=best

# Random Restart 攻击
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="list(range(0,50))"' \
    attack=random_restart \
    attacks.random_restart.num_steps=500 \
    attacks.random_restart.log_embeddings=true
```

#### A5. 不同 epsilon 的 PGD (消融用)

```bash
# 小 epsilon (更接近可实现集合)
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="list(range(0,50))"' \
    attack=pgd \
    attacks.pgd.epsilon=0.5 \
    attacks.pgd.log_embeddings=true \
    generation_config.generate_completions=best

# 中等 epsilon
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="list(range(0,50))"' \
    attack=pgd \
    attacks.pgd.epsilon=1.0 \
    attacks.pgd.log_embeddings=true \
    generation_config.generate_completions=best

# 大 epsilon (更 off-manifold)
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="list(range(0,50))"' \
    attack=pgd \
    attacks.pgd.epsilon=2.0 \
    attacks.pgd.log_embeddings=true \
    generation_config.generate_completions=best

# 非常大 epsilon
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="list(range(0,50))"' \
    attack=pgd \
    attacks.pgd.epsilon=5.0 \
    attacks.pgd.log_embeddings=true \
    generation_config.generate_completions=best
```

#### A6. 批量运行脚本

```bash
#!/bin/bash
# scripts/collect_realizability_data.sh

MODEL="Qwen/Qwen3-8B"
SAMPLE_RANGE='list(range(0,50))'

echo "=== Phase A: 数据收集 ==="

# A1. Benign
echo "[A1] 收集 Benign 数据..."
python run_attacks.py \
    model=$MODEL \
    dataset=alpaca \
    attack=direct

# A2. Refusal
echo "[A2] 收集 Refusal 数据..."
python run_attacks.py \
    model=$MODEL \
    dataset=adv_behaviors \
    "datasets.adv_behaviors.idx=\"$SAMPLE_RANGE\"" \
    attack=direct

# A3. Token-space attacks
echo "[A3] 运行 Token-space 攻击..."
for attack in gcg beast random_search; do
    echo "  Running $attack..."
    python run_attacks.py \
        model=$MODEL \
        dataset=adv_behaviors \
        "datasets.adv_behaviors.idx=\"$SAMPLE_RANGE\"" \
        attack=$attack
done

# A4. Embedding-space attacks
echo "[A4] 运行 Embedding-space 攻击..."

# PGD
python run_attacks.py \
    model=$MODEL \
    dataset=adv_behaviors \
    "datasets.adv_behaviors.idx=\"$SAMPLE_RANGE\"" \
    attack=pgd \
    attacks.pgd.num_steps=100 \
    attacks.pgd.log_embeddings=true \
    generation_config.generate_completions=best

# Random Restart
python run_attacks.py \
    model=$MODEL \
    dataset=adv_behaviors \
    "datasets.adv_behaviors.idx=\"$SAMPLE_RANGE\"" \
    attack=random_restart \
    attacks.random_restart.num_steps=500 \
    attacks.random_restart.log_embeddings=true

# A5. Different epsilon
echo "[A5] 运行不同 epsilon 的 PGD..."
for eps in 0.5 1.0 2.0 5.0; do
    echo "  Running PGD with epsilon=$eps..."
    python run_attacks.py \
        model=$MODEL \
        dataset=adv_behaviors \
        "datasets.adv_behaviors.idx=\"$SAMPLE_RANGE\"" \
        attack=pgd \
        attacks.pgd.epsilon=$eps \
        attacks.pgd.log_embeddings=true \
        generation_config.generate_completions=best
done

echo "=== 数据收集完成 ==="
```

---

### Phase B: 特征提取

创建特征提取脚本 `scripts/realizability/extract_features.py`:

```python
"""
特征提取脚本：从攻击结果中提取可实现性特征

使用方法:
    python scripts/realizability/extract_features.py \
        --data-dir outputs/2026-01-06 \
        --model-path /mnt/public/open_source_model/Qwen3-8B \
        --output features.json \
        --level 1,2
"""

import argparse
import json
import torch
import safetensors.torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_embedding_matrix(model_path):
    """加载模型的 embedding 矩阵"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # 获取 embedding 矩阵
    if hasattr(model.model, 'embed_tokens'):
        embed_matrix = model.model.embed_tokens.weight.detach()
    elif hasattr(model, 'get_input_embeddings'):
        embed_matrix = model.get_input_embeddings().weight.detach()
    else:
        raise ValueError("Cannot find embedding matrix")

    return embed_matrix, model


def compute_level1_features(embeddings, embed_matrix):
    """
    Level 1: 逐 token 最近邻重构误差

    Args:
        embeddings: (seq_len, hidden_dim) 输入 embedding
        embed_matrix: (vocab_size, hidden_dim) 词表 embedding

    Returns:
        dict with mean_nn_l2, max_nn_l2, p90_nn_l2, p99_nn_l2
    """
    # 计算每个位置到词表所有 token 的 L2 距离
    # embeddings: (T, D), embed_matrix: (V, D)
    # distances: (T, V)
    distances = torch.cdist(embeddings.float(), embed_matrix.float(), p=2)

    # 每个位置的最近邻距离
    min_distances, _ = distances.min(dim=1)  # (T,)

    return {
        "mean_nn_l2": min_distances.mean().item(),
        "max_nn_l2": min_distances.max().item(),
        "p90_nn_l2": torch.quantile(min_distances, 0.9).item(),
        "p99_nn_l2": torch.quantile(min_distances, 0.99).item(),
        "min_nn_l2": min_distances.min().item(),
        "std_nn_l2": min_distances.std().item(),
    }


def compute_level2_features(embeddings, embed_matrix, tokenizer, top_k=10):
    """
    Level 2: 序列级全局解释 (简化版 Viterbi)

    使用 Top-K 近邻和贪心解码
    """
    # 计算距离
    distances = torch.cdist(embeddings.float(), embed_matrix.float(), p=2)

    # 贪心选择最近邻
    min_distances, closest_tokens = distances.min(dim=1)
    total_cost = min_distances.sum().item()

    # 解码重构的 token 序列
    reconstructed_text = tokenizer.decode(closest_tokens.tolist())

    return {
        "seq_total_cost": total_cost,
        "seq_mean_cost": total_cost / len(embeddings),
        "reconstructed_text": reconstructed_text[:100],  # 截断
    }


def compute_level3_features(embeddings, embed_matrix, attention_mask=None):
    """
    Level 3: 可逆一致性检查

    检查特殊 token 位置的 embedding 是否精确匹配
    """
    # 计算每个位置的最近邻
    distances = torch.cdist(embeddings.float(), embed_matrix.float(), p=2)
    min_distances, closest_tokens = distances.min(dim=1)

    # 检查是否有完全匹配的 token (距离接近 0)
    exact_matches = (min_distances < 1e-5).sum().item()
    total_tokens = len(embeddings)

    return {
        "exact_match_ratio": exact_matches / total_tokens,
        "exact_match_count": exact_matches,
        "total_tokens": total_tokens,
    }


def extract_features_from_run(run_path, embed_matrix, tokenizer, levels=[1, 2, 3]):
    """从单个 run 中提取特征"""
    run_path = Path(run_path)

    # 读取 run.json
    with open(run_path / "run.json", "r") as f:
        run_data = json.load(f)

    features_list = []

    for run in run_data.get("runs", []):
        for step in run.get("steps", []):
            embed_path = step.get("model_input_embeddings")

            if embed_path is None:
                continue

            # 加载 embedding
            if isinstance(embed_path, str):
                # 从 safetensors 文件加载
                tensors = safetensors.torch.load_file(embed_path)
                embeddings = tensors["embeddings"]
            else:
                embeddings = torch.tensor(embed_path)

            # 确保在正确的设备上
            embeddings = embeddings.to(embed_matrix.device)

            features = {
                "step": step.get("step"),
                "loss": step.get("loss"),
                "attack": run_data["config"]["attack"],
            }

            # 计算各级特征
            if 1 in levels:
                features.update(compute_level1_features(embeddings, embed_matrix))
            if 2 in levels:
                features.update(compute_level2_features(embeddings, embed_matrix, tokenizer))
            if 3 in levels:
                features.update(compute_level3_features(embeddings, embed_matrix))

            features_list.append(features)

    return features_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="攻击结果目录")
    parser.add_argument("--model-path", required=True, help="模型路径")
    parser.add_argument("--output", default="features.json", help="输出文件")
    parser.add_argument("--level", default="1,2,3", help="要计算的特征级别")
    args = parser.parse_args()

    levels = [int(l) for l in args.level.split(",")]

    # 加载模型
    print(f"加载模型: {args.model_path}")
    embed_matrix, model = load_embedding_matrix(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 遍历所有 run 目录
    data_dir = Path(args.data_dir)
    all_features = []

    for run_dir in tqdm(list(data_dir.glob("**/run.json"))):
        run_path = run_dir.parent
        try:
            features = extract_features_from_run(run_path, embed_matrix, tokenizer, levels)
            all_features.extend(features)
        except Exception as e:
            print(f"处理 {run_path} 失败: {e}")

    # 保存特征
    with open(args.output, "w") as f:
        json.dump(all_features, f, indent=2)

    print(f"提取了 {len(all_features)} 个样本的特征，保存到 {args.output}")


if __name__ == "__main__":
    main()
```

使用方法:

```bash
# 提取 Level 1 特征 (最快)
python scripts/realizability/extract_features.py \
    --data-dir outputs/2026-01-06 \
    --model-path /mnt/public/open_source_model/Qwen3-8B \
    --output features_level1.json \
    --level 1

# 提取所有级别特征
python scripts/realizability/extract_features.py \
    --data-dir outputs/2026-01-06 \
    --model-path /mnt/public/open_source_model/Qwen3-8B \
    --output features_all.json \
    --level 1,2,3
```

---

### Phase C: 训练检测器

创建检测器训练脚本 `scripts/realizability/train_detector.py`:

```python
"""
检测器训练脚本

使用方法:
    python scripts/realizability/train_detector.py \
        --features features.json \
        --output detector_results.json
"""

import argparse
import json
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


def load_features(path):
    """加载特征"""
    with open(path) as f:
        return json.load(f)


def prepare_data(features):
    """
    准备训练数据

    标签:
        - benign/refusal/token-space attacks -> 0 (正常/可实现)
        - embedding-space attacks (pgd, random_restart) -> 1 (异常/不可实现)
    """
    X = []
    y = []

    for f in features:
        attack = f.get("attack", "")

        # 特征向量
        feature_vec = [
            f.get("mean_nn_l2", 0),
            f.get("max_nn_l2", 0),
            f.get("p90_nn_l2", 0),
            f.get("p99_nn_l2", 0),
            f.get("std_nn_l2", 0),
        ]

        # 标签
        if attack in ["pgd", "random_restart"]:
            label = 1  # embedding-space attack
        else:
            label = 0  # benign or token-space attack

        X.append(feature_vec)
        y.append(label)

    return np.array(X), np.array(y)


def evaluate_threshold_detector(X, y, feature_idx=0, feature_name="mean_nn_l2"):
    """阈值检测器评估"""
    scores = X[:, feature_idx]

    # ROC-AUC
    auc_score = roc_auc_score(y, scores)

    # 找到不同 FPR 下的阈值和 TPR
    results = {"feature": feature_name, "auc": auc_score}

    # 计算在不同 FPR 目标下的 TPR
    for target_fpr in [0.01, 0.05, 0.10]:
        # 找到使 FPR <= target_fpr 的最大阈值
        benign_scores = scores[y == 0]
        threshold = np.percentile(benign_scores, 100 * (1 - target_fpr))

        # 计算 TPR
        attack_scores = scores[y == 1]
        tpr = (attack_scores > threshold).mean()

        results[f"tpr@fpr{int(target_fpr*100)}%"] = tpr
        results[f"threshold@fpr{int(target_fpr*100)}%"] = threshold

    return results


def train_ml_detector(X, y, model_type="logistic"):
    """训练 ML 检测器"""
    if model_type == "logistic":
        model = LogisticRegression(random_state=42)
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "lgbm":
        model = lgb.LGBMClassifier(random_state=42, verbose=-1)

    # 简单的 train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, y_pred_proba)

    return {
        "model": model_type,
        "auc": auc_score,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", default="detector_results.json")
    args = parser.parse_args()

    # 加载特征
    features = load_features(args.features)
    X, y = prepare_data(features)

    print(f"数据: {len(X)} 样本, {y.sum()} 正样本 (embedding attacks)")

    results = {"n_samples": len(X), "n_positive": int(y.sum())}

    # 阈值检测器
    feature_names = ["mean_nn_l2", "max_nn_l2", "p90_nn_l2", "p99_nn_l2", "std_nn_l2"]
    results["threshold_detectors"] = []
    for i, name in enumerate(feature_names):
        res = evaluate_threshold_detector(X, y, feature_idx=i, feature_name=name)
        results["threshold_detectors"].append(res)
        print(f"{name}: AUC={res['auc']:.4f}, TPR@FPR5%={res['tpr@fpr5%']:.4f}")

    # ML 检测器
    results["ml_detectors"] = []
    for model_type in ["logistic", "rf", "lgbm"]:
        try:
            res = train_ml_detector(X, y, model_type)
            results["ml_detectors"].append(res)
            print(f"{model_type}: AUC={res['auc']:.4f}")
        except Exception as e:
            print(f"{model_type} 失败: {e}")

    # 保存结果
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"结果保存到 {args.output}")


if __name__ == "__main__":
    main()
```

使用方法:

```bash
python scripts/realizability/train_detector.py \
    --features features_level1.json \
    --output detector_results.json
```

---

### Phase D: 消融实验

#### D1: 攻击强度 vs 可实现性

```bash
# 分析不同 epsilon 下的可实现性分数
python scripts/realizability/ablation_epsilon.py \
    --features features_all.json \
    --output ablation_epsilon.png
```

#### D2: 投影净化防御

```bash
# 测试投影净化后的攻击成功率
python scripts/realizability/ablation_projection.py \
    --data-dir outputs/2026-01-06 \
    --model-path /mnt/public/open_source_model/Qwen3-8B \
    --output projection_results.json
```

#### D3: 自适应攻击 (可选)

修改 PGD 添加可实现性正则项:

```bash
# TODO: 实现自适应攻击
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    attack=pgd_adaptive \
    attacks.pgd_adaptive.realizability_weight=0.5
```

---

## 代码结构

```
src/
├── detectors/                  # 检测器模块
│   ├── __init__.py
│   ├── detector.py            # Detector 基类
│   └── realizability.py       # RealizabilityDetector
├── defense/                    # 防御模块
│   ├── __init__.py
│   └── projection.py          # EmbeddingProjector

scripts/realizability/          # 实验脚本
├── collect_data.sh            # Phase A - 数据收集
├── extract_features.py        # Phase B - 特征提取
├── train_detector.py          # Phase C - 检测器训练
├── ablation_epsilon.py        # Phase D1 - epsilon 消融
├── ablation_projection.py     # Phase D2 - 投影净化
└── analyze_results.py         # 分析和可视化
```

---

## 评估指标

- **ROC-AUC / PR-AUC**: 检测性能
- **FPR @ TPR=95%**: 假阳性率（控制在 5% 以下）
- **TPR @ FPR=5%**: 在低假阳性下的检出率
- **Success Rate**: 攻击成功率（投影净化后应显著下降）
- **Utility Drop**: 对 benign 的性能影响（应 < 5%）

---

## 预期结果

- Level 1: ROC-AUC > 0.9, FPR@TPR95 < 5%
- 投影净化: 攻击成功率下降 50%+
- epsilon↑ → 可实现性分数↑ (trade-off)

---

## Embedding 保存说明

### 已完成的代码修改

1. **PGD** (`src/attacks/pgd.py`):
   - 添加 `log_embeddings` 配置选项
   - 在 `generate_completions="all"/"best"/"last"` 三种模式下都支持保存 embedding
   - Embedding 保存为完整的输入序列 (包含 attack tokens)

2. **Random Restart** (`src/attacks/random_restart.py`):
   - 添加 `log_embeddings` 配置选项
   - 在每个 checkpoint 保存完整的 embedding (user + adv + target)
   - 保存连续优化后的 embedding (用于计算可实现性)

### Embedding 存储格式

- Embedding 会自动保存为 `.safetensors` 文件
- 路径记录在 `run.json` 的 `model_input_embeddings` 字段
- 使用 `safetensors.torch.load_file(path)["embeddings"]` 加载

### 重要配置

```yaml
# attacks.yaml
pgd:
  log_embeddings: true  # 必须设置为 true

random_restart:
  log_embeddings: true  # 必须设置为 true
```

---

## 快速开始

```bash
# 1. 收集数据 (保存 embedding)
bash scripts/realizability/collect_data.sh

# 2. 提取特征
python scripts/realizability/extract_features.py \
    --data-dir outputs/$(date +%Y-%m-%d) \
    --model-path /mnt/public/open_source_model/Qwen3-8B \
    --output features.json

# 3. 训练检测器
python scripts/realizability/train_detector.py \
    --features features.json \
    --output results.json
```

---

## 时间线

- Week 1: 数据收集 + Level 1 实现
- Week 2: Level 2-3 + 初步检测
- Week 3: 消融实验
- Week 4: 整合和撰写

---

## 当前进度

- [x] PGD 代码修改 - 支持保存 embedding
- [x] Random Restart 代码修改 - 支持保存 embedding
- [x] 配置文件更新
- [ ] 数据收集脚本
- [ ] 特征提取脚本
- [ ] 检测器训练脚本
- [ ] 消融实验
