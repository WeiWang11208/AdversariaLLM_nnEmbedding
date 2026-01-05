# Idea 3 实现计划：Embedding 可实现性检测

## 核心目标

检测 **embedding-space 攻击**（如 PGD）：通过判断输入 embedding 是否"可以被合法 token 序列解释"来识别 off-manifold 攻击。

## 核心假设

1. **H1**: PGD 等 embedding-space 攻击的 embedding 在可实现性分数上显著异于正常输入
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

收集四类数据：

```bash
# 1. Benign (正常查询) - 使用 alpaca 数据集
python run_attacks.py \
    model=Qwen/Qwen3-8B \
    dataset=alpaca \
    attack=direct

# 2. Refusal (直接问有害问题)
python run_attacks.py --multirun \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    attack=direct

# 3. Token-space attacks
python run_attacks.py -m \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    attack=gcg,beast,random_search

# 4. Embedding-space attacks
# 一次性运行（可能显存不足）
python run_attacks.py --multirun \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    attack=pgd \
    attacks.pgd.attack_space=embedding \
    attacks.pgd.num_steps=100

# 5. 不同 epsilon 的 PGD (消融用)
# 一次性运行
python run_attacks.py -m \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    attack=pgd \
    attacks.pgd.attack_space=embedding \
    attacks.pgd.epsilon=0.5,1.0,2.0,5.0

### Phase B: 特征提取

```bash
# 收集和组织数据
python scripts/realizability/collect_data.py \
    --model Qwen/Qwen3-8B \
    --output outputs/realizability_data

# 提取特征 (三档)
python scripts/realizability/extract_features.py \
    --data-dir outputs/realizability_data \
    --level 1,2,3
```

### Phase C: 训练检测器

```bash
# 阈值检测器
python scripts/realizability/train_detector.py \
    --data-dir outputs/realizability_data \
    --level 1 \
    --fpr-target 0.01,0.05,0.10

# ML 检测器 (可选)
python scripts/realizability/train_ml_detector.py \
    --data-dir outputs/realizability_data \
    --level 1 \
    --model logistic,rf,lgbm
```

### Phase D: 消融实验

```bash
# D1: 攻击强度 vs 可实现性
python scripts/realizability/ablation_epsilon.py \
    --data-dir outputs/realizability_data

# D2: 投影净化防御
python scripts/realizability/ablation_projection.py \
    --data-dir outputs/realizability_data \
    --projection nearest,weighted

# D3: 自适应攻击
python run_attacks.py -m \
    model=Qwen/Qwen3-8B \
    dataset=adv_behaviors \
    datasets.adv_behaviors.idx="range(0,50)" \
    attack=pgd_adaptive \
    attacks.pgd_adaptive.realizability_weight=0.1,0.5,1.0
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
├── collect_data.py            # Phase A
├── extract_features.py        # Phase B
├── train_detector.py          # Phase C
├── train_ml_detector.py       # Phase C (ML)
├── ablation_epsilon.py        # Phase D1
├── ablation_projection.py     # Phase D2
└── analyze_results.py         # 分析和可视化
```

---

## 评估指标

- **ROC-AUC / PR-AUC**: 检测性能
- **FPR @ TPR=95%**: 假阳性率（控制在 5% 以下）
- **Success Rate**: 攻击成功率（投影净化后应显著下降）
- **Utility Drop**: 对 benign 的性能影响（应 < 5%）

---

## 预期结果

- Level 1: ROC-AUC > 0.9, FPR@TPR95 < 5%
- 投影净化: 攻击成功率下降 50%+
- 自适应攻击: 存在 realizability vs effectiveness trade-off

---

## 时间线

- Week 1: 数据收集 + Level 1 实现
- Week 2: Level 2-3 + 初步检测
- Week 3: 消融实验
- Week 4: 整合和撰写
