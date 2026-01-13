#!/bin/bash

# Random Restart Attack 测试脚本
# 用于快速验证实现是否正确

echo "=========================================="
echo "Random Restart Attack 测试脚本"
echo "=========================================="
echo ""

# 设置错误时退出
set -e

# 默认使用 Qwen/Qwen3-8B 模型（可通过环境变量覆盖）
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
echo "使用模型: $MODEL"
echo ""

# 测试1: 基本功能测试 (在小数据集上)
echo "[测试1] 基本功能测试 - 在前2个样本上运行"
echo "命令: python run_attacks.py model=$MODEL attack=random_restart dataset=adv_behaviors 'datasets.adv_behaviors.idx=\"list(range(0,2))\"' attacks.random_restart.num_steps=50"
echo ""

python run_attacks.py \
    model=$MODEL \
    attack=random_restart \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="list(range(0,2))"' \
    attacks.random_restart.num_steps=50 \
    attacks.random_restart.checkpoints=[5.0,1.0]

echo ""
echo "✓ 测试1 通过"
echo ""

# 测试2: 参数覆盖测试
echo "[测试2] 参数覆盖测试 - 测试不同的学习率和步数"
echo "命令: python run_attacks.py model=$MODEL attack=random_restart dataset=adv_behaviors 'datasets.adv_behaviors.idx=\"list(range(0,1))\"' attacks.random_restart.num_steps=20 attacks.random_restart.initial_lr=0.05"
echo ""

python run_attacks.py \
    model=$MODEL \
    attack=random_restart \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="list(range(0,1))"' \
    attacks.random_restart.num_steps=20 \
    attacks.random_restart.initial_lr=0.05 \
    overwrite=true

echo ""
echo "✓ 测试2 通过"
echo ""

# 测试3: Allow non-ASCII测试
echo "[测试3] Non-ASCII字符测试"
echo "命令: python run_attacks.py model=$MODEL attack=random_restart dataset=adv_behaviors 'datasets.adv_behaviors.idx=\"list(range(0,1))\"' attacks.random_restart.allow_non_ascii=True attacks.random_restart.num_steps=20"
echo ""

python run_attacks.py \
    model=$MODEL \
    attack=random_restart \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="list(range(0,1))"' \
    attacks.random_restart.allow_non_ascii=True \
    attacks.random_restart.num_steps=20 \
    overwrite=true

echo ""
echo "✓ 测试3 通过"
echo ""

# 测试4: 多个checkpoints测试
echo "[测试4] 多checkpoint测试"
echo "命令: python run_attacks.py model=$MODEL attack=random_restart dataset=adv_behaviors 'datasets.adv_behaviors.idx=\"list(range(0,1))\"' attacks.random_restart.checkpoints=[10.0,5.0,2.0] attacks.random_restart.num_steps=30"
echo ""

python run_attacks.py \
    model=$MODEL \
    attack=random_restart \
    dataset=adv_behaviors \
    'datasets.adv_behaviors.idx="list(range(0,1))"' \
    attacks.random_restart.checkpoints=[10.0,5.0,2.0] \
    attacks.random_restart.num_steps=30 \
    overwrite=true

echo ""
echo "✓ 测试4 通过"
echo ""

echo "=========================================="
echo "所有测试通过! ✓"
echo "=========================================="
echo ""
echo "Random Restart攻击已成功集成到框架中"
echo ""
echo "使用方法:"
echo "  python run_attacks.py model=Qwen/Qwen3-8B attack=random_restart dataset=adv_behaviors"
echo ""
echo "快速测试 (2个样本):"
echo "  python run_attacks.py model=Qwen/Qwen3-8B attack=random_restart dataset=adv_behaviors 'datasets.adv_behaviors.idx=\"list(range(0,2))\"' attacks.random_restart.num_steps=100"
echo ""
echo "查看配置:"
echo "  cat conf/attacks/attacks.yaml | grep -A 20 random_restart"
echo ""
echo "查看文档:"
echo "  cat docs/random_restart_attack.md"
echo ""
