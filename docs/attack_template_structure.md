# AdversariaLLM 攻击方法代码结构文档

本文档详细说明了项目中攻击方法的代码结构，为添加新的攻击方法提供模板指导。

---

## 目录

1. [项目整体结构](#项目整体结构)
2. [核心流程](#核心流程)
3. [Attack基类体系](#attack基类体系)
4. [添加新攻击方法的步骤](#添加新攻击方法的步骤)
5. [配置文件结构](#配置文件结构)
6. [完整示例](#完整示例)

---

## 1. 项目整体结构

```
AdversariaLLM/
├── conf/                          # 配置文件目录
│   ├── config.yaml               # 主配置文件
│   ├── attacks/
│   │   └── attacks.yaml          # 攻击方法配置
│   ├── datasets/
│   │   └── datasets.yaml         # 数据集配置
│   ├── models/
│   │   └── models.yaml           # 模型配置
│   └── paths.yaml                # 路径配置
├── src/
│   ├── attacks/                  # 攻击实现目录
│   │   ├── __init__.py          # 导出Attack和Result类
│   │   ├── attack.py            # Attack基类和结果类
│   │   ├── direct.py            # Direct攻击实现
│   │   ├── gcg.py               # GCG攻击实现
│   │   ├── pair.py              # PAIR攻击实现
│   │   └── ...                  # 其他攻击方法
│   ├── dataset/                 # 数据集实现
│   ├── io_utils/                # IO工具
│   └── lm_utils/                # LM工具函数
├── run_attacks.py               # 主运行脚本
└── run_judges.py                # 评判脚本
```

---

## 2. 核心流程

### 2.1 运行流程图

```
run_attacks.py (main)
    ↓
collect_configs()  # 收集所有配置组合
 

### 2.2 run_attacks.py 关键函数

```python
def collect_configs(cfg: DictConfig) -> list[RunConfig]:
    """收集所有model×dataset×attack的配置组合"""
    models_to_run = select_configs(cfg.models, cfg.model)
    datasets_to_run = select_configs(cfg.datasets, cfg.dataset)
    attacks_to_run = select_configs(cfg.attacks, cfg.attack)

    all_run_configs = []
    for model, model_params in models_to_run:
        for dataset, dataset_params in datasets_to_run:
            for attack, attack_params in attacks_to_run:
                run_config = RunConfig(
                    model, dataset, attack,
                    model_params, dataset_params, attack_params,
                    batch_size=cfg.batch_size
                )
                # 过滤已完成的任务(除非overwrite=True)
                run_config = filter_config(run_config, dset_len, overwrite=cfg.overwrite)
                if run_config is not None:
                    all_run_configs.append(run_config)
    return all_run_configs
```

```python
def run_attacks(all_run_configs: list[RunConfig], cfg: DictConfig, date_time_string: str):
    """执行所有攻击配置"""
    for run_config in all_run_configs:
        # 加载模型(如果变化)
        if last_model != run_config.model:
            model, tokenizer = load_model_and_tokenizer(run_config.model_params)

        # 加载数据集(如果变化)
        if last_dataset != run_config.dataset:
            dataset = PromptDataset.from_name(run_config.dataset)(run_config.dataset_params)

        # 创建Attack实例
        attack: Attack = Attack.from_name(run_config.attack)(run_config.attack_params)

        # 执行攻击(可能分批)
        if batch_size > 0:
            results = AttackResult()
            for start in range(0, len(dataset), batch_size):
                sub_dataset = _DatasetView(dataset, list(range(start, end)))
                sub_results = attack.run(model, tokenizer, sub_dataset)
                results.runs.extend(sub_results.runs)
        else:
            results = attack.run(model, tokenizer, dataset)

        # 记录结果
        log_attack(run_config, results, cfg, date_time_string)
```

---

## 3. Attack基类体系

### 3.1 核心数据类

#### AttackStepResult
存储单步攻击结果，包括：
```python
@dataclass(kw_only=True)
class AttackStepResult:
    step: int                                      # 步骤编号
    model_completions: list[str]                   # 模型生成的完成结果
    scores: dict[str, dict[str, list[float]]]     # 评判分数
    time_taken: float = 0.0                        # 单步耗时
    flops: Optional[int] = None                    # 计算量

    # 可选字段
    loss: Optional[float] = None                   # 损失值
    model_input: Optional[Conversation] = None     # 模型输入(对话)
    model_input_tokens: Optional[list[int]] = None # 输入tokens
    model_input_embeddings: Optional[Union[Tensor, str]] = None  # 输入embeddings
```

#### SingleAttackRunResult
存储单个数据实例的完整攻击结果：
```python
@dataclass
class SingleAttackRunResult:
    original_prompt: Conversation              # 原始prompt
    steps: list[AttackStepResult]             # 所有步骤的结果
    total_time: float = 0.0                   # 总耗时
```

#### AttackResult
存储整个数据集的攻击结果：
```python
@dataclass
class AttackResult:
    runs: list[SingleAttackRunResult] = field(default_factory=list)
```

#### GenerationConfig
生成配置：
```python
@dataclass
class GenerationConfig:
    generate_completions: Literal["all", "best", "last"] = "all"
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    num_return_sequences: int = 1
```

### 3.2 Attack基类

```python
class Attack(Generic[AttRes]):
    def __init__(self, config):
        self.config = config
        transformers.set_seed(config.seed)

    @classmethod
    def from_name(cls, name: str) -> type["Attack"]:
        """根据名称动态加载Attack类"""
        match name:
            case "direct":
                from .direct import DirectAttack
                return DirectAttack
            case "gcg":
                from .gcg import GCGAttack
                return GCGAttack
            case "pair":
                from .pair import PAIRAttack
                return PAIRAttack
            # ... 更多攻击方法
            case _:
                raise ValueError(f"Unknown attack: {name}")

    @abstractmethod
    def run(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataset: PromptDataset,
    ) -> AttRes:
        """执行攻击的核心方法"""
        raise NotImplementedError
```

---

## 4. 添加新攻击方法的步骤

### 步骤1: 创建Config类

在 `src/attacks/your_attack.py` 中定义配置类：

```python
from dataclasses import dataclass, field
from .attack import GenerationConfig

@dataclass
class YourAttackConfig:
    # 必需字段
    name: str = "your_attack"
    type: str = "discrete"  # 或 "continuous" 或 "hybrid"
    version: str = "0.0.1"
    seed: int = 0
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)

    # 攻击特定参数
    num_steps: int = 100
    your_param1: float = 0.5
    your_param2: int = 10
    # ... 更多参数
```

### 步骤2: 实现Attack类

```python
import time
import torch
import transformers
from .attack import Attack, AttackResult, SingleAttackRunResult, AttackStepResult

class YourAttack(Attack):
    def __init__(self, config: YourAttackConfig):
        super().__init__(config)
        # 初始化攻击特定的属性
        self.your_attribute = config.your_param1

    @torch.no_grad  # 如果不需要梯度
    def run(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataset: PromptDataset,
    ) -> AttackResult:
        """
        执行攻击的主要方法

        Parameters:
        -----------
        model: 目标模型
        tokenizer: tokenizer
        dataset: 数据集，每个元素是一个Conversation
                 Conversation = list[dict] with keys 'role' and 'content'

        Returns:
        --------
        AttackResult: 包含所有runs的结果
        """
        runs = []

        # 遍历数据集中的每个conversation
        for conversation in dataset:
            run_result = self._attack_single_conversation(
                model, tokenizer, conversation
            )
            runs.append(run_result)

        return AttackResult(runs=runs)

    def _attack_single_conversation(
        self,
        model,
        tokenizer,
        conversation
    ) -> SingleAttackRunResult:
        """
        对单个conversation执行攻击

        conversation格式:
        [
            {"role": "user", "content": "原始prompt"},
            {"role": "assistant", "content": "目标回复"}
        ]
        """
        t0 = time.time()
        steps = []

        # 提取原始prompt和目标
        original_prompt = conversation
        user_prompt = conversation[0]["content"]
        target_response = conversation[1]["content"]

        # 执行多步优化
        for step_idx in range(self.config.num_steps):
            # 1. 生成/优化adversarial prompt
            adversarial_prompt = self._generate_adversarial_prompt(
                user_prompt, target_response, step_idx
            )

            # 2. 构造模型输入
            model_input = [
                {"role": "user", "content": adversarial_prompt},
                {"role": "assistant", "content": ""}
            ]

            # 3. 使用工具函数准备tokens
            from ..lm_utils import prepare_conversation
            token_tensors = prepare_conversation(tokenizer, model_input)[0]
            prompt_tokens = torch.cat(token_tensors[:-1]).unsqueeze(0).to(model.device)

            # 4. 生成模型回复
            from ..lm_utils import generate_ragged_batched
            completions = generate_ragged_batched(
                model,
                tokenizer,
                token_list=[prompt_tokens[0]],
                max_new_tokens=self.config.generation_config.max_new_tokens,
                temperature=self.config.generation_config.temperature,
                top_p=self.config.generation_config.top_p,
                top_k=self.config.generation_config.top_k,
                num_return_sequences=self.config.generation_config.num_return_sequences,
            )

            # 5. 计算损失(可选)
            loss = self._compute_loss(model, prompt_tokens, target_response)

            # 6. 记录这一步的结果
            step_result = AttackStepResult(
                step=step_idx,
                model_completions=completions[0],
                time_taken=time.time() - t0,
                loss=loss,
                flops=0,  # 如果能计算FLOPS的话
                model_input=model_input,
                model_input_tokens=prompt_tokens[0].tolist(),
            )
            steps.append(step_result)

            # 7. 早停条件(可选)
            if self._should_stop(completions[0], target_response):
                break

        t1 = time.time()
        return SingleAttackRunResult(
            original_prompt=original_prompt,
            steps=steps,
            total_time=t1 - t0
        )

    def _generate_adversarial_prompt(self, user_prompt, target_response, step_idx):
        """生成adversarial prompt的逻辑"""
        # 实现你的攻击算法
        # 例如: 添加后缀、修改prompt、使用优化算法等
        return user_prompt + " your_suffix"

    def _compute_loss(self, model, tokens, target):
        """计算损失(可选)"""
        # 如果你的攻击需要计算损失
        return None

    def _should_stop(self, completions, target):
        """早停条件(可选)"""
        return False
```

### 步骤3: 注册到Attack.from_name()

在 `src/attacks/attack.py` 的 `from_name()` 方法中添加：

```python
@classmethod
def from_name(cls, name: str) -> type["Attack"]:
    match name:
        # ... 现有的case ...
        case "your_attack":
            from .your_attack import YourAttack
            return YourAttack
        case _:
            raise ValueError(f"Unknown attack: {name}")
```

### 步骤4: 添加配置文件

在 `conf/attacks/attacks.yaml` 中添加配置：

```yaml
your_attack:
  name: your_attack
  type: discrete  # discrete/continuous/hybrid
  version: 0.0.1
  seed: ${attacks._default.seed}
  generation_config: ${attacks._default.generation_config}

  # 攻击特定参数
  num_steps: 100
  your_param1: 0.5
  your_param2: 10

  # 如果需要额外的模型(如PAIR)
  # attack_model:
  #   id: model_id
  #   tokenizer_id: tokenizer_id
  #   # ... 其他模型配置
```

### 步骤5: 运行测试

```bash
# 测试单个攻击
python run_attacks.py attack=your_attack dataset=adv_behaviors model=google/gemma-2-2b-it

# 测试多个配置
python run_attacks.py attack=your_attack dataset=adv_behaviors model=google/gemma-2-2b-it,meta-llama/Meta-Llama-3.1-8B-Instruct

# 覆盖参数
python run_attacks.py attack=your_attack attacks.your_attack.num_steps=50
```

---

## 5. 配置文件结构

### 5.1 主配置文件 (conf/config.yaml)

```yaml
defaults:
  - attacks: attacks      # 加载attacks.yaml
  - datasets: datasets    # 加载datasets.yaml
  - models: models        # 加载models.yaml
  - paths                 # 加载paths.yaml
  - override hydra/launcher: basic
  - _self_

hydra:
  run:
    dir: ${root_dir}/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}/
  job:
    chdir: true

name: testing
save_dir: ${root_dir}/outputs/
embed_dir: ${root_dir}/embeddings/
overwrite: false
batch_size: 0  # 0表示不分批，>0表示分批大小

classifiers: ["strong_reject"]  # 评判器列表

# 顶层生成配置(方便覆盖)
generation_config:
  generate_completions: all  # "all", "best", "last"
  temperature: 0.0
  top_p: 1.0
  top_k: 0
  max_new_tokens: 256
  num_return_sequences: 1

# 运行选择(null表示运行所有)
attack: null    # 或指定具体攻击: "direct" 或列表 ["direct", "gcg"]
dataset: adv_behaviors
model: null
```

### 5.2 攻击配置 (conf/attacks/attacks.yaml)

```yaml
_default:
  generation_config: ${generation_config}
  seed: 0

direct:
  name: direct
  type: discrete
  version: 0.0.1
  generation_config: ${attacks._default.generation_config}
  seed: ${attacks._default.seed}

gcg:
  name: gcg
  type: discrete
  version: 0.0.1
  placement: suffix
  generation_config: ${attacks._default.generation_config}
  seed: ${attacks._default.seed}
  num_steps: 50
  batch_size: 512
  optim_str_init: "x x x x x x x x x x x x x x x x x x x x"
  search_width: 512
  topk: 256
  n_replace: 1
  buffer_size: 0
  loss: ce
  use_prefix_cache: True
  allow_non_ascii: False
  allow_special: False
  # ... 更多GCG参数

pair:
  name: pair
  type: discrete
  version: 0.0.2
  seed: ${attacks._default.seed}
  generation_config: ${attacks._default.generation_config}
  num_streams: 1
  keep_last_num: 3
  num_steps: 20
  attack_model:
    id: lmsys/vicuna-13b-v1.5
    tokenizer_id: lmsys/vicuna-13b-v1.5
    dtype: bfloat16
    # ... 更多模型配置
  target_model:
    max_new_tokens: 256
    temperature: 0
    top_p: 1
  judge_model:
    id: null
    # ... judge配置
```

### 5.3 数据集配置 (conf/datasets/datasets.yaml)

```yaml
adv_behaviors:
  name: adv_behaviors
  messages_path: ${root_dir}/data/behavior_datasets/harmbench_behaviors_text_all.csv
  targets_path: ${root_dir}/data/optimizer_targets/harmbench_targets_text.json
  categories: ['chemical_biological', 'illegal', 'misinformation_disinformation', 'harmful', 'harassment_bullying', 'cybercrime_intrusion']
  seed: 0
  idx: null      # null表示全部，或指定索引列表/range
  shuffle: true

alpaca:
  name: alpaca
  seed: 0
  idx: null
  shuffle: true
```

### 5.4 模型配置 (conf/models/models.yaml)

```yaml
google/gemma-2-2b-it:
  id: google/gemma-2-2b-it
  tokenizer_id: google/gemma-2-2b-it
  short_name: Gemma
  developer_name: Google
  compile: False
  dtype: bfloat16
  chat_template: gemma-it
  trust_remote_code: True

meta-llama/Meta-Llama-3.1-8B-Instruct:
  id: meta-llama/Meta-Llama-3.1-8B-Instruct
  tokenizer_id: meta-llama/Meta-Llama-3.1-8B-Instruct
  short_name: Llama
  developer_name: Meta
  compile: False
  dtype: bfloat16
  chat_template: llama-3-instruct
  trust_remote_code: True
```

---

## 6. 完整示例

### 6.1 简单攻击示例: Direct Attack

```python
"""Direct Attack: 直接使用原始prompt，不做任何修改"""

import copy
import time
from dataclasses import dataclass, field
import torch
import transformers

from .attack import (Attack, AttackResult, AttackStepResult,
                     GenerationConfig, SingleAttackRunResult)
from ..lm_utils import (generate_ragged_batched, get_losses_batched,
                        prepare_conversation)
from ..types import Conversation


@dataclass
class DirectConfig:
    name: str = "direct"
    type: str = "discrete"
    version: str = ""
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0


class DirectAttack(Attack):
    def __init__(self, config: DirectConfig):
        super().__init__(config)

    @torch.no_grad
    def run(
        self,
        model: transformers.AutoModelForCausalLM,
        tokenizer: transformers.AutoTokenizer,
        dataset: torch.utils.data.Dataset,
    ) -> AttackResult:
        t0 = time.time()

        # 1. 准备输入
        original_conversations: list[Conversation] = []
        prompt_token_tensors_list: list[torch.Tensor] = []

        for conversation in dataset:
            assert len(conversation) == 2
            original_conversations.append(conversation)

            # 准备tokens
            token_tensors = prepare_conversation(tokenizer, conversation)
            flat_tokens = [t for turn_tokens in token_tensors for t in turn_tokens]
            prompt_token_tensors_list.append(torch.cat(flat_tokens[:-1]))

        # 2. 计算损失(可选)
        # ... 损失计算逻辑 ...

        # 3. 生成completions
        completions = generate_ragged_batched(
            model,
            tokenizer,
            token_list=prompt_token_tensors_list,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
        )

        # 4. 组装结果
        runs = []
        t1 = time.time()
        for i in range(len(original_conversations)):
            model_input = copy.deepcopy(original_conversations[i])
            model_input[-1]["content"] = ""

            step_result = AttackStepResult(
                step=0,
                model_completions=completions[i],
                time_taken=(t1 - t0) / len(original_conversations),
                loss=None,
                flops=0,
                model_input=model_input,
                model_input_tokens=prompt_token_tensors_list[i].tolist(),
            )

            run_result = SingleAttackRunResult(
                original_prompt=original_conversations[i],
                steps=[step_result],
                total_time=t1 - t0,
            )
            runs.append(run_result)

        return AttackResult(runs=runs)
```

### 6.2 复杂攻击示例: GCG Attack (简化版)

```python
"""GCG Attack: 基于梯度的后缀优化攻击"""

import time
from dataclasses import dataclass, field
import torch
from .attack import Attack, AttackResult, SingleAttackRunResult, AttackStepResult

@dataclass
class GCGConfig:
    name: str = "gcg"
    type: str = "discrete"
    version: str = ""
    seed: int = 0
    num_steps: int = 250
    optim_str_init: str = "x x x x x x x x x x"
    search_width: int = 512
    topk: int = 256
    # ... 更多参数

class GCGAttack(Attack):
    def __init__(self, config: GCGConfig):
        super().__init__(config)

    def run(self, model, tokenizer, dataset) -> AttackResult:
        runs = []
        for conversation in dataset:
            runs.append(self._attack_single_conversation(model, tokenizer, conversation))
        return AttackResult(runs=runs)

    def _attack_single_conversation(self, model, tokenizer, conversation):
        t0 = time.time()

        # 1. 准备attack conversation (添加后缀)
        attack_conversation = [
            {"role": "user", "content": conversation[0]["content"] + self.config.optim_str_init},
            {"role": "assistant", "content": conversation[1]["content"]},
        ]

        # 2. 准备tokens
        from ..lm_utils import prepare_conversation
        pre_ids, _, prompt_ids, attack_suffix_ids, post_ids, target_ids = \
            prepare_conversation(tokenizer, conversation, attack_conversation)[0]

        # 移到GPU
        prompt_ids = prompt_ids.unsqueeze(0).to(model.device)
        attack_ids = attack_suffix_ids.unsqueeze(0).to(model.device)
        target_ids = target_ids.unsqueeze(0).to(model.device)
        # ...

        # 3. 迭代优化
        optim_strings = []
        losses = []
        for step in range(self.config.num_steps):
            # a. 计算梯度
            grad = self._compute_token_gradient(model, attack_ids, target_ids)

            # b. 采样candidate sequences
            sampled_ids = self._sample_from_gradient(grad, attack_ids)

            # c. 计算candidate losses
            candidate_losses = self._compute_candidates_loss(model, sampled_ids)

            # d. 选择最佳candidate
            best_idx = candidate_losses.argmin()
            attack_ids = sampled_ids[best_idx].unsqueeze(0)
            losses.append(candidate_losses[best_idx].item())

            # e. 记录
            optim_str = tokenizer.decode(attack_ids[0])
            optim_strings.append(optim_str)

        # 4. 生成最终completions
        from ..lm_utils import generate_ragged_batched
        attack_conversations = []
        token_list = []
        for attack_str in optim_strings:
            attack_conv = [
                {"role": "user", "content": conversation[0]["content"] + attack_str},
                {"role": "assistant", "content": ""},
            ]
            attack_conversations.append(attack_conv)
            tokens = prepare_conversation(tokenizer, conversation, attack_conv)[0]
            token_list.append(torch.cat(tokens[:5]))

        batch_completions = generate_ragged_batched(
            model, tokenizer,
            token_list=token_list,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            # ... 其他参数
        )

        # 5. 组装结果
        steps = []
        for i in range(len(optim_strings)):
            step = AttackStepResult(
                step=i,
                model_completions=batch_completions[i],
                time_taken=0,  # 计算实际时间
                loss=losses[i],
                model_input=attack_conversations[i],
                model_input_tokens=token_list[i].tolist(),
            )
            steps.append(step)

        return SingleAttackRunResult(
            original_prompt=conversation,
            steps=steps,
            total_time=time.time() - t0,
        )

    def _compute_token_gradient(self, model, attack_ids, target_ids):
        """计算token gradient"""
        # 实现梯度计算逻辑
        pass

    def _sample_from_gradient(self, grad, attack_ids):
        """从梯度采样candidates"""
        # 实现采样逻辑
        pass

    def _compute_candidates_loss(self, model, sampled_ids):
        """计算候选序列的loss"""
        # 实现loss计算
        pass
```

---

## 7. 常用工具函数

### 7.1 lm_utils中的工具函数

```python
from ..lm_utils import (
    prepare_conversation,      # 准备conversation tokens
    generate_ragged_batched,   # 批量生成(支持不同长度)
    get_losses_batched,        # 批量计算loss
    get_flops,                 # 计算FLOPS
    filter_suffix,             # 过滤无效后缀
    get_disallowed_ids,        # 获取不允许的token ids
)
```

#### prepare_conversation
```python
def prepare_conversation(
    tokenizer,
    original_conversation,
    attack_conversation=None
) -> list[tuple[torch.Tensor, ...]]:
    """
    准备conversation的tokens

    Returns:
    --------
    list of tuples:
        (pre_ids, attack_prefix_ids, prompt_ids, attack_suffix_ids, post_ids, target_ids)
    """
```

#### generate_ragged_batched
```python
def generate_ragged_batched(
    model,
    tokenizer,
    token_list: list[torch.Tensor],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    num_return_sequences: int = 1,
    initial_batch_size: int = None,
    return_tokens: bool = False,
    verbose: bool = False,
) -> list[list[str]]:
    """
    支持不同长度输入的批量生成

    Returns:
    --------
    list[list[str]]: 每个输入对应的多个生成结果
    """
```

### 7.2 类型定义

```python
from ..types import Conversation

# Conversation类型定义
Conversation = list[dict]
# 格式: [{"role": "user"/"assistant"/"system", "content": str}, ...]
```

---

## 8. 调试和日志

### 8.1 日志记录

```python
import logging

# 在你的Attack类中
logging.info("Starting attack...")
logging.warning("Warning message")
logging.error("Error occurred")
```

### 8.2 结果存储

结果会自动保存到:
```
${root_dir}/outputs/
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            └── {model}_{dataset}_{attack}.json
```

JSON格式:
```json
{
    "config": {...},
    "results": {
        "runs": [
            {
                "original_prompt": [...],
                "steps": [
                    {
                        "step": 0,
                        "model_completions": ["..."],
                        "loss": 0.5,
                        "time_taken": 1.2,
                        ...
                    }
                ],
                "total_time": 120.5
            }
        ]
    }
}
```

---

## 9. 最佳实践

1. **配置管理**: 所有超参数都应该在Config类中定义，方便通过命令行覆盖
2. **错误处理**: 在关键位置添加try-except和assertion
3. **内存管理**: 对于大模型，及时清理中间结果
4. **批处理**: 利用`batch_size`参数支持大数据集
5. **可复现性**: 使用`seed`确保结果可复现
6. **文档**: 在代码中添加清晰的注释和docstring
7. **测试**: 先在小数据集上测试，确认无误后再运行完整实验

---

## 10. 命令行使用示例

```bash
# 基本用法
python run_attacks.py attack=your_attack dataset=adv_behaviors model=google/gemma-2-2b-it

# 选择数据集子集
python run_attacks.py attack=gcg dataset=adv_behaviors datasets.adv_behaviors.idx="range(0,10)"

# 覆盖攻击参数
python run_attacks.py attack=gcg attacks.gcg.num_steps=100 attacks.gcg.search_width=256

# 运行多个配置
python run_attacks.py -m attack=direct,gcg dataset=adv_behaviors model=google/gemma-2-2b-it

# 覆盖写入
python run_attacks.py attack=gcg overwrite=true

# 批处理
python run_attacks.py attack=gcg batch_size=32

# 指定分类器
python run_attacks.py attack=gcg classifiers=[strong_reject,harmbench]
```

---

## 附录: 项目特点

1. **Hydra配置管理**: 灵活的配置组合和命令行覆盖
2. **模块化设计**: Attack/Dataset/Model独立管理
3. **结果缓存**: 避免重复运行(通过`overwrite`控制)
4. **批量处理**: 支持大数据集的分批处理
5. **多重评判**: 支持多个classifier同时评判
6. **FLOPS追踪**: 记录计算成本
7. **流式保存**: 对于长时间运行的攻击，支持流式保存中间结果

---

**文档版本**: v1.0
**最后更新**: 2026-01-05
**维护者**: AdversariaLLM Team
