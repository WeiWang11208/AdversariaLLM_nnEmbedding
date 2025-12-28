"""
@article{chao2024jailbreakbench,
  title={Jailbreakbench: An open robustness benchmark for jailbreaking large language models},
  author={Chao, Patrick and Debenedetti, Edoardo and Robey, Alexander and Andriushchenko, Maksym and Croce, Francesco and Sehwag, Vikash and Dobriban, Edgar and Flammarion, Nicolas and Pappas, George J and Tramer, Florian and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={55005--55029},
  year={2024}
}
"""

from dataclasses import dataclass
import os
import pandas as pd
# Removed import jailbreakbench to avoid litellm dependency issues

from src.types import Conversation
from .prompt_dataset import PromptDataset

@dataclass
class JBBBehaviorsConfig:
    name: str = "jbb_behaviors"
    seed: int = 0
    idx: list[int] | int | str | None = None
    shuffle: bool = True

@PromptDataset.register("jbb_behaviors")
class JBBBehaviorsDataset(PromptDataset):
    def __init__(self, config: JBBBehaviorsConfig):
        super().__init__(config)
        
        # Scheme B: Load directly from HuggingFace or local cache without jailbreakbench
        cache_path = "data/behavior_datasets/jbb_behaviors.csv"
        
        if os.path.exists(cache_path):
            dataset = pd.read_csv(cache_path)
        else:
            try:
                from datasets import load_dataset
                print(f"Downloading JBB-Behaviors from HuggingFace...")
                hf_dataset = load_dataset("dedeswim/JBB-Behaviors", "behaviors", split="harmful")
                dataset = hf_dataset.to_pandas()
                # Optional: cache it locally for future offline use
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                dataset.to_csv(cache_path, index=False)
            except Exception as e:
                raise ImportError(
                    f"Could not load JBB-Behaviors dataset. Network error or datasets not installed. "
                    f"Original error: {e}. \n"
                    f"Please ensure you have a stable network or provide {cache_path} manually."
                ) from e

        self.idx, self.config_idx = self._select_idx(config, len(dataset))

        self.messages = dataset.Goal.iloc[self.idx].reset_index(drop=True)
        self.targets = dataset.Target.iloc[self.idx].reset_index(drop=True)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> Conversation:
        msg = self.messages.iloc[idx]
        target = self.targets.iloc[idx]
        conversation = [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": target},
        ]
        return conversation
