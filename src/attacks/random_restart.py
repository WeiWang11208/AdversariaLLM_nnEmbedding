"""
Random Restart (RR) Attack: Embedding-based gradient optimization attack

This attack optimizes adversarial suffixes in the continuous embedding space,
then projects back to discrete tokens. It uses checkpoints to save best results
at different loss thresholds.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from transformers import (
    LlamaForCausalLM,
    FalconForCausalLM,
    MistralForCausalLM,
)

from .attack import (
    Attack,
    AttackResult,
    AttackStepResult,
    GenerationConfig,
    SingleAttackRunResult,
)
from ..dataset import PromptDataset
from ..lm_utils import generate_ragged_batched, prepare_conversation
from ..types import Conversation


@dataclass
class RandomRestartConfig:
    """Configuration for Random Restart attack."""
    name: str = "random_restart"
    type: str = "continuous"
    version: str = "0.0.1"
    placement: str = "suffix"
    seed: int = 0
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)

    # Optimization parameters
    num_steps: int = 500
    initial_lr: float = 0.1
    weight_decay: float = 0.0
    decay_rate: float = 0.99
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"

    # Checkpoint thresholds for saving best results
    checkpoints: list[float] = field(
        default_factory=lambda: [10.0, 5.0, 2.0, 1.0, 0.5]
    )

    # Token filtering
    allow_non_ascii: bool = False
    allow_special: bool = False

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Initialization noise
    init_noise_std: float = 0.1

    # Logging embeddings for realizability analysis
    log_embeddings: bool = False


class RandomRestartAttack(Attack):
    """Random Restart attack using continuous embedding optimization."""

    def __init__(self, config: RandomRestartConfig):
        super().__init__(config)
        self.logger = logging.getLogger("random_restart")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def run(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataset: PromptDataset,
    ) -> AttackResult:
        """Run the Random Restart attack on the dataset."""
        runs = []
        for conversation in dataset:
            run_result = self._attack_single_conversation(
                model, tokenizer, conversation
            )
            runs.append(run_result)
        return AttackResult(runs=runs)

    def _attack_single_conversation(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        conversation: Conversation,
    ) -> SingleAttackRunResult:
        """Run the attack on a single conversation."""
        t0 = time.time()

        # Prepare tokens
        embed_weights = self._get_embedding_matrix(model)

        # Get user prompt tokens
        user_prompt = conversation[0]["content"]
        user_prompt_ids = tokenizer(user_prompt, return_tensors="pt")["input_ids"].to(model.device)

        # Get adversarial string initialization tokens
        adv_string_init_ids = tokenizer(
            self.config.optim_str_init, return_tensors="pt"
        )["input_ids"].to(model.device)

        # Get target tokens
        target = conversation[1]["content"]
        target_ids = tokenizer(target, return_tensors="pt")["input_ids"].to(model.device)

        # Remove BOS token for certain models
        if not isinstance(model, FalconForCausalLM):
            if adv_string_init_ids.size(1) > 1:
                adv_string_init_ids = adv_string_init_ids[:, 1:]
            if target_ids.size(1) > 1:
                target_ids = target_ids[:, 1:]

        # Create embeddings
        embeddings_user = embed_weights[user_prompt_ids[0]]
        embeddings_adv = embed_weights[adv_string_init_ids[0]]
        embeddings_target = embed_weights[target_ids[0]]

        # Add initialization noise
        embeddings_adv = embeddings_adv + torch.normal(
            0, self.config.init_noise_std, embeddings_adv.size()
        ).to(model.device)

        embeddings_adv.requires_grad = True

        # Setup optimizer
        optimizer = optim.AdamW(
            [embeddings_adv],
            lr=self.config.initial_lr,
            weight_decay=self.config.weight_decay,
        )

        # Get non-ASCII tokens if needed
        if not self.config.allow_non_ascii:
            non_ascii_toks = self._get_nonascii_toks(tokenizer, model.device)
        else:
            non_ascii_toks = None

        # Initialize tracking variables
        best_discrete_losses = [float("inf")] * len(self.config.checkpoints)
        best_strings = [""] * len(self.config.checkpoints)
        distances = [float("inf")] * len(self.config.checkpoints)
        # Store continuous embeddings at each checkpoint for realizability analysis
        checkpoint_embeddings: list[torch.Tensor | None] = [None] * len(self.config.checkpoints)

        all_losses = []
        all_suffixes = []

        # Optimization loop
        for iteration in range(self.config.num_steps):
            optimizer.zero_grad()

            # Calculate loss on continuous embeddings
            loss = self._calc_ce_loss(
                model,
                embeddings_user.unsqueeze(0),
                embeddings_adv.unsqueeze(0),
                embeddings_target.unsqueeze(0),
                target_ids[0],
            )
            loss_value = loss.detach().cpu().item()
            loss.backward()

            # Find closest discrete embeddings
            with torch.no_grad():
                closest_distances, closest_indices, closest_embeddings = (
                    self._find_closest_embeddings(
                        embeddings_adv.unsqueeze(0),
                        embed_weights,
                        model.device,
                        self.config.allow_non_ascii,
                        non_ascii_toks,
                    )
                )

                current_suffix = tokenizer.decode(closest_indices[0])
                current_mean_distance = closest_distances.mean().cpu().item()

                # Calculate discrete loss
                discrete_loss = self._calc_ce_loss(
                    model,
                    embeddings_user.unsqueeze(0),
                    closest_embeddings,
                    embeddings_target.unsqueeze(0),
                    target_ids[0],
                )
                discrete_loss_value = discrete_loss.cpu().item()

            all_losses.append(discrete_loss_value)
            all_suffixes.append(current_suffix)

            # Save checkpoints when loss crosses thresholds
            for i, checkpoint in enumerate(self.config.checkpoints):
                if loss_value < checkpoint and best_discrete_losses[i] == float("inf"):
                    best_discrete_losses[i] = discrete_loss_value
                    best_strings[i] = current_suffix
                    distances[i] = current_mean_distance
                    # Save continuous embeddings for realizability analysis
                    if self.config.log_embeddings:
                        # Concatenate full input: user + adv + target embeddings
                        full_embeds = torch.cat([
                            embeddings_user,
                            embeddings_adv.detach(),
                            embeddings_target
                        ], dim=0).cpu()
                        checkpoint_embeddings[i] = full_embeds
                    self.logger.info(
                        f"Checkpoint {i}: loss={loss_value:.4f}, "
                        f"discrete_loss={discrete_loss_value:.4f}, "
                        f"suffix={current_suffix[:50]}"
                    )
                    # Early stop if all checkpoints are filled
                    if best_discrete_losses[-1] != float("inf"):
                        self.logger.info("All checkpoints reached. Stopping early.")
                        break

            # Check if we should stop early
            if best_discrete_losses[-1] != float("inf"):
                break

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [embeddings_adv], max_norm=self.config.max_grad_norm
            )

            # Optimization step
            optimizer.step()

            # Adjust learning rate
            optimizer.param_groups[0]["lr"] = self._adjust_learning_rate(
                self.config.initial_lr, iteration, self.config.decay_rate
            )

            if iteration % 50 == 0:
                self.logger.info(
                    f"Iteration {iteration}: loss={loss_value:.4f}, "
                    f"discrete_loss={discrete_loss_value:.4f}"
                )

        # Fill remaining checkpoints with last values if not reached
        last_discrete_loss = all_losses[-1] if all_losses else float("inf")
        last_suffix = all_suffixes[-1] if all_suffixes else ""
        last_distance = distances[-1] if any(d != float("inf") for d in distances) else 0.0

        for i in range(len(self.config.checkpoints)):
            if best_discrete_losses[i] == float("inf"):
                best_discrete_losses[i] = last_discrete_loss
                best_strings[i] = last_suffix
                distances[i] = last_distance
                # Also save last embedding if logging is enabled and not yet saved
                if self.config.log_embeddings and checkpoint_embeddings[i] is None:
                    full_embeds = torch.cat([
                        embeddings_user,
                        embeddings_adv.detach(),
                        embeddings_target
                    ], dim=0).cpu()
                    checkpoint_embeddings[i] = full_embeds

        # Generate completions for best suffixes
        steps = self._generate_completions(
            model,
            tokenizer,
            conversation,
            best_strings,
            best_discrete_losses,
            distances,
            checkpoint_embeddings if self.config.log_embeddings else None,
        )

        t1 = time.time()
        return SingleAttackRunResult(
            original_prompt=conversation,
            steps=steps,
            total_time=t1 - t0,
        )

    def _generate_completions(
        self,
        model,
        tokenizer,
        conversation,
        suffixes,
        losses,
        distances,
        embeddings=None,
    ):
        """Generate completions for all checkpoint suffixes."""
        steps = []
        attack_conversations = []
        token_list = []

        for suffix in suffixes:
            attack_conv = [
                {"role": "user", "content": conversation[0]["content"] + suffix},
                {"role": "assistant", "content": ""},
            ]
            attack_conversations.append(attack_conv)

            # Prepare tokens
            tokens = prepare_conversation(tokenizer, conversation, attack_conv)[0]
            token_list.append(torch.cat(tokens[:5]))

        # Generate completions in batch
        batch_completions = generate_ragged_batched(
            model,
            tokenizer,
            token_list=token_list,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
            initial_batch_size=len(token_list),
        )

        # Create step results
        for i, (suffix, loss, distance) in enumerate(zip(suffixes, losses, distances)):
            # Get embedding for this checkpoint if available
            step_embedding = embeddings[i] if embeddings is not None else None
            step = AttackStepResult(
                step=i,
                model_completions=batch_completions[i],
                time_taken=0.0,  # Time is tracked per conversation
                loss=loss,
                flops=0,
                model_input=attack_conversations[i],
                model_input_tokens=token_list[i].tolist(),
                model_input_embeddings=step_embedding,
            )
            steps.append(step)

        return steps

    def _get_embedding_matrix(self, model):
        """Get embedding matrix from model."""
        if isinstance(model, LlamaForCausalLM):
            return model.model.embed_tokens.weight
        elif isinstance(model, FalconForCausalLM):
            return model.get_input_embeddings().weight.data
        elif isinstance(model, MistralForCausalLM):
            return model.model.embed_tokens.weight
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            # MPT and similar models
            return model.transformer.wte.weight
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            # Generic fallback
            return model.model.embed_tokens.weight
        else:
            raise ValueError(f"Unknown model type: {type(model)}")

    def _get_nonascii_toks(self, tokenizer, device):
        """Get non-ASCII token IDs."""
        def is_ascii(s):
            return s.isascii() and s.isprintable()

        non_ascii_toks = []
        for i in range(3, tokenizer.vocab_size):
            try:
                decoded = tokenizer.decode([i])
                if not is_ascii(decoded):
                    non_ascii_toks.append(i)
            except Exception:
                non_ascii_toks.append(i)

        # Add special tokens
        if tokenizer.bos_token_id is not None:
            non_ascii_toks.append(tokenizer.bos_token_id)
        if tokenizer.eos_token_id is not None:
            non_ascii_toks.append(tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            non_ascii_toks.append(tokenizer.pad_token_id)
        if tokenizer.unk_token_id is not None:
            non_ascii_toks.append(tokenizer.unk_token_id)

        return torch.tensor(non_ascii_toks, device=device)

    def _find_closest_embeddings(
        self,
        embeddings_adv,
        embed_weights,
        device,
        allow_non_ascii=True,
        non_ascii_toks=None,
    ):
        """Find closest discrete embeddings to continuous embeddings."""
        def normalize(v):
            return v / (torch.norm(v, p=2, dim=-1, keepdim=True) + 1e-8)

        embeddings_adv_norm = normalize(embeddings_adv)
        embed_weights_norm = normalize(embed_weights)

        # Calculate distances
        distances = torch.cdist(embeddings_adv_norm, embed_weights_norm, p=2)

        # Mask non-ASCII tokens if needed
        if not allow_non_ascii and non_ascii_toks is not None:
            distances[0][:, non_ascii_toks] = float("inf")

        # Find closest
        closest_distances, closest_indices = torch.min(distances, dim=-1)
        closest_embeddings = embed_weights[closest_indices]

        return closest_distances, closest_indices, closest_embeddings

    def _calc_ce_loss(
        self,
        model,
        embeddings_user,
        embeddings_adv,
        embeddings_target,
        target_ids,
    ):
        """Calculate cross-entropy loss."""
        # Concatenate embeddings
        full_embeddings = torch.cat(
            [embeddings_user, embeddings_adv, embeddings_target], dim=1
        ).to(dtype=model.dtype)

        # Forward pass
        logits = model(inputs_embeds=full_embeddings).logits

        # Calculate loss on target tokens
        loss_slice_start = embeddings_user.size(1) + embeddings_adv.size(1)
        loss = nn.CrossEntropyLoss()(
            logits[0, loss_slice_start - 1 : -1, :], target_ids
        )

        return loss

    def _adjust_learning_rate(self, initial_lr, iteration, decay_rate):
        """Adjust learning rate with exponential decay."""
        return initial_lr * (decay_rate ** iteration)
