import torch
import numpy as np
from typing import Optional
from .detector import Detector, DetectionResult


class RealizabilityDetector(Detector):
    """Detectability detector using realizability scores

    Three levels of detection:
    - Level 1: Per-token nearest neighbor distance
    - Level 2: Sequence-level Viterbi decoding
    - Level 3: Reconstruction consistency
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        level: int = 1,
        topk: int = 20,
        embed_scale: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize realizability detector

        Args:
            embedding_matrix: Vocabulary embedding matrix (vocab_size, hidden_dim)
            level: Detection level (1, 2, or 3)
            topk: Top-K nearest neighbors for level 2
            embed_scale: Embedding scale factor (for Gemma, etc.)
            device: Device to run on
        """
        self.embedding_matrix = embedding_matrix.to(device)
        self.level = level
        self.topk = topk
        self.embed_scale = embed_scale
        self.device = device
        self.threshold = None

    def extract_features(
        self,
        embeddings: torch.Tensor,
        tokens: Optional[list[int]] = None,
        **kwargs
    ) -> dict[str, float]:
        """Extract realizability features"""
        embeddings = embeddings.to(self.device).float()
        features = {}

        # Level 1: Per-token nearest neighbor distances
        if self.level >= 1:
            nn_distances = self._compute_nn_distances(embeddings)
            features.update({
                'mean_nn_l2': nn_distances.mean().item(),
                'max_nn_l2': nn_distances.max().item(),
                'p90_nn_l2': torch.quantile(nn_distances, 0.9).item(),
                'p95_nn_l2': torch.quantile(nn_distances, 0.95).item(),
                'std_nn_l2': nn_distances.std().item(),
            })

        # Level 2: Sequence-level realizability (Viterbi)
        if self.level >= 2:
            seq_cost = self._compute_sequence_realizability(embeddings)
            features['seq_realizability_cost'] = seq_cost
            features['seq_realizability_cost_normalized'] = seq_cost / len(embeddings)

        # Level 3: Reconstruction consistency
        if self.level >= 3 and tokens is not None:
            consistency = self._compute_reconstruction_consistency(embeddings, tokens)
            features['reconstruction_consistency'] = consistency

        return features

    def _compute_nn_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute nearest neighbor distances for each token

        Args:
            embeddings: (seq_len, hidden_dim)

        Returns:
            Tensor of distances (seq_len,)
        """
        # Apply scaling
        embeddings_scaled = embeddings * self.embed_scale
        embed_matrix_scaled = self.embedding_matrix * self.embed_scale

        # Compute pairwise L2 distances
        # Use chunked computation to avoid OOM
        chunk_size = 1024
        min_distances = []

        for i in range(0, len(embeddings), chunk_size):
            chunk = embeddings_scaled[i:i+chunk_size]
            distances = torch.cdist(
                chunk.unsqueeze(0),
                embed_matrix_scaled.unsqueeze(0),
                p=2
            ).squeeze(0)  # (chunk_size, vocab_size)

            chunk_min_distances = distances.min(dim=1)[0]
            min_distances.append(chunk_min_distances)

        return torch.cat(min_distances)

    def _compute_sequence_realizability(self, embeddings: torch.Tensor) -> float:
        """Compute sequence-level realizability using Viterbi-like DP

        Args:
            embeddings: (seq_len, hidden_dim)

        Returns:
            Total sequence cost (lower = more realizable)
        """
        seq_len = len(embeddings)
        embeddings_scaled = embeddings * self.embed_scale
        embed_matrix_scaled = self.embedding_matrix * self.embed_scale

        # Compute emission costs (squared L2 distances)
        # For efficiency, only compute for top-K nearest neighbors
        distances = torch.cdist(
            embeddings_scaled.unsqueeze(0),
            embed_matrix_scaled.unsqueeze(0),
            p=2
        ).squeeze(0)  # (seq_len, vocab_size)

        # Get top-K nearest for each position
        topk_distances, topk_indices = distances.topk(
            self.topk, dim=1, largest=False
        )  # (seq_len, topk)

        emission_costs = topk_distances ** 2  # (seq_len, topk)

        # Viterbi DP (uniform transition costs for simplicity)
        # dp[t][k] = min cost to reach position t with k-th candidate
        dp = torch.full((seq_len, self.topk), float('inf'), device=self.device)
        dp[0] = emission_costs[0]

        for t in range(1, seq_len):
            # For each current candidate, find min cost from all previous
            for k in range(self.topk):
                # Uniform transition: just add emission cost
                costs = dp[t-1] + emission_costs[t, k]
                dp[t, k] = costs.min()

        # Final cost is minimum at last position
        seq_cost = dp[-1].min().item()
        return seq_cost

    def _compute_reconstruction_consistency(
        self,
        embeddings: torch.Tensor,
        tokens: list[int]
    ) -> float:
        """Compute reconstruction consistency

        Args:
            embeddings: (seq_len, hidden_dim)
            tokens: List of token IDs

        Returns:
            Consistency score (higher = more consistent)
        """
        # Get original embeddings from tokens
        tokens_tensor = torch.tensor(tokens, device=self.device)
        original_embeds = self.embedding_matrix[tokens_tensor]

        # Compute L2 distances
        distances = torch.norm(
            embeddings * self.embed_scale - original_embeds * self.embed_scale,
            p=2,
            dim=1
        )

        # Consistency score (inverse of distance)
        consistency = 1.0 / (1.0 + distances.mean().item())
        return consistency

    def detect(
        self,
        embeddings: torch.Tensor,
        **kwargs
    ) -> DetectionResult:
        """Detect if embeddings are off-manifold (attack)"""
        features = self.extract_features(embeddings, **kwargs)

        # Select main score based on level
        if self.level == 1:
            score = features['p90_nn_l2']
        elif self.level == 2:
            score = features['seq_realizability_cost_normalized']
        else:  # level 3
            # For level 3, lower consistency = higher attack score
            if 'reconstruction_consistency' in features:
                score = 1.0 - features['reconstruction_consistency']
            else:
                score = features['seq_realizability_cost_normalized']

        if self.threshold is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        is_attack = score > self.threshold
        confidence = abs(score - self.threshold) / (self.threshold + 1e-8)

        return DetectionResult(
            is_attack=is_attack,
            confidence=min(confidence, 1.0),
            features=features,
            threshold=self.threshold
        )

    def fit(
        self,
        benign_data: list[torch.Tensor],
        fpr_target: float = 0.01
    ):
        """Fit threshold on benign data

        Args:
            benign_data: List of benign embedding tensors
            fpr_target: Target false positive rate (default 1%)
        """
        all_scores = []

        for embeddings in benign_data:
            features = self.extract_features(embeddings)

            # Select main score
            if self.level == 1:
                score = features['p90_nn_l2']
            elif self.level == 2:
                score = features['seq_realizability_cost_normalized']
            else:
                if 'reconstruction_consistency' in features:
                    score = 1.0 - features['reconstruction_consistency']
                else:
                    score = features['seq_realizability_cost_normalized']

            all_scores.append(score)

        # Set threshold at (1-fpr_target) percentile
        self.threshold = float(np.percentile(all_scores, (1 - fpr_target) * 100))
        print(f"Fitted threshold: {self.threshold:.6f} at FPR={fpr_target:.2%}")
