from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class DetectionResult:
    """Detection result for a single input"""
    is_attack: bool
    confidence: float
    features: dict[str, float]
    threshold: Optional[float] = None


class Detector(ABC):
    """Base class for embedding anomaly detectors"""

    @abstractmethod
    def extract_features(
        self,
        embeddings: torch.Tensor,
        tokens: Optional[list[int]] = None,
        **kwargs
    ) -> dict[str, float]:
        """Extract feature vector from embeddings

        Args:
            embeddings: Input embeddings (seq_len, hidden_dim)
            tokens: Optional corresponding token IDs

        Returns:
            Dictionary of feature name -> value
        """
        pass

    @abstractmethod
    def detect(
        self,
        embeddings: torch.Tensor,
        **kwargs
    ) -> DetectionResult:
        """Detect if embeddings are from an attack

        Args:
            embeddings: Input embeddings (seq_len, hidden_dim)

        Returns:
            DetectionResult with is_attack flag and features
        """
        pass

    @abstractmethod
    def fit(self, benign_data: list[torch.Tensor], **kwargs):
        """Fit detector on benign data (e.g., set thresholds)

        Args:
            benign_data: List of benign embedding tensors
        """
        pass
