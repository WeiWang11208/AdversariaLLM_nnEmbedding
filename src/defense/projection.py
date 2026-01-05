import torch
import torch.nn.functional as F


class EmbeddingProjector:
    """Project embeddings back to realizable set (vocabulary embeddings)"""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        method: str = "nearest",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize projector

        Args:
            embedding_matrix: Vocabulary embedding matrix (vocab_size, hidden_dim)
            method: Projection method ("nearest" or "weighted")
            device: Device to run on
        """
        self.embedding_matrix = embedding_matrix.to(device)
        self.method = method
        self.device = device

    def project(
        self,
        embeddings: torch.Tensor,
        temperature: float = 1.0,
        topk: int = 10
    ) -> torch.Tensor:
        """Project embeddings to realizable set

        Args:
            embeddings: Input embeddings (seq_len, hidden_dim)
            temperature: Temperature for weighted projection
            topk: Number of top neighbors for weighted projection

        Returns:
            Projected embeddings (seq_len, hidden_dim)
        """
        embeddings = embeddings.to(self.device).float()

        if self.method == "nearest":
            return self._project_nearest(embeddings)
        elif self.method == "weighted":
            return self._project_weighted(embeddings, temperature, topk)
        else:
            raise ValueError(f"Unknown projection method: {self.method}")

    def _project_nearest(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Hard projection: replace each embedding with nearest neighbor

        Args:
            embeddings: (seq_len, hidden_dim)

        Returns:
            Projected embeddings (seq_len, hidden_dim)
        """
        # Compute distances in chunks to avoid OOM
        chunk_size = 1024
        projected_chunks = []

        for i in range(0, len(embeddings), chunk_size):
            chunk = embeddings[i:i+chunk_size]
            distances = torch.cdist(
                chunk.unsqueeze(0),
                self.embedding_matrix.unsqueeze(0),
                p=2
            ).squeeze(0)  # (chunk_size, vocab_size)

            # Get nearest neighbor indices
            nearest_indices = distances.argmin(dim=1)  # (chunk_size,)

            # Replace with nearest embeddings
            projected_chunk = self.embedding_matrix[nearest_indices]
            projected_chunks.append(projected_chunk)

        return torch.cat(projected_chunks, dim=0)

    def _project_weighted(
        self,
        embeddings: torch.Tensor,
        temperature: float = 1.0,
        topk: int = 10
    ) -> torch.Tensor:
        """Soft projection: weighted average of top-K nearest neighbors

        Args:
            embeddings: (seq_len, hidden_dim)
            temperature: Temperature for softmax weights
            topk: Number of nearest neighbors to average

        Returns:
            Projected embeddings (seq_len, hidden_dim)
        """
        chunk_size = 1024
        projected_chunks = []

        for i in range(0, len(embeddings), chunk_size):
            chunk = embeddings[i:i+chunk_size]
            distances = torch.cdist(
                chunk.unsqueeze(0),
                self.embedding_matrix.unsqueeze(0),
                p=2
            ).squeeze(0)  # (chunk_size, vocab_size)

            # Convert to similarities
            similarities = torch.exp(-distances / temperature)

            # Get top-K
            topk_similarities, topk_indices = similarities.topk(topk, dim=1)

            # Normalize weights
            weights = F.softmax(topk_similarities, dim=1)  # (chunk_size, topk)

            # Get top-K embeddings
            topk_embeddings = self.embedding_matrix[topk_indices]  # (chunk_size, topk, hidden_dim)

            # Weighted average
            projected_chunk = (weights.unsqueeze(-1) * topk_embeddings).sum(dim=1)
            projected_chunks.append(projected_chunk)

        return torch.cat(projected_chunks, dim=0)
