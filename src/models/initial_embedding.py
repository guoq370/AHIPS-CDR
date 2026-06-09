import torch
import torch.nn as nn
from typing import Dict

class HeteroNodeEmbedding(nn.Module):
    """
    Initializes and manages structural low-dimensional embedding lookup vectors 
    for all 6 heterogeneous node types in the unified HIN.
    """
    def __init__(self, node_counts: Dict[str, int], embedding_dim: int = 64):
        super(HeteroNodeEmbedding, self).__init__()
        self.node_counts = node_counts
        self.embedding_dim = embedding_dim
        
        # Create separate embedding dictionaries for each unique entity type
        self.embeddings = nn.ModuleDict({
            node_type: nn.Embedding(num_embeddings=count, embedding_dim=embedding_dim)
            for node_type, count in node_counts.items()
        })
        self._init_weights()

    def _init_weights(self):
        """Applies Xavier uniform initialization to optimize gradient convergence."""
        for embedding in self.embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, node_type: str, node_ids: torch.Tensor) -> torch.Tensor:
        """Retrieves raw latent representations for a specific slice of entity indices."""
        if node_type not in self.embeddings:
            raise KeyError(f"Node type '{node_type}' is not registered in the HIN configuration.")
        return self.embeddings[node_type](node_ids)

    def get_all_embeddings_by_type(self, node_type: str) -> torch.Tensor:
        """Fetches the entire parameter weights partition for global structural downstream processing."""
        return self.embeddings[node_type].weight
