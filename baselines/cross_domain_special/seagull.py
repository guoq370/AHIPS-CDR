import torch
import torch.nn as nn
import torch.nn.functional as F

class SEAGULLDisentangler(nn.Module):
    """
    SEAGULL Baseline Framework: Disentangles node representations into 
    orthogonal shared components and domain-specific specific components.
    """
    def __init__(self, embedding_dim: int):
        super(SEAGULLDisentangler, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Shared component extractor maps
        self.shared_extractor = nn.Linear(embedding_dim, embedding_dim)
        
        # Specific component extractor maps
        self.specific_extractor = nn.Linear(embedding_dim, embedding_dim)
        
        # Final combination projector
        self.reconstructor = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, raw_node_embeddings: torch.Tensor) -> tuple:
        """
        Extracts orthogonal factors from unified hidden representations.
        Args:
            raw_node_embeddings: Input node features from graph convolutions [N, embedding_dim]
        Returns:
            Tuple of (shared_feats, specific_feats, combined_out)
        """
        # Step 1: Extract domain-shared invariant semantic tracks
        shared_feats = torch.tanh(self.shared_extractor(raw_node_embeddings))
        
        # Step 2: Extract domain-specific personalized behavior tracks
        specific_feats = torch.relu(self.specific_extractor(raw_node_embeddings))
        
        # Step 3: Reconstruct integrated representations for ranking steps
        concatenated_space = torch.cat([shared_feats, specific_feats], dim=-1) # [N, embedding_dim * 2]
        combined_out = self.reconstructor(concatenated_space)
        
        return shared_feats, specific_feats, combined_out

    def compute_orthogonality_loss(self, shared_feats: torch.Tensor, specific_feats: torch.Tensor) -> torch.Tensor:
        """
        Mathematical constraints ensuring zero correlation between disentangled tracks (Cosine penalty).
        """
        cos_sim = torch.sum(shared_feats * specific_feats, dim=-1) / (
            torch.norm(shared_feats, dim=-1) * torch.norm(specific_feats, dim=-1) + 1e-9
        )
        return torch.mean(cos_sim ** 2)
