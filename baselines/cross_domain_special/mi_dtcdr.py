import torch
import torch.nn as nn
import torch.nn.functional as F

class MIDTCDRLayer(nn.Module):
    """
    Core implementation of Multi-Interactive Deep Target Cross-Domain Recommendation (MI-DTCDR).
    Uses non-linear multi-layer interaction matrices to transfer preferences across domains.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super(MIDTCDRLayer, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Linear projection maps for cross-domain deep interactions
        self.src_interaction = nn.Linear(embedding_dim, hidden_dim)
        self.tgt_interaction = nn.Linear(embedding_dim, hidden_dim)
        
        # Interactive translation matrix (The core of MI-DTCDR mechanism)
        self.translation_matrix = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.translation_matrix.data)
        
        # Fusion layer combining original target states and transferred source intent
        self.deep_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, source_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Deep multi-interactive forward alignment pass.
        Args:
            source_embeddings: Source domain user features [Batch_size, embedding_dim]
            target_embeddings: Target domain user features [Batch_size, embedding_dim]
        Returns:
            Enhanced interactive target embeddings [Batch_size, embedding_dim]
        """
        # Step 1: Project into interactive latent subspaces
        h_src = F.relu(self.src_interaction(source_embeddings)) # [B, hidden_dim]
        h_tgt = F.relu(self.tgt_interaction(target_embeddings)) # [B, hidden_dim]
        
        # Step 2: Multi-interactive transfer multiplication
        # Translate source preferences into target-aligned semantic signals
        transferred_src_signal = torch.matmul(h_src, self.translation_matrix) # [B, hidden_dim]
        
        # Step 3: Deep fusion bottleneck coupling translated signals with original behaviors
        combined_features = torch.cat([h_tgt, transferred_src_signal], dim=-1) # [B, hidden_dim * 2]
        enhanced_target_representations = self.deep_fusion(combined_features)
        
        return enhanced_target_representations
