import torch
import torch.nn as nn
import torch.nn.functional as F

class DualLevelAttention(nn.Module):
    """
    Implements a two-layer MLP Attention network to adaptively evaluate semantic
    importance coefficients across varied meta-path representations while suppressing structural noise.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super(DualLevelAttention, self).__init__()
        # Layer 1: Project input into non-linear semantic attention space
        self.mlp_layer1 = nn.Linear(input_dim, hidden_dim)
        # Layer 2: Compute scalar weight coefficient
        self.mlp_layer2 = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, metapath_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            metapath_embeddings: Tensor of shape (batch_size, num_metapaths, input_dim)
        Returns:
            Attended fused semantic representation tensor of shape (batch_size, input_dim)
        """
        # Step 1: Compute attention scores using the dual-layer MLP structure
        # (batch_size, num_metapaths, hidden_dim)
        score_space = F.relu(self.mlp_layer1(metapath_embeddings))
        score_space = self.dropout(score_space)
        
        # (batch_size, num_metapaths, 1)
        raw_scores = self.mlp_layer2(score_space)
        
        # Step 2: Normalize coefficients across all parallel meta-paths via Softmax
        attention_weights = F.softmax(raw_scores, dim=1) # (batch_size, num_metapaths, 1)
        
        # Step 3: Weighted aggregation of meta-path embedding spaces
        fused_representation = torch.sum(attention_weights * metapath_embeddings, dim=1)
        return fused_representation


class IndependentParallelStrategy(nn.Module):
    """
    Core AHIPS-CDR Architecture: Decouples feature processing into dual streams
    (Intra-domain Stream & Cross-domain Stream) to prevent catastrophic feature cancellation.
    """
    def __init__(self, embedding_dim: int, attention_hidden_dim: int = 128, dropout: float = 0.1):
        super(IndependentParallelStrategy, self).__init__()
        
        # Stream A: Dedicated Intra-domain Attention Flow
        self.intra_attention = DualLevelAttention(embedding_dim, attention_hidden_dim, dropout)
        
        # Stream B: Dedicated Cross-domain (Inter-domain) Attention Flow
        self.cross_attention = DualLevelAttention(embedding_dim, attention_hidden_dim, dropout)
        
        # Final Fusion Layer: Harmonizes parallel outputs into a unified semantic space
        self.fusion_layer = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, intra_path_embeddings: torch.Tensor, cross_path_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            intra_path_embeddings: (batch_size, num_intra_paths, embedding_dim)
            cross_path_embeddings: (batch_size, num_cross_paths, embedding_dim)
        Returns:
            Decoupled and integrated multi-source HIN embedding vector (batch_size, embedding_dim)
        """
        # Execute Stream A processing independently
        intra_fused = self.intra_attention(intra_path_embeddings) # (batch_size, embedding_dim)
        
        # Execute Stream B processing independently 
        cross_fused = self.cross_attention(cross_path_embeddings) # (batch_size, embedding_dim)
        
        # Concatenate outputs along the hidden dimension without mutual interference
        concatenated_features = torch.cat([intra_fused, cross_fused], dim=-1) # (batch_size, embedding_dim * 2)
        
        # Linear projection back to target latent dimension size
        final_hin_embedding = self.fusion_layer(concatenated_features)
        return final_hin_embedding
