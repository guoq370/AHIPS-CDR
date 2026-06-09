import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class MetaPathAttentionStream(nn.Module):
    """
    Implements a single semantically homogeneous attention pipeline for a specific stream 
    (either Intra-domain or Inter-domain) to prevent in-stream semantic conflict.
    
    Reference: Manuscript Section 3.4, Equations (5) and (6).
    """
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128, dropout_rate: float = 0.1):
        super(MetaPathAttentionStream, self).__init__()
        # First layer of the attention MLP: Projects node embeddings into latent attention space
        self.attn_fc1 = nn.Linear(embedding_dim, hidden_dim)
        # Second layer of the attention MLP: Projects hidden states into a scalar importance score
        self.attn_fc2 = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Weight initialization following standard academic conventions
        nn.init.xavier_uniform_(self.attn_fc1.weight)
        nn.init.zeros_(self.attn_fc1.bias)
        nn.init.xavier_uniform_(self.attn_fc2.weight)

    def forward(self, path_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for computing adaptive weights across diverse meta-paths.
        
        Args:
            path_embeddings (torch.Tensor): Shape [B, M, d], where:
                                            B = Batch size (number of entities)
                                            M = Number of meta-paths assigned to this stream
                                            d = Node embedding dimensionality (typically 64)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - aggregated_rep: Compressed semantic matrix of shape [B, d]
                - attention_weights: Normalized alignment vector of shape [B, M, 1]
        """
        B, M, d = path_embeddings.size()
        
        # Step 1: Flatten batch and path dimensions to execute efficient concurrent dense layers
        # Flat shape: [B * M, d]
        flat_embeddings = path_embeddings.view(B * M, d)
        
        # Step 2: Pass through the two-layer perceptron with ReLU activation
        # Equation (5): s = W2 * ReLU(W1 * e + b1)
        hidden_states = F.relu(self.attn_fc1(flat_embeddings))
        hidden_states = self.dropout(hidden_states)
        raw_scores = self.attn_fc2(hidden_states) # Shape: [B * M, 1]
        
        # Step 3: Reshape back to decouple the distinct meta-path context channels
        # Shape: [B, M, 1]
        raw_scores = raw_scores.view(B, M, 1)
        
        # Step 4: Apply Softmax normalization along the path dimension (dim=1)
        # Equation (6): beta = exp(s) / sum(exp(s))
        attention_weights = F.softmax(raw_scores, dim=1)
        
        # Step 5: Execute weighted aggregation to build the final aggregated vector
        # Shape translation: [B, d] <- sum([B, M, d] * [B, M, 1])
        aggregated_rep = torch.sum(path_embeddings * attention_weights, dim=1)
        
        return aggregated_rep, attention_weights


class IPSAttentionFusion(nn.Module):
    """
    Independent Parallel Strategy (IPS) Fusion Module.
    Structurally decouples representation learning into two separate streams 
    to preserve domain-specific behaviors and weak cross-domain transfer signals.
    
    Reference: Manuscript Section 3.4, Equations (7) to (10).
    """
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128, dropout_rate: float = 0.1):
        super(IPSAttentionFusion, self).__init__()
        
        # Completely isolated neural parameters for the intra-domain semantic stream
        self.intra_stream_handler = MetaPathAttentionStream(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate
        )
        
        # Completely isolated neural parameters for the cross-domain (inter-domain) semantic stream
        self.inter_stream_handler = MetaPathAttentionStream(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate
        )

    def forward(
        self, 
        intra_path_embeddings: torch.Tensor, 
        inter_path_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Executes parallel, conflict-avoidant dual-level attention aggregation.
        
        Args:
            intra_path_embeddings (torch.Tensor): Shape [B, M_intra, d] from domain-specific walks.
            inter_path_embeddings (torch.Tensor): Shape [B, M_inter, d] from bridging cross-domain walks.
            
        Returns:
            Dict[str, torch.Tensor]: A structured packet containing refined, isolated representations:
                - 'intra_rep': [B, d] -> Unified intra-domain representation (Eq. 7 / Eq. 8)
                - 'inter_rep': [B, d] -> Unified cross-domain transfer representation (Eq. 9 / Eq. 10)
                - 'intra_weights': [B, M_intra, 1] -> Diagnostic path weights for structural validation
                - 'inter_weights': [B, M_inter, 1] -> Diagnostic path weights for structural validation
        """
        # Execute parallel processing path pipelines independently to bypass premature feature pollution
        intra_rep, intra_weights = self.intra_stream_handler(intra_path_embeddings)
        inter_rep, inter_weights = self.inter_stream_handler(inter_path_embeddings)
        
        return {
            "intra_rep": intra_rep,
            "inter_rep": inter_rep,
            "intra_weights": intra_weights,
            "inter_weights": inter_weights
        }


if __name__ == "__main__":
    # Rigorous academic sanity test block to verify shapes and pipeline flow
    print("[TEST] Initializing IPS Attention Fusion module sanity check...")
    
    # Setup structural dimensions according to Hyperparameter Settings (Table 4)
    batch_size = 32
    d_dim = 64
    h_dim = 128
    
    # Assume Coursera dataset scenario: 5 intra-domain paths, 3 cross-domain paths activated
    num_intra_paths = 5
    num_inter_paths = 3
    
    # Generate mock tensor inputs representing Metapath2Vec++ initial base node features
    mock_intra_input = torch.randn(batch_size, num_intra_paths, d_dim)
    mock_inter_input = torch.randn(batch_size, num_inter_paths, d_dim)
    
    # Instantiate the structural network model
    ips_layer = IPSAttentionFusion(embedding_dim=d_dim, hidden_dim=h_dim, dropout_rate=0.1)
    ips_layer.eval()
    
    # Execute forward propagation
    with torch.no_grad():
        output_bundle = ips_layer(mock_intra_input, mock_inter_input)
        
    # Assert structural integrity for robust deployment
    assert output_bundle["intra_rep"].shape == (batch_size, d_dim), "Intra representation mismatch!"
    assert output_bundle["inter_rep"].shape == (batch_size, d_dim), "Inter representation mismatch!"
    assert output_bundle["intra_weights"].shape == (batch_size, num_intra_paths, 1), "Intra weight tensor shape mismatch!"
    assert output_bundle["inter_weights"].shape == (batch_size, num_inter_paths, 1), "Inter weight tensor shape mismatch!"
    
    print("[SUCCESS] IPS Fusion Module passed structural shape and parallel-flow validation.")
    print(f"-> Extracted Intra Shape: {output_bundle['intra_rep'].shape}")
    print(f"-> Extracted Inter Shape: {output_bundle['inter_rep'].shape}")
