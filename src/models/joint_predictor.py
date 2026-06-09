import torch
import torch.nn as nn

class EnhancedMatrixFactorization(nn.Module):
    """
    Enhanced Matrix Factorization module incorporating traditional collaborative filtering
    latent factors, spatial entity biases, and high-level structural HIN representations.
    """
    def __init__(self, num_users: int, num_items: int, mf_dim: int = 15, hin_dim: int = 64):
        super(EnhancedMatrixFactorization, self).__init__()
        
        # 1. Standard Matrix Factorization Parameters
        self.user_latent_factors = nn.Embedding(num_users, mf_dim)
        self.item_latent_factors = nn.Embedding(num_items, mf_dim)
        
        # 2. Bias Parameters
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # 3. Projection Layer to map deep HIN embeddings into the rating prediction space
        self.hin_projection = nn.Linear(hin_dim, 1, bias=False)
        
        self._init_weights()

    def _init_weights(self):
        """Initializes parameter spaces to ensure training stability."""
        nn.init.normal_(self.user_latent_factors.weight, std=0.01)
        nn.init.normal_(self.item_latent_factors.weight, std=0.01)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)

    def forward(
        self, 
        user_ids: torch.Tensor, 
        item_ids: torch.Tensor, 
        user_hin_embed: torch.Tensor, 
        item_hin_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates joint prediction scores.
        Args:
            user_ids: (batch_size,)
            item_ids: (batch_size,)
            user_hin_embed: From IPS module output (batch_size, hin_dim)
            item_hin_embed: From IPS module output (batch_size, hin_dim)
        Returns:
            Scalar prediction score vectors (batch_size,)
        """
        # Fetch MF vectors
        p_u = self.user_latent_factors(user_ids) # (batch_size, mf_dim)
        q_i = self.item_latent_factors(item_ids) # (batch_size, mf_dim)
        
        # Fetch entity biases
        b_u = self.user_biases(user_ids).squeeze(-1) # (batch_size,)
        b_i = self.item_biases(item_ids).squeeze(-1) # (batch_size,)
        
        # Base Collaborative Filtering Dot Product Component
        cf_signal = torch.sum(p_u * q_i, dim=-1) # (batch_size,)
        
        # Interaction Component derived from Deep HIN Embeddings (Element-wise multiplication then linear projected)
        hin_interaction = user_hin_embed * item_hin_embed # (batch_size, hin_dim)
        hin_signal = self.hin_projection(hin_interaction).squeeze(-1) # (batch_size,)
        
        # Harmonize components according to the Enhanced MF joint scoring function
        prediction_score = self.global_bias + b_u + b_i + cf_signal + hin_signal
        return prediction_score
