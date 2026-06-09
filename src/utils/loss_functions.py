import torch
import torch.nn as nn
from typing import Dict

class AHIPSCDRJointLoss(nn.Module):
    """
    Implements the multi-task joint optimization loss framework for AHIPS-CDR model.
    Formula: Total_Loss = F1 (MF Error) + alpha * F2 (Structure Loss) + beta * F3 (HeteSim) + lambda * F4 (L2 Reg)
    """
    def __init__(self, alpha: float = 0.1, beta: float = 0.1, reg_lambda: float = 0.1):
        super(AHIPSCDRJointLoss, self).__init__()
        self.alpha = alpha          # Control coefficient for Metapath2Vec++ structure loss (F2)
        self.beta = beta            # Control coefficient for HeteSim proximity regularizer (F3)
        self.reg_lambda = reg_lambda  # Regularization penalty weight for weight parameters (F4)
        
        # Prediction Loss Component (F1): Mean Squared Error for continuous rating preference matching
        self.prediction_criterion = nn.MSELoss()

    def compute_structure_loss(self, center_embeddings: torch.Tensor, context_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes the F2 Skip-Gram loss component with negative sampling 
        to retain meta-path structural topology proximity.
        """
        # Element-wise dot-product matching between core nodes and contextual walk sequences
        score = torch.sum(center_embeddings * context_embeddings, dim=-1)
        # Minimize the negative log-sigmoid of structural target trajectories
        structure_loss = -torch.mean(torch.log(torch.sigmoid(score) + 1e-9))
        return structure_loss

    def compute_hetesim_regularization(self, user_hin_embed: torch.Tensor, item_hin_embed: torch.Tensor, hetesim_scores: torch.Tensor) -> torch.Tensor:
        """
        Computes the F3 HeteSim structural alignment loss constraint. 
        Forces deep HIN projections to align with mathematical meta-path similarity matrices.
        """
        # Calculate latent cosine similarity between processed user and item HIN spaces
        cos_sim = torch.sum(user_hin_embed * item_hin_embed, dim=-1) / (
            torch.norm(user_hin_embed, dim=-1) * torch.norm(item_hin_embed, dim=-1) + 1e-9
        )
        # Penalize spatial divergence away from precalculated HeteSim semantic scores
        hetesim_loss = torch.mean((cos_sim - hetesim_scores) ** 2)
        return hetesim_loss

    def forward(
        self, 
        pred_ratings: torch.Tensor, 
        true_ratings: torch.Tensor,
        center_nodes: torch.Tensor,
        context_nodes: torch.Tensor,
        user_hin_embed: torch.Tensor,
        item_hin_embed: torch.Tensor,
        hetesim_scores: torch.Tensor,
        model_parameters_dict: Dict
    ) -> torch.Tensor:
        """
        Executes unified joint backward loss computation.
        """
        # Task 1: Collaborative Filtering Rating Prediction Loss (F1 Component)
        f1_prediction_loss = self.prediction_criterion(pred_ratings, true_ratings)
        
        # Task 2: Metapath-constrained Skip-Gram Structure Loss (F2 Component)
        f2_structure_loss = self.compute_structure_loss(center_nodes, context_nodes)
        
        # Task 3: HeteSim Structural Proximity Regularization Loss (F3 Component)
        f3_hetesim_loss = self.compute_hetesim_regularization(user_hin_embed, item_hin_embed, hetesim_scores)
        
        # Task 4: Structural L2 Parameter Weight Regularization (F4 Component)
        f4_l2_reg = torch.tensor(0.0, device=pred_ratings.device)
        for param in model_parameters_dict.values():
            f4_l2_reg += torch.norm(param, p=2) ** 2
            
        # Unified aggregation step mapping to final mathematical objective formulation
        total_joint_loss = (
            f1_prediction_loss + 
            (self.alpha * f2_structure_loss) + 
            (self.beta * f3_hetesim_loss) + 
            (self.reg_lambda * f4_l2_reg)
        )
        return total_joint_loss
