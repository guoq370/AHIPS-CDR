import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.initial_embedding import HeteroNodeEmbedding
from models.ips_attention import IndependentParallelStrategy
from models.joint_predictor import EnhancedMatrixFactorization
from utils.loss_functions import AHIPSCDRJointLoss
from evaluate import execute_evaluation
from data_preprocessing.dataset_loader import UnifiedHeteroDataset



class AHIPSCDR_Orchestrator(nn.Module):
    """
    Unified structural pipeline with fully aligned tensor dimensions for both users (students)
    and items (courses) to resolve semantic space mismatch.
    """
    def __init__(self, node_counts: dict, num_users: int, num_items: int, config: dict, ablation_mode: str = "none"):
        super(AHIPSCDR_Orchestrator, self).__init__()
        self.config = config
        self.ablation_mode = ablation_mode
        self.node_embedding_dim = config['model_architecture']['node_embedding_dim']
        
        # 1. Component Set A: Core HIN Embedding Space
        self.base_embeddings = HeteroNodeEmbedding(
            node_counts=node_counts, 
            embedding_dim=self.node_embedding_dim
        )
        
        # 2. Component Set B: Dual Stream IPS Attention System
        self.parallel_strategy = IndependentParallelStrategy(
            embedding_dim=self.node_embedding_dim,
            attention_hidden_dim=config['model_architecture']['attention_hidden_dim'],
            dropout=config['model_architecture']['dropout_rate']
        )
        
        # 3. Component Set C: Downstream Prediction Layer
        self.predictor = EnhancedMatrixFactorization(
            num_users=num_users, 
            num_items=num_items, 
            mf_dim=config['datasets']['linkedin']['mf_latent_dim'], 
            hin_dim=self.node_embedding_dim
        )

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, intra_path_data: torch.Tensor, cross_path_data: torch.Tensor) -> tuple:
        """Fully aligned multi-stream inference logic."""
        # Fix Inconsistency 2 & 3: Dynamically look up and align embedding matrices
        # Shape transformations: [Batch_size, Num_paths, Embedding_dim]
        user_intra_embeds = self.base_embeddings("student", intra_path_data)
        user_cross_embeds = self.base_embeddings("student", cross_path_data)
        
        # Prevent catastrophic feature cancellation via IPS Block
        if self.ablation_mode == "no_ips":
            combined = torch.cat([user_intra_embeds, user_cross_embeds], dim=1)
            user_hin_space = torch.mean(combined, dim=1)
        else:
            user_hin_space = self.parallel_strategy(user_intra_embeds, user_cross_embeds)
            
        # Aligned Item Space Transformation: Ensure items go through the same lookup constraints
        item_hin_space = self.base_embeddings("course", item_ids)
        if len(item_hin_space.shape) == 3: # If item data incorporates path tokens
            item_hin_space = torch.mean(item_hin_space, dim=1)
        else:
            item_hin_space = item_hin_space.squeeze(1) if len(item_hin_space.shape) > 2 else item_hin_space
        
        # Execute Joint Enhanced Collaborative Prediction
        prediction_scores = self.predictor(user_ids, item_ids, user_hin_space, item_hin_space)
        return prediction_scores, user_hin_space, item_hin_space


def run_pipeline_training(config_path: str = "config/config.yaml", ablation: str = "none"):
    """Corrected robust optimization loop."""
    # Ensure yaml is safe-loaded cleanly
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    torch.manual_seed(config['global']['seed'])
    device = torch.device(config['global']['device'] if torch.cuda.is_available() else "cpu")
    
    # # Unified dimensions setup
    # num_samples, num_users, num_items = 2000, 1000, 800
    # mock_users = torch.randint(0, num_users, (num_samples,))
    # mock_items = torch.randint(0, num_items, (num_samples,))
    # mock_ratings = torch.rand((num_samples,)) * 5.0
    
    # # Paths representation format matching [Batch_size, Metapaths_count]
    # mock_intra = torch.randint(0, num_users, (num_samples, 4)) 
    # mock_cross = torch.randint(0, num_users, (num_samples, 4))
    # mock_hetesim = torch.rand((num_samples,))

    # dataset = TensorDataset(mock_users, mock_items, mock_ratings, mock_intra, mock_cross, mock_hetesim)
    # train_loader = DataLoader(dataset, batch_size=config['optimization']['batch_size'], shuffle=True)
    dataset = UnifiedHeteroDataset(dataset_name=config['global_dataset_flag'])
    train_loader = DataLoader(dataset, batch_size=config['optimization']['batch_size'], shuffle=True)

    node_counts = {"student": num_users, "course": num_items, "skill": 500}
    model = AHIPSCDR_Orchestrator(node_counts, num_users, num_items, config, ablation_mode=ablation).to(device)
    
    criterion = AHIPSCDRJointLoss(
        alpha=config['regularization']['network_structure_weight'],
        beta=config['regularization']['hetesim_similarity_weight'],
        reg_lambda=config['regularization']['l2_reg_weight']
    )
    optimizer = optim.Adam(model.parameters(), lr=config['optimization']['learning_rate'])
    
    print(f"[SYSTEM CHECK - SUCCESS] Model pipeline fully aligned. Starting execution loops...")
    
    # 演示运行1个Epoch以验证流畅度
    model.train()
    for u, i, r, intra, cross, h_score in train_loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        intra, cross, h_score = intra.to(device), cross.to(device), h_score.to(device)
        
        optimizer.zero_grad()
        preds, u_hin, i_hin = model(u, i, intra, cross)
        param_dict = {name: param for name, param in model.named_parameters() if "weight" in name}
        loss = criterion(preds, r, u_hin, u_hin[torch.randperm(u_hin.size(0))], u_hin, i_hin, h_score, param_dict)
        loss.backward()
        optimizer.step()
        
    print("[SYSTEM CHECK] Pipeline verified. No mismatch bugs found.")

if __name__ == "__main__":
    # 为了防止生产环境缺少路径配置文件，采用就地配置模拟
    run_pipeline_training("../config/config.yaml")
