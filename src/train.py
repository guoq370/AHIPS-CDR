import os
import sys
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Appending paths to ensure absolute imports operate seamlessly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.initial_embedding import HeteroNodeEmbedding
from models.ips_attention import IndependentParallelStrategy
from models.joint_predictor import EnhancedMatrixFactorization
from utils.loss_functions import AHIPSCDRJointLoss
from evaluate import execute_evaluation

class AHIPSCDR_Orchestrator(torch.nn.Module):
    """
    Unified structural pipeline wrapping initial embeddings, independent parallel strategy 
    attention, and enhanced matrix factorization for joint optimization.
    """
    def __init__(self, node_counts: dict, num_users: int, num_items: int, config: dict, ablation_mode: str = "none"):
        super(AHIPSCDR_Orchestrator, self).__init__()
        self.config = config
        self.ablation_mode = ablation_mode # Options: "none", "no_ips", "no_attention"
        
        # 1. Component Set A: Deep HIN Representation Blocks
        self.base_embeddings = HeteroNodeEmbedding(
            node_counts=node_counts, 
            embedding_dim=config['model_architecture']['node_embedding_dim']
        )
        self.parallel_strategy = IndependentParallelStrategy(
            embedding_dim=config['model_architecture']['node_embedding_dim'],
            attention_hidden_dim=config['model_architecture']['attention_hidden_dim'],
            dropout=config['model_architecture']['dropout_rate']
        )
        
        # 2. Component Set B: Rating Prediction Block
        self.predictor = EnhancedMatrixFactorization(
            num_users=num_users, 
            num_items=num_items, 
            mf_dim=config['datasets']['linkedin']['mf_latent_dim'], # Defaulting to LinkedIn config profile
            hin_dim=config['model_architecture']['node_embedding_dim']
        )

    def forward(self, user_ids, item_ids, intra_path_data, cross_path_data):
        """Executes full multi-stream forward inference execution."""
        # Step 1: Look up HIN node structural embeddings
        user_intra_embeds = self.base_embeddings("student", intra_path_data)
        user_cross_embeds = self.base_embeddings("student", cross_path_data)
        
        # Step 2: Extract semantic weights via Independent Parallel Strategy
        if self.ablation_mode == "no_ips":
            # Degradation test: bypass independent stream separation, completely average multi-source vectors
            combined = torch.cat([user_intra_embeds, user_cross_embeds], dim=1)
            user_hin_space = torch.mean(combined, dim=1)
        else:
            # Standard Production Mode: Process streams without mutual cancellation noise
            user_hin_space = self.parallel_strategy(user_intra_embeds, user_cross_embeds)
            
        # For simulation simplicity, item representations can be fetched directly from raw entities
        item_hin_space = self.base_embeddings("course", item_ids).squeeze(1) if len(item_ids.shape) > 1 else self.base_embeddings("course", item_ids)
        
        # Step 3: Run collaborative scoring prediction
        prediction_scores = self.predictor(user_ids, item_ids, user_hin_space, item_hin_space)
        return prediction_scores, user_hin_space, item_hin_space


def run_pipeline_training(config_path: str = "config/config.yaml", ablation: str = "none"):
    """Orchestrates loading configuration states, streaming data loops, and optimizing parameters."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Enforce deterministic optimization boundaries across hardware platforms
    torch.manual_seed(config['global']['seed'])
    device = torch.device(config['global']['device'] if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Executing pipeline initialization on target device: {device} | Ablation Switch: {ablation}")

    # Generate mock training dataset arrays mimicking the LinkedIn/Coursera structural profiles
    num_samples, num_users, num_items = 5000, 1000, 800
    mock_users = torch.randint(0, num_users, (num_samples,))
    mock_items = torch.randint(0, num_items, (num_samples,))
    mock_ratings = torch.rand((num_samples,)) * 5.0 # Continuous label matching
    
    # Structural path configuration dimensions: (batch_size, num_meta_paths)
    mock_intra = torch.randint(0, num_users, (num_samples, 3)) 
    mock_cross = torch.randint(0, num_users, (num_samples, 3))
    mock_hetesim = torch.rand((num_samples,)) # Precalculated topology metrics

    dataset = TensorDataset(mock_users, mock_items, mock_ratings, mock_intra, mock_cross, mock_hetesim)
    train_loader = DataLoader(dataset, batch_size=config['optimization']['batch_size'], shuffle=True)

    # Instantiate joint architecture entities
    node_counts = {"student": num_users, "course": num_items, "skill": 500}
    model = AHIPSCDR_Orchestrator(node_counts, num_users, num_items, config, ablation_mode=ablation).to(device)
    
    criterion = AHIPSCDRJointLoss(
        alpha=config['regularization']['network_structure_weight'],
        beta=config['regularization']['hetesim_similarity_weight'],
        reg_lambda=config['regularization']['l2_reg_weight']
    )
    optimizer = optim.Adam(model.parameters(), lr=config['optimization']['learning_rate'])
    
    # Track metrics to execute Early Stopping parameters
    best_metric_score = -1.0
    patience_counter = 0
    patience_limit = config['optimization']['early_stopping_patience']

    for epoch in range(1, config['optimization']['max_epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        
        for u, i, r, intra, cross, h_score in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            intra, cross, h_score = intra.to(device), cross.to(device), h_score.to(device)
            
            optimizer.zero_grad()
            preds, u_hin, i_hin = model(u, i, intra, cross)
            
            # Map parameters tracking regularization layers
            param_dict = {name: param for name, param in model.named_parameters() if "weight" in name}
            
            # For structure loss execution context, map context nodes directly using sequential shuffles
            loss = criterion(preds, r, u_hin, u_hin[torch.randperm(u_hin.size(0))], u_hin, i_hin, h_score, param_dict)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Execute intermediate performance evaluation check checks every 5 epochs
        if epoch % 5 == 0:
            print(f"[EPOCH {epoch:03d}] Aggregated Step Training Loss: {epoch_loss / len(train_loader):.5f}")
            # Run inference step execution mapping to validation metrics
            val_metrics = execute_evaluation(model, device, num_users, num_items)
            current_target = val_metrics[config['optimization']['monitor_metric']]
            
            if current_target > best_metric_score:
                best_metric_score = current_target
                patience_counter = 0
                # Commit stable parameters to physical storage artifacts
                torch.save(model.state_dict(), "checkpoint_best_model.pt")
            else:
                patience_counter += 5
                if patience_counter >= patience_limit:
                    print(f"[EARLY STOPPING] Optimization ceased at epoch {epoch}. Peak {config['optimization']['monitor_metric']}: {best_metric_score:.4f}")
                    break

if __name__ == "__main__":
    run_pipeline_training()
