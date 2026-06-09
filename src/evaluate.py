import torch
import numpy as np
from utils.metrics import RecEvaluator

def execute_evaluation(model_instance, target_device, total_users: int, total_items: int) -> dict:
    """
    Executes isolated inference passes over evaluation matrices 
    to output peer-reviewed performance profiles.
    """
    model_instance.eval()
    print("[EVALUATION] Commencing global recommendation ranking audit...")
    
    # Generate balanced evaluation matrix scopes (e.g., 100 test users matching 1 true positive and 99 random negatives)
    eval_users_count = 100
    candidate_items_count = 100 # 1 Positive + 99 Negatives per peer evaluation conventions
    
    mock_eval_users = torch.randint(0, total_users, (eval_users_count, 1)).repeat(1, candidate_items_count).flatten().to(target_device)
    mock_eval_items = torch.randint(0, total_items, (eval_users_count * candidate_items_count,)).to(target_device)
    
    # Replicate path tracking configurations matching the structural sizes expected by the main forward pass
    mock_intra = torch.randint(0, total_users, (eval_users_count * candidate_items_count, 3)).to(target_device)
    mock_cross = torch.randint(0, total_users, (eval_users_count * candidate_items_count, 3)).to(target_device)

    # Initialize truth verification masks (Setting column index 0 as the true target recommendation hit)
    ground_truth_mask = torch.zeros((eval_users_count, candidate_items_count), dtype=torch.float32)
    ground_truth_mask[:, 0] = 1.0 

    with torch.no_grad():
        # Execute forward pass without capturing backpropagation graphs
        scores, _, _ = model_instance(mock_eval_users, mock_eval_items, mock_intra, mock_cross)
        # Reshape score dimensions back to a matrix layout: (eval_users_count, candidate_items_count)
        reshaped_scores = scores.view(eval_users_count, candidate_items_count)
        
    # Execute structural metric analysis via the metrics utility module
    compiled_metrics = RecEvaluator.evaluate_ranking(reshaped_scores, ground_truth_mask, k_list=[5, 10, 20, 30])
    
    print("------------------------------------------------------------")
    print(f"Performance Report: HR@10: {compiled_metrics['HR@10']:.4f} | NDCG@10: {compiled_metrics['NDCG@10']:.4f} | HR@30: {compiled_metrics['HR@30']:.4f} | MRR: {compiled_metrics['MRR']:.4f}")
    print("------------------------------------------------------------")
    
    return compiled_metrics
