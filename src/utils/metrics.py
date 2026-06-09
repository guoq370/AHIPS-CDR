import numpy as np
import torch
from typing import List, Dict

class RecEvaluator:
    """
    Standard evaluation metrics calculator for top-K recommendation performance,
    supporting Hit Rate (HR), Precision, NDCG, and Mean Reciprocal Rank (MRR).
    """
    @staticmethod
    def evaluate_ranking(
        predicted_scores: torch.Tensor, 
        ground_truth_mask: torch.Tensor, 
        k_list: List[int] = [10, 30, 50]
    ) -> Dict[str, float]:
        """
        Computes recommendation metrics across multiple top-K cutoff thresholds.
        
        Args:
            predicted_scores: Tensor of shape (batch_size, num_items) with model score outputs.
            ground_truth_mask: Binary tensor of shape (batch_size, num_items) where 1 indicates positive interaction.
            k_list: List of truncation points for top-K evaluation.
            
        Returns:
            A dictionary containing compiled performance metrics (e.g., 'HR@10', 'NDCG@30').
        """
        metrics_results = {}
        batch_size = predicted_scores.shape[0]
        
        # Step 1: Sort candidate items in descending order based on prediction scores
        # _, topk_indices shape: (batch_size, max(k_list))
        max_k = max(k_list)
        _, topk_indices = torch.topk(predicted_scores, k=max_k, dim=-1)
        
        # Gather the ground truth status of the top-K recommended items
        # sorted_hits shape: (batch_size, max_k) -> binary array indicating hits
        sorted_hits = torch.gather(ground_truth_mask, dim=-1, index=topk_indices).cpu().numpy()
        
        # Step 2: Iterate through each specified K threshold to compile metrics
        for k in k_list:
            hits_at_k = sorted_hits[:, :k] # Truncate slice to current top-K rank
            
            # 1. Hit Rate @ K (HR@K): Proportion of users who received at least one correct recommendation
            user_hits = np.any(hits_at_k > 0, axis=1)
            metrics_results[f"HR@{k}"] = float(np.mean(user_hits))
            
            # 2. Precision @ K (P@K): Fraction of recommended items that are relevant
            metrics_results[f"Precision@{k}"] = float(np.mean(np.sum(hits_at_k, axis=1) / k))
            
            # 3. Normalized Discounted Cumulative Gain (NDCG@K)
            ndcg_list = []
            for i in range(batch_size):
                user_hit_trace = hits_at_k[i]
                dcg = np.sum(user_hit_trace / np.log2(np.arange(2, k + 2)))
                # Ideal DCG calculation assuming all true positive hits are ranked at the top
                total_positives = int(np.sum(sorted_hits[i]))
                idcg = np.sum(1.0 / np.log2(np.arange(2, min(k, total_positives) + 2))) if total_positives > 0 else 1.0
                ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
            metrics_results[f"NDCG@{k}"] = float(np.mean(ndcg_list))
            
        # 4. Mean Reciprocal Rank (MRR): Evaluates where the first true positive occurs globally
        mrr_list = []
        for i in range(batch_size):
            hit_positions = np.where(sorted_hits[i] > 0)[0]
            if len(hit_positions) > 0:
                first_hit_rank = hit_positions[0] + 1 # Convert 0-indexed position to 1-indexed rank
                mrr_list.append(1.0 / first_hit_rank)
            else:
                mrr_list.append(0.0)
        metrics_results["MRR"] = float(np.mean(mrr_list))
        
        return metrics_results

if __name__ == "__main__":
    # Sanity check block with simulated batch outputs
    mock_scores = torch.tensor([[0.1, 0.8, 0.4, 0.9], [0.7, 0.2, 0.3, 0.1]])
    mock_truth = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]]) # First item hit at index 1 and 0
    results = RecEvaluator.evaluate_ranking(mock_scores, mock_truth, k_list=[2, 3])
    print("Compiled metrics output verification:")
    for metric, val in results.items():
        print(f"{metric}: {val:.4f}")
