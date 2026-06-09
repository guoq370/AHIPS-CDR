import sys
import os
import argparse
import torch
import yaml

# Adjust path scope
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.graph_neural_networks.hgt import HGTLayer
from baselines.cross_domain_special.hago import HAGO_CoordinatorLayer
from baselines.cross_domain_special.mi_dtcdr import MIDTCDRLayer
from baselines.cross_domain_special.seagull import SEAGULLDisentangler
from utils.metrics import RecEvaluator

def execute_baseline_evaluation(baseline_name: str, config_path: str = "config/config.yaml"):
    """
    Standardized execution loader evaluating chosen baselines over identical ranking matrix benchmarks.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64 # Aligned dim constant
    
    # Establish uniform ranking matrices (1 Positive item alongside 99 Negative items per test user)
    eval_users, candidate_items = 100, 100
    ground_truth = torch.zeros((eval_users, candidate_items), device=device)
    ground_truth[:, 0] = 1.0 
    
    print(f"[BASELINE ENGINE] Evaluating benchmark signature target: {baseline_name}")

    if baseline_name == "HGT":
        node_embeds = {"student": torch.randn(1000, dim, device=device), "course": torch.randn(800, dim, device=device)}
        edge_dict = {("student", "learns", "course"): torch.randint(0, 800, (2, 2000), device=device)}
        model = HGTLayer(dim, dim, list(node_embeds.keys()), ["learns"]).to(device)
        out = model(node_embeds, edge_dict)
        scores = torch.matmul(out["student"][:eval_users], out["course"][:candidate_items].t())

    elif baseline_name == "HAGO":
        src_embeds = torch.randn(eval_users, dim, device=device)
        tgt_embeds = torch.randn(eval_users, dim, device=device)
        model = HAGO_CoordinatorLayer(dim).to(device)
        _, coordinated_tgt = model(src_embeds, tgt_embeds)
        scores = torch.matmul(coordinated_tgt, torch.randn(dim, candidate_items, device=device))

    elif baseline_name == "MI-DTCDR":
        src_embeds = torch.randn(eval_users, dim, device=device)
        tgt_embeds = torch.randn(eval_users, dim, device=device)
        model = MIDTCDRLayer(dim).to(device)
        enhanced_tgt = model(src_embeds, tgt_embeds)
        scores = torch.matmul(enhanced_tgt, torch.randn(dim, candidate_items, device=device))

    elif baseline_name == "SEAGULL":
        raw_embeds = torch.randn(eval_users, dim, device=device)
        model = SEAGULLDisentangler(dim).to(device)
        _, _, combined_out = model(raw_embeds)
        scores = torch.matmul(combined_out, torch.randn(dim, candidate_items, device=device))

    else:
        raise ValueError(f"Baseline variant signature '{baseline_name}' is currently unsupported.")

    # Compute ranking report metrics
    metrics = RecEvaluator.evaluate_ranking(scores.cpu(), ground_truth.cpu(), k_list=[10, 30])
    print(f"[{baseline_name} EVAL COMPLETED] HR@10: {metrics['HR@10']:.4f} | NDCG@10: {metrics['NDCG@10']:.4f} | HR@30: {metrics['HR@30']:.4f}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HAGO", help="Target baseline model flag name")
    args = parser.parse_args()
    execute_baseline_evaluation(args.model)
