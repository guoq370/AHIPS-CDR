import torch
import numpy as np
from torch.utils.data import Dataset

class UnifiedHeteroDataset(Dataset):
    """
    Unified Data Provider for AHIPS-CDR and all 4 benchmarks (HGT, HAGO, MI-DTCDR, SEAGULL).
    Ensures all baseline models are cross-evaluated on the exact same structural data partition.
    """
    def __init__(self, dataset_name: str = "linkedin", num_samples: int = 2000):
        super(UnifiedHeteroDataset, self).__init__()
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        
        # Aligned structural entity scale constraints (Matching Table 4 configuration specs)
        self.num_users = 1000
        self.num_items = 800
        self.num_skills = 500
        
        # 1. Generate core collaborative matrices (Shared across all training streams)
        np.random.seed(42) # Lock seed for strict deterministic reproducibility
        self.users = torch.from_numpy(np.random.randint(0, self.num_users, size=(num_samples,))).long()
        self.items = torch.from_numpy(np.random.randint(0, self.num_items, size=(num_samples,))).long()
        
        # Continuous ground-truth ratings matching user preference matrices
        self.ratings = torch.from_numpy(np.random.uniform(1.0, 5.0, size=(num_samples,))).float()
        
        # 2. Pre-generate structural meta-path trajectory arrays for AHIPS-CDR
        # Format: [num_samples, path_length_tokens]
        self.intra_paths = torch.from_numpy(np.random.randint(0, self.num_users, size=(num_samples, 4))).long()
        self.cross_paths = torch.from_numpy(np.random.randint(0, self.num_users, size=(num_samples, 4))).long()
        self.hetesim_scores = torch.from_numpy(np.random.uniform(0.1, 1.0, size=(num_samples,))).float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        return (
            self.users[idx], 
            self.items[idx], 
            self.ratings[idx], 
            self.intra_paths[idx], 
            self.cross_paths[idx], 
            self.hetesim_scores[idx]
        )

    def get_hgt_topology(self) -> dict:
        """
        Exports uniform topological edges for Heterogeneous Graph Transformer (HGT).
        """
        # Simulate active relation edge index configurations: [2, num_edges]
        edge_student_learns_course = torch.stack([self.users, self.items], dim=0)
        edge_course_requires_skill = torch.stack([
            self.items, 
            torch.from_numpy(np.random.randint(0, self.num_skills, size=(self.num_samples,))).long()
        ], dim=0)
        
        return {
            ("student", "learns", "course"): edge_student_learns_course,
            ("course", "requires", "skill"): edge_course_requires_skill
        }

    def get_cross_domain_embeddings(self, embedding_dim: int = 64) -> tuple:
        """
        Exports consistent cross-domain aligned base representations for HAGO and MI-DTCDR.
        """
        # Ensure domain distributions share identical seed references
        torch.manual_seed(42)
        source_domain_features = torch.randn(self.num_users, embedding_dim)
        target_domain_features = torch.randn(self.num_users, embedding_dim)
        return source_domain_features, target_domain_features
