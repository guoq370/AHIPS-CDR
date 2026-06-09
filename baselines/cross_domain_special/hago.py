import torch
import torch.nn as nn
import torch.nn.functional as F

class HAGO_CoordinatorLayer(nn.Module):
    """
    Heterogeneous Adaptive Graph coOrdinators (HAGO) baseline mechanism.
    Dynamically transfers collaborative preferences across disparate domains via virtual hub coordinators.
    """
    def __init__(self, embedding_dim: int, num_coordinators: int = 5):
        super(HAGO_CoordinatorLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_coordinators = num_coordinators
        
        # Virtual latent semantic coordinator hubs
        self.coordinators = nn.Parameter(torch.randn(num_coordinators, embedding_dim))
        nn.init.xavier_uniform_(self.coordinators.data)
        
        # Dimension scaling layers for domains
        self.src_projection = nn.Linear(embedding_dim, embedding_dim)
        self.tgt_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Adaptive information routing gates to eliminate negative transfer
        self.src_gate = nn.Sequential(nn.Linear(embedding_dim * 2, 1), nn.Sigmoid())
        self.tgt_gate = nn.Sequential(nn.Linear(embedding_dim * 2, 1), nn.Sigmoid())

    def forward(self, source_user_embeddings: torch.Tensor, target_user_embeddings: torch.Tensor) -> tuple:
        """
        Routes and coordinates cross-domain parameters across hub bottlenecks.
        Args:
            source_user_embeddings: User states inside Source Domain (e.g., Coursera) [N_users, dim]
            target_user_embeddings: User states inside Target Domain (e.g., LinkedIn) [N_users, dim]
        """
        # Step 1: Align domain representation metrics via alignment weights
        proj_src = self.src_projection(source_user_embeddings) # [N, dim]
        proj_tgt = self.tgt_projection(target_user_embeddings) # [N, dim]
        
        # Step 2: Compute Adaptive Coordination Attention maps between users and hubs
        # Source to Coordinators alignment matrix
        src_coord_scores = torch.matmul(proj_src, self.coordinators.t()) / (self.embedding_dim ** 0.5)
        src_coord_weights = F.softmax(src_coord_scores, dim=-1) # [N, num_coordinators]
        
        # Target to Coordinators alignment matrix
        tgt_coord_scores = torch.matmul(proj_tgt, self.coordinators.t()) / (self.embedding_dim ** 0.5)
        tgt_coord_weights = F.softmax(tgt_coord_scores, dim=-1) # [N, num_coordinators]
        
        # Step 3: Extract shared latent semantic contexts via the coordinator bottleneck
        shared_src_context = torch.matmul(src_coord_weights, self.coordinators) # [N, dim]
        shared_tgt_context = torch.matmul(tgt_coord_weights, self.coordinators) # [N, dim]
        
        # Step 4: Gated Fusion to control inter-domain preference pollution
        # Source domain filtered enhancement
        src_gate_input = torch.cat([source_user_embeddings, shared_tgt_context], dim=-1)
        g_src = self.src_gate(src_gate_input)
        coordinated_source = source_user_embeddings + g_src * shared_tgt_context
        
        # Target domain filtered enhancement (Critical for course-to-job prediction)
        tgt_gate_input = torch.cat([target_user_embeddings, shared_src_context], dim=-1)
        g_tgt = self.tgt_gate(tgt_gate_input)
        coordinated_target = target_user_embeddings + g_tgt * shared_src_context
        
        return coordinated_source, coordinated_target
