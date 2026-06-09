import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HGTLayer(nn.Module):
    """
    Classic Heterogeneous Graph Transformer (HGT) Layer.
    Computes type-specific attention weights for heterogeneous nodes and relations.
    """
    def __init__(self, in_dim: int, out_dim: int, node_types: list, rel_types: list, num_heads: int = 4):
        super(HGTLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_types = node_types
        self.rel_types = rel_types
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        
        # Type-specific linear projections for Query, Key, Value, and Target Residuals
        self.k_linear = nn.ModuleDict({n_type: nn.Linear(in_dim, out_dim) for n_type in node_types})
        self.q_linear = nn.ModuleDict({n_type: nn.Linear(in_dim, out_dim) for n_type in node_types})
        self.v_linear = nn.ModuleDict({n_type: nn.Linear(in_dim, out_dim) for n_type in node_types})
        self.a_linear = nn.ModuleDict({n_type: nn.Linear(out_dim, out_dim) for n_type in node_types})
        
        # Relation-specific parameter matrices (Priori relation matrices)
        self.relation_att = nn.ParameterDict({
            rel: nn.Parameter(torch.Tensor(num_heads, self.d_k, self.d_k)) for rel in rel_types
        })
        self.relation_msg = nn.ParameterDict({
            rel: nn.Parameter(torch.Tensor(num_heads, self.d_k, self.d_k)) for rel in rel_types
        })
        
        # Skip connection gains
        self.skip = nn.ParameterDict({n_type: nn.Parameter(torch.ones(1)) for n_type in node_types})
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights using the Xavier uniform scheme."""
        for rel in self.rel_types:
            nn.init.xavier_uniform_(self.relation_att[rel])
            nn.init.xavier_uniform_(self.relation_msg[rel])

    def forward(self, node_embeddings_dict: dict, adjacency_dict: dict) -> dict:
        """
        Forward logic passing messages across disparate relation configurations.
        Args:
            node_embeddings_dict: Dict mapping node type strings to tensor blocks (shape: [N, in_dim])
            adjacency_dict: Dict mapping relation tuples (src_type, rel_name, tgt_type) to edge index sparse layouts.
        """
        updated_embeddings = {n_type: torch.zeros_like(node_embeddings_dict[n_type]) for n_type in self.node_types}
        attention_summary = {n_type: torch.zeros(node_embeddings_dict[n_type].size(0), 1, device=node_embeddings_dict[n_type].device) for n_type in self.node_types}
        
        # Project all nodes into multi-head spaces
        k_projections = {n_type: self.k_linear[n_type](embed).view(-1, self.num_heads, self.d_k) for n_type, embed in node_embeddings_dict.items()}
        q_projections = {n_type: self.q_linear[n_type](embed).view(-1, self.num_heads, self.d_k) for n_type, embed in node_embeddings_dict.items()}
        v_projections = {n_type: self.v_linear[n_type](embed).view(-1, self.num_heads, self.d_k) for n_type, embed in node_embeddings_dict.items()}

        # Iterate over distinct meta-relations
        for (src_type, rel_name, tgt_type), edge_index in adjacency_dict.items():
            if edge_index.numel() == 0:
                continue
                
            src_idx, tgt_idx = edge_index[0], edge_index[1]
            
            # Extract interacting nodes
            k_src = k_projections[src_type][src_idx]
            q_tgt = q_projections[tgt_type][tgt_idx]
            v_src = v_projections[src_type][src_idx]
            
            # Step 1: Compute Relation-Specific Heterogeneous Attention
            # Multiply source key with relation matrix
            r_att = self.relation_att[rel_name] # [heads, d_k, d_k]
            k_src_r = torch.einsum('bhd,hdc->bhc', k_src, r_att)
            # Dot product with target query
            att_scores = torch.sum(k_src_r * q_tgt, dim=-1) / math.sqrt(self.d_k) # [edges, heads]
            
            # Step 2: Extract Relation-Specific Value Messages
            r_msg = self.relation_msg[rel_name]
            v_src_r = torch.einsum('bhd,hdc->bhc', v_src, r_msg) # [edges, heads, d_k]
            
            # Step 3: Aggregate message packets to target nodes
            exp_att = torch.exp(att_scores)
            for head in range(self.num_heads):
                # Scatter add for weighted message values and attention denominators
                updated_embeddings[tgt_type].index_add_(
                    0, tgt_idx, v_src_r[:, head, :] * exp_att[:, head].unsqueeze(-1)
                )
                attention_summary[tgt_type].index_add_(
                    0, tgt_idx, exp_att[:, head].unsqueeze(-1)
                )
                
        # Step 4: Final Normalization and Target Residual Skip Connections
        final_out = {}
        for n_type in self.node_types:
            raw_agg = updated_embeddings[n_type] / (attention_summary[n_type] + 1e-9)
            # Squash heads back into singular representation dimensions
            linear_agg = self.a_linear[n_type](raw_agg.view(raw_agg.size(0), -1))
            
            # Integrate original identity states via alphabetic scale factors
            alpha = torch.sigmoid(self.skip[n_type])
            final_out[n_type] = alpha * linear_agg + (1.0 - alpha) * node_embeddings_dict[n_type]
            
        return final_out
