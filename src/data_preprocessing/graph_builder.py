import dgl
import torch
import numpy as np
from typing import Dict, Tuple

class CourseJobHINBuilder:
    """
    Constructs a unified unified Course-Job Heterogeneous Information Network (HIN)
    consisting of 6 node types and their multi-relational edges using DGL framework.
    """
    def __init__(self):
        self.graph_data = {}

    def add_relation_edges(self, src_type: str, relation: str, dst_type: str, src_ids: np.ndarray, dst_ids: np.ndarray):
        """Adds a directed relation and its inverse counterpart into the raw edge buffer."""
        src_tensors = torch.tensor(src_ids, dtype=torch.int64)
        dst_tensors = torch.tensor(dst_ids, dtype=torch.int64)
        
        # Register canonical directed forward edge
        self.graph_data[(src_type, relation, dst_type)] = (src_tensors, dst_tensors)
        # Register corresponding inverse relation edge to support bi-directional structural message passing
        inverse_relation = f"rev_{relation}"
        self.graph_data[(dst_type, inverse_relation, src_type)] = (dst_tensors, src_tensors)

    def build_dgl_graph(self) -> dgl.DGLHeteroGraph:
        """Compiles registered relational buffers into a concrete DGL HeteroGraph instance."""
        print("[INFO] Initializing DGL HeteroGraph compilation process...")
        hin_graph = dgl.heterograph(self.graph_data)
        
        # Log basic structural metrics for pipeline auditability
        print("====== HIN Structural Summary ======")
        for ntype in hin_graph.ntypes:
            print(f"Node Type: [{ntype:<15}] Count: {hin_graph.num_nodes(ntype)}")
        for etype in hin_graph.canonical_etypes:
            print(f"Edge Schema: {etype} | Count: {hin_graph.num_edges(etype)}")
        print("====================================")
        return hin_graph

if __name__ == "__main__":
    # Generate mock interaction data to verify correct graph synthesis topology
    builder = CourseJobHINBuilder()
    
    # 1. Course Domain Interactions
    builder.add_relation_edges('student', 'takes', 'course', np.array([0, 0, 1]), np.array([10, 11, 11]))
    builder.add_relation_edges('course', 'precedes', 'course', np.array([10]), np.array([11]))
    builder.add_relation_edges('course', 'teaches', 'course_skill', np.array([10, 11]), np.array([0, 1]))
    
    # 2. Cross-Domain Bridging Connections (from BERT alignment script output)
    builder.add_relation_edges('course_skill', 'aligned_with', 'job_skill', np.array([0, 1]), np.array([2, 3]))
    
    # 3. Job Domain Interactions
    builder.add_relation_edges('employee', 'applies', 'job', np.array([0, 1]), np.array([5, 6]))
    builder.add_relation_edges('job', 'requires', 'job_skill', np.array([5, 6]), np.array([2, 3]))
    
    test_graph = builder.build_dgl_graph()
