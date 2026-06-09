import dgl
import torch
import random
from typing import List, Dict

class MetapathRandomWalker:
    """
    Executes meta-path-constrained random walks over a compiled DGL HeteroGraph
    to extract structural node sequence chains for Metapath2Vec++ representation learning.
    """
    def __init__(self, graph: dgl.DGLHeteroGraph):
        self.graph = graph

    def run_single_walk(self, start_node_id: int, metapath_schema: List[str]) -> List[str]:
        """
        Executes a single walk trace following a specified structural meta-path schema sequence.
        Example Schema Format: ['takes', 'teaches', 'aligned_with', 'required_by']
        Returns a trace list of mapped string tokens formatted as 'node_type_id'.
        """
        current_node = start_node_id
        # Extract the source entity node type directly from the first edge definition
        current_type = self.graph.to_canonical_etype(metapath_schema[0])[0]
        
        walk_trace = [f"{current_type}_{current_node}"]

        for edge_relation in metapath_schema:
            canonical_etype = self.graph.to_canonical_etype(edge_relation)
            src_type, rel, dst_type = canonical_etype
            
            if current_type != src_type:
                raise ValueError(f"Schema violation! Expected source '{src_type}', got current context '{current_type}'")

            # Retrieve actual structural neighbor indices from the graph topology
            neighbors = self.graph.successors(current_node, etype=canonical_etype).tolist()
            
            if not neighbors:
                # Handle structural dead-ends safely via an early exit back-off mechanism
                break
                
            current_node = random.choice(neighbors)
            current_type = dst_type
            walk_trace.append(f"{current_type}_{current_node}")
            
        return walk_trace

    def generate_metapath_corpus(
        self, 
        start_node_type: str, 
        metapath_schema: List[str], 
        walks_per_node: int = 10
    ) -> List[List[str]]:
        """Iterates across all available target nodes to compile an aggregated corpus log."""
        corpus = []
        num_nodes = self.graph.num_nodes(start_node_type)
        print(f"[PROCESS] Executing metapath walks for pattern: {metapath_schema}")
        
        for node_id in range(num_nodes):
            for _ in range(walks_per_node):
                walk_trace = self.run_single_walk(node_id, metapath_schema)
                if len(walk_trace) > 1: # Guard against zero-length structural dropouts
                    corpus.append(walk_trace)
        return corpus

if __name__ == "__main__":
    # Test script instantiation using simulated graph topology built above
    from graph_builder import CourseJobHINBuilder
    import numpy as np
    
    sim_builder = CourseJobHINBuilder()
    sim_builder.add_relation_edges('student', 'takes', 'course', np.array([0, 1]), np.array([10, 11]))
    sim_builder.add_relation_edges('course', 'teaches', 'course_skill', np.array([10, 11]), np.array([0, 1]))
    g = sim_builder.build_dgl_graph()
    
    walker = MetapathRandomWalker(g)
    # Define a target sample meta-path schema: Student -> Course -> Course Skill
    sample_path_schema = ['takes', 'teaches']
    generated_corpus = walker.generate_metapath_corpus('student', sample_path_schema, walks_per_node=2)
    print("Sample compiled traces:\n", generated_corpus[:3])
