import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from typing import List, Tuple, Dict

class SkillSemanticAligner:
    """
    Leverages a pre-trained BERT engine to encode unstructured skills text
    from different domains and discover high-confidence semantic cross-domain mappings.
    """
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _get_embeddings(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """Generates mean-pooled BERT representations for a given list of strings."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded_input = self.tokenizer(
                batch_texts, padding=True, truncation=True, 
                max_length=64, return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Apply mean pooling over token embeddings
                attention_mask = encoded_input['attention_mask'].unsqueeze(-1)
                token_embeddings = model_output.last_hidden_state
                num_tokens = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
                pooled = (token_embeddings * attention_mask).sum(dim=1) / num_tokens
                all_embeddings.append(pooled.cpu().numpy())
                
        return np.vstack(all_embeddings)

    def compute_alignment_edges(
        self, 
        course_skills: List[str], 
        job_skills: List[str], 
        threshold: float = 0.75
    ) -> List[Tuple[int, int, float]]:
        """
        Calculates pairwise cosine similarity between Coursera and LinkedIn skills.
        Returns a list of high-confidence alignment tuples (course_skill_idx, job_skill_idx, similarity).
        """
        print(f"[INFO] Encoding {len(course_skills)} course skills and {len(job_skills)} job skills...")
        cs_embeds = self._get_embeddings(course_skills)
        js_embeds = self._get_embeddings(job_skills)

        # Normalize representations for efficient cosine similarity calculation
        cs_norm = cs_embeds / np.linalg.norm(cs_embeds, axis=1, keepdims=True)
        js_norm = js_embeds / np.linalg.norm(js_embeds, axis=1, keepdims=True)

        print("[INFO] Computing pairwise similarity matrix...")
        similarity_matrix = np.dot(cs_norm, js_norm.T)
        
        # Filter connections according to the predefined threshold
        matching_indices = np.argwhere(similarity_matrix >= threshold)
        alignment_edges = []
        
        for idx in matching_indices:
            cs_idx, js_idx = int(idx[0]), int(idx[1])
            score = float(similarity_matrix[cs_idx, js_idx])
            alignment_edges.append((cs_idx, js_idx, score))
            
        print(f"[SUCCESS] Extracted {len(alignment_edges)} semantic alignment edges between domains.")
        return alignment_edges

if __name__ == "__main__":
    # Sanity verification block
    aligner = SkillSemanticAligner(device='cpu')
    c_skills = ["Python programming", "Database indexing", "Machine learning basics"]
    j_skills = ["Python Developer", "SQL optimization", "Data Science Engineer"]
    edges = aligner.compute_alignment_edges(c_skills, j_skills, threshold=0.70)
    print("Sample generated edges:", edges)
