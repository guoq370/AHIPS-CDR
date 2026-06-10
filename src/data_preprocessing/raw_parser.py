import pandas as pd
import torch
import numpy as np

class RawDataPipeline:
    """
    Parses original Coursera and LinkedIn text-based CSV tables 
    and compiles them into multi-relational tensors for AHIPS-CDR.
    """
    def __init__(self, coursera_path: str, linkedin_path: str):
        self.course_behaviors = pd.read_csv(os.path.join(coursera_path, "raw_course_behaviors.csv"))
        self.course_features = pd.read_csv(os.path.join(coursera_path, "raw_course_features.csv"))
        self.job_interactions = pd.read_csv(os.path.join(linkedin_path, "raw_job_interactions.csv"))
        self.job_features = pd.read_csv(os.path.join(linkedin_path, "raw_job_features.csv"))

    def build_semantic_bridge(self) -> dict:
        """
        Extracts and aligns skills across domains to construct the semantic cross-domain bridge.
        """
        # Collect all unique skill tokens across both platforms
        coursera_skills = set(";".join(self.course_features['skills_tagged'].dropna()).split(";"))
        linkedin_skills = set(";".join(self.job_features['skills_required'].dropna()).split(";"))
        
        shared_skills = list(coursera_skills.intersection(linkedin_skills))
        skill_to_id = {skill: idx for idx, skill in enumerate(shared_skills)}
        
        print(f"[PREPROCESSING] Aligned {len(shared_skills)} shared skill bridging entities safely.")
        return skill_to_id

    def generate_training_tensors(self) -> tuple:
        """
        Transforms aligned logs into clean torch tensors for downstream dataloaders.
        """
        # Encode string IDs into clean integer continuous categorical indices
        user_map = {uid: idx for idx, uid in enumerate(self.course_behaviors['student_id'].unique())}
        item_map = {iid: idx for idx, iid in enumerate(self.course_features['course_id'].unique())}
        
        encoded_users = self.course_behaviors['student_id'].map(user_map).values
        encoded_items = self.course_behaviors['course_id'].map(item_map).values
        ratings = self.course_behaviors['rating'].values
        
        # Convert to torch tensor space
        user_tensor = torch.tensor(encoded_users, dtype=torch.long)
        item_tensor = torch.tensor(encoded_items, dtype=torch.long)
        rating_tensor = torch.tensor(ratings, dtype=torch.float32)
        
        return user_tensor, item_tensor, rating_tensor
