from typing import Dict, List
from datetime import datetime

class FeedbackLearner:
    """Learn from low-confidence predictions and user corrections"""
    
    def __init__(self, pipeline, similarity_threshold: float = 0.7):
        self.pipeline = pipeline
        self.similarity_threshold = similarity_threshold
        self.feedback_buffer = []
    
    def collect_feedback(self, query: str, result: Dict, user_correction: str = None):
        """Collect queries with low similarity or user corrections"""
        max_similarity = max([r["similarity_score"] for r in result["context"]["results"]], default=0)
        
        if max_similarity < self.similarity_threshold or user_correction:
            self.feedback_buffer.append({
                "query": query,
                "predicted_category": result["classifications"][0]["category"],
                "correct_category": user_correction or result["classifications"][0]["category"],
                "max_similarity": max_similarity,
                "timestamp": datetime.now()
            })
    
    def retrain_from_feedback(self, batch_size: int = 10):
        """Re-ingest corrected queries to improve embeddings"""
        if len(self.feedback_buffer) >= batch_size:
            queries_to_add = [f["query"] for f in self.feedback_buffer[:batch_size]]
            self.pipeline.ingest_queries(queries_to_add)
            self.feedback_buffer = self.feedback_buffer[batch_size:]
            return len(queries_to_add)
        return 0