import random
from typing import (List)
class QueryAugmenter:
    """Generate variations of training queries to improve coverage"""
    
    TEMPLATES = [
        "{query}",
        "Cho tôi biết {query}",
        "Hỏi về {query}",
        "Giải thích {query}",
        "Tôi muốn tìm hiểu {query}",
    ]
    
    @staticmethod
    def augment_query(query: str, num_variations: int = 3) -> List[str]:
        """Generate paraphrased versions of a query"""
        variations = [query]  # Original
        
        # Template-based augmentation
        for _ in range(num_variations - 1):
            template = random.choice(QueryAugmenter.TEMPLATES)
            variations.append(template.format(query=query.lower()))
        
        return variations
    
    @staticmethod
    def augment_training_data(queries: List[str]) -> List[str]:
        """Augment all training queries"""
        augmented = []
        for q in queries:
            augmented.extend(QueryAugmenter.augment_query(q))
        return augmented