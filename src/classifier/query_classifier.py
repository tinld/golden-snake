"""
Query Classifier with category prediction using Qdrant
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import sys
import json
from pathlib import Path
sys.path.insert(0, '../src')

from .embedder import PhoBertEmbedder, get_embedder
from src.config import SIMILARITY_THRESHOLD, EMBEDDING_DIM
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Classify Vietnamese queries into categories using Qdrant vector database
    """
    
    def __init__(self, embedder: Optional[PhoBertEmbedder] = None, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """
        Initialize query classifier with Qdrant
        
        Args:
            embedder: PhoBertEmbedder instance (creates new if None)
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
        """
        self.embedder = embedder or get_embedder()
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = "categories"
        # self._init_qdrant_collection()
        self.sync_categories_from_json()
    
    def _init_qdrant_collection(self):
        """Initialize Qdrant collection for categories"""
        try:
            self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
    
    def sync_categories_from_json(self, json_path: Optional[str] = None):
        """
        Sync categories from JSON file to Qdrant
        
        Args:
            json_path: Path to categories.json (defaults to same directory)
        """
        if json_path is None:
            json_path = Path(__file__).parent / "categories.json"
        else:
            json_path = Path(json_path)
        
        if not json_path.exists():
            logger.error(f"Categories file not found: {json_path}")
            return
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            categories = data.get("categories", {})
        
        logger.info(f"Syncing {len(categories)} categories to Qdrant...")
        
        points = []
        for idx, (category_name, category_info) in enumerate(categories.items()):
            examples = category_info.get("examples", [])
            keywords = category_info.get("keywords", [])
            
            if not examples:
                logger.warning(f"Category '{category_name}' has no examples, skipping")
                continue
            
            # Generate embedding from examples
            embeddings = self.embedder.embed_texts(examples)
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Create point for Qdrant
            points.append(
                PointStruct(
                    id=idx,
                    vector=avg_embedding.tolist(),
                    payload={
                        "category_name": category_name,
                        "keywords": keywords,
                        "examples": examples
                    }
                )
            )
            logger.info(f"Prepared category: {category_name}")
        
        # Upsert all categories to Qdrant
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully synced {len(points)} categories to Qdrant")
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def _keyword_match_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword match score"""
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        return matches / len(keywords) if keywords else 0.0
    
    def classify(self, query: str, top_k: int = 3, use_keywords: bool = True) -> List[Dict]:
        """
        Classify a query into categories using Qdrant
        
        Args:
            query: Vietnamese query text
            top_k: Number of top categories to return
            use_keywords: Whether to boost score using keyword matching
            
        Returns:
            List of dicts with 'category', 'score', and 'confidence'
        """
        # Get query embedding
        query_embedding = self.embedder.embed_single(query)
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        if not search_results:
            logger.warning("No categories found in Qdrant. Run sync_categories_from_json() first.")
            return []
        
        # Process results
        results = []
        for i, hit in enumerate(search_results):
            category_name = hit.payload.get("category_name", "Unknown")
            keywords = hit.payload.get("keywords", [])
            semantic_score = hit.score
            
            # Keyword matching boost
            keyword_score = 0.0
            if use_keywords and keywords:
                keyword_score = self._keyword_match_score(query, keywords)
            
            # Combined score (weighted average)
            combined_score = 0.7 * semantic_score + 0.3 * keyword_score
            confidence = max(0, min(1, combined_score))
            
            results.append({
                "rank": i + 1,
                "category": category_name,
                "score": combined_score,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "confidence": confidence,
                "is_confident": confidence >= SIMILARITY_THRESHOLD
            })
        
        # Re-sort by combined score
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result["rank"] = i + 1
        
        return results
    
    def batch_classify(self, queries: List[str], top_k: int = 1) -> List[List[Dict]]:
        """
        Classify multiple queries
        
        Args:
            queries: List of Vietnamese queries
            top_k: Number of top categories per query
            
        Returns:
            List of classification results for each query
        """
        results = []
        for query in queries:
            results.append(self.classify(query, top_k=top_k))
        return results
    
    def get_categories(self) -> List[str]:
        """Get list of all defined categories from Qdrant"""
        try:
            # Scroll through all points to get category names
            records, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100
            )
            return [record.payload.get("category_name", "Unknown") for record in records]
        except Exception as e:
            logger.error(f"Error fetching categories: {e}")
            return []


if __name__ == "__main__":
    # Test classifier
    classifier = QueryClassifier()
    
    test_queries = [
        "Làm sao sửa lỗi này?",
        "Giá sản phẩm bao nhiêu?",
        "Tính năng nào được hỗ trợ?",
        "Đặt lại mật khẩu thế nào?"
    ]
    
    for query in test_queries:
        results = classifier.classify(query, top_k=3)
        print(f"\nQuery: {query}")
        for result in results:
            print(f"  {result['rank']}. {result['category']} - Score: {result['score']:.4f} (Confident: {result['is_confident']})")
