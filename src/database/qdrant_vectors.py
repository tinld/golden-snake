"""
Qdrant Vector Database - Simplified to store only embeddings
Links to MongoDB via document IDs
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional
import uuid
import sys
sys.path.insert(0, '..')

from src.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_VECTORS,
    EMBEDDING_DIM
)


class QdrantVectorDB:
    """
    Simplified Qdrant client for vector storage only
    Metadata is stored in MongoDB, only vectors + MongoDB ID stored here
    """
    
    def __init__(self, collection_name: str = QDRANT_COLLECTION_VECTORS):
        self.client = self._get_client()
        self.collection_name = collection_name
    
    def _get_client(self) -> QdrantClient:
        """Create Qdrant client"""
        if QDRANT_API_KEY:
            return QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                api_key=QDRANT_API_KEY
            )
        else:
            return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    def create_collection(self):
        """Create vector collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' already exists")
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
    
    def collection_exists(self) -> bool:
        """Check if collection exists"""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except:
            return False
        
    def insert_vector(
        self,
        vector: List[float],
        mongodb_id: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Insert a single vector with minimal metadata
        
        Args:
            vector: Embedding vector
            mongodb_id: Reference to MongoDB document ID
            metadata: Optional minimal metadata (e.g., category for filtering)
            
        Returns:
            Qdrant point ID
        """
        point_id = str(uuid.uuid4())
        
        # Minimal payload - just MongoDB reference and optional category
        payload = {
            "mongodb_id": mongodb_id,
            **(metadata or {})
        }
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        return point_id
    
    def upsert_vectors(self, collection_name, points: List[PointStruct]):
        """Upsert multiple vectors"""
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    def insert_vectors_batch(
        self,
        vectors: List[List[float]],
        mongodb_ids: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Insert multiple vectors in batch
        
        Args:
            vectors: List of embedding vectors
            mongodb_ids: List of MongoDB document IDs
            metadatas: Optional list of minimal metadata dicts
            
        Returns:
            List of Qdrant point IDs
        """
        if metadatas is None:
            metadatas = [{}] * len(vectors)
        
        points = []
        point_ids = []
        
        for vector, mongodb_id, metadata in zip(vectors, mongodb_ids, metadatas):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            payload = {
                "mongodb_id": mongodb_id,
                **(metadata or {})
            }
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return point_ids
    
    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding
            limit: Number of results
            score_threshold: Minimum similarity score
            filter_dict: Optional filter (e.g., {"category": "technical"})
            
        Returns:
            List of results with MongoDB IDs and scores
        """
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_vector,
            "limit": limit
        }
        
        if score_threshold:
            search_params["score_threshold"] = score_threshold
        
        if filter_dict:
            # Build Qdrant filter
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            search_params["query_filter"] = models.Filter(must=conditions)
        
        results = self.client.search(**search_params)
        
        return [
            {
                "qdrant_id": str(hit.id),
                "mongodb_id": hit.payload.get("mongodb_id"),
                "score": hit.score,
                "metadata": hit.payload
            }
            for hit in results
        ]
    
    def delete_by_mongodb_id(self, mongodb_id: str):
        """Delete vector by MongoDB document ID"""
        # Find points with this MongoDB ID
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="mongodb_id",
                        match=models.MatchValue(value=mongodb_id)
                    )
                ]
            ),
            limit=100
        )
        
        points_to_delete = [str(point.id) for point, _ in [results] for point in results[0]]
        
        if points_to_delete:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=points_to_delete)
            )
    
    def get_count(self) -> int:
        """Get total number of vectors"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except:
            return 0
    # ...existing code...
    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Delete a collection"""
        name = collection_name or self.collection_name
        try:
            self.client.delete_collection(name)
            print(f"Deleted collection: {name}")
            return True
        except Exception as e:
            print(f"Failed to delete collection '{name}': {e}")
            return False

    def delete_all_points(self, collection_name: Optional[str] = None) -> bool:
        """Delete all points but keep the collection"""
        name = collection_name or self.collection_name
        try:
            from qdrant_client.http.models import MatchAll
            self.client.delete(collection_name=name, points_selector=MatchAll())
            print(f"Deleted all points in collection: {name}")
            return True
        except Exception as e:
            print(f"Failed to delete points in '{name}': {e}")
            return False

# Global Qdrant instance
_qdrant_instance = None


def get_qdrant() -> QdrantVectorDB:
    """Get Qdrant vector DB instance"""
    global _qdrant_instance
    if _qdrant_instance is None:
        _qdrant_instance = QdrantVectorDB()
    return _qdrant_instance


def init_qdrant():
    """Initialize Qdrant collection"""
    qdrant = get_qdrant()
    qdrant.create_collection()
    print("Qdrant vector database initialized successfully!")
    return qdrant


if __name__ == "__main__":
    # Test Qdrant
    qdrant = init_qdrant()
    
    # Test insert
    test_vector = [0.1] * EMBEDDING_DIM
    point_id = qdrant.insert_vector(
        vector=test_vector,
        mongodb_id="test_mongo_123",
        metadata={"category": "test"}
    )
    print(f"\nInserted vector: {point_id}")
    
    # Test search
    results = qdrant.search_similar(test_vector, limit=1)
    print(f"Search results: {results}")
    
    # Test delete
    qdrant.delete_by_mongodb_id("test_mongo_123")
    print("Deleted test vector")
