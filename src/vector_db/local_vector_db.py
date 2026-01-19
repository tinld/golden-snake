"""
Local Vector Database wrapper using Qdrant
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import uuid
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, EMBEDDING_DIM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalVectorDB:
    """
    Local vector database using Qdrant for storing and retrieving embeddings
    """
    
    def __init__(self, collection_name: str = "vector_collection"):
        """
        Initialize local vector database
        
        Args:
            collection_name: Name of the collection
        """
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        if QDRANT_API_KEY:
            self.client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                api_key=QDRANT_API_KEY
            )
        else:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Create collection if it doesn't exist
        self._init_collection()
        
        logger.info(f"Initialized Qdrant vector DB with collection '{collection_name}'")
    
    def _init_collection(self):
        """Initialize collection if it doesn't exist"""
        existing_collections = {col.name for col in self.client.get_collections().collections}
        
        if self.collection_name not in existing_collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add embeddings to the vector database
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            texts: List of original texts
            metadatas: List of metadata dicts for each embedding
            ids: Custom IDs for documents (generates if None)
            
        Returns:
            List of document IDs
        """
        if len(embeddings) != len(texts):
            raise ValueError(f"Embeddings ({len(embeddings)}) and texts ({len(texts)}) length mismatch")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in range(len(texts))]
        
        # Create points
        points = []
        for i, (embedding, text, metadata, point_id) in enumerate(zip(embeddings, texts, metadatas, ids)):
            payload = {
                "text": text,
                **metadata
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    payload=payload
                )
            )
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Added {len(texts)} embeddings to vector DB")
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filter (e.g., {"category": "technical"})
            
        Returns:
            Dict with 'ids', 'documents', 'distances', 'metadatas'
        """
        # Convert numpy array to list
        query_vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Build filter if provided
        search_filter = None
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            search_filter = Filter(must=conditions)
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=n_results,
            query_filter=search_filter
        )
        
        # Format results
        ids = []
        documents = []
        distances = []
        metadatas = []
        
        for result in search_results:
            ids.append(str(result.id))
            documents.append(result.payload.get("text", ""))
            distances.append(1 - result.score)  # Convert similarity to distance
            
            # Extract metadata (exclude 'text' field)
            metadata = {k: v for k, v in result.payload.items() if k != "text"}
            metadatas.append(metadata)
        
        return {
            "ids": ids,
            "documents": documents,
            "distances": distances,
            "metadatas": metadatas
        }
    
    def search_by_text(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Search with query text and embedding
        
        Args:
            query_text: Original query text
            query_embedding: Query embedding vector
            n_results: Number of results
            where: Optional metadata filter
            
        Returns:
            Dict with search results including text and distances
        """
        results = self.search(query_embedding, n_results=n_results, where=where)
        
        # Calculate similarity from distance
        distances = results["distances"]
        similarities = [1 - d for d in distances]  # Convert distance back to similarity
        
        return {
            "query_text": query_text,
            "results": [
                {
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "distance": distances[i],
                    "similarity": similarities[i],
                    "metadata": results["metadatas"][i]
                }
                for i in range(len(results["ids"]))
            ]
        }
    
    def update_embeddings(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None
    ):
        """
        Update existing embeddings
        
        Args:
            ids: Document IDs to update
            embeddings: New embeddings
            texts: New texts (optional)
            metadatas: New metadatas (optional)
        """
        points = []
        for i, (point_id, embedding) in enumerate(zip(ids, embeddings)):
            payload = {}
            if texts:
                payload["text"] = texts[i]
            if metadatas:
                payload.update(metadatas[i])
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    payload=payload
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Updated {len(ids)} embeddings in vector DB")
    
    def delete_embeddings(self, ids: List[str]):
        """Delete embeddings by ID"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=ids
            )
        )
        logger.info(f"Deleted {len(ids)} embeddings from vector DB")
    
    def get_by_id(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID"""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id]
            )
            
            if result:
                point = result[0]
                return {
                    "id": str(point.id),
                    "text": point.payload.get("text", ""),
                    "embedding": point.vector,
                    "metadata": {k: v for k, v in point.payload.items() if k != "text"}
                }
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
        
        return None
    
    def count(self) -> int:
        """Get total number of embeddings in collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error getting count: {e}")
            return 0
    
    def clear(self):
        """Clear all embeddings from collection"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._init_collection()
            logger.info(f"Cleared collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def persist(self):
        """Persist database to disk (Qdrant auto-persists)"""
        logger.info("Qdrant vector DB auto-persists to disk")


def get_vector_db(collection_name: str = "vector_collection") -> LocalVectorDB:
    """Factory function to get vector DB instance"""
    return LocalVectorDB(collection_name)


if __name__ == "__main__":
    # Test vector DB
    from classifier.embedder import get_embedder
    
    embedder = get_embedder()
    vector_db = LocalVectorDB()
    
    # Sample texts
    sample_texts = [
        "Làm sao sửa lỗi này?",
        "Giá sản phẩm bao nhiêu?",
        "Tính năng nào được hỗ trợ?"
    ]
    
    # Generate embeddings
    embeddings = embedder.embed_texts(sample_texts)
    
    # Add to DB
    ids = vector_db.add_embeddings(embeddings, sample_texts)
    print(f"Added {len(ids)} documents")
    print(f"Total documents in DB: {vector_db.count()}")
    
    # Search
    query = "Vấn đề kỹ thuật là gì?"
    query_embedding = embedder.embed_single(query)
    results = vector_db.search_by_text(query, query_embedding, n_results=2)
    
    print(f"\nSearch for: {query}")
    for result in results["results"]:
        print(f"  - {result['text']} (similarity: {result['similarity']:.4f})")