"""
RAG (Retrieval-Augmented Generation) Pipeline - Hybrid Architecture
MongoDB for rich metadata + Qdrant for vector similarity
"""
import logging
from typing import Dict, List, Optional
import uuid
from datetime import datetime
import sys
sys.path.insert(0, '..')

from src.classifier import QueryClassifier, get_embedder
from src.vector_db import LocalVectorDB
from src.database import get_mongodb, get_qdrant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline with hybrid database architecture
    - MongoDB: Stores all query/session metadata
    - Qdrant: Stores only embeddings for similarity search
    """
    
    def __init__(
        self,
        classifier: Optional[QueryClassifier] = None,
        vector_db: Optional[LocalVectorDB] = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            classifier: QueryClassifier instance
            vector_db: LocalVectorDB instance (ChromaDB for backward compatibility)
        """
        self.embedder = get_embedder()
        self.classifier = classifier or QueryClassifier(self.embedder)
        self.vector_db = vector_db or LocalVectorDB()
        
        # Hybrid databases
        self.mongodb = get_mongodb()
        self.qdrant = get_qdrant()
    
    def ingest_queries(
        self,
        queries: List[str],
        category: Optional[str] = None
    ) -> List[str]:
        """
        Ingest and process queries
        
        Args:
            queries: List of queries to process
            category: Override category (if None, classify automatically)
            
        Returns:
            List of MongoDB document IDs
        """
        ids = []
        
        for query_text in queries:
            # Classify query
            classification = self.classifier.classify(query_text, top_k=1)[0]
            
            # Generate embedding
            embedding = self.embedder.embed_single(query_text)
            
            # Create MongoDB document with auto-generated ID
            mongo_id = str(uuid.uuid4())
            query_doc = {
                "_id": mongo_id,
                "original_query": query_text,
                "normalized_query": query_text.lower().strip(),
                "primary_category": category or classification["category"],
                "confidence_score": classification["confidence"],
                "language": "vi",
                "intent": classification.get("intent", ""),
                "keywords": [classification["category"]],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Save metadata to MongoDB
            self.mongodb.insert_query(query_doc)
            
            # Save vector to Qdrant with MongoDB ID reference
            self.qdrant.insert_vector(
                vector=embedding.tolist(),
                mongodb_id=mongo_id,
                metadata={"category": classification["category"]}
            )
            
            # Also add to ChromaDB for backward compatibility
            self.vector_db.add_embeddings(
                embedding.reshape(1, -1),
                [query_text],
                metadatas=[{
                    "category": classification["category"],
                    "confidence": str(classification["confidence"]),
                    "query_id": mongo_id
                }],
                ids=[mongo_id]
            )
            
            ids.append(mongo_id)
            logger.info(f"Ingested query [{mongo_id}]: {query_text[:50]}... -> {classification['category']}")
        
        return ids
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """
        Retrieve similar queries using Qdrant vector search + MongoDB metadata
        
        Args:
            query: Query text
            top_k: Number of results to retrieve
            
        Returns:
            Dict with retrieval results including full metadata from MongoDB
        """
        # Generate embedding for query
        query_embedding = self.embedder.embed_single(query)
        
        # Search in Qdrant for similar vectors
        qdrant_results = self.qdrant.search_similar(
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        # Fetch full metadata from MongoDB
        enriched_results = []
        for result in qdrant_results:
            mongodb_id = result["mongodb_id"]
            mongo_doc = self.mongodb.find_query_by_id(mongodb_id)
            
            if mongo_doc:
                enriched_results.append({
                    "mongodb_id": mongodb_id,
                    "qdrant_id": result["qdrant_id"],
                    "similarity_score": result["score"],
                    "query": mongo_doc.get("original_query", ""),
                    "category": mongo_doc.get("primary_category", ""),
                    "confidence": mongo_doc.get("confidence_score", 0.0),
                    "created_at": mongo_doc.get("created_at", ""),
                    "full_metadata": mongo_doc
                })
        
        return {
            "query": query,
            "results": enriched_results,
            "num_results": len(enriched_results)
        }
    
    def process_query(self, query: str, retrieve_context: bool = True, top_k: int = 5) -> Dict:
        """
        Complete query processing pipeline
        
        Args:
            query: User query
            retrieve_context: Whether to retrieve context
            top_k: Number of context documents to retrieve
            
        Returns:
            Dict with classification, context, and metadata
        """
        # Step 1: Classify query
        classifications = self.classifier.classify(query, top_k=3)
        
        # Step 2: Retrieve context if requested
        context = None
        if retrieve_context:
            context = self.retrieve(query, top_k=top_k)
        
        # Step 3: Create processing result
        result = {
            "query": query,
            "classifications": classifications,
            "primary_category": classifications[0]["category"] if classifications else "unknown",
            "context": context,
            "num_results": len(context["results"]) if context else 0
        }
        
        return result
    
    def save_session(self, query_text: str, response_text: str, num_retrieved: int = 0) -> str:
        """
        Save RAG session to MongoDB
        
        Args:
            query_text: Original query
            response_text: Generated response
            num_retrieved: Number of documents retrieved
            
        Returns:
            Session MongoDB ID
        """
        # Find or create query record
        query_doc = self.mongodb.find_query_by_text(query_text)
        
        if not query_doc:
            # Create new query record
            classifications = self.classifier.classify(query_text, top_k=1)
            query_embedding = self.embedder.embed_single(query_text)
            
            query_id = str(uuid.uuid4())
            query_doc = {
                "_id": query_id,
                "original_query": query_text,
                "normalized_query": query_text.lower().strip(),
                "primary_category": classifications[0]["category"] if classifications else "unknown",
                "confidence_score": classifications[0]["confidence"] if classifications else 0.0,
                "language": "vi"
            }
            self.mongodb.insert_query(query_doc)
            
            # Save vector
            self.qdrant.insert_vector(
                vector=query_embedding.tolist(),
                mongodb_id=query_id,
                metadata={"category": query_doc["primary_category"]}
            )
        
        # Create session record
        session_id = str(uuid.uuid4())
        session_doc = {
            "_id": session_id,
            "session_id": f"session_{query_doc['_id']}_{int(datetime.utcnow().timestamp() * 1000)}",
            "query_id": query_doc["_id"],
            "num_retrieved": num_retrieved,
            "retrieval_method": "similarity",
            "generated_response": response_text,
            "created_at": datetime.utcnow()
        }
        
        self.mongodb.insert_session(session_doc)
        
        logger.info(f"Saved RAG session: {session_doc['session_id']}")
        return session_id
    
    def get_query_history(self, category: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        Get query history from MongoDB
        
        Args:
            category: Filter by category (optional)
            limit: Number of records to return
            
        Returns:
            List of query records
        """
        filter_dict = {}
        if category:
            filter_dict["primary_category"] = category
        
        records = self.mongodb.find_queries(
            filter_dict=filter_dict,
            limit=limit,
            sort_by="created_at",
            ascending=False
        )
        
        return [
            {
                "id": str(r["_id"]),
                "query": r.get("original_query", ""),
                "category": r.get("primary_category", ""),
                "confidence": r.get("confidence_score", 0.0),
                "created_at": r.get("created_at", "")
            }
            for r in records
        ]
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics from MongoDB"""
        mongo_stats = self.mongodb.get_statistics()
        vector_count = self.qdrant.get_count()
        
        return {
            "total_queries": mongo_stats["total_queries"],
            "total_rag_sessions": mongo_stats["total_sessions"],
            "total_vectors_qdrant": vector_count,
            "categories": mongo_stats["categories"],
            "confidence_by_category": mongo_stats.get("confidence_by_category", {}),
            "classifier_categories": self.classifier.get_categories()
        }


if __name__ == "__main__":
    # Test RAG pipeline with hybrid architecture
    pipeline = RAGPipeline()
    
    # Sample queries
    sample_queries = [
        "Làm sao sửa lỗi này?",
        "Giá sản phẩm bao nhiêu?",
        "Tính năng nào được hỗ trợ?"
    ]
    
    # Ingest
    print("Ingesting queries...")
    ids = pipeline.ingest_queries(sample_queries)
    print(f"Ingested {len(ids)} queries")
    print(f"MongoDB IDs: {ids}")
    
    # Process query
    print("\nProcessing new query...")
    result = pipeline.process_query("Vấn đề kỹ thuật là gì?")
    print(f"Category: {result['primary_category']}")
    print(f"Retrieved: {result['num_results']} similar queries")
    
    # Show retrieved context
    if result['context'] and result['context']['results']:
        print("\nSimilar queries found:")
        for i, res in enumerate(result['context']['results'][:3], 1):
            print(f"  {i}. {res['query']} (score: {res['similarity_score']:.3f}, category: {res['category']})")
    
    # Statistics
    print("\nPipeline Statistics:")
    stats = pipeline.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
