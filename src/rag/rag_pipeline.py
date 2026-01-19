"""
RAG (Retrieval-Augmented Generation) Pipeline
Combines query classification, vector search, and response generation
"""
import logging
from typing import Dict, List, Optional
import uuid
import sys
sys.path.insert(0, '../src')

from src.classifier import QueryClassifier, get_embedder
from src.vector_db import LocalVectorDB
from src.database import get_mongodb, get_qdrant
from src.database import QueryRecord, RagSession


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for Vietnamese query processing
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
            vector_db: LocalVectorDB instance
        """
        self.embedder = get_embedder()
        self.classifier = classifier or QueryClassifier(self.embedder)
        self.vector_db = vector_db or LocalVectorDB()
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
            List of query IDs
        """
        ids = []
        
        for query_text in queries:
            # Classify query
            classification = self.classifier.classify(query_text, top_k=1)[0]
            
            # Generate embedding
            embedding = self.embedder.embed_single(query_text)
            
            # Create database record with auto-generated ID
            record_id = str(uuid.uuid4())
            record = QueryRecord(
                id=record_id,
                original_query=query_text,
                normalized_query=query_text.lower().strip(),
                primary_category=category or classification["category"],
                confidence_score=classification["confidence"],
                language="vi",
                intent=classification.get("intent", ""),
                keywords=",".join([classification["category"]])
            )
            
            # Save to Qdrant (data is persisted immediately, no commit needed)
            self.mongodb.insert_query(record, vector=embedding.tolist())
            
            # Add to vector DB (use plain UUID as ID)
            vector_ids = self.vector_db.add_embeddings(
                embedding.reshape(1, -1),
                [query_text],
                metadatas=[{
                    "category": classification["category"],
                    "confidence": str(classification["confidence"]),
                    "query_id": record_id
                }],
                ids=[record_id]  # Use plain UUID without prefix
            )
            
            ids.append(vector_ids[0])
            logger.info(f"Ingested query: {query_text[:50]}... -> {classification['category']}")
        
        return ids
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """
        Retrieve similar queries from vector DB
        
        Args:
            query: Query text
            top_k: Number of results to retrieve
            
        Returns:
            Dict with retrieval results
        """
        # Generate embedding for query
        query_embedding = self.embedder.embed_single(query)
        
        # Search in vector DB
        search_results = self.vector_db.search_by_text(query, query_embedding, n_results=top_k)
        
        return search_results
    
    def process_query(self, query: str, retrieve_context: bool = True, top_k: int = 5) -> Dict:
        """
        Complete query processing pipeline
        
        Args:
            query: User query
            retrieve_context: Whether to retrieve context from vector DB
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
        Save RAG session to database
        
        Args:
            query_text: Original query
            response_text: Generated response
            num_retrieved: Number of documents retrieved
            
        Returns:
            Session ID
        """
        # Generate embedding and search for existing query
        query_embedding = self.embedder.embed_single(query_text)
        search_results = self.db.search_queries(query_embedding.tolist(), limit=1)
        
        # Get or create query record
        if search_results and search_results[0]["record"].original_query == query_text:
            query_record = search_results[0]["record"]
        else:
            # Create new record if not found
            classifications = self.classifier.classify(query_text, top_k=1)
            query_id = str(uuid.uuid4())
            query_record = QueryRecord(
                id=query_id,
                original_query=query_text,
                normalized_query=query_text.lower().strip(),
                primary_category=classifications[0]["category"] if classifications else "unknown",
                confidence_score=classifications[0]["confidence"] if classifications else 0.0
            )
            self.db.insert_query(query_record, vector=query_embedding.tolist())
        
        # Create session record with auto-generated ID
        session_id = str(uuid.uuid4())
        session = RagSession(
            id=session_id,
            session_id=f"session_{query_record.id}_{int(__import__('time').time() * 1000)}",
            query_id=query_record.id,
            num_retrieved=num_retrieved,
            generated_response=response_text
        )
        
        self.db.insert_session(session)
        
        logger.info(f"Saved RAG session: {session.session_id}")
        return session_id
    
    def get_query_history(self, category: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        Get query history
        
        Args:
            category: Filter by category (optional)
            limit: Number of records to return
            
        Returns:
            List of query records
        """
        from qdrant_client.http import models as qmodels
        
        # Build filter if category specified
        scroll_filter = None
        if category:
            scroll_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="primary_category",
                        match=qmodels.MatchValue(value=category)
                    )
                ]
            )
        
        # Scroll through records
        records, _ = self.db.client.scroll(
            collection_name="query_records",
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        return [
            {
                "id": str(r.id),
                "query": r.payload.get("original_query", ""),
                "category": r.payload.get("primary_category", ""),
                "confidence": r.payload.get("confidence_score", 0.0),
                "created_at": r.payload.get("created_at", "")
            }
            for r in records
        ]
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        # Get counts from Qdrant collections
        try:
            query_collection = self.db.client.get_collection("query_records")
            total_queries = query_collection.points_count
        except:
            total_queries = 0
        
        try:
            session_collection = self.db.client.get_collection("rag_sessions")
            total_sessions = session_collection.points_count
        except:
            total_sessions = 0
        
        total_vectors = self.vector_db.count()
        
        # Category distribution - scroll through all queries
        categories = {}
        try:
            records, _ = self.db.client.scroll(
                collection_name="query_records",
                limit=1000,
                with_payload=["primary_category"],
                with_vectors=False
            )
            for record in records:
                cat = record.payload.get("primary_category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
        except:
            pass
        
        return {
            "total_queries": total_queries,
            "total_rag_sessions": total_sessions,
            "total_vectors": total_vectors,
            "categories": categories,
            "classifier_categories": self.classifier.get_categories()
        }


if __name__ == "__main__":
    # Test RAG pipeline
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
    
    # Process query
    print("\nProcessing new query...")
    result = pipeline.process_query("Vấn đề kỹ thuật là gì?")
    print(f"Category: {result['primary_category']}")
    print(f"Retrieved: {result['num_results']} similar queries")
    
    # Statistics
    print("\nPipeline Statistics:")
    stats = pipeline.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
