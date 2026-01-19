"""
MongoDB models for storing queries and sessions with rich metadata
"""
from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys
sys.path.insert(0, '..')

from src.config import MONGODB_URI, MONGODB_DATABASE, MONGODB_COLLECTION_QUERIES, MONGODB_COLLECTION_SESSIONS, MONGODB_COLLECTION_CATEGORIES


class MongoDBClient:
    """MongoDB client wrapper for managing collections"""
    
    def __init__(self, uri: str = MONGODB_URI, database: str = MONGODB_DATABASE):
        """Initialize MongoDB client"""
        self.client = MongoClient(uri)
        self.db = self.client[database]
        
        # Collections
        self.queries = self.db[MONGODB_COLLECTION_QUERIES]
        self.sessions = self.db[MONGODB_COLLECTION_SESSIONS]
        self.categories = self.db[MONGODB_COLLECTION_CATEGORIES]
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes for efficient querying"""
        # Query indexes
        self.queries.create_index([("original_query", ASCENDING)])
        self.queries.create_index([("primary_category", ASCENDING)])
        self.queries.create_index([("created_at", DESCENDING)])
        self.queries.create_index([("confidence_score", DESCENDING)])
        
        # Session indexes
        self.sessions.create_index([("session_id", ASCENDING)], unique=True)
        self.sessions.create_index([("query_id", ASCENDING)])
        self.sessions.create_index([("created_at", DESCENDING)])
        
        # Category indexes
        self.categories.create_index([("category_name", ASCENDING)], unique=True)
    
    def insert_query(self, query_data: Dict) -> str:
        """
        Insert a query record
        
        Args:
            query_data: Dictionary with query information
            
        Returns:
            Inserted document ID
        """
        if "created_at" not in query_data:
            query_data["created_at"] = datetime.utcnow()
        if "updated_at" not in query_data:
            query_data["updated_at"] = datetime.utcnow()
        
        result = self.queries.insert_one(query_data)
        return str(result.inserted_id)
    
    def insert_session(self, session_data: Dict) -> str:
        """
        Insert a RAG session record
        
        Args:
            session_data: Dictionary with session information
            
        Returns:
            Inserted document ID
        """
        if "created_at" not in session_data:
            session_data["created_at"] = datetime.utcnow()
        
        result = self.sessions.insert_one(session_data)
        return str(result.inserted_id)
    
    def insert_category(self, category_data: Dict) -> str:
        """
        Insert a category definition
        
        Args:
            category_data: Dictionary with category information
            
        Returns:
            Inserted document ID
        """
        if "created_at" not in category_data:
            category_data["created_at"] = datetime.utcnow()
        
        result = self.categories.insert_one(category_data)
        return str(result.inserted_id)
    
    def find_query_by_id(self, query_id: str) -> Optional[Dict]:
        """Find a query by its ID"""
        return self.queries.find_one({"_id": query_id})
    
    def find_query_by_text(self, query_text: str) -> Optional[Dict]:
        """Find a query by exact text match"""
        return self.queries.find_one({"original_query": query_text})
    
    def find_queries(
        self, 
        filter_dict: Optional[Dict] = None, 
        limit: int = 10,
        sort_by: str = "created_at",
        ascending: bool = False
    ) -> List[Dict]:
        """
        Find queries with filters
        
        Args:
            filter_dict: MongoDB filter dictionary
            limit: Number of results
            sort_by: Field to sort by
            ascending: Sort direction
            
        Returns:
            List of query documents
        """
        filter_dict = filter_dict or {}
        sort_direction = ASCENDING if ascending else DESCENDING
        
        cursor = self.queries.find(filter_dict).sort(sort_by, sort_direction).limit(limit)
        return list(cursor)
    
    def find_sessions(
        self,
        filter_dict: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Find sessions with filters"""
        filter_dict = filter_dict or {}
        cursor = self.sessions.find(filter_dict).sort("created_at", DESCENDING).limit(limit)
        return list(cursor)
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        total_queries = self.queries.count_documents({})
        total_sessions = self.sessions.count_documents({})
        
        # Category distribution using aggregation
        category_pipeline = [
            {"$group": {
                "_id": "$primary_category",
                "count": {"$sum": 1}
            }}
        ]
        category_results = list(self.queries.aggregate(category_pipeline))
        categories = {item["_id"]: item["count"] for item in category_results}
        
        # Average confidence by category
        confidence_pipeline = [
            {"$group": {
                "_id": "$primary_category",
                "avg_confidence": {"$avg": "$confidence_score"},
                "count": {"$sum": 1}
            }}
        ]
        confidence_results = list(self.queries.aggregate(confidence_pipeline))
        
        return {
            "total_queries": total_queries,
            "total_sessions": total_sessions,
            "categories": categories,
            "confidence_by_category": {
                item["_id"]: {
                    "avg": round(item["avg_confidence"], 3),
                    "count": item["count"]
                }
                for item in confidence_results
            }
        }
    
    def update_query(self, query_id: str, update_data: Dict) -> bool:
        """Update a query record"""
        update_data["updated_at"] = datetime.utcnow()
        result = self.queries.update_one(
            {"_id": query_id},
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    def delete_query(self, query_id: str) -> bool:
        """Delete a query record"""
        result = self.queries.delete_one({"_id": query_id})
        return result.deleted_count > 0
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()


# Global MongoDB instance
_mongodb_instance = None


def get_mongodb() -> MongoDBClient:
    """Get MongoDB client instance"""
    global _mongodb_instance
    if _mongodb_instance is None:
        _mongodb_instance = MongoDBClient()
    return _mongodb_instance


def init_mongodb():
    """Initialize MongoDB collections and indexes"""
    client = get_mongodb()
    print(f"MongoDB initialized successfully!")
    print(f"  Database: {client.db.name}")
    print(f"  Collections: {client.db.list_collection_names()}")
    return client


if __name__ == "__main__":
    # Test MongoDB connection
    client = init_mongodb()
    
    # Test insert
    test_query = {
        "original_query": "Test query",
        "primary_category": "test",
        "confidence_score": 0.95
    }
    print(test_query)
    
    query_id = client.insert_query(test_query)
    print(f"\nInserted test query: {query_id}")
    
    # Test find
    found = client.find_query_by_id("test_123")
    print(f"Found query: {found}")
    
    # Test delete
    deleted = client.delete_query("test_123")
    print(f"Deleted: {deleted}")
    
    client.close()
