"""
Database models for MongoDB storage
Simple dataclasses for queries, sessions, and categories
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import uuid


@dataclass
class QueryRecord:
    """
    Model for storing classified queries in MongoDB
    """
    original_query: str
    normalized_query: str
    primary_category: str
    confidence_score: float = 0.0
    id: Optional[str] = None
    secondary_category: Optional[str] = None
    user_id: Optional[str] = None
    language: str = "vi"
    source: str = "user_input"
    intent: Optional[str] = None
    keywords: Optional[List[str]] = field(default_factory=list)
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """Convert to MongoDB document format"""
        doc = asdict(self)
        doc["_id"] = doc.pop("id")  # MongoDB uses _id
        return doc
    
    @classmethod
    def from_dict(cls, doc: Dict):
        """Create instance from MongoDB document"""
        doc = doc.copy()
        if "_id" in doc:
            doc["id"] = str(doc.pop("_id"))
        return cls(**doc)


@dataclass
class CategoryDefinition:
    """
    Define query categories and their descriptions for MongoDB
    """
    category_name: str
    id: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[List[str]] = field(default_factory=list)
    is_active: bool = True
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """Convert to MongoDB document format"""
        doc = asdict(self)
        doc["_id"] = doc.pop("id")
        return doc
    
    @classmethod
    def from_dict(cls, doc: Dict):
        """Create instance from MongoDB document"""
        doc = doc.copy()
        if "_id" in doc:
            doc["id"] = str(doc.pop("_id"))
        return cls(**doc)


@dataclass
class RagSession:
    """
    Track RAG retrieval sessions in MongoDB
    """
    query_id: str
    session_id: str
    num_retrieved: int = 0
    id: Optional[str] = None
    retrieval_method: str = "similarity"
    top_k: int = 5
    generated_response: Optional[str] = None
    relevance_score: float = 0.0
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """Convert to MongoDB document format"""
        doc = asdict(self)
        doc["_id"] = doc.pop("id")
        return doc
    
    @classmethod
    def from_dict(cls, doc: Dict):
        """Create instance from MongoDB document"""
        doc = doc.copy()
        if "_id" in doc:
            doc["id"] = str(doc.pop("_id"))
        return cls(**doc)

if __name__ == "__main__":
    # Test MongoDB-compatible models
    print("Testing MongoDB-compatible models...")
    
    # Test QueryRecord
    query = QueryRecord(
        original_query="Test query",
        normalized_query="test query",
        primary_category="test",
        confidence_score=0.95
    )
    print(f"\nQueryRecord:")
    print(f"  ID: {query.id}")
    print(f"  Dict: {query.to_dict()}")
    
    # Test CategoryDefinition
    category = CategoryDefinition(
        category_name="Test Category",
        description="Test description",
        keywords=["test", "category"]
    )
    print(f"\nCategoryDefinition:")
    print(f"  ID: {category.id}")
    print(f"  Dict: {category.to_dict()}")
    
    # Test RagSession
    session = RagSession(
        query_id="test_query_123",
        session_id="session_123",
        num_retrieved=5
    )
    print(f"\nRagSession:")
    print(f"  ID: {session.id}")
    print(f"  Dict: {session.to_dict()}")
    
    print("\n✓ All models work with MongoDB!")

