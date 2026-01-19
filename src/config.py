"""
Configuration settings for RAG Query Classifier
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# PhoBERT Model settings
PHOBERT_MODEL = "vinai/phobert-base"  # or "vinai/phobert-large"
EMBEDDING_DIM = 768
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256

# Vector DB settings
CHROMA_COLLECTION_NAME = "vietnamese_queries"
SIMILARITY_THRESHOLD = 0.7

# Qdrant Database settings (vectors only)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # Optional for cloud/authentication
QDRANT_COLLECTION_VECTORS = "query_vectors"  # Only store embeddings

# MongoDB settings (metadata storage)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "rag_business")
MONGODB_COLLECTION_QUERIES = "query_records"
MONGODB_COLLECTION_SESSIONS = "rag_sessions"
MONGODB_COLLECTION_CATEGORIES = "category_definitions"

# Logging
LOG_LEVEL = "INFO"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)
