# Vietnamese RAG Question Classifier

A production-ready RAG (Retrieval-Augmented Generation) system for classifying Vietnamese user questions and documents' answers, storing them in a local vector database, and analyzing them with semantic search capabilities.

## Features

✨ **Core Features:**
- **PhoBERT-based Embeddings**: Vietnamese language model for semantic understanding
- **Query Classification**: Automatically classify questions into predefined categories
- **Local Vector Database**: Chroma-based storage for fast semantic search
- **SQLite Integration**: Persistent storage for query history and metadata
- **RAG Pipeline**: Complete workflow from ingestion to retrieval and analysis
- **Flexible Categories**: Support for custom category definitions
- **Keyword Boosting**: Hybrid approach combining semantic similarity with keyword matching

## Project Structure

```
rag-business-snake/
├── src/
│   ├── classifier/           # PhoBERT embeddings and query classification
│   │   ├── embedder.py       # Embedding generation
│   │   ├── query_classifier.py
│   │   └── __init__.py
│   ├── vector_db/            # Local vector database (Chroma)
│   │   ├── local_vector_db.py
│   │   └── __init__.py
│   ├── database/             # SQLAlchemy models and operations
│   │   ├── models.py
│   │   └── __init__.py
│   ├── rag/                  # RAG pipeline orchestration
│   │   ├── rag_pipeline.py
│   │   └── __init__.py
│   ├── utils/                # Utility functions
│   │   ├── text_utils.py
│   │   └── __init__.py
│   └── config.py             # Configuration settings
├── examples/                 # Example scripts
│   ├── example_1_classification.py
│   ├── example_2_vector_db.py
│   └── example_3_rag_pipeline.py
├── data/                     # Data directory (created at runtime)
│   ├── chroma_db/            # Vector database storage
│   └── queries.db            # SQLite database
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### 1. Clone and Setup

```bash
cd rag-business-snake
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `torch` & `transformers` - Deep learning framework
- `sentence-transformers` - Semantic embeddings
- `chromadb` - Vector database
- `sqlalchemy` - ORM for database operations
- `langchain` - RAG utilities
- And other supporting libraries

### 4. Initialize Database

```bash
python -c "from src.database import init_db; init_db()"
```

## Quick Start

### Example 1: Query Classification

```python
from src.classifier import QueryClassifier

classifier = QueryClassifier()

query = "Làm sao sửa lỗi ứng dụng bị crash?"
results = classifier.classify(query, top_k=2)

for result in results:
    print(f"{result['category']} - Confidence: {result['confidence']:.2%}")
```

Run the example:
```bash
cd examples
python example_1_classification.py
```

### Example 2: Vector Database Operations

```python
from src.classifier import get_embedder
from src.vector_db import LocalVectorDB

embedder = get_embedder()
vector_db = LocalVectorDB()

# Generate embeddings and store
texts = ["Câu hỏi 1", "Câu hỏi 2", "Câu hỏi 3"]
embeddings = embedder.embed_texts(texts)
ids = vector_db.add_embeddings(embeddings, texts)

# Search
query_embedding = embedder.embed_single("Tìm kiếm tương tự")
results = vector_db.search(query_embedding, n_results=5)
```

Run the example:
```bash
cd examples
python example_2_vector_db.py
```

### Example 3: Complete RAG Pipeline

```python
from src.rag import RAGPipeline
from src.database import init_db

init_db()
pipeline = RAGPipeline()

# Ingest queries
queries = ["Câu hỏi 1", "Câu hỏi 2"]
pipeline.ingest_queries(queries)

# Process new query with context retrieval
result = pipeline.process_query("Câu hỏi mới?", retrieve_context=True)
print(result["classifications"])
print(result["context"])

# Save session
pipeline.save_session("Câu hỏi", "Phản hồi")

# Get statistics
stats = pipeline.get_statistics()
```

Run the example:
```bash
cd examples
python example_3_rag_pipeline.py
```

## Configuration

Edit `src/config.py` to customize:

```python
# Model settings
PHOBERT_MODEL = "vinai/phobert-base"  # or "vinai/phobert-large"
EMBEDDING_DIM = 768
BATCH_SIZE = 32

# Vector DB
CHROMA_COLLECTION_NAME = "vietnamese_queries"
SIMILARITY_THRESHOLD = 0.7

# Database
DB_URL = f"sqlite:///{SQLITE_DB_PATH}"
```

## API Reference

### QueryClassifier

**Methods:**

- `classify(query: str, top_k: int = 1) -> List[Dict]`
  - Classify a query into categories
  - Returns category, confidence score, and semantic similarity

- `batch_classify(queries: List[str], top_k: int = 1) -> List[List[Dict]]`
  - Classify multiple queries

- `add_categories(categories: Dict[str, Dict])`
  - Add custom categories with examples

- `get_categories() -> List[str]`
  - Get all available categories

### LocalVectorDB

**Methods:**

- `add_embeddings(embeddings, texts, metadatas=None, ids=None) -> List[str]`
  - Store embeddings and return IDs

- `search(query_embedding, n_results=5, where=None) -> Dict`
  - Semantic search for similar embeddings

- `search_by_text(query_text, query_embedding, n_results=5) -> Dict`
  - Search with text and embedding

- `update_embeddings(ids, embeddings, texts=None, metadatas=None)`
  - Update existing embeddings

- `delete_embeddings(ids: List[str])`
  - Delete embeddings by ID

- `count() -> int`
  - Get total embeddings count

### RAGPipeline

**Methods:**

- `ingest_queries(queries: List[str], category=None) -> List[str]`
  - Ingest and classify queries

- `retrieve(query: str, top_k: int = 5) -> Dict`
  - Retrieve similar queries from vector DB

- `process_query(query: str, retrieve_context=True, top_k=5) -> Dict`
  - Complete query processing with classification and retrieval

- `save_session(query_text, response_text, num_retrieved=0) -> int`
  - Save RAG session to database

- `get_query_history(category=None, limit=10) -> List[Dict]`
  - Retrieve query history

- `get_statistics() -> Dict`
  - Get pipeline statistics

## Database Schema

### QueryRecord Table

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| original_query | String | User's original question |
| normalized_query | String | Lowercased query |
| primary_category | String | Main classification category |
| secondary_category | String | Alternative category |
| confidence_score | Float | Classification confidence |
| embedding_id | String | Reference to vector DB |
| created_at | DateTime | Record creation time |
| updated_at | DateTime | Last update time |
| intent | String | Detected intent |
| keywords | Text | JSON array of keywords |

### RagSession Table

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| session_id | String | Unique session identifier |
| query_id | Integer | Reference to QueryRecord |
| num_retrieved | Integer | Retrieved document count |
| generated_response | Text | RAG-generated response |
| relevance_score | Float | Response relevance score |
| created_at | DateTime | Session start |
| completed_at | DateTime | Session completion |

## Default Categories

The system comes with 5 default Vietnamese question categories:

1. **技术问题** (Technical Issues)
   - Keywords: lỗi, không hoạt động, gặp vấn đề, bug, crash

2. **定价与计费** (Pricing & Billing)
   - Keywords: giá, chi phí, tiền, thanh toán, hóa đơn

3. **产品特性与功能** (Product Features)
   - Keywords: tính năng, có thể, hỗ trợ, khả năng, chức năng

4. **账户与登录** (Account & Login)
   - Keywords: tài khoản, đăng nhập, mật khẩu, đăng ký, profile

5. **一般问询** (General Inquiry)
   - Keywords: là gì, những gì, thế nào, cách nào

## Advanced Usage

### Custom Categories

```python
from src.classifier import QueryClassifier

classifier = QueryClassifier()

custom_categories = {
    "Bảo vệ Dữ liệu": {
        "keywords": ["an toàn", "bảo mật", "mã hóa", "dữ liệu"],
        "examples": [
            "Dữ liệu của tôi có an toàn không?",
            "Làm sao bạn bảo vệ thông tin cá nhân?"
        ]
    }
}

classifier.add_categories(custom_categories)
```

### Metadata Filtering

```python
vector_db = LocalVectorDB()

# Filter by category in metadata
results = vector_db.search(
    query_embedding,
    n_results=10,
    where={"category": {"$eq": "Kỹ thuật"}}
)
```

### Batch Processing

```python
classifier = QueryClassifier()
queries = [...]  # Large list of queries
results = classifier.batch_classify(queries, top_k=3)
```

## Performance Tips

1. **Batch Processing**: Use `batch_classify()` for multiple queries to leverage GPU
2. **Embedding Cache**: Store frequently used embeddings to avoid recomputation
3. **Vector DB Indexing**: Chroma automatically indexes for fast search
4. **Model Selection**: Use `phobert-base` (faster) or `phobert-large` (more accurate)

## Troubleshooting

### Memory Issues
- Reduce `BATCH_SIZE` in config.py
- Use CPU instead of GPU

### Slow Embedding Generation
- Use smaller model: `vinai/phobert-base`
- Reduce `MAX_SEQ_LENGTH`

### Database Errors
- Ensure `data/` directory exists
- Check SQLite file permissions

## Dependencies

- **torch**: Deep learning framework
- **transformers**: Pre-trained models
- **sentence-transformers**: Embedding generation
- **chromadb**: Vector database
- **sqlalchemy**: ORM
- **langchain**: RAG utilities
- **pydantic**: Data validation

## License

This project is open-source and available under the MIT License.

## Support

For issues, questions, or improvements, please refer to the example scripts and inline documentation in the source code.

## Roadmap

- [ ] Web API interface (FastAPI)
- [ ] Real-time query analytics dashboard
- [ ] Multi-language support
- [ ] Fine-tuning capabilities
- [ ] LLM integration for response generation
- [ ] Batch inference optimization
- [ ] Docker containerization

## Contributing

Contributions are welcome! Areas for improvement:
- Additional Vietnamese categories
- Performance optimizations
- Enhanced text preprocessing
- Integration with other LLMs
