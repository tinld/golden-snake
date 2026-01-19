# Architecture & API Reference

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Application Layer                        │
│                    (Your Code / API)                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                                  │
│              (src/rag/rag_pipeline.py)                          │
│                                                                  │
│  • Query Ingestion    • Context Retrieval                       │
│  • Classification     • Session Management                      │
│  • Analytics          • History Tracking                        │
└─────────────────────┬─────────────────────────────────────────┬─┘
                      │                                           │
        ┌─────────────▼─────────────┐         ┌──────────────────▼─┐
        │  Query Classifier         │         │  Vector Database   │
        │  (PhoBERT Embeddings)     │         │  (Chroma)          │
        │                           │         │                    │
        │ • embed_texts()           │         │ • add_embeddings() │
        │ • classify()              │         │ • search()         │
        │ • batch_classify()        │         │ • update()         │
        │ • add_categories()        │         │ • delete()         │
        └─────────────┬─────────────┘         └──────────────────┬─┘
                      │                                           │
                      └───────────────┬───────────────────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │   Database Layer (SQLAlchemy) │
                      │                               │
                      │   • QueryRecord               │
                      │   • CategoryDefinition        │
                      │   • RagSession                │
                      │                               │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │   SQLite Database             │
                      │   (data/queries.db)           │
                      │                               │
                      │   + Chroma Vector Store       │
                      │   (data/chroma_db/)           │
                      └───────────────────────────────┘
```

## Data Flow

```
User Input (Vietnamese Question)
        │
        ▼
┌─────────────────────────────┐
│  Text Normalization         │
│  • Lowercase                │
│  • Remove special chars     │
│  • Trim whitespace          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  PhoBERT Embedding          │
│  • Tokenize with PhoBERT    │
│  • Generate 768-dim vector  │
│  • Normalize embedding      │
└─────────────┬───────────────┘
              │
              ├─────────────────────┬────────────────────┐
              ▼                     ▼                    ▼
        ┌──────────┐      ┌──────────────┐    ┌─────────────────┐
        │Classifier│      │Vector Search │    │Store in Database│
        │           │      │              │    │                 │
        │Semantic   │──►   │Find similar  │    │QueryRecord +    │
        │Similarity │      │in Chroma DB  │    │RagSession       │
        │+ Keywords │      │              │    │                 │
        │           │      │Return Top-K  │    │Record metadata  │
        └─────┬─────┘      └──────┬───────┘    └────────┬────────┘
              │                   │                     │
              └─────────┬─────────┴─────────┬───────────┘
                        │                   │
                        ▼                   ▼
            ┌──────────────────┐  ┌──────────────────┐
            │Classification    │  │Retrieved Context │
            │Results           │  │Similar Questions │
            │                  │  │                  │
            │• Category        │  │• Top-5 similar   │
            │• Confidence      │  │• Similarity score│
            │• Score           │  │• Metadata        │
            └──────────────────┘  └──────────────────┘
                        │                   │
                        └────────┬──────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │ RAG Output           │
                    │                      │
                    │ • Category + Score   │
                    │ • Similar Questions  │
                    │ • Context for Gen    │
                    └──────────────────────┘
```

## Core API Reference

### 1. QueryClassifier

**Location**: `src/classifier/query_classifier.py`

```python
from src.classifier import QueryClassifier

# Initialize
classifier = QueryClassifier(embedder=None)
```

**Methods:**

#### `classify(query: str, top_k: int = 1, use_keywords: bool = True) -> List[Dict]`
Classify a single query.

**Parameters:**
- `query` (str): Vietnamese question to classify
- `top_k` (int): Number of top categories to return (default: 1)
- `use_keywords` (bool): Use keyword matching for boosting (default: True)

**Returns:**
```python
[
    {
        'rank': 1,
        'category': '技术问题',
        'score': 0.892,
        'semantic_score': 0.89,
        'keyword_score': 0.80,
        'confidence': 0.89,
        'is_confident': True
    }
]
```

**Example:**
```python
results = classifier.classify("Làm sao sửa lỗi?", top_k=2)
for r in results:
    print(f"{r['category']}: {r['confidence']:.2%}")
```

---

#### `batch_classify(queries: List[str], top_k: int = 1) -> List[List[Dict]]`
Classify multiple queries efficiently.

**Parameters:**
- `queries` (List[str]): List of Vietnamese questions
- `top_k` (int): Top categories per query

**Returns:**
```python
[
    [{'category': '...', 'confidence': ...}],  # Results for query 1
    [{'category': '...', 'confidence': ...}],  # Results for query 2
    ...
]
```

**Example:**
```python
queries = ["Lỗi?", "Giá?", "Tính năng?"]
batch_results = classifier.batch_classify(queries, top_k=1)
```

---

#### `add_categories(categories: Dict[str, Dict])`
Add custom categories for classification.

**Parameters:**
- `categories` (Dict): Format:
  ```python
  {
      "Category Name": {
          "keywords": ["kw1", "kw2"],
          "examples": ["Example Q 1", "Example Q 2"]
      }
  }
  ```

**Example:**
```python
classifier.add_categories({
    "Bảo vệ": {
        "keywords": ["an toàn", "bảo mật"],
        "examples": ["Dữ liệu an toàn?"]
    }
})
```

---

#### `get_categories() -> List[str]`
Get all available categories.

**Returns:** List of category names

**Example:**
```python
cats = classifier.get_categories()
print(f"Available: {cats}")
```

---

### 2. LocalVectorDB

**Location**: `src/vector_db/local_vector_db.py`

```python
from src.vector_db import LocalVectorDB

# Initialize
vector_db = LocalVectorDB(db_path="./data/chroma_db", collection_name="vietnamese_queries")
```

**Methods:**

#### `add_embeddings(embeddings: np.ndarray, texts: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None) -> List[str]`
Add embeddings to vector database.

**Parameters:**
- `embeddings` (np.ndarray): Shape (n, embedding_dim)
- `texts` (List[str]): Original texts
- `metadatas` (List[Dict], optional): Metadata for each embedding
- `ids` (List[str], optional): Custom IDs (auto-generated if None)

**Returns:** List of document IDs

**Example:**
```python
texts = ["Question 1", "Question 2"]
embeddings = embedder.embed_texts(texts)
ids = vector_db.add_embeddings(
    embeddings, 
    texts,
    metadatas=[{"category": "Tech"}, {"category": "Price"}]
)
```

---

#### `search(query_embedding: np.ndarray, n_results: int = 5, where: Optional[Dict] = None) -> Dict`
Search for similar embeddings.

**Parameters:**
- `query_embedding` (np.ndarray): Query embedding vector
- `n_results` (int): Number of results (default: 5)
- `where` (Dict, optional): Metadata filter

**Returns:**
```python
{
    'ids': ['id1', 'id2', ...],
    'documents': ['text1', 'text2', ...],
    'distances': [0.1, 0.2, ...],
    'metadatas': [{...}, {...}, ...]
}
```

**Example:**
```python
query_emb = embedder.embed_single("Search term")
results = vector_db.search(query_emb, n_results=5)
for doc in results['documents']:
    print(doc)
```

---

#### `search_by_text(query_text: str, query_embedding: np.ndarray, n_results: int = 5, where: Optional[Dict] = None) -> Dict`
Search with text and embedding.

**Parameters:** Same as `search()`, plus `query_text`

**Returns:**
```python
{
    'query_text': 'Search term',
    'results': [
        {
            'id': 'doc_id',
            'text': 'Document text',
            'distance': 0.1,
            'similarity': 0.95,
            'metadata': {...}
        }
    ]
}
```

**Example:**
```python
results = vector_db.search_by_text(
    "My query",
    query_embedding,
    n_results=3
)
for r in results['results']:
    print(f"{r['text']} (similarity: {r['similarity']:.2%})")
```

---

#### `update_embeddings(ids: List[str], embeddings: np.ndarray, texts: Optional[List[str]] = None, metadatas: Optional[List[Dict]] = None)`
Update existing embeddings.

**Example:**
```python
vector_db.update_embeddings(['id1', 'id2'], new_embeddings, texts=['New text 1', 'New text 2'])
```

---

#### `delete_embeddings(ids: List[str])`
Delete embeddings by ID.

**Example:**
```python
vector_db.delete_embeddings(['id1', 'id2'])
```

---

#### `count() -> int`
Get total embeddings in collection.

**Example:**
```python
total = vector_db.count()
print(f"Database has {total} documents")
```

---

### 3. RAGPipeline

**Location**: `src/rag/rag_pipeline.py`

```python
from src.rag import RAGPipeline

# Initialize
pipeline = RAGPipeline(classifier=None, vector_db=None)
```

**Methods:**

#### `ingest_queries(queries: List[str], category: Optional[str] = None) -> List[str]`
Ingest and classify queries.

**Parameters:**
- `queries` (List[str]): Questions to ingest
- `category` (str, optional): Override category

**Returns:** List of query IDs

**Example:**
```python
ids = pipeline.ingest_queries([
    "Làm sao sửa lỗi?",
    "Giá bao nhiêu?"
])
```

---

#### `retrieve(query: str, top_k: int = 5) -> Dict`
Retrieve similar queries from vector DB.

**Parameters:**
- `query` (str): Query text
- `top_k` (int): Number of results

**Returns:**
```python
{
    'query_text': 'Your query',
    'results': [
        {
            'id': 'id1',
            'text': 'Similar text',
            'distance': 0.1,
            'similarity': 0.95,
            'metadata': {...}
        }
    ]
}
```

---

#### `process_query(query: str, retrieve_context: bool = True, top_k: int = 5) -> Dict`
Complete query processing.

**Parameters:**
- `query` (str): Vietnamese question
- `retrieve_context` (bool): Retrieve context from vector DB
- `top_k` (int): Number of context docs

**Returns:**
```python
{
    'query': 'Your question',
    'classifications': [...],  # Classification results
    'primary_category': 'Tech Issue',
    'context': {...},  # Retrieved similar queries
    'num_results': 5
}
```

**Example:**
```python
result = pipeline.process_query("Ứng dụng bị lỗi?", retrieve_context=True)
print(f"Category: {result['primary_category']}")
for ctx in result['context']['results']:
    print(f"  Similar: {ctx['text']}")
```

---

#### `save_session(query_text: str, response_text: str, num_retrieved: int = 0) -> int`
Save RAG session to database.

**Returns:** Session ID

**Example:**
```python
session_id = pipeline.save_session(
    query_text="User question",
    response_text="Generated response",
    num_retrieved=5
)
```

---

#### `get_query_history(category: Optional[str] = None, limit: int = 10) -> List[Dict]`
Get query history.

**Returns:**
```python
[
    {
        'id': 1,
        'query': 'Question text',
        'category': 'Category',
        'confidence': 0.89,
        'created_at': '2024-01-01T12:00:00'
    }
]
```

---

#### `get_statistics() -> Dict`
Get pipeline statistics.

**Returns:**
```python
{
    'total_queries': 100,
    'total_rag_sessions': 50,
    'total_vectors': 100,
    'categories': {'Tech': 30, 'Price': 20, ...},
    'classifier_categories': ['Tech', 'Price', ...]
}
```

---

## Configuration Reference

**File**: `src/config.py`

```python
# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
SQLITE_DB_PATH = DATA_DIR / "queries.db"

# PhoBERT Model
PHOBERT_MODEL = "vinai/phobert-base"  # or "vinai/phobert-large"
EMBEDDING_DIM = 768
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256

# Vector DB
CHROMA_COLLECTION_NAME = "vietnamese_queries"
SIMILARITY_THRESHOLD = 0.7

# Database
DB_URL = f"sqlite:///{SQLITE_DB_PATH}"
DB_ECHO = False

# Logging
LOG_LEVEL = "INFO"
```

## Usage Patterns

### Pattern 1: Simple Classification
```python
from src.classifier import QueryClassifier

classifier = QueryClassifier()
result = classifier.classify("Làm sao sửa lỗi?", top_k=1)
print(result[0]['category'])
```

### Pattern 2: Bulk Processing
```python
queries = [...]  # 1000 queries
results = classifier.batch_classify(queries, top_k=1)
```

### Pattern 3: Complete Workflow
```python
pipeline = RAGPipeline()
pipeline.ingest_queries(training_queries)
result = pipeline.process_query(new_query)
stats = pipeline.get_statistics()
```

### Pattern 4: Custom Categories
```python
classifier.add_categories({
    "Custom": {
        "keywords": ["kw1", "kw2"],
        "examples": ["ex1", "ex2"]
    }
})
```

### Pattern 5: Vector Search
```python
db = LocalVectorDB()
ids = db.add_embeddings(embeddings, texts)
results = db.search(query_emb, n_results=5)
```

---

## Error Handling

```python
from src.rag import RAGPipeline

try:
    pipeline = RAGPipeline()
    result = pipeline.process_query("Question?")
except Exception as e:
    print(f"Error: {e}")
```

Common issues:
- Model download timeout → Increase timeout or pre-download
- Memory errors → Reduce BATCH_SIZE
- Database locked → Check file permissions or restart
