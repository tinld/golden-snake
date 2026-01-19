# Quick Start Guide

## Installation (5 minutes)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Initialize the Database
```bash
python -c "from src.database import init_db; init_db()"
```

## Run Examples (10 minutes)

### Example 1: Query Classification
```bash
cd examples
python example_1_classification.py
```

**What it does:**
- Loads PhoBERT model
- Classifies Vietnamese questions into categories
- Shows confidence scores
- Demonstrates custom category addition

**Expected output:**
```
Query: Làm sao sửa lỗi này?
└─ Primary: 技术问题 (confidence: 95.2%)
```

### Example 2: Vector Database
```bash
cd examples
python example_2_vector_db.py
```

**What it does:**
- Creates embeddings for documents
- Stores in local Chroma database
- Performs semantic search
- Updates and deletes documents

**Key operations:**
- Generate embeddings
- Store in vector DB
- Semantic search
- Document updates

### Example 3: Complete RAG Pipeline
```bash
cd examples
python example_3_rag_pipeline.py
```

**What it does:**
- Ingests training queries
- Classifies them automatically
- Stores in database + vector DB
- Retrieves context for new queries
- Shows statistics

## Core Modules Overview

### 1. **QueryClassifier** (`src/classifier/query_classifier.py`)
```python
from src.classifier import QueryClassifier

classifier = QueryClassifier()
results = classifier.classify("Câu hỏi của bạn?")
```

- Semantic similarity classification
- Keyword matching support
- Configurable categories
- Batch processing

### 2. **LocalVectorDB** (`src/vector_db/local_vector_db.py`)
```python
from src.vector_db import LocalVectorDB

vector_db = LocalVectorDB()
ids = vector_db.add_embeddings(embeddings, texts)
results = vector_db.search(query_embedding, n_results=5)
```

- Store embeddings
- Semantic search
- Metadata filtering
- Persistent storage

### 3. **RAGPipeline** (`src/rag/rag_pipeline.py`)
```python
from src.rag import RAGPipeline

pipeline = RAGPipeline()
pipeline.ingest_queries(queries)
results = pipeline.process_query("Câu hỏi?")
```

- End-to-end workflow
- Query classification
- Context retrieval
- Session management

## Basic Workflow

```python
from src.rag import RAGPipeline
from src.database import init_db

# 1. Initialize
init_db()
pipeline = RAGPipeline()

# 2. Ingest training data
training_queries = [
    "Làm sao sửa lỗi?",
    "Giá bao nhiêu?"
]
pipeline.ingest_queries(training_queries)

# 3. Process new query
result = pipeline.process_query(
    "Ứng dụng bị lỗi",
    retrieve_context=True,
    top_k=5
)

# 4. Access results
print(result["classifications"])  # Category predictions
print(result["context"])           # Similar queries
```

## Configuration

Edit `src/config.py`:

```python
# PhoBERT model (base or large)
PHOBERT_MODEL = "vinai/phobert-base"

# Vector DB collection name
CHROMA_COLLECTION_NAME = "vietnamese_queries"

# Similarity threshold for confidence
SIMILARITY_THRESHOLD = 0.7

# Batch size for processing
BATCH_SIZE = 32
```

## Common Tasks

### Add Custom Categories
```python
from src.classifier import QueryClassifier

classifier = QueryClassifier()
classifier.add_categories({
    "Bảo vệ Dữ liệu": {
        "keywords": ["an toàn", "bảo mật"],
        "examples": ["Dữ liệu có an toàn không?"]
    }
})
```

### Search Similar Queries
```python
from src.rag import RAGPipeline

pipeline = RAGPipeline()
results = pipeline.retrieve("Vấn đề kỹ thuật", top_k=5)

for result in results["results"]:
    print(f"- {result['text']} (similarity: {result['similarity']:.2%})")
```

### Get Statistics
```python
stats = pipeline.get_statistics()
print(f"Total queries: {stats['total_queries']}")
print(f"Total vectors: {stats['total_vectors']}")
print(f"Categories: {stats['categories']}")
```

### Query History
```python
history = pipeline.get_query_history(category="技术问题", limit=10)
for record in history:
    print(f"- {record['query']} ({record['category']})")
```

## Database Files

After running examples, you'll find:

```
data/
├── chroma_db/           # Vector embeddings (Chroma)
│   ├── chroma.parquet
│   └── data_level0.bin
└── queries.db           # Query metadata (SQLite)
```

## Troubleshooting

### Issue: Model download slow
**Solution:** The first run downloads PhoBERT (~500MB)
- Use `PHOBERT_MODEL = "vinai/phobert-base"` (smaller)
- Or pre-download with: `from transformers import AutoModel; AutoModel.from_pretrained("vinai/phobert-base")`

### Issue: Out of memory
**Solution:** Reduce batch size in `config.py`
```python
BATCH_SIZE = 8  # default: 32
```

### Issue: Database locked
**Solution:** Remove `data/` folder and reinitialize
```bash
rmdir /s data
python -c "from src.database import init_db; init_db()"
```

## Next Steps

1. **Run Examples**: Execute all 3 examples to understand the system
2. **Explore API**: Check each module's docstrings
3. **Add Data**: Ingest your own Vietnamese questions
4. **Custom Categories**: Define domain-specific categories
5. **Integrate**: Use in your application (FastAPI, etc.)

## Project Structure
```
rag-business-snake/
├── src/
│   ├── classifier/       # PhoBERT embeddings & classification
│   ├── vector_db/        # Chroma vector database
│   ├── database/         # SQLAlchemy models
│   ├── rag/              # RAG pipeline orchestration
│   ├── utils/            # Text processing utilities
│   └── config.py         # Configuration
├── examples/             # 3 working examples
├── data/                 # Generated at runtime
├── requirements.txt      # Dependencies
└── README.md             # Full documentation
```

## Performance Notes

- **First run**: ~2-3 minutes (PhoBERT model download)
- **Classification**: ~100ms per query (GPU) / ~500ms (CPU)
- **Vector search**: <10ms per query (Chroma is fast!)
- **Batch inference**: Much faster for multiple queries

## Support

For detailed information:
- See `README.md` for full API reference
- Check example files for implementation patterns
- Review source code docstrings for detailed docs
