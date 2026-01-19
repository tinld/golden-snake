# Project File Manifest

## Complete File Listing

### Root Level Files

```
rag-business-snake/
├── README.md                    # Full API documentation & guides
├── QUICKSTART.md               # 10-minute quick start guide
├── PROJECT_SUMMARY.md          # High-level overview
├── ARCHITECTURE.md             # System architecture & API reference
├── requirements.txt            # Python dependencies
├── setup.py                    # Setup script for initialization
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
└── [Directories listed below]
```

### Source Code (`src/`)

#### Configuration
- **src/config.py** (60 lines)
  - Configuration settings for models, DB, vector store
  - Default category definitions
  - Path setup

#### Classifier Module (`src/classifier/`)
- **src/classifier/embedder.py** (120 lines)
  - `PhoBertEmbedder` class for Vietnamese embeddings
  - PhoBERT model loading and inference
  - Batch and single text embedding
  - Mean pooling and normalization

- **src/classifier/query_classifier.py** (250 lines)
  - `QueryClassifier` class for semantic classification
  - Category management with examples
  - Cosine similarity + keyword matching
  - Batch classification support
  - Configurable confidence thresholds

- **src/classifier/__init__.py**
  - Module exports

#### Vector Database (`src/vector_db/`)
- **src/vector_db/local_vector_db.py** (260 lines)
  - `LocalVectorDB` class using Chroma
  - Add/update/delete embeddings
  - Semantic search with multiple methods
  - Metadata filtering
  - Document retrieval by ID

- **src/vector_db/__init__.py**
  - Module exports

#### Database Layer (`src/database/`)
- **src/database/models.py** (200 lines)
  - `QueryRecord` SQLAlchemy model
  - `CategoryDefinition` model
  - `RagSession` model
  - Database initialization functions
  - Session management

- **src/database/__init__.py**
  - Module exports

#### RAG Pipeline (`src/rag/`)
- **src/rag/rag_pipeline.py** (320 lines)
  - `RAGPipeline` class orchestrating the full workflow
  - Query ingestion and classification
  - Context retrieval from vector DB
  - Session management and tracking
  - Statistics and analytics
  - Query history

- **src/rag/__init__.py**
  - Module exports

#### Utilities (`src/utils/`)
- **src/utils/text_utils.py** (100 lines)
  - `normalize_text()` for Vietnamese text
  - `tokenize()` function
  - `extract_keywords()` with Vietnamese stopwords

- **src/utils/__init__.py**
  - Module exports

#### Module Init
- **src/__init__.py**
  - Empty init file

---

### Examples (`examples/`)

- **examples/example_1_classification.py** (80 lines)
  - Demonstrates query classification
  - Tests default categories
  - Custom category addition
  - Multiple classification results
  - **Run time**: ~5 minutes (includes model download)

- **examples/example_2_vector_db.py** (120 lines)
  - Demonstrates vector database operations
  - Embedding generation and storage
  - Semantic search
  - Update and delete operations
  - Metadata filtering
  - **Run time**: ~2-3 minutes

- **examples/example_3_rag_pipeline.py** (150 lines)
  - Complete RAG workflow demonstration
  - Query ingestion
  - Context retrieval
  - Session management
  - Statistics generation
  - Query history
  - **Run time**: ~3-5 minutes

---

### Documentation Files

- **README.md** (500+ lines)
  - Installation instructions
  - Feature overview
  - Quick start examples
  - API reference for all classes
  - Database schema
  - Configuration options
  - Performance tips
  - Troubleshooting guide
  - Dependencies list

- **QUICKSTART.md** (300+ lines)
  - 10-minute quick start
  - Example walkthrough
  - Common tasks
  - Configuration guide
  - Performance notes
  - Troubleshooting

- **PROJECT_SUMMARY.md** (350+ lines)
  - Project overview
  - Key components description
  - Default categories
  - Usage examples
  - Installation steps
  - Feature highlights
  - File checklist

- **ARCHITECTURE.md** (400+ lines)
  - System architecture diagrams (ASCII art)
  - Data flow visualization
  - Complete API reference
  - Configuration reference
  - Usage patterns
  - Error handling examples

---

### Configuration Files

- **.env.example** (10 lines)
  - Environment variables template
  - Model configuration
  - Database settings
  - API settings template

- **.gitignore** (35 lines)
  - Python cache files
  - IDE settings
  - Data and database files
  - Environment files
  - Temporary files

---

### Setup Files

- **requirements.txt** (13 packages)
  - torch==2.0.1
  - transformers==4.35.2
  - sentence-transformers==2.2.2
  - chromadb==0.4.18
  - sqlalchemy==2.0.23
  - And 8 more dependencies

- **setup.py** (150 lines)
  - Dependency checking
  - Directory initialization
  - Database setup
  - Module import verification
  - Comprehensive output messages

---

### Generated Directories (Runtime)

- **data/** (created on first run)
  - **chroma_db/** - Chroma vector store
    - `chroma.parquet` - Vector embeddings
    - `data_level0.bin` - Index data
  - **queries.db** - SQLite database
    - QueryRecord table
    - CategoryDefinition table
    - RagSession table

- **logs/** (optional)
  - Logging output

---

## File Statistics

| Category | Count | Lines of Code |
|----------|-------|---------------|
| Source Code (.py) | 10 | ~1,700 |
| Examples | 3 | ~350 |
| Documentation | 4 | ~1,500+ |
| Config Files | 3 | ~60 |
| **Total** | **20** | **~3,610+** |

---

## Core Module Sizes

| Module | Files | Size | Purpose |
|--------|-------|------|---------|
| Classifier | 2 | ~370 | PhoBERT embeddings & classification |
| Vector DB | 1 | ~260 | Chroma wrapper & search |
| Database | 1 | ~200 | SQLAlchemy models |
| RAG Pipeline | 1 | ~320 | Orchestration & workflow |
| Utilities | 1 | ~100 | Text processing helpers |

---

## Documentation Breakdown

| Document | Pages | Content |
|----------|-------|---------|
| README.md | 12+ | Complete reference, API docs, troubleshooting |
| QUICKSTART.md | 8+ | 10-min quick start, common tasks |
| PROJECT_SUMMARY.md | 9+ | High-level overview, features, architecture |
| ARCHITECTURE.md | 10+ | System diagrams, detailed API, patterns |

---

## Key Features by File

### Configuration Management
- **src/config.py**: Central configuration
- **.env.example**: Environment template
- **setup.py**: Automatic setup validation

### Classification System
- **src/classifier/embedder.py**: PhoBERT embedding
- **src/classifier/query_classifier.py**: Category classification

### Storage System
- **src/vector_db/local_vector_db.py**: Vector embeddings (Chroma)
- **src/database/models.py**: Structured data (SQLite)

### RAG System
- **src/rag/rag_pipeline.py**: Complete workflow orchestration
- **src/utils/text_utils.py**: Vietnamese text processing

### Examples & Documentation
- **examples/example_*.py**: Working demonstrations
- **README.md**: Complete reference
- **ARCHITECTURE.md**: Technical details

---

## Import Structure

```python
# Classifier
from src.classifier import QueryClassifier, get_embedder, PhoBertEmbedder

# Vector DB
from src.vector_db import LocalVectorDB, get_vector_db

# Database
from src.database import QueryRecord, CategoryDefinition, RagSession, init_db

# RAG Pipeline
from src.rag import RAGPipeline

# Utilities
from src.utils import normalize_text, tokenize, extract_keywords

# Configuration
from src.config import PHOBERT_MODEL, EMBEDDING_DIM, etc.
```

---

## File Dependencies

```
setup.py
├── src/config.py
├── src/database/models.py
│   └── SQLAlchemy, sqlite
├── src/classifier/embedder.py
│   └── transformers, torch
└── src/classifier/query_classifier.py
    └── embedder.py

RAGPipeline
├── src/classifier/
├── src/vector_db/
│   └── chromadb
└── src/database/
    └── SQLAlchemy

Examples
├── src/classifier/
├── src/vector_db/
├── src/rag/
└── src/database/
```

---

## Testing the Installation

After `pip install -r requirements.txt` and `python setup.py`:

```bash
# Test classifier
python -c "from src.classifier import QueryClassifier; print('✓ Classifier')"

# Test vector DB
python -c "from src.vector_db import LocalVectorDB; print('✓ Vector DB')"

# Test database
python -c "from src.database import init_db; print('✓ Database')"

# Test RAG
python -c "from src.rag import RAGPipeline; print('✓ RAG')"
```

---

## Next Steps

1. **Install**: `pip install -r requirements.txt`
2. **Setup**: `python setup.py`
3. **Learn**: Read QUICKSTART.md
4. **Run Examples**: Execute all 3 example scripts
5. **Customize**: Modify for your use case
6. **Integrate**: Use in your application

---

## Support Resources

- **API Reference**: See ARCHITECTURE.md for complete API
- **Quick Start**: See QUICKSTART.md for common tasks
- **Examples**: See examples/ folder for working code
- **Configuration**: Edit src/config.py for customization
- **Troubleshooting**: See README.md for common issues
