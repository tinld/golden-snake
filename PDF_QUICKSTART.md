# PDF Processing Feature - Quick Start Guide

## 🎯 What This Feature Does

Automatically processes PDF documents into searchable chunks:
1. **Reads** PDF files and extracts text
2. **Chunks** text into smaller pieces (with overlap for context)
3. **Vectorizes** each chunk using PhoBERT embeddings
4. **Stores** in Qdrant vector database with metadata
5. **Queries** documents with semantic search

## 📁 Files Created

### Core Implementation
- **`src/media/pdf_processor.py`** - Main processing engine
  - `PDFChunker`: Splits text into overlapping chunks
  - `PDFDocumentProcessor`: Complete pipeline (read → chunk → vectorize → store)

### API Endpoints (in `api.py`)
- `POST /pdf/process` - Process a PDF document
- `POST /pdf/query` - Query PDF documents
- `GET /pdf/documents` - List all documents
- `GET /pdf/documents/{id}` - Get document info

### Examples & Documentation
- **`examples/example_4_pdf_processing.py`** - Complete usage example
- **`PDF_PROCESSING.md`** - Full documentation
- **`test_pdf_feature.py`** - Quick test script

## 🚀 Quick Start

### 1. Test the Feature

```bash
# Run quick tests
python test_pdf_feature.py
```

This verifies:
- PDF file exists in media folder
- Required modules are available
- Qdrant is running
- Basic functionality works

### 2. Process Your First PDF

```python
from src.media.pdf_processor import PDFDocumentProcessor

# Initialize processor
processor = PDFDocumentProcessor(
    collection_name="pdf_documents",
    chunk_size=500,       # 500 characters per chunk
    chunk_overlap=50      # 50 character overlap
)

# Process the PDF in media folder
result = processor.process_pdf(
    pdf_path="media/vpcp_1.pdf",
    max_pages=None,       # Process all pages
    metadata={
        "category": "business",
        "language": "vietnamese"
    }
)

print(f"✓ Processed {result['total_chunks']} chunks from {result['total_pages']} pages")
print(f"Document ID: {result['document_id']}")
```

### 3. Query the Document

```python
# Search for information
results = processor.query_document(
    query="thông tin về công ty",
    top_k=5
)

# Display results
for i, result in enumerate(results, 1):
    print(f"\nResult {i} (Score: {result['score']:.2%}):")
    print(f"  Page: {result['page']}")
    print(f"  {result['text'][:200]}...")
```

### 4. Use the API

```bash
# Start the API server
python api.py
```

Then process PDFs via HTTP:

```bash
# Process a PDF
curl -X POST http://localhost:8000/pdf/process \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "media/vpcp_1.pdf",
    "metadata": {"category": "business"}
  }'

# Query documents
curl -X POST http://localhost:8000/pdf/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "thông tin về công ty",
    "top_k": 5
  }'

# List all documents
curl http://localhost:8000/pdf/documents
```

## 🔧 Configuration

### Chunk Size Selection

```python
# Small chunks (200-300): More precise, more chunks
processor = PDFDocumentProcessor(chunk_size=250, chunk_overlap=25)

# Medium chunks (400-600): Balanced (RECOMMENDED)
processor = PDFDocumentProcessor(chunk_size=500, chunk_overlap=50)

# Large chunks (800-1000): More context, fewer chunks
processor = PDFDocumentProcessor(chunk_size=1000, chunk_overlap=100)
```

### Collection Management

```python
# Use different collections for different document types
legal_docs = PDFDocumentProcessor(collection_name="legal_documents")
contracts = PDFDocumentProcessor(collection_name="contracts")
manuals = PDFDocumentProcessor(collection_name="manuals")
```

## 📊 Stored Data Structure

Each chunk in Qdrant contains:

```json
{
  "vector": [0.123, 0.456, ...],  // 768-dim embedding
  "payload": {
    "text": "Chunk text content...",
    "document_id": "abc123def456",
    "document_name": "vpcp_1.pdf",
    "document_path": "media/vpcp_1.pdf",
    "page": 5,
    "chunk_index": 2,
    "global_chunk_index": 23,
    "total_chunks": 45,
    "processed_at": "2026-01-09T10:30:00",
    // Custom metadata
    "category": "business",
    "language": "vietnamese"
  }
}
```

## 🎯 Use Cases

### 1. Document Q&A
```python
# Ask questions about documents
answer = processor.query_document("Quy trình nghỉ phép?", top_k=1)
print(answer[0]['text'])
```

### 2. Semantic Search
```python
# Find similar content
results = processor.query_document("chính sách công ty", top_k=10)
```

### 3. Document-Specific Search
```python
# Search within one document only
results = processor.query_document(
    query="điều khoản",
    document_name="contract.pdf"
)
```

### 4. Multi-Document Knowledge Base
```python
# Process multiple documents
for pdf in ["manual.pdf", "faq.pdf", "guide.pdf"]:
    processor.process_pdf(f"media/{pdf}")

# Search across all
results = processor.query_document("cách sử dụng", top_k=5)
```

## 📖 Full Examples

### Run Complete Example
```bash
python examples/example_4_pdf_processing.py
```

This demonstrates:
- Processing a PDF
- Querying with different questions
- Getting document information
- Listing all documents
- Cross-document search

## 🔍 Query Features

### Basic Query
```python
results = processor.query_document(query="search term", top_k=5)
```

### Filter by Document ID
```python
results = processor.query_document(
    query="search term",
    document_id="abc123def456"
)
```

### Filter by Document Name
```python
results = processor.query_document(
    query="search term",
    document_name="vpcp_1.pdf"
)
```

## 🎨 Advanced Features

### Custom Metadata
```python
processor.process_pdf(
    "media/contract.pdf",
    metadata={
        "type": "legal",
        "year": 2026,
        "department": "legal",
        "confidential": True,
        "tags": ["contract", "agreement"]
    }
)
```

### Batch Processing
```python
from pathlib import Path

for pdf_path in Path("media").glob("*.pdf"):
    print(f"Processing {pdf_path.name}...")
    result = processor.process_pdf(str(pdf_path))
    print(f"✓ {result['total_chunks']} chunks created")
```

### Document Management
```python
# List all documents
docs = processor.list_documents()

# Get specific document info
info = processor.get_document_info(document_id)

# Count total chunks
total = sum(doc['total_chunks'] for doc in docs)
print(f"Total chunks across all documents: {total}")
```

## 📚 API Reference

See **`API_DOCUMENTATION.md`** for complete API reference including:
- Request/response schemas
- Error handling
- Status codes
- Example requests

See **`PDF_PROCESSING.md`** for detailed documentation including:
- Architecture details
- Performance optimization
- Troubleshooting
- Advanced workflows

## ✅ Verification

After implementation, verify:

1. **Imports work**
   ```python
   from src.media.pdf_processor import PDFDocumentProcessor
   ```

2. **Qdrant is accessible**
   - Visit: http://localhost:6333/dashboard

3. **PDF is readable**
   ```python
   from src.media.pdf import read_pdf_text
   text = read_pdf_text("media/vpcp_1.pdf", max_pages=1)
   print(len(text))
   ```

4. **API is functional**
   - Visit: http://localhost:8000/docs

## 🐛 Troubleshooting

### "Qdrant connection failed"
```bash
# Start Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant
```

### "PDF not found"
```python
# Use absolute path
from pathlib import Path
pdf_path = Path(__file__).parent / "media" / "vpcp_1.pdf"
```

### "No text extracted"
- PDF might be image-based (needs OCR)
- Try with different PDF

### "Import error"
```bash
# Install dependencies
pip install -r requirements.txt
```

## 📝 Next Steps

1. **Test**: Run `python test_pdf_feature.py`
2. **Example**: Run `python examples/example_4_pdf_processing.py`
3. **Integrate**: Use in your RAG pipeline
4. **Customize**: Adjust chunk sizes and metadata
5. **Scale**: Process your document library

## 🎓 Learning Resources

- **Example Script**: `examples/example_4_pdf_processing.py`
- **Full Docs**: `PDF_PROCESSING.md`
- **API Docs**: `API_DOCUMENTATION.md`
- **Interactive API**: http://localhost:8000/docs

## 💡 Tips

1. **Start small**: Test with 1-2 page PDFs first
2. **Tune chunk size**: Experiment with 300-800 range
3. **Use metadata**: Add useful tags for filtering
4. **Monitor performance**: Check query speeds with different filters
5. **Batch smartly**: Process documents sequentially, not in parallel

---

**Ready to process PDFs? Run the test script to get started!**

```bash
python test_pdf_feature.py
```
