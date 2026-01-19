# PDF Processing Feature Documentation

## Overview

The PDF Processing feature allows you to:
1. **Extract text** from PDF documents
2. **Chunk text** into manageable pieces with overlap for better context
3. **Vectorize chunks** using PhoBERT embeddings
4. **Store in Qdrant** vector database with rich metadata
5. **Query documents** with semantic search across all or specific PDFs

## Architecture

```
PDF File → Text Extraction → Chunking → Vectorization → Qdrant Storage
                                                              ↓
                                                         Query Interface
```

### Key Components

1. **PDFReadingTransformer** (`src/media/pdf.py`)
   - Extracts text from PDF pages
   - Cleans and normalizes text
   - Handles large documents page-by-page

2. **PDFChunker** (`src/media/pdf_processor.py`)
   - Splits text into chunks with configurable size
   - Maintains overlap between chunks for context preservation
   - Preserves page metadata for each chunk

3. **PDFDocumentProcessor** (`src/media/pdf_processor.py`)
   - Complete processing pipeline
   - Generates embeddings for all chunks
   - Stores in Qdrant with rich metadata
   - Provides query interface

## Quick Start

### 1. Process a PDF Document

```python
from src.media.pdf_processor import PDFDocumentProcessor

# Initialize processor
processor = PDFDocumentProcessor(
    collection_name="pdf_documents",
    chunk_size=500,        # characters per chunk
    chunk_overlap=50       # overlap between chunks
)

# Process PDF
result = processor.process_pdf(
    pdf_path="media/vpcp_1.pdf",
    max_pages=None,  # Process all pages
    metadata={
        "category": "business",
        "language": "vietnamese"
    }
)

print(f"Processed {result['total_chunks']} chunks from {result['total_pages']} pages")
print(f"Document ID: {result['document_id']}")
```

### 2. Query the Document

```python
# Query specific document
results = processor.query_document(
    query="thông tin về công ty",
    top_k=5,
    document_id=result['document_id']  # Optional: search only this document
)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"  Page: {result['page']}")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Text: {result['text'][:100]}...")
```

### 3. List All Documents

```python
# Get all processed documents
documents = processor.list_documents()

for doc in documents:
    print(f"- {doc['document_name']}")
    print(f"  ID: {doc['document_id']}")
    print(f"  Chunks: {doc['total_chunks']}")
```

## API Endpoints

### 1. Process PDF Document

**POST /pdf/process**

Upload and process a PDF document.

**Request:**
```json
{
  "pdf_path": "media/vpcp_1.pdf",
  "max_pages": null,
  "metadata": {
    "category": "business",
    "language": "vietnamese"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "document_id": "abc123def456",
  "document_name": "vpcp_1.pdf",
  "total_pages": 10,
  "total_chunks": 45,
  "chunks_stored": 45,
  "collection": "pdf_documents"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/pdf/process \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "media/vpcp_1.pdf",
    "metadata": {"category": "business"}
  }'
```

---

### 2. Query PDF Documents

**POST /pdf/query**

Search for relevant content across all or specific PDF documents.

**Request:**
```json
{
  "query": "thông tin về công ty",
  "top_k": 5,
  "document_id": null,
  "document_name": "vpcp_1.pdf"
}
```

**Response:**
```json
[
  {
    "id": "abc123_chunk_0",
    "text": "Thông tin về công ty...",
    "score": 0.92,
    "document_name": "vpcp_1.pdf",
    "document_id": "abc123def456",
    "page": 1,
    "chunk_index": 0,
    "metadata": {
      "category": "business",
      "processed_at": "2026-01-09T10:30:00"
    }
  }
]
```

**Example:**
```bash
curl -X POST http://localhost:8000/pdf/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "thông tin về công ty",
    "top_k": 5
  }'
```

---

### 3. List All Documents

**GET /pdf/documents**

Get a list of all processed PDF documents.

**Response:**
```json
[
  {
    "document_id": "abc123def456",
    "document_name": "vpcp_1.pdf",
    "document_path": "media/vpcp_1.pdf",
    "total_chunks": 45,
    "processed_at": "2026-01-09T10:30:00"
  }
]
```

**Example:**
```bash
curl http://localhost:8000/pdf/documents
```

---

### 4. Get Document Info

**GET /pdf/documents/{document_id}**

Get detailed information about a specific document.

**Response:**
```json
{
  "document_id": "abc123def456",
  "document_name": "vpcp_1.pdf",
  "document_path": "media/vpcp_1.pdf",
  "total_chunks": 45,
  "processed_at": "2026-01-09T10:30:00"
}
```

**Example:**
```bash
curl http://localhost:8000/pdf/documents/abc123def456
```

## Configuration

### Chunking Parameters

```python
processor = PDFDocumentProcessor(
    chunk_size=500,      # Maximum characters per chunk
    chunk_overlap=50,    # Overlap between consecutive chunks
)
```

**Chunk Size Guidelines:**
- **Small (200-300)**: Better for precise queries, more chunks
- **Medium (400-600)**: Balanced approach (recommended)
- **Large (800-1000)**: Better context, fewer chunks

**Overlap Guidelines:**
- **10-15%**: Minimal overlap for distinct chunks
- **20-25%**: Recommended for context preservation
- **30%+**: High overlap for maximum context

### Qdrant Collection

Each PDF chunk is stored with:
- **Vector**: 768-dimensional PhoBERT embedding
- **Payload**:
  - `text`: Chunk text content
  - `document_id`: Unique document identifier
  - `document_name`: Original filename
  - `document_path`: File path
  - `page`: Source page number
  - `chunk_index`: Index within page
  - `global_chunk_index`: Index across entire document
  - `total_chunks`: Total chunks in document
  - `processed_at`: Processing timestamp
  - Custom metadata fields

## Advanced Usage

### Batch Processing

```python
from pathlib import Path

processor = PDFDocumentProcessor()

# Process all PDFs in a folder
pdf_folder = Path("media")
for pdf_path in pdf_folder.glob("*.pdf"):
    print(f"Processing {pdf_path.name}...")
    result = processor.process_pdf(
        pdf_path=str(pdf_path),
        metadata={"source": "batch_import"}
    )
    print(f"✓ {result['total_chunks']} chunks created")
```

### Filtered Queries

```python
# Query only specific document
results = processor.query_document(
    query="chính sách",
    top_k=10,
    document_name="vpcp_1.pdf"  # Search only this file
)

# Query by document ID
results = processor.query_document(
    query="quy trình",
    top_k=5,
    document_id="abc123def456"
)
```

### Custom Metadata

```python
# Add custom metadata during processing
result = processor.process_pdf(
    pdf_path="media/contract.pdf",
    metadata={
        "category": "legal",
        "department": "legal_dept",
        "year": 2026,
        "confidential": True,
        "tags": ["contract", "agreement"]
    }
)

# Metadata is stored with each chunk and returned in queries
```

### Pagination for Large Results

```python
# Get results in pages
page_size = 10
for page in range(5):
    offset = page * page_size
    results = processor.query_document(
        query="thông tin",
        top_k=page_size
    )
    
    # Process results
    for result in results:
        print(f"Page {result['page']}: {result['text'][:50]}...")
```

## Performance Optimization

### Processing Speed

| Pages | Chunks | Processing Time | Storage Time |
|-------|--------|----------------|--------------|
| 10    | ~40    | 2-5 seconds    | <1 second    |
| 50    | ~200   | 10-20 seconds  | 2-3 seconds  |
| 100   | ~400   | 20-40 seconds  | 5-8 seconds  |

### Query Speed

- **Single document query**: 50-200ms
- **All documents query**: 100-500ms (depends on collection size)
- **Filtering improves performance** by ~30-50%

### Best Practices

1. **Batch Processing**: Process multiple PDFs sequentially, not in parallel
2. **Chunk Size**: Use 400-600 for best balance
3. **Index Management**: Let Qdrant handle indexing automatically
4. **Metadata**: Keep metadata concise for faster queries
5. **Filters**: Use document_id/document_name filters for faster searches

## Example Workflows

### 1. Document QA System

```python
# Process all company documents
for pdf in ["handbook.pdf", "policies.pdf", "procedures.pdf"]:
    processor.process_pdf(f"media/{pdf}")

# Answer questions
question = "Quy trình nghỉ phép như thế nào?"
answers = processor.query_document(query=question, top_k=3)

# Present answer with sources
for i, answer in enumerate(answers, 1):
    print(f"\nAnswer {i} (Score: {answer['score']:.2%}):")
    print(f"Source: {answer['document_name']}, Page {answer['page']}")
    print(f"Content: {answer['text']}")
```

### 2. Contract Search

```python
# Process contracts
result = processor.process_pdf(
    "media/contract.pdf",
    metadata={"type": "contract", "year": 2026}
)

# Search for specific clauses
clauses = processor.query_document(
    query="điều khoản thanh toán",
    top_k=5,
    document_id=result['document_id']
)
```

### 3. Knowledge Base

```python
# Build knowledge base from PDFs
documents = [
    ("guides/user_manual.pdf", {"type": "manual"}),
    ("guides/faq.pdf", {"type": "faq"}),
    ("guides/troubleshooting.pdf", {"type": "troubleshooting"})
]

for pdf_path, metadata in documents:
    processor.process_pdf(pdf_path, metadata=metadata)

# Query across all guides
results = processor.query_document(
    query="Làm thế nào để đặt lại mật khẩu?",
    top_k=3
)
```

## Troubleshooting

### Issue: "PDF file not found"
**Solution**: Check file path is absolute or relative to project root

### Issue: "No text extracted from PDF"
**Solution**: PDF might be image-based. Use OCR preprocessing first.

### Issue: "Collection already exists"
**Solution**: Normal behavior. Processor will use existing collection.

### Issue: Slow processing
**Solution**: 
- Reduce chunk_size for faster processing
- Limit max_pages for testing
- Check Qdrant server is running locally

### Issue: Low quality results
**Solution**:
- Increase chunk_size for more context
- Increase chunk_overlap for better continuity
- Refine query wording

## Running the Example

```bash
# Run the example script
python examples/example_4_pdf_processing.py
```

This will:
1. Process the PDF in `media/vpcp_1.pdf`
2. Create chunks and store in Qdrant
3. Run sample queries
4. Display results with scores

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Next Steps

1. **Try the example**: Run `example_4_pdf_processing.py`
2. **Process your PDFs**: Use API or Python interface
3. **Build your application**: Integrate into your RAG pipeline
4. **Customize**: Adjust chunk size, metadata, and queries

## Support

For issues:
1. Check server logs for detailed errors
2. Verify Qdrant is running: `http://localhost:6333/dashboard`
3. Test with the example script first
4. Review chunk quality with small test documents
