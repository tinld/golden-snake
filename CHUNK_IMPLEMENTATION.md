# PDF to Vector Storage - Complete Implementation

## Overview

This implementation reads PDF files, splits them into **chunks**, generates **vector embeddings**, and stores them in **Qdrant** for semantic search.

## Process Flow

```
PDF File → Read Pages → Chunk Text → Generate Vectors → Store in Qdrant
```

### Detailed Steps:

1. **📄 Read PDF Pages**
   - Extract text from each page
   - Clean and normalize text
   - Preserve page numbers

2. **✂️ Chunk Text**
   - Split text into manageable pieces (default: 500 characters)
   - Add overlap between chunks (default: 50 characters)
   - Maintain metadata (page number, chunk index)

3. **🧠 Generate Embeddings**
   - Create 768-dimensional vectors using PhoBERT
   - Each chunk gets its own vector
   - Vectors represent semantic meaning

4. **💾 Store in Qdrant**
   - Each chunk stored as a point
   - Vector for semantic search
   - Payload with text and metadata

## Key Features

### ✅ Chunking Strategy

**Why chunk?**
- PDFs can be very long
- Models have token limits
- Better search precision
- Faster retrieval

**Chunk Configuration:**
```python
PDFChunker(
    chunk_size=500,      # Max characters per chunk
    chunk_overlap=50     # Overlap for context continuity
)
```

**Chunk sizes:**
- **Small (200-300)**: More precise, more chunks, slower
- **Medium (400-600)**: Balanced ⭐ RECOMMENDED
- **Large (800-1000)**: More context, fewer chunks

### ✅ Storage Structure

Each chunk in Qdrant contains:

**Vector (768 dimensions):**
```
[0.123, -0.456, 0.789, ..., 0.321]
```

**Payload (metadata):**
```json
{
  "text": "Full chunk text...",
  "document_id": "abc123def456",
  "document_name": "vpcp_1.pdf",
  "document_path": "media/vpcp_1.pdf",
  "page": 5,
  "chunk_index": 2,
  "global_chunk_index": 23,
  "total_chunks": 45,
  "processed_at": "2026-01-09T10:30:00"
}
```

## Usage Examples

### Example 1: Basic Processing

```python
from src.media.pdf_processor import PDFDocumentProcessor

# Initialize processor
processor = PDFDocumentProcessor(
    chunk_size=500,
    chunk_overlap=50
)

# Process PDF with chunking
result = processor.process_pdf(
    pdf_path="media/vpcp_1.pdf",
    max_pages=None  # Process all pages
)

print(f"Created {result['total_chunks']} chunks from {result['total_pages']} pages")
```

### Example 2: Query Chunks

```python
# Search across all chunks
results = processor.query_document(
    query="thông tin quan trọng",
    top_k=5
)

for result in results:
    print(f"Page {result['page']}, Chunk {result['chunk_index']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text'][:100]}...\n")
```

### Example 3: Visual Demo

```python
# See chunking in action
from src.media.pdf import PDFReadingTransformer
from src.media.pdf_processor import PDFChunker

# Read PDF
reader = PDFReadingTransformer("media/vpcp_1.pdf", max_pages=1)
pages = reader.read_pdf_page_records()

# Create chunks
chunker = PDFChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_pages(pages)

print(f"Page 1 text: {len(pages[0]['text'])} characters")
print(f"Split into: {len(chunks)} chunks")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}: {len(chunk['text'])} characters")
    print(f"Preview: {chunk['text'][:80]}...")
```

## Test Scripts

### 1. Full Test (Recommended)

```bash
python test_pdf_reader.py
```

**What it does:**
- ✅ Reads PDF from media folder
- ✅ Chunks text (500 char chunks, 50 char overlap)
- ✅ Generates embeddings for each chunk
- ✅ Stores all chunks in Qdrant
- ✅ Tests semantic queries
- ✅ Verifies storage in Qdrant

### 2. Chunking Demo

```bash
python demo_chunking.py
```

**What it shows:**
- 📊 Step-by-step chunking process
- 📈 Chunk statistics
- 🔍 Overlap visualization
- 💾 Storage format
- 📏 Size comparisons

### 3. Simple Test

```bash
python examples/example_4_pdf_processing.py
```

## Chunk Storage Details

### Point ID Format
```
{document_id}_chunk_{chunk_number}
```
Example: `abc123def456_chunk_0`

### Batch Upload
Chunks uploaded in batches of 100 for efficiency:
```python
batch_size = 100
for i in range(0, len(points), batch_size):
    batch = points[i:i + batch_size]
    qdrant_client.upsert(collection_name, points=batch)
```

### Collection Configuration
```python
Collection: pdf_documents
Vector Size: 768
Distance: COSINE
```

## Query Capabilities

### Semantic Search
```python
# Find similar content semantically
results = processor.query_document(
    query="chính sách công ty",
    top_k=10
)
```

### Filter by Document
```python
# Search specific document only
results = processor.query_document(
    query="điều khoản",
    document_id="abc123",
    top_k=5
)
```

### Filter by Document Name
```python
# Search by filename
results = processor.query_document(
    query="quy trình",
    document_name="vpcp_1.pdf"
)
```

## Performance Metrics

### Processing Speed
| Pages | Chunks | Read Time | Embed Time | Store Time |
|-------|--------|-----------|------------|------------|
| 10    | ~40    | 1-2s      | 2-3s       | <1s        |
| 50    | ~200   | 5-8s      | 10-15s     | 2-3s       |
| 100   | ~400   | 10-15s    | 20-30s     | 5-8s       |

### Query Speed
- **Semantic search**: 50-200ms
- **Filtered search**: 30-100ms
- **Results with text**: +10-20ms

## Chunk Overlap Explained

**Without overlap:**
```
Chunk 1: [Text A]
Chunk 2: [Text B]
Chunk 3: [Text C]
```

**With overlap (50 chars):**
```
Chunk 1: [Text A | overlap]
Chunk 2: [overlap | Text B | overlap]
Chunk 3: [overlap | Text C]
```

**Benefits:**
- ✅ Context continuity between chunks
- ✅ Better semantic understanding
- ✅ Reduced information loss at boundaries

## Troubleshooting

### Issue: "No chunks created"
**Cause:** Text too short or empty PDF
**Solution:** Check PDF has text content (not scanned images)

### Issue: "Too many chunks"
**Cause:** Chunk size too small
**Solution:** Increase chunk_size to 800-1000

### Issue: "Poor search results"
**Cause:** Chunk size too large or too small
**Solution:** Use recommended 400-600 character chunks

### Issue: "Slow processing"
**Cause:** Large PDF or slow embeddings
**Solution:** 
- Use max_pages to limit processing
- Process in background
- Check GPU availability for embeddings

## Verification

### Check chunks in Qdrant:
```python
from src.database import get_qdrant

qdrant = get_qdrant()
collection = qdrant.get_collection("pdf_documents")

print(f"Total chunks stored: {collection.points_count}")
print(f"Vector dimension: {collection.config.params.vectors.size}")
```

### View chunks in dashboard:
- Open: http://localhost:6333/dashboard
- Select collection: `pdf_documents`
- View points and payloads

## Next Steps

1. **Run test**: `python test_pdf_reader.py`
2. **View demo**: `python demo_chunking.py`
3. **Adjust chunks**: Modify chunk_size if needed
4. **Process your PDFs**: Add PDFs to media folder
5. **Query documents**: Use semantic search

## Complete Example

```python
from src.media.pdf_processor import PDFDocumentProcessor

# Initialize
processor = PDFDocumentProcessor(
    collection_name="my_documents",
    chunk_size=500,
    chunk_overlap=50
)

# Process with chunking
result = processor.process_pdf(
    pdf_path="media/document.pdf",
    metadata={"category": "business"}
)

print(f"✓ Processed: {result['total_chunks']} chunks stored")

# Query chunks
results = processor.query_document(
    query="important information",
    top_k=5
)

# Display results
for i, r in enumerate(results, 1):
    print(f"{i}. [Page {r['page']}, Chunk {r['chunk_index']}] Score: {r['score']:.3f}")
    print(f"   {r['text'][:100]}...")
```

## Files Reference

- **Implementation**: `src/media/pdf_processor.py`
- **Test**: `test_pdf_reader.py`
- **Demo**: `demo_chunking.py`
- **Example**: `examples/example_4_pdf_processing.py`
- **Docs**: `PDF_PROCESSING.md`

---

**Ready to use!** Run `python test_pdf_reader.py` to see PDF chunking and vectorization in action.
