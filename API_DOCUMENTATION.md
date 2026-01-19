# RAG Query API Documentation

## Overview

This API provides endpoints for processing Vietnamese queries using a Retrieval-Augmented Generation (RAG) pipeline. It combines query classification with context retrieval to provide intelligent query processing with confidence scores.

## Quick Start

### Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Server

```bash
python api.py
```

The server will start on `http://localhost:8000`

### Access the Interactive API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Purpose:** Check API and RAG pipeline status

**Response:**
```json
{
  "status": "ok",
  "rag_pipeline_ready": true
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### 2. Process Query (Full Pipeline)

**Endpoint:** `POST /query`

**Purpose:** Process a query with classification and context retrieval

**Request Body:**
```json
{
  "query": "Tôi cần tìm kiếm thông tin về sản phẩm",
  "retrieve_context": true,
  "top_k": 5
}
```

**Parameters:**
- `query` (string, required): The query text to process
- `retrieve_context` (boolean, optional, default: true): Whether to retrieve context documents
- `top_k` (integer, optional, default: 5): Number of context documents to retrieve (1-20)

**Response:**
```json
{
  "query": "Tôi cần tìm kiếm thông tin về sản phẩm",
  "primary_category": "product_search",
  "primary_score": 0.95,
  "all_classifications": [
    {
      "category": "product_search",
      "confidence": 0.95,
      "intent": "search"
    },
    {
      "category": "general_inquiry",
      "confidence": 0.04,
      "intent": null
    },
    {
      "category": "complaint",
      "confidence": 0.01,
      "intent": null
    }
  ],
  "context": [
    {
      "id": "doc1",
      "text": "Sample context document",
      "score": 0.92,
      "metadata": {
        "category": "product_search",
        "confidence": "0.95",
        "query_id": "abc123"
      }
    }
  ],
  "num_results": 1
}
```

**Status Codes:**
- `200 OK`: Query processed successfully
- `400 Bad Request`: Invalid request parameters
- `500 Internal Server Error`: Server error during processing
- `503 Service Unavailable`: RAG pipeline not initialized

**Example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tôi cần tìm kiếm thông tin về sản phẩm",
    "retrieve_context": true,
    "top_k": 5
  }'
```

---

### 3. Classify Query Only

**Endpoint:** `POST /classify`

**Purpose:** Classify a query without retrieving context

**Request Body:**
```json
{
  "query": "Tôi cần tìm kiếm thông tin về sản phẩm"
}
```

**Parameters:**
- `query` (string, required): The query text to classify

**Response:**
```json
[
  {
    "category": "product_search",
    "confidence": 0.95,
    "intent": "search"
  },
  {
    "category": "general_inquiry",
    "confidence": 0.04,
    "intent": null
  },
  {
    "category": "complaint",
    "confidence": 0.01,
    "intent": null
  }
]
```

**Status Codes:**
- `200 OK`: Classification successful
- `400 Bad Request`: Invalid request parameters
- `500 Internal Server Error`: Server error during processing
- `503 Service Unavailable`: RAG pipeline not initialized

**Example:**
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "Tôi cần tìm kiếm thông tin về sản phẩm"}'
```

---

### 4. Retrieve Context Only

**Endpoint:** `POST /retrieve`

**Purpose:** Retrieve context documents for a query without classification

**Request Body:**
```json
{
  "query": "Tôi cần tìm kiếm thông tin về sản phẩm",
  "top_k": 5
}
```

**Parameters:**
- `query` (string, required): The query text
- `top_k` (integer, optional, default: 5): Number of documents to retrieve (1-20)

**Response:**
```json
[
  {
    "id": "doc1",
    "text": "Sample context document",
    "score": 0.92,
    "metadata": {
      "category": "product_search",
      "confidence": "0.95",
      "query_id": "abc123"
    }
  },
  {
    "id": "doc2",
    "text": "Another context document",
    "score": 0.88,
    "metadata": {
      "category": "product_search",
      "confidence": "0.90",
      "query_id": "def456"
    }
  }
]
```

**Status Codes:**
- `200 OK`: Retrieval successful
- `400 Bad Request`: Invalid request parameters
- `500 Internal Server Error`: Server error during processing
- `503 Service Unavailable`: RAG pipeline not initialized

**Example:**
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tôi cần tìm kiếm thông tin về sản phẩm",
    "top_k": 5
  }'
```

---

## Response Fields Explained

### Classification Results
- **category**: The predicted category for the query
- **confidence**: Confidence score between 0 and 1 (higher = more confident)
- **intent**: Detected intent (e.g., "search", "complaint", "inquiry")

### Context Documents
- **id**: Unique identifier for the document
- **text**: The text content of the document
- **score**: Similarity score between 0 and 1 (higher = more similar to query)
- **metadata**: Additional metadata about the document

---

## Usage Examples

### Python Example

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/query"

# Request payload
payload = {
    "query": "Tôi cần tìm kiếm thông tin về sản phẩm",
    "retrieve_context": True,
    "top_k": 5
}

# Send request
response = requests.post(url, json=payload)

# Parse response
result = response.json()

print(f"Query: {result['query']}")
print(f"Primary Category: {result['primary_category']}")
print(f"Primary Score: {result['primary_score']:.2%}")
print(f"Number of Results: {result['num_results']}")

for i, classification in enumerate(result['all_classifications']):
    print(f"  {i+1}. {classification['category']}: {classification['confidence']:.2%}")

for i, doc in enumerate(result['context']):
    print(f"  Document {i+1}: {doc['text'][:50]}... (Score: {doc['score']:.2%})")
```

### JavaScript/Node.js Example

```javascript
// Using fetch API
const queryData = {
    query: "Tôi cần tìm kiếm thông tin về sản phẩm",
    retrieve_context: true,
    top_k: 5
};

fetch('http://localhost:8000/query', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(queryData)
})
.then(response => response.json())
.then(result => {
    console.log(`Query: ${result.query}`);
    console.log(`Primary Category: ${result.primary_category}`);
    console.log(`Primary Score: ${(result.primary_score * 100).toFixed(2)}%`);
    console.log(`Number of Results: ${result.num_results}`);
})
.catch(error => console.error('Error:', error));
```

### cURL Examples

**Classification Only:**
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tôi cần giúp đỡ"
  }' | python -m json.tool
```

**Full Query Processing:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tôi cần giúp đỡ",
    "retrieve_context": true,
    "top_k": 3
  }' | python -m json.tool
```

**Retrieval Only:**
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tôi cần giúp đỡ",
    "top_k": 5
  }' | python -m json.tool
```

---

## Error Handling

All endpoints return structured error responses:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Scenarios

1. **RAG Pipeline Not Initialized (503)**
   ```json
   {
     "detail": "RAG Pipeline not initialized. Please check server logs."
   }
   ```
   - **Solution**: Ensure Qdrant and MongoDB are running

2. **Invalid Request (400)**
   ```json
   {
     "detail": "Query text cannot be empty"
   }
   ```
   - **Solution**: Provide a non-empty query string

3. **Server Error (500)**
   ```json
   {
     "detail": "Error processing query: [specific error message]"
   }
   ```
   - **Solution**: Check server logs for detailed error information

---

## Performance Notes

- **Query Classification**: ~100-500ms
- **Context Retrieval**: ~50-200ms (depends on database size)
- **Full Query Processing**: ~200-700ms

---

## Deployment

### Production Deployment with Gunicorn

```bash
pip install gunicorn

gunicorn api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Environment Configuration

Set these environment variables if needed:

```bash
# Qdrant Connection
QDRANT_HOST=localhost
QDRANT_PORT=6333

# MongoDB Connection
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=rag_db

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Support

For issues or questions:
1. Check the server logs for detailed error messages
2. Verify all backend services (Qdrant, MongoDB) are running
3. Review the API documentation at `/docs`

---

## Version History

- **v1.0.0** (2026-01-09): Initial release with classification and retrieval endpoints
