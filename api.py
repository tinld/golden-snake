"""
FastAPI REST API for RAG Query Processing
Provides endpoints to query and get results with confidence scores
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag.rag_pipeline import RAGPipeline
from src.classifier import QueryClassifier, get_embedder
from src.media.pdf_processor import PDFDocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Query API",
    description="API for processing Vietnamese queries with classification and retrieval",
    version="1.0.0"
)

# Initialize RAG pipeline globally
try:
    rag_pipeline = RAGPipeline()
    logger.info("RAG Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG Pipeline: {e}")
    rag_pipeline = None

# Initialize PDF processor
try:
    pdf_processor = PDFDocumentProcessor(
        collection_name="pdf_documents",
        chunk_size=500,
        chunk_overlap=50
    )
    logger.info("PDF Processor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize PDF Processor: {e}")
    pdf_processor = None


# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for query processing"""
    query: str = Field(..., min_length=1, description="The query text to process")
    retrieve_context: bool = Field(default=True, description="Whether to retrieve context from vector DB")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context documents to retrieve")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Tôi cần tìm kiếm thông tin về sản phẩm",
                "retrieve_context": True,
                "top_k": 5
            }
        }


class ClassificationResult(BaseModel):
    """Classification result for a query"""
    category: str = Field(..., description="Predicted category")
    confidence: float = Field(..., description="Confidence score (0-1)")
    intent: Optional[str] = Field(None, description="Detected intent")


class ContextDocument(BaseModel):
    """A retrieved context document"""
    id: str = Field(..., description="Document ID")
    text: str = Field(..., description="Document text")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class QueryResponse(BaseModel):
    """Response model for query processing"""
    query: str = Field(..., description="The original query")
    primary_category: str = Field(..., description="Primary category classification")
    primary_score: float = Field(..., description="Confidence score for primary category")
    all_classifications: List[ClassificationResult] = Field(..., description="Top 3 classifications")
    context: List[ContextDocument] = Field(default_factory=list, description="Retrieved context documents")
    num_results: int = Field(..., description="Number of retrieved context documents")

    class Config:
        json_schema_extra = {
            "example": {
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
                        "intent": None
                    }
                ],
                "context": [
                    {
                        "id": "doc1",
                        "text": "Sample context document",
                        "score": 0.92,
                        "metadata": {}
                    }
                ],
                "num_results": 1
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    rag_pipeline_ready: bool = Field(..., description="Whether RAG pipeline is initialized")
    pdf_processor_ready: bool = Field(..., description="Whether PDF processor is initialized")


# PDF Processing Models
class PDFProcessRequest(BaseModel):
    """Request model for PDF processing"""
    pdf_path: str = Field(..., description="Path to PDF file")
    max_pages: Optional[int] = Field(None, description="Maximum number of pages to process")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pdf_path": "media/vpcp_1.pdf",
                "max_pages": None,
                "metadata": {
                    "category": "business",
                    "language": "vietnamese"
                }
            }
        }


class PDFProcessResponse(BaseModel):
    """Response model for PDF processing"""
    status: str = Field(..., description="Processing status")
    document_id: Optional[str] = Field(None, description="Generated document ID")
    document_name: Optional[str] = Field(None, description="Document name")
    total_pages: Optional[int] = Field(None, description="Total pages processed")
    total_chunks: Optional[int] = Field(None, description="Total chunks created")
    chunks_stored: Optional[int] = Field(None, description="Chunks stored in database")
    collection: Optional[str] = Field(None, description="Collection name")
    message: Optional[str] = Field(None, description="Error message if failed")


class PDFQueryRequest(BaseModel):
    """Request model for PDF document queries"""
    query: str = Field(..., min_length=1, description="Query text")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    document_id: Optional[str] = Field(None, description="Filter by document ID")
    document_name: Optional[str] = Field(None, description="Filter by document name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "thông tin về công ty",
                "top_k": 5,
                "document_id": None,
                "document_name": "vpcp_1.pdf"
            }
        }


class PDFChunkResult(BaseModel):
    """A chunk result from PDF query"""
    id: str = Field(..., description="Chunk ID")
    text: str = Field(..., description="Chunk text")
    score: float = Field(..., description="Similarity score")
    document_name: str = Field(..., description="Source document name")
    document_id: str = Field(..., description="Source document ID")
    page: int = Field(..., description="Page number")
    chunk_index: int = Field(..., description="Chunk index within page")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentInfo(BaseModel):
    """Document information"""
    document_id: str = Field(..., description="Document ID")
    document_name: str = Field(..., description="Document name")
    document_path: str = Field(..., description="Document path")
    total_chunks: int = Field(..., description="Total chunks in document")
    processed_at: str = Field(..., description="Processing timestamp")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns:
        HealthResponse with status and pipeline readiness
    """
    return HealthResponse(
        status="ok",
        rag_pipeline_ready=rag_pipeline is not None,
        pdf_processor_ready=pdf_processor is not None
    )


@app.post("/query", response_model=QueryResponse, tags=["Query Processing"])
async def process_query(request: QueryRequest):
    """
    Process a query and return classification, score, and context
    
    Args:
        request: QueryRequest containing query text and parameters
        
    Returns:
        QueryResponse with classifications, scores, and retrieved context
        
    Raises:
        HTTPException: If RAG pipeline is not initialized or query processing fails
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG Pipeline not initialized. Please check server logs."
        )
    
    try:
        # Process query through RAG pipeline
        result = rag_pipeline.process_query(
            query=request.query,
            retrieve_context=request.retrieve_context,
            top_k=request.top_k
        )
        
        # Extract classifications
        classifications = result.get("classifications", [])
        
        # Build classification results
        classification_results = []
        for clf in classifications:
            classification_results.append(
                ClassificationResult(
                    category=clf["category"],
                    confidence=clf["confidence"],
                    intent=clf.get("intent")
                )
            )
        
        # Build context results
        context_results = []
        if result.get("context"):
            for doc in result["context"].get("results", []):
                context_results.append(
                    ContextDocument(
                        id=doc.get("id", ""),
                        text=doc.get("text", ""),
                        score=doc.get("score", 0.0),
                        metadata=doc.get("metadata", {})
                    )
                )
        
        # Create response
        response = QueryResponse(
            query=request.query,
            primary_category=result.get("primary_category", "unknown"),
            primary_score=classifications[0]["confidence"] if classifications else 0.0,
            all_classifications=classification_results,
            context=context_results,
            num_results=result.get("num_results", 0)
        )
        
        logger.info(f"Successfully processed query: {request.query[:50]}...")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/classify", response_model=List[ClassificationResult], tags=["Classification"])
async def classify_query(request: QueryRequest):
    """
    Classify a query without retrieving context
    
    Args:
        request: QueryRequest containing query text
        
    Returns:
        List of ClassificationResult objects with top 3 categories and scores
        
    Raises:
        HTTPException: If RAG pipeline is not initialized or classification fails
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG Pipeline not initialized. Please check server logs."
        )
    
    try:
        # Get classifications
        classifications = rag_pipeline.classifier.classify(request.query, top_k=3)
        
        # Convert to response format
        results = []
        for clf in classifications:
            results.append(
                ClassificationResult(
                    category=clf["category"],
                    confidence=clf["confidence"],
                    intent=clf.get("intent")
                )
            )
        
        logger.info(f"Successfully classified query: {request.query[:50]}...")
        return results
        
    except Exception as e:
        logger.error(f"Error classifying query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error classifying query: {str(e)}"
        )


@app.post("/retrieve", response_model=List[ContextDocument], tags=["Retrieval"])
async def retrieve_context(request: QueryRequest):
    """
    Retrieve context documents for a query
    
    Args:
        request: QueryRequest containing query text and top_k
        
    Returns:
        List of ContextDocument objects with similarity scores
        
    Raises:
        HTTPException: If RAG pipeline is not initialized or retrieval fails
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG Pipeline not initialized. Please check server logs."
        )
    
    try:
        # Retrieve context
        result = rag_pipeline.retrieve(request.query, top_k=request.top_k)
        
        # Convert to response format
        context_results = []
        for doc in result.get("results", []):
            context_results.append(
                ContextDocument(
                    id=doc.get("id", ""),
                    text=doc.get("text", ""),
                    score=doc.get("score", 0.0),
                    metadata=doc.get("metadata", {})
                )
            )
        
        logger.info(f"Retrieved {len(context_results)} documents for query: {request.query[:50]}...")
        return context_results
        
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving context: {str(e)}"
        )


# ============================================================================
# PDF Processing Endpoints
# ============================================================================

@app.post("/pdf/process", response_model=PDFProcessResponse, tags=["PDF Processing"])
async def process_pdf_document(request: PDFProcessRequest):
    """
    Process a PDF document: extract text, chunk, vectorize, and store
    
    Args:
        request: PDFProcessRequest with PDF path and options
        
    Returns:
        PDFProcessResponse with processing results
        
    Raises:
        HTTPException: If PDF processor is not initialized or processing fails
    """
    if pdf_processor is None:
        raise HTTPException(
            status_code=503,
            detail="PDF Processor not initialized. Please check server logs."
        )
    
    try:
        # Process the PDF
        result = pdf_processor.process_pdf(
            pdf_path=request.pdf_path,
            max_pages=request.max_pages,
            metadata=request.metadata
        )
        
        # Convert to response model
        response = PDFProcessResponse(**result)
        
        logger.info(f"Successfully processed PDF: {request.pdf_path}")
        return response
        
    except FileNotFoundError as e:
        logger.error(f"PDF file not found: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )


@app.post("/pdf/query", response_model=List[PDFChunkResult], tags=["PDF Processing"])
async def query_pdf_documents(request: PDFQueryRequest):
    """
    Query PDF documents and retrieve relevant chunks
    
    Args:
        request: PDFQueryRequest with query and filters
        
    Returns:
        List of PDFChunkResult objects with matching chunks
        
    Raises:
        HTTPException: If PDF processor is not initialized or query fails
    """
    if pdf_processor is None:
        raise HTTPException(
            status_code=503,
            detail="PDF Processor not initialized. Please check server logs."
        )
    
    try:
        # Query the documents
        results = pdf_processor.query_document(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id,
            document_name=request.document_name
        )
        
        # Convert to response format
        chunk_results = []
        for result in results:
            chunk_results.append(
                PDFChunkResult(
                    id=result["id"],
                    text=result["text"],
                    score=result["score"],
                    document_name=result["document_name"],
                    document_id=result["document_id"],
                    page=result["page"],
                    chunk_index=result["chunk_index"],
                    metadata=result["metadata"]
                )
            )
        
        logger.info(f"Found {len(chunk_results)} results for query: {request.query[:50]}...")
        return chunk_results
        
    except Exception as e:
        logger.error(f"Error querying PDF documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error querying PDF documents: {str(e)}"
        )


@app.get("/pdf/documents", response_model=List[DocumentInfo], tags=["PDF Processing"])
async def list_pdf_documents():
    """
    List all processed PDF documents
    
    Returns:
        List of DocumentInfo objects
        
    Raises:
        HTTPException: If PDF processor is not initialized or listing fails
    """
    if pdf_processor is None:
        raise HTTPException(
            status_code=503,
            detail="PDF Processor not initialized. Please check server logs."
        )
    
    try:
        documents = pdf_processor.list_documents()
        
        # Convert to response format
        doc_infos = []
        for doc in documents:
            doc_infos.append(
                DocumentInfo(
                    document_id=doc["document_id"],
                    document_name=doc["document_name"],
                    document_path=doc["document_path"],
                    total_chunks=doc["total_chunks"],
                    processed_at=doc["processed_at"]
                )
            )
        
        logger.info(f"Listed {len(doc_infos)} documents")
        return doc_infos
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )


@app.get("/pdf/documents/{document_id}", response_model=DocumentInfo, tags=["PDF Processing"])
async def get_pdf_document_info(document_id: str):
    """
    Get information about a specific PDF document
    
    Args:
        document_id: Document ID
        
    Returns:
        DocumentInfo object
        
    Raises:
        HTTPException: If document not found or retrieval fails
    """
    if pdf_processor is None:
        raise HTTPException(
            status_code=503,
            detail="PDF Processor not initialized. Please check server logs."
        )
    
    try:
        doc_info = pdf_processor.get_document_info(document_id)
        
        if doc_info.get("status") == "not_found":
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        return DocumentInfo(
            document_id=doc_info["document_id"],
            document_name=doc_info["document_name"],
            document_path=doc_info["document_path"],
            total_chunks=doc_info["total_chunks"],
            processed_at=doc_info["processed_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting document info: {str(e)}"
        )


# ============================================================================
# Root endpoint
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information
    """
    return {
        "name": "RAG Query API",
        "version": "1.0.0",
        "description": "API for processing Vietnamese queries with classification and retrieval",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "classify": "/classify",
            "retrieve": "/retrieve",
            "pdf_process": "/pdf/process",
            "pdf_query": "/pdf/query",
            "pdf_documents": "/pdf/documents"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
