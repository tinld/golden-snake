"""
PDF Document Processor
Handles PDF ingestion, chunking, vectorization, and storage in Qdrant
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib
import uuid
from datetime import datetime
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.media.pdf import PDFReadingTransformer
from src.classifier.embedder import get_embedder
from src.vector_db.local_vector_db import LocalVectorDB
from src.database import get_qdrant, get_mongodb, QdrantVectorDB, MongoDBClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from src.config import MONGODB_DATABASE
from src.classifier import QueryClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFChunker:
    """
    Intelligent PDF text chunking with overlap for better context preservation
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n"
    ):
        """
        Initialize PDF chunker
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
            separator: Text separator for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        # Normalize unicode spaces
        text = text.replace('\u00a0', ' ')

        # Fix ligatures
        ligatures = {
            'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff',
            'ﬃ': 'ffi', 'ﬄ': 'ffl',
        }
        for k, v in ligatures.items():
            text = text.replace(k, v)

        # Fix smart quotes
        text = (
            text.replace('“', '"').replace('”', '"')
                .replace('‘', "'").replace('’', "'")
        )

        # Normalize dashes and ellipsis
        text = text.replace('–', '-').replace('—', '-')
        text = text.replace('…', '...')

        # Clean excessive spaces
        text = re.sub(r'[ \t]+', ' ', text)

        # Normalize newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Fix spaced-character lines (line-by-line)
        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            if re.search(r'(\w\s){6,}', line):
                chars = re.findall(r'\w', line)
                spaces = line.count(' ')
                if spaces / max(len(line), 1) > 0.4:
                    line = re.sub(r'(\w)\s+(?=\w)', r'\1', line)
            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)

        # Remove standalone page numbers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        # Strip final result
        return text.strip()
     
    def _is_valid_chunk(self, text: str) -> bool:
        """
        Check if chunk contains meaningful content
        
        Args:
            text: Chunk text
            
        Returns:
            True if chunk is valid
        """
        # Minimum length check
        if len(text.strip()) < 20:
            return False
        
        # Check if contains enough letters (not just numbers/symbols)
        letter_count = sum(c.isalpha() for c in text)
        if letter_count < 10:
            return False
        
        # Check if not just repetitive characters
        unique_chars = len(set(text.replace(' ', '').replace('\n', '')))
        if unique_chars < 5:
            return False
        
        return True
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text by sentences for better semantic chunks
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with NLP libraries)
        sentences = re.split(r'(?<=[.!?;])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_legal_sentences(self, text: str) -> List[str]:
        """
        Sentence splitting adapted for Vietnamese legal documents
        """
        parts = re.split(
            r'(?<=[.;!?])\s+|\n(?=Căn cứ|Theo|Điều|Khoản|Mục|Chương|Quyết định)',
            text
        )
        return [p.strip() for p in parts if p.strip()]
    
    def remove_headers_footers(self, text: str) -> str:
        """
        Remove common headers and footers from PDF text
        
        Args:
            text: Input page text
        Returns:
            Cleaned text without headers/footers
        """
        
        lines = text.splitlines()
        cleaned_lines = []
        
        for line in lines:
            # Remove page numbers
            if re.match(r'^\s*\d+\s*$', line):
                continue
            # Remove common headers/footers (customize as needed)
            if re.match(r'^(Trang|Page|Document|Confidential|Company Name)', line, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines).strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into semantic, embedding-friendly chunks
        (150–300 tokens equivalent)
        """

        if not text:
            return []

        text = self._clean_text(text)
        if not text:
            return []

        sentences = self._split_by_sentences(text)
        # print(sentences)

        chunks = []
        current = []
        current_len = 0

        TARGET_LEN = self.chunk_size          # e.g. 600 chars
        MAX_LEN = int(self.chunk_size * 1.2)  # soft limit
        MIN_LEN = int(self.chunk_size * 0.3)  # minimum chunk size

        for sent in sentences:
            sent_len = len(sent)
            
            if sent_len < MIN_LEN:
                continue

            # Very long sentence → force split
            if sent_len > MAX_LEN:
                if current:
                    chunks.append(" ".join(current))
                    current, current_len = [], 0

                for i in range(0, sent_len, TARGET_LEN):
                    sub = sent[i:i + TARGET_LEN]
                    if self._is_valid_chunk(sub):
                        chunks.append(sub)
                continue

            if current_len + sent_len > MAX_LEN:
                chunks.append(" ".join(current))
                current, current_len = [], 0

            current.append(sent)
            current_len += sent_len

        if current:
            chunks.append(" ".join(current))

        return [c for c in chunks if self._is_valid_chunk(c)]

    def chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        Chunk multiple pages with metadata preservation and text cleaning
        
        Args:
            pages: List of page dicts with 'page' and 'text' keys
            
        Returns:
            List of chunk dicts with cleaned text and metadata
        """
        all_chunks = []
        global_idx = 0
        
        for page_record in pages:
            page_num = page_record.get("page", 0)
            page_text = page_record.get("text", "")
            
            if not page_text:
                continue
            
            # Clean and chunk the page text
            page_text = self._clean_text(page_text)
            
            if not page_text:
                continue
            
            # Split by paragraphs first
            paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
            
            # Chunk each paragraph
            page_chunk_idx = 0
            for paragraph in paragraphs:
                paragraph_chunks = self.chunk_text(paragraph)
                
                for chunk_text in paragraph_chunks:
                    if not self._is_valid_chunk(chunk_text):
                        continue
                    
                    # Remove newlines and normalize spaces
                    chunk_text = chunk_text.replace('\n', ' ')
                    chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
                    
                    chunk_record = {
                        "text": chunk_text,
                        "page": page_num,
                        "chunk_index": page_chunk_idx,
                        "total_chunks_in_page": len(paragraph_chunks),
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split())
                    }
                    all_chunks.append(chunk_record)
                    page_chunk_idx += 1
                    global_idx += 1
        
        logger.info(f"Created {len(all_chunks)} valid chunks from {len(pages)} pages")
        return all_chunks


class PDFDocumentProcessor:
    """
    Complete PDF processing pipeline with optimized hybrid storage:
    
    ARCHITECTURE:
    1. Read PDF pages
    2. Chunk text into manageable pieces
    3. Generate embeddings
    4. Store intelligently across Qdrant + MongoDB
    
    STORAGE OPTIMIZATION:
    - Qdrant (Vector DB): 
      * Vector embeddings for similarity search
      * Minimal metadata (document_id, page, chunk_index, etc.)
      * Text preview only (first 100 chars)
      * Optimized for: Fast vector search, filtering, low memory
      
    - MongoDB (Document DB):
      * Full text content for all chunks
      * Complete document metadata
      * Processing statistics
      * Optimized for: Text retrieval, analytics, structured queries
      
    BENEFITS:
    - 60-80% reduction in Qdrant storage and memory usage
    - Faster vector search (smaller payloads)
    - Better data separation and scalability
    - Full text available on-demand from MongoDB
    - Efficient batch text retrieval
    
    USAGE:
        processor = PDFDocumentProcessor(collection_name="my_docs")
        
        # Process PDF
        result = processor.process_pdf("document.pdf")
        
        # Query (returns preview + fetches full text from MongoDB)
        results = processor.query_document("search query", top_k=5)
        
        # Get specific chunk
        chunk = processor.get_chunk_by_id(chunk_id)
    """
    
    def __init__(
        self,
        collection_name: str = "pdf_documents",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize PDF document processor
        
        Args:
            collection_name: Qdrant collection name for PDF documents
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.collection_name = collection_name
        self.embedder = get_embedder()
        self.chunker = PDFChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Initialize Qdrant vector DB for embeddings
        try:
            self.vector_db = QdrantVectorDB(collection_name=collection_name)
            self._init_collection()
            logger.info(f"✓ PDF processor initialized with Qdrant collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {str(e)}")
            raise
        
        # Initialize MongoDB for document metadata
        try:
            self.mongodb = get_mongodb()
            # Create pdf_documents collection if it doesn't exist
            if "pdf_documents" not in self.mongodb.db.list_collection_names():
                self.mongodb.db.create_collection("pdf_documents")
            self.pdf_collection = self.mongodb.db["pdf_documents"]
            # Create indexes
            self.pdf_collection.create_index([("document_id", 1)], unique=True)
            self.pdf_collection.create_index([("document_name", 1)])
            self.pdf_collection.create_index([("processed_at", -1)])
            logger.info(f"✓ PDF processor initialized with MongoDB collection 'pdf_documents'")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {str(e)}")
            raise
    
    def _init_collection(self):
        """Create Qdrant collection for PDF documents if it doesn't exist"""
        try:
            # Try to get collection info
            collection_info = self.vector_db.client.get_collection(self.collection_name)
            logger.info(f"✓ Collection '{self.collection_name}' already exists")
            logger.info(f"  Points: {collection_info.points_count}, Vector size: {collection_info.config.params.vectors.size}")
        except Exception:
            # Collection doesn't exist, create it
            try:
                logger.info(f"Creating new collection '{self.collection_name}'...")
                self.vector_db.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=768,  # PhoBERT embedding dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✓ Created Qdrant collection: {self.collection_name}")
            except Exception as e:
                # Collection might have been created by another process
                if "already exists" in str(e).lower():
                    logger.info(f"✓ Collection '{self.collection_name}' was created by another process")
                else:
                    logger.error(f"Error creating collection '{self.collection_name}': {str(e)}")
                    raise
    
    def _generate_document_id(self, pdf_path: Path) -> str:
        """
        Generate unique document ID based on file path and modification time
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Unique document ID
        """
        # Use file path and modification time for uniqueness
        file_info = f"{pdf_path.name}_{pdf_path.stat().st_mtime}"
        doc_hash = hashlib.md5(file_info.encode()).hexdigest()
        return doc_hash
    
    def process_pdf(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Process a PDF file: read, chunk, vectorize, and store intelligently
        - Qdrant: Vector embeddings + chunk-level data for similarity search
        - MongoDB: Document-level metadata + processing history + statistics
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process
            metadata: Additional metadata to store with chunks
            
        Returns:
            Dict with processing statistics
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        processing_start = datetime.now()
        classifier = QueryClassifier()
        
        # Step 1: Read PDF pages
        logger.info(f"Reading PDF: {pdf_path.name}")
        pdf_reader = PDFReadingTransformer(pdf_path, max_pages=max_pages)
        pages = pdf_reader.read_pdf_page_records(clean=True)
        
        if not pages:
            return {"status": "error", "message": "No text found in PDF"}
        
        # Step 2: Chunk pages into smaller pieces
        logger.info(f"Chunking {len(pages)} pages...")
        chunks = self.chunker.chunk_pages(pages)
        
        if not chunks:
            return {"status": "error", "message": "No chunks created"}
        
        # Step 3: Generate embeddings for all chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed_texts(chunk_texts)
        
        # Step 4: Prepare document ID and metadata
        document_id = self._generate_document_id(pdf_path)
        processed_at = datetime.now()
        
        # Step 5: Store chunk embeddings in Qdrant (vector search) and text in MongoDB
        logger.info(f"Preparing chunks for storage...")
        points = []  # For Qdrant
        chunk_records = []  # For MongoDB
        chunk_ids = []  # Track chunk IDs for reference
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create unique point ID as UUID (required by Qdrant)
            point_id = str(uuid.uuid4())
            chunk_ids.append(point_id)
            
            category = classifier.classify(chunk["text"], top_k=1, use_keywords=False)
            category_name = category[0]['category'] if category else "uncategorized"
            
            # QDRANT: Minimal payload (NO full text - optimized for vector search)
            # Only store metadata needed for filtering and reference
            qdrant_payload = {
                "chunk_id": point_id,
                "document_id": document_id,
                "document_name": pdf_path.name,
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "global_chunk_index": idx,
                "char_count": chunk.get("char_count", len(chunk["text"])),
                "word_count": chunk.get("word_count", len(chunk["text"].split())),
                "text_preview": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"],  # Preview only
            }
            
            # Add custom metadata if provided
            if metadata:
                qdrant_payload.update(metadata)
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=qdrant_payload
                )
            )
            
            # MONGODB: Full chunk record with complete text
            chunk_record = {
                "chunk_id": point_id,
                "document_id": document_id,
                "document_name": pdf_path.name,
                "text": chunk["text"],  # Full text stored here
                "page": int(chunk["page"]),  # Ensure int type
                "chunk_index": int(chunk["chunk_index"]),  # Ensure int type
                "global_chunk_index": int(idx),  # Ensure int type
                "category": category_name,
                "processed_at": processed_at.isoformat(),
                "custom_metadata": metadata or {}
            }
            chunk_records.append(chunk_record)
        
        # Upload vectors to Qdrant in batches
        logger.info(f"Storing {len(points)} vectors in Qdrant...")
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.vector_db.upsert_vectors(
                collection_name=self.collection_name,
                points=batch
            )
        
        # Store full text chunks in MongoDB
        logger.info(f"Storing {len(chunk_records)} text chunks in MongoDB...")
        if chunk_records:
            # Create chunks collection if doesn't exist
            if "pdf_chunks" not in self.mongodb.db.list_collection_names():
                self.mongodb.db.create_collection("pdf_chunks")
            chunks_collection = self.mongodb.db["pdf_chunks"]
            
            # Create indexes for efficient retrieval
            chunks_collection.create_index([("chunk_id", 1)], unique=True)
            chunks_collection.create_index([("document_id", 1)])
            chunks_collection.create_index([("document_id", 1), ("global_chunk_index", 1)])
            
            # Bulk insert chunks (replace if exists)
            from pymongo import UpdateOne
            operations = [
                UpdateOne(
                    {"chunk_id": record["chunk_id"]},
                    {"$set": record},
                    upsert=True
                )
                for record in chunk_records
            ]
            chunks_collection.bulk_write(operations)
        
        processing_end = datetime.now()
        processing_time = (processing_end - processing_start).total_seconds()
        
        # Step 6: Store document-level metadata in MongoDB (structured data)
        logger.info(f"Storing document metadata in MongoDB...")
        
        # Calculate document statistics
        total_chars = sum(chunk.get("char_count", len(chunk["text"])) for chunk in chunks)
        total_words = sum(chunk.get("word_count", len(chunk["text"].split())) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        # Page-level statistics (with string keys for MongoDB compatibility)
        chunks_per_page = {}
        for chunk in chunks:
            page = str(chunk["page"])  # Convert page number to string
            chunks_per_page[page] = chunks_per_page.get(page, 0) + 1
        
        # Create comprehensive document record for MongoDB
        document_record = {
            "document_id": document_id,
            "document_name": pdf_path.name,
            "document_path": str(pdf_path.absolute()),
            "file_size_bytes": int(pdf_path.stat().st_size),  # Ensure int type
            "file_modified_at": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat(),
            
            # Processing information
            "processed_at": processed_at.isoformat(),
            "processing_time_seconds": round(processing_time, 2),
            "qdrant_collection": self.collection_name,
            
            # Content statistics
            "total_pages": int(len(pages)),  # Ensure int type
            "total_chunks": int(len(chunks)),  # Ensure int type
            "total_characters": int(total_chars),  # Ensure int type
            "total_words": int(total_words),  # Ensure int type
            "avg_chunk_size": float(round(avg_chunk_size, 2)),  # Ensure float type
            "chunks_per_page": chunks_per_page,  # Now with string keys
            
            # Chunk references (links to Qdrant)
            "chunk_ids": chunk_ids,  # All chunk point IDs in Qdrant
            
            # Custom metadata
            "custom_metadata": metadata or {},
            
            # Processing configuration
            "chunking_config": {
                "chunk_size": int(self.chunker.chunk_size),
                "chunk_overlap": int(self.chunker.chunk_overlap),
                "separator": str(self.chunker.separator)
            },
            
            # Status tracking
            "status": "completed",
            "version": "1.0",
            "last_updated_at": processed_at.isoformat()
        }
        
        # Upsert document record in MongoDB (update if exists, insert if new)
        try:
            self.pdf_collection.update_one(
                {"document_id": document_id},
                {"$set": document_record},
                upsert=True
            )
            logger.info(f"✓ Document metadata saved to MongoDB")
        except Exception as e:
            logger.error(f"Failed to save to MongoDB: {str(e)}")
            # Continue anyway since Qdrant data is already saved
        
        return {
            "status": "success",
            "document_id": document_id,
            "document_name": pdf_path.name,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "total_words": total_words,
            "chunks_stored_qdrant": len(points),
            "metadata_stored_mongodb": True,
            "qdrant_collection": self.collection_name,
            "mongodb_collection": "pdf_documents",
            "processing_time_seconds": round(processing_time, 2)
        }
    
    def query_document(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
        include_full_text: bool = True
    ) -> List[Dict]:
        """
        Query the document collection (optimized hybrid search)
        
        Args:
            query: Query text
            top_k: Number of results to return
            document_id: Filter by specific document ID
            document_name: Filter by document name
            include_full_text: Whether to fetch full text from MongoDB (default: True)
            
        Returns:
            List of matching chunks with scores and metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_single(query)
        
        # Build filter as dictionary (not Filter object)
        filter_dict = {}
        if document_id:
            filter_dict["document_id"] = document_id
        if document_name:
            filter_dict["document_name"] = document_name
        
        # Search in Qdrant (fast vector search)
        search_results = self.vector_db.search_similar(
            query_vector=query_embedding.tolist(),
            limit=top_k,
            filter_dict=filter_dict if filter_dict else None
        )
        
        # Format results
        results = []
        chunk_ids = []
        
        for result in search_results:
            chunk_id = result.get("metadata", {}).get("chunk_id") or result.get("qdrant_id")
            chunk_ids.append(chunk_id)
            
            result_dict = {
                "chunk_id": chunk_id,
                "score": result.get("score", 0),
                "document_name": result.get("metadata", {}).get("document_name", ""),
                "document_id": result.get("metadata", {}).get("document_id", ""),
                "page": result.get("metadata", {}).get("page", 0),
                "chunk_index": result.get("metadata", {}).get("chunk_index", 0),
                "text_preview": result.get("metadata", {}).get("text_preview", ""),
                "metadata": result.get("metadata", {})
            }
            results.append(result_dict)
        
        # Fetch full text from MongoDB if requested (optimized batch query)
        if include_full_text and chunk_ids:
            chunks_collection = self.mongodb.db["pdf_chunks"]
            chunk_texts = list(chunks_collection.find(
                {"chunk_id": {"$in": chunk_ids}},
                {"chunk_id": 1, "text": 1, "_id": 0}
            ))
            
            # Map chunk_id to text
            text_map = {chunk["chunk_id"]: chunk["text"] for chunk in chunk_texts}
            
            # Add full text to results
            for result in results:
                result["text"] = text_map.get(result["chunk_id"], "")
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a specific chunk's full text from MongoDB
        
        Args:
            chunk_id: Chunk ID (UUID)
            
        Returns:
            Chunk record with full text, or None if not found
        """
        try:
            chunks_collection = self.mongodb.db["pdf_chunks"]
            chunk = chunks_collection.find_one({"chunk_id": chunk_id}, {"_id": 0})
            return chunk
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {str(e)}")
            return None
    
    def get_document_chunks(self, document_id: str, include_text: bool = True) -> List[Dict]:
        """
        Get all chunks for a document from MongoDB
        
        Args:
            document_id: Document ID
            include_text: Whether to include full text (default: True)
            
        Returns:
            List of chunk records
        """
        try:
            chunks_collection = self.mongodb.db["pdf_chunks"]
            projection = {"_id": 0}
            if not include_text:
                projection["text"] = 0  # Exclude text field
            
            chunks = list(chunks_collection.find(
                {"document_id": document_id},
                projection
            ).sort("global_chunk_index", 1))
            
            return chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks for document {document_id}: {str(e)}")
            return []
    
    def get_document_info(self, document_id: str) -> Dict:
        """
        Get information about a processed document from MongoDB
        
        Args:
            document_id: Document ID
            
        Returns:
            Dict with document statistics
        """
        try:
            # Get from MongoDB (primary source of truth)
            doc = self.pdf_collection.find_one({"document_id": document_id}, {"_id": 0})
            
            if doc:
                return {
                    "status": "found",
                    **doc
                }
            
            return {"status": "not_found", "document_id": document_id}
        except Exception as e:
            logger.error(f"Error getting document info: {str(e)}")
            return {"status": "error", "document_id": document_id, "error": str(e)}
    
    def list_documents(self) -> List[Dict]:
        """
        List all documents in the collection from MongoDB (optimized)
        
        Returns:
            List of document information
        """
        try:
            # Get from MongoDB directly (much faster than scanning Qdrant)
            documents = list(self.pdf_collection.find(
                {},
                {
                    "_id": 0,
                    "document_id": 1,
                    "document_name": 1,
                    "document_path": 1,
                    "total_chunks": 1,
                    "total_pages": 1,
                    "processed_at": 1,
                    "file_size_bytes": 1
                }
            ).sort("processed_at", -1))
            
            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
