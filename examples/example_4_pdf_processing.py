"""
Example: PDF Document Processing
Demonstrates how to ingest and query PDF documents
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.media.pdf_processor import PDFDocumentProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate PDF processing"""
    
    # Initialize PDF processor
    processor = PDFDocumentProcessor(
        collection_name="pdf_documents",
        chunk_size=500,  # characters per chunk
        chunk_overlap=50  # overlap between chunks
    )
    
    # Path to PDF file in media folder
    pdf_path = Path(__file__).parent.parent / "media" / "vpcp_1.pdf"
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    # =========================================================================
    # STEP 1: Process the PDF document
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Processing PDF document")
    logger.info("=" * 60)
    
    result = processor.process_pdf(
        pdf_path=str(pdf_path),
        max_pages=None,  # Process all pages (set to number to limit)
        metadata={
            "category": "business",
            "language": "vietnamese",
            "source": "media_folder"
        }
    )
    
    logger.info(f"Processing result: {result}")
    
    if result["status"] == "success":
        document_id = result["document_id"]
        logger.info(f"✓ Document processed successfully!")
        logger.info(f"  - Document ID: {document_id}")
        logger.info(f"  - Total pages: {result['total_pages']}")
        logger.info(f"  - Total chunks: {result['total_chunks']}")
        logger.info(f"  - Collection: {result['collection']}")
    else:
        logger.error(f"✗ Processing failed: {result.get('message')}")
        return
    
    # =========================================================================
    # STEP 2: Query the document
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Querying document")
    logger.info("=" * 60)
    
    # Example queries
    queries = [
        "thông tin về công ty",
        "quy trình làm việc",
        "chính sách và quy định"
    ]
    
    for query in queries:
        logger.info(f"\nQuery: '{query}'")
        logger.info("-" * 60)
        
        results = processor.query_document(
            query=query,
            top_k=3,
            document_id=document_id  # Search only in this document
        )
        
        if results:
            for i, result in enumerate(results, 1):
                logger.info(f"\nResult {i}:")
                logger.info(f"  Score: {result['score']:.4f}")
                logger.info(f"  Page: {result['page']}")
                logger.info(f"  Chunk: {result['chunk_index']}")
                logger.info(f"  Text preview: {result['text'][:150]}...")
        else:
            logger.info("  No results found")
    
    # =========================================================================
    # STEP 3: Get document information
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Document information")
    logger.info("=" * 60)
    
    doc_info = processor.get_document_info(document_id)
    logger.info(f"Document info: {doc_info}")
    
    # =========================================================================
    # STEP 4: List all documents
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: List all documents in collection")
    logger.info("=" * 60)
    
    all_docs = processor.list_documents()
    logger.info(f"Total documents in collection: {len(all_docs)}")
    
    for doc in all_docs:
        logger.info(f"\n  - {doc['document_name']}")
        logger.info(f"    ID: {doc['document_id']}")
        logger.info(f"    Chunks: {doc['total_chunks']}")
        logger.info(f"    Processed: {doc['processed_at']}")
    
    # =========================================================================
    # STEP 5: Query across all documents (without filter)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Query across all documents")
    logger.info("=" * 60)
    
    global_query = "chính sách công ty"
    logger.info(f"\nGlobal query: '{global_query}'")
    logger.info("-" * 60)
    
    global_results = processor.query_document(
        query=global_query,
        top_k=5  # Get top 5 results from all documents
    )
    
    for i, result in enumerate(global_results, 1):
        logger.info(f"\nResult {i}:")
        logger.info(f"  Document: {result['document_name']}")
        logger.info(f"  Score: {result['score']:.4f}")
        logger.info(f"  Page: {result['page']}")
        logger.info(f"  Text preview: {result['text'][:100]}...")
    
    logger.info("\n" + "=" * 60)
    logger.info("PDF processing example completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
