"""
Test PDF vectorization and storage in Qdrant
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.media.pdf_processor import PDFDocumentProcessor
from src.database import get_qdrant

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_pdf_to_vector():
    """Test complete PDF processing pipeline with optimized hybrid storage"""
    processor = PDFDocumentProcessor(
        collection_name="pdf_documents",
        chunk_size=500,
        chunk_overlap=50
    )
    
    test_queries = [
        "Thông tin về luật đất đai",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 80)

        try:
            results = processor.query_document(
                query=query,
                top_k=3,
                include_full_text=True
            )
            
            if results:
                print(f"   ✓ Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"\n   Result {i}:")
                    print(f"      Chunk ID: {result['chunk_id']}")
                    print(f"      Score: {result['score']:.4f}")
                    print(f"      Document: {result['document_name']}")
                    print(f"      Page: {result['page']}, Chunk Index: {result['chunk_index']}")
                    print(f"      Word Count: {result['metadata'].get('word_count', 'N/A')} words")
                    print(f"      Full Text: {result['text'][:120]}...")
            else:
                print("   No results found")
                
        except Exception as e:
            print(f"   ✗ Query error: {str(e)}")
            import traceback
            traceback.print_exc()


def test_query_without_filter():
    """Test querying across all documents with optimized storage"""    
    processor = PDFDocumentProcessor(collection_name="pdf_documents")
    
    query = "important information"
    try:
        results = processor.query_document(
            query=query,
            top_k=5,
            include_full_text=True  # Fetch full text from MongoDB
        )
        
        if results:
            print(f"✓ Found {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Similarity Score: {result['score']:.4f}")
                print(f"     └─ Chunk ID: {result['chunk_id']}")
                print(f"     └─ Document: {result['document_name']}")
                print(f"     └─ Location: Page {result['page']}, Chunk {result['chunk_index']}")
                print(f"     └─ Word Count: {result['metadata'].get('word_count', 'N/A')} words")
                print(f"     └─ Full Text: {result['text'][:120]}...")
                print()
        else:
            print("❌ No results found")
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def test_retrieve_document_info():
    """Test retrieving document metadata (from MongoDB)"""
    print("\n" + "=" * 80)
    print("TESTING DOCUMENT METADATA RETRIEVAL")
    print("=" * 80)
    print("Retrieving document info from MongoDB (optimized for metadata queries)\n")
    
    processor = PDFDocumentProcessor(collection_name="pdf_documents")
    
    # List all documents
    print("📚 All Processed Documents:")
    documents = processor.list_documents()
    
    if documents:
        for i, doc in enumerate(documents, 1):
            print(f"\n  {i}. {doc['document_name']}")
            print(f"     └─ Document ID: {doc['document_id']}")
            print(f"     └─ Pages: {doc['total_pages']}, Chunks: {doc['total_chunks']}")
            print(f"     └─ File Size: {doc['file_size_bytes'] / 1024:.2f} KB")
            print(f"     └─ Processed: {doc['processed_at']}")
            
            # Get full document info
            doc_info = processor.get_document_info(doc['document_id'])
            if doc_info['status'] == 'found':
                print(f"     └─ Processing Time: {doc_info['processing_time_seconds']}s")
                print(f"     └─ Total Words: {doc_info['total_words']}")
                print(f"     └─ Chunks per Page: {doc_info['chunks_per_page']}")
    else:
        print("  No documents found")


def test_retrieve_specific_chunk():
    """Test retrieving a specific chunk from MongoDB"""
    processor = PDFDocumentProcessor(collection_name="pdf_documents")
    
    # First, query to get a chunk_id
    print("1️⃣  Searching for a chunk...\n")
    results = processor.query_document(
        query="example query",
        top_k=1,
        include_full_text=False  # Don't fetch yet
    )
    
    if results:
        chunk_id = results[0]['chunk_id']
        print(f"   Found chunk ID: {chunk_id}\n")
        
        # Now retrieve the full chunk
        print(f"2️⃣  Retrieving full chunk by ID...\n")
        chunk = processor.get_chunk_by_id(chunk_id)
        
        if chunk:
            print(f"   ✓ Chunk retrieved from MongoDB:")
            print(f"     └─ ID: {chunk['chunk_id']}")
            print(f"     └─ Document: {chunk['document_name']}")
            print(f"     └─ Page: {chunk['page']}")
            print(f"     └─ Word Count: {chunk['word_count']}")
            print(f"     └─ Full Text: {chunk['text']}")
        else:
            print(f"   ❌ Chunk not found")
    else:
        print("   No results to test with")


if __name__ == "__main__":        
    test_pdf_to_vector()