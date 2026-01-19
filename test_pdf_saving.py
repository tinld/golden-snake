"""
Test PDF Document Saving to Qdrant
Comprehensive test for PDF processing, chunking, vectorization, and storage
"""

import logging
from src.database.qdrant_vectors import QdrantVectorDB
from src.media.pdf_processor import PDFDocumentProcessor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_pdf_saving():
    """Test complete PDF to Qdrant pipeline"""

    # Step 1: Initialize processor
    print("\n[1] Initializing PDF processor...")
    processor = PDFDocumentProcessor(
        collection_name="pdf_documents",
        chunk_size=500,
        chunk_overlap=50
    )

    # Step 2: Find the specific PDF file
    print("\n[2] Locating PDF file 'luat_dat_dai.pdf'...")
    media_folder = Path(__file__).parent / "media"
    pdf_path = media_folder / "luat_dat_dai.pdf"

    if not pdf_path.exists():
        print("✗ PDF file 'luat_dat_dai.pdf' not found in media folder")
        return False
    # media_folder = Path(__file__).parent / "media"
    # pdf_files = list(media_folder.glob("*.pdf"))

    # if not pdf_files:
    #     return False

    # pdf_path = pdf_files[0]
    
    print("\n[3] Processing PDF (read → chunk → vectorize → save)...")
    try:
        result = processor.process_pdf(
            pdf_path=str(pdf_path),
            max_pages=10
        )

        if result["status"] != "success":
            print(f"✗ Processing failed: {result.get('message')}")
            return False

    except Exception as e:
        print(f"✗ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_direct_qdrant_access():
    """Test direct access to Qdrant collection"""

    print("\n" + "=" * 80)
    print("DIRECT QDRANT ACCESS TEST")
    print("=" * 80)

    try:
        qdrant = QdrantVectorDB(collection_name="pdf_documents")
        if not qdrant.collection_exists():
            # Create collection if it doesn't exist
            qdrant.create_collection()
        # Get collection info
        if qdrant.collection_exists():
            print("✓ Collection 'pdf_documents' exists")

            count = qdrant.get_count()
            print(f"✓ Total points in collection: {count}")

            if count > 0:
                # Get sample point
                scroll_result = qdrant.client.scroll(
                    collection_name="pdf_documents",
                    limit=1
                )

                if scroll_result and scroll_result[0]:
                    point = scroll_result[0][0]
                    print(f"\n  Sample Point:")
                    print(f"    ID: {point.id}")
                    print(f"    Vector length: {len(point.vector)}")
                    print(f"    Payload keys: {list(point.payload.keys())}")
                    print(f"\n  Payload Preview:")
                    for key, value in point.payload.items():
                        if key == "text":
                            print(f"    {key}: {str(value)[:60]}...")
                        else:
                            print(f"    {key}: {value}")
            else:
                print("ℹ Collection is empty")

    except Exception as e:
        print(f"✗ Error accessing Qdrant: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    print("=" * 80)
    return True


if __name__ == "__main__":
    # Run tests
    success = test_pdf_saving()
