"""
Test the hybrid MongoDB + Qdrant architecture
"""
import sys
sys.path.insert(0, './src')

from rag.rag_pipeline_hybrid import RAGPipeline

if __name__ == "__main__":
    # Test RAG pipeline with hybrid architecture
    print("="*60)
    print("Testing Hybrid Architecture: MongoDB + Qdrant")
    print("="*60)
    
    pipeline = RAGPipeline()
    
    # Sample queries
    sample_queries = [
        "Làm sao sửa lỗi này?",
        "Giá sản phẩm bao nhiêu?",
        "Tính năng nào được hỗ trợ?"
    ]
    
    # Ingest
    print("\n[1] Ingesting queries...")
    ids = pipeline.ingest_queries(sample_queries)
    print(f"✓ Ingested {len(ids)} queries")
    print(f"  MongoDB IDs: {[id[:8]+'...' for id in ids]}")
    
    # Process query
    print("\n[2] Processing new query...")
    result = pipeline.process_query("Vấn đề kỹ thuật là gì?")
    print(f"✓ Category: {result['primary_category']}")
    print(f"✓ Retrieved: {result['num_results']} similar queries")
    
    # Show retrieved context
    if result['context'] and result['context']['results']:
        print("\n[3] Similar queries found:")
        for i, res in enumerate(result['context']['results'][:3], 1):
            print(f"  {i}. {res['query']}")
            print(f"     Score: {res['similarity_score']:.3f} | Category: {res['category']}")
    
    # Statistics
    print("\n[4] Pipeline Statistics:")
    stats = pipeline.get_statistics()
    print(f"  Total queries (MongoDB): {stats['total_queries']}")
    print(f"  Total sessions (MongoDB): {stats['total_rag_sessions']}")
    print(f"  Total vectors (Qdrant): {stats['total_vectors_qdrant']}")
    print(f"  Total vectors (ChromaDB): {stats['total_vectors_chroma']}")
    print(f"\n  Categories distribution:")
    for cat, count in stats['categories'].items():
        print(f"    - {cat}: {count}")
    
    print("\n" + "="*60)
    print("✓ Hybrid architecture test completed!")
    print("="*60)
