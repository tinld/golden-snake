"""
Example 3: Complete RAG Pipeline
Demonstrates the full RAG workflow with classification, storage, and retrieval
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import RAGPipeline
from src.database import init_mongodb, init_qdrant


def main():
    print("=" * 60)
    print("Example 3: Complete RAG Pipeline")
    print("=" * 60)
    
    # Initialize database
    print("\n1. Initializing database...")
    print("   Database initialized")
    
    # Initialize RAG pipeline
    print("\n2. Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    print("   RAG pipeline ready")
    
    # Ingest training queries
    print("\n3. Ingesting training queries...")
    training_queries = [
        "Thông tư số 15 quy định về gì?",
        "Nghị định 99/2023 có hiệu lực từ khi nào?",
        "Luật lao động sửa đổi có điều khoản gì mới?",
        "Thủ tục đăng ký kinh doanh cần những giấy tờ gì?",
        "Chính sách thuế GTGT mới nhất là gì?",
        "Làm sao xin giấy phép kinh doanh?",
        "Thuế thu nhập cá nhân tính như thế nào?",
        "Quyền của công dân được quy định ở đâu?",
        "Thời hạn giải quyết khiếu nại là bao lâu?",
        "Tra cứu văn bản pháp luật ở đâu?"
    ]
    
    ingested_ids = pipeline.ingest_queries(training_queries)
    print(f"   Ingested {len(ingested_ids)} queries")
    
    # Process new queries with RAG
    print("\n4. Processing new queries with RAG...\n")
    
    new_queries = [
        "Thông tư 20 về thuế có nội dung gì?",
        "Nghị định về bảo hiểm xã hội quy định thế nào?",
        "Làm sao tra cứu văn bản luật lao động?",
        "Thủ tục xin cấp giấy phép xuất nhập khẩu như thế nào?",
        "Chính sách miễn thuế cho doanh nghiệp nhỏ ra sao?"
    ]
    
    for query in new_queries:
        print(f"   Processing: {query}")
        result = pipeline.process_query(query, retrieve_context=True, top_k=3)
        
        # Classification
        classification = result["classifications"][0]
        print(f"   ├─ Category: {classification['category']}")
        print(f"   ├─ Confidence: {classification['confidence']:.2%}")
        
        # Retrieved context
        if result["context"]:
            print(f"   ├─ Retrieved {result['num_results']} similar queries:")
            for i, ctx in enumerate(result["context"]["results"], 1):
                ctx_text = ctx['query']
                similarity = ctx["similarity_score"]
                print(f"   │  {i}. \"{ctx_text}\" (similarity: {similarity:.4f})")
        
        # Save session
        response = f"{classification['category']}"
        session_id = pipeline.save_session(
            query,
            response,
            num_retrieved=result["num_results"]
        )
        print(f"   └─ Session ID: {session_id}\n")


if __name__ == "__main__":
    main()
