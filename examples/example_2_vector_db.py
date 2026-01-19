"""
Example 2: Local Vector Database Operations
Demonstrates storing and retrieving embeddings with Chroma
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import get_embedder
from src.vector_db import LocalVectorDB
import numpy as np


def main():
    print("=" * 60)
    print("Example 2: Local Vector Database with Chroma")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing embedder and vector DB...")
    embedder = get_embedder()
    vector_db = LocalVectorDB()
    
    # Sample documents
    documents = [
        "Làm sao sửa lỗi ứng dụng bị crash?",
        "Giá gói dịch vụ hàng tháng là bao nhiêu?",
        "Ứng dụng có hỗ trợ tính năng tìm kiếm nâng cao không?",
        "Tôi quên mật khẩu, làm sao để đặt lại?",
        "Sản phẩm của bạn là gì? Nó làm được gì?",
        "API của bạn hỗ trợ những gì?",
        "Làm sao để nâng cấp gói dịch vụ?",
        "Các phương thức thanh toán là gì?"
    ]
    
    # Generate embeddings
    print(f"\n2. Generating embeddings for {len(documents)} documents...")
    embeddings = embedder.embed_texts(documents)
    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    
    # Add to vector DB
    print("\n3. Adding documents to vector DB...")
    metadatas = [
        {"category": "Kỹ thuật", "type": "vấn đề"},
        {"category": "Giá cả", "type": "chi phí"},
        {"category": "Tính năng", "type": "khả năng"},
        {"category": "Tài khoản", "type": "bảo mật"},
        {"category": "Chung", "type": "thông tin"},
        {"category": "Kỹ thuật", "type": "tích hợp"},
        {"category": "Thanh toán", "type": "gói dịch vụ"},
        {"category": "Thanh toán", "type": "phương thức"}
    ]
    
    ids = vector_db.add_embeddings(embeddings, documents, metadatas=metadatas)
    print(f"   Added {len(ids)} documents")
    print(f"   Total documents in DB: {vector_db.count()}")
    
    # Perform searches
    print("\n4. Performing semantic searches...\n")
    
    search_queries = [
        "Lỗi ứng dụng không hoạt động",
        "Chi phí dịch vụ hàng năm",
        "Làm sao trả tiền?"
    ]
    
    for search_query in search_queries:
        print(f"   Query: {search_query}")
        search_embedding = embedder.embed_single(search_query)
        results = vector_db.search_by_text(search_query, search_embedding, n_results=3)
        
        for i, result in enumerate(results["results"], 1):
            print(f"   {i}. \"{result['text']}\"")
            print(f"      Similarity: {result['similarity']:.4f}")
            print(f"      Category: {result['metadata'].get('category', 'N/A')}")
        print()
    
    # Update and delete operations
    print("5. Update and delete operations...")
    
    # Update first document
    if ids:
        updated_text = "Làm sao khắc phục sự cố ứng dụng?"
        updated_embedding = embedder.embed_single(updated_text)
        vector_db.update_embeddings(
            [ids[0]],
            updated_embedding.reshape(1, -1),
            texts=[updated_text]
        )
        print(f"   Updated document {ids[0]}")
    
    # Get by ID
    if ids:
        doc = vector_db.get_by_id(ids[0])
        if doc:
            print(f"   Retrieved: {doc['text']}")
    
    print(f"\n   Current DB size: {vector_db.count()} documents")
    
    print("\n" + "=" * 60)
    print("Example 2 completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
