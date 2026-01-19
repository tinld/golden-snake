"""
Example 1: Basic Query Classification
Demonstrates PhoBERT-based query classification
"""
import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classifier import QueryClassifier


def main():
    print("=" * 60)
    print("Example 1: Vietnamese Query Classification with PhoBERT")
    print("=" * 60)
    
    # Initialize classifier
    print("\n1. Initializing query classifier...")
    classifier = QueryClassifier()
    print(f"   Categories available: {len(classifier.get_categories())}")
    
    # Test queries
    test_queries = [
        "Làm sao sửa lỗi ứng dụng bị crash?",
        "Giá gói dịch vụ hàng tháng là bao nhiêu?",
        "Ứng dụng có hỗ trợ tính năng tìm kiếm nâng cao không?",
        "Tôi quên mật khẩu, làm sao để đặt lại?",
        "Sản phẩm của bạn là gì? Nó làm được gì?",
        "Tại sao tôi lại gặp lỗi 404?"
    ]
    
    # Classify queries
    print("\n2. Classifying queries:\n")
    for query in test_queries:
        results = classifier.classify(query, top_k=2)
        primary = results[0]
        
        print(f"   Query: {query}")
        print(f"   ├─ Primary: {primary['category']} (confidence: {primary['confidence']:.2%})")
        if len(results) > 1:
            secondary = results[1]
            print(f"   └─ Secondary: {secondary['category']} (confidence: {secondary['confidence']:.2%})")
        print()
    
    # Custom categories
    print("3. Adding custom categories...")
    custom_categories = {
        "Khiếu nại & Phàn nàn": {
            "keywords": ["khiếu nại", "phàn nàn", "không hài lòng", "rất tệ", "tồi tệ"],
            "examples": [
                "Tôi rất không hài lòng với dịch vụ này",
                "Khiếu nại về chất lượng sản phẩm"
            ]
        }
    }
    
    # classifier.add_categories(custom_categories)
    
    complaint_query = "Tôi muốn khiếu nại về chất lượng dịch vụ!"
    result = classifier.classify(complaint_query, top_k=1)
    print(f"   Query: {complaint_query}")
    print(f"   Classified as: {result[0]['category']}\n")
    
    print("=" * 60)
    print("Example 1 completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
