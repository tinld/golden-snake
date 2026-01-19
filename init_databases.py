"""
Quick setup script to initialize both databases
"""
import sys
sys.path.insert(0, './src')

from database import init_mongodb, init_qdrant

if __name__ == "__main__":
    print("="*60)
    print("Initializing Hybrid Database Architecture")
    print("="*60)
    
    # Initialize MongoDB
    print("\n[1] Setting up MongoDB...")
    try:
        mongo = init_mongodb()
        print("✓ MongoDB ready!")
    except Exception as e:
        print(f"✗ MongoDB error: {e}")
        print("  Make sure MongoDB is running on localhost:27017")
    
    # Initialize Qdrant
    print("\n[2] Setting up Qdrant...")
    try:
        qdrant = init_qdrant()
        print("✓ Qdrant ready!")
    except Exception as e:
        print(f"✗ Qdrant error: {e}")
        print("  Make sure Qdrant is running on localhost:6333")
    
    print("\n" + "="*60)
    print("Architecture:")
    print("  📦 MongoDB (localhost:27017) - Rich metadata storage")
    print("  🔍 Qdrant (localhost:6333) - Vector embeddings")
    print("="*60)
