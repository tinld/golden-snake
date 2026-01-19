"""
Quick script to check items in Qdrant collections using REST API
"""
import requests
import json

QDRANT_URL = "http://localhost:6333"

def check_collection(collection_name: str, limit: int = 10):
    """Check items in a collection"""
    print(f"\n{'='*60}")
    print(f"Collection: {collection_name}")
    print('='*60)
    
    try:
        # Get collection info
        info_response = requests.get(f"{QDRANT_URL}/collections/{collection_name}")
        if info_response.status_code == 200:
            info = info_response.json()
            points_count = info.get("result", {}).get("points_count", 0)
            print(f"\nTotal points: {points_count}")
        
        # Scroll through points
        scroll_response = requests.post(
            f"{QDRANT_URL}/collections/{collection_name}/points/scroll",
            json={
                "limit": limit,
                "with_payload": True,
                "with_vector": False
            }
        )
        
        if scroll_response.status_code == 200:
            data = scroll_response.json()
            points = data.get("result", {}).get("points", [])
            
            print(f"\nShowing {len(points)} items:\n")
            for i, point in enumerate(points, 1):
                print(f"{i}. ID: {point.get('id')}")
                payload = point.get('payload', {})
                for key, value in payload.items():
                    print(f"   {key}: {value}")
                print()
        else:
            print(f"Error: {scroll_response.status_code} - {scroll_response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check all collections
    print("Available collections:")
    try:
        response = requests.get(f"{QDRANT_URL}/collections")
        if response.status_code == 200:
            collections = response.json().get("result", {}).get("collections", [])
            for col in collections:
                print(f"  - {col.get('name')}")
    except Exception as e:
        print(f"Error getting collections: {e}")
    
    # Check specific collections
    check_collection("rag_sessions", limit=5)
    check_collection("query_records", limit=5)
