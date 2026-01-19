"""
Sync categories from JSON to Qdrant
Run this script once to populate the 'categories' collection
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.classifier.query_classifier import QueryClassifier

def main():
    print("Initializing QueryClassifier...")
    classifier = QueryClassifier()
    
    print("\nSyncing categories from JSON to Qdrant...")
    classifier.sync_categories_from_json()
    
    print("\nVerifying categories in Qdrant...")
    categories = classifier.get_categories()
    print(f"Total categories stored: {len(categories)}")
    for i, category in enumerate(categories, 1):
        print(f"  {i}. {category}")
    
    print("\n✓ Categories successfully synced to Qdrant!")
    print("You can now use classifier.classify() to classify queries.")

if __name__ == "__main__":
    main()
