from src.database.qdrant_vectors import QdrantVectorDB

if __name__ == "__main__":
    db = QdrantVectorDB(collection_name="pdf_documents")
    if not db.delete_collection():
        raise SystemExit(1)