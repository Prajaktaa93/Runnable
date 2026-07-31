# qdrant db setup and creating collection
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os 

def init_qdrant():
    print("Connecting to Qdrant Database...")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        client = QdrantClient(url=qdrant_url)
    
    collection_name = "running_knowledge"
    
    # 2. Check if our collection already exists
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if exists:
        print(f"Collection '{collection_name}' already exists. Setup skipped.")
    else:
        print(f"Creating collection '{collection_name}'...")
        # 3. Create collection with vector config
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,  # 384 dimensions is the size for BAAI/bge-small-en-v1.5 embeddings
                distance=Distance.COSINE  # Cosine similarity is standard for text search
            )
        )
        print(f"Collection '{collection_name}' successfully created!")
        
    return client

if __name__ == "__main__":
    init_qdrant()
