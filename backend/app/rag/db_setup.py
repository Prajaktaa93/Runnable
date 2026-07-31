# qdrant db setup and creating collection
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType
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
        try:
            info = client.get_collection(collection_name)
            vectors = info.config.params.vectors
            current_size = getattr(vectors, "size", None)
            # In case it is a dictionary of named vectors (e.g. {"": VectorParams(...)})
            if current_size is None and isinstance(vectors, dict):
                first_vector = list(vectors.values())[0]
                current_size = getattr(first_vector, "size", None)

            if current_size is not None and current_size != 768:
                print(f"Collection size is {current_size}, deleting and recreating with 768 dimensions...")
                client.delete_collection(collection_name)
                exists = False
            else:
                print(f"Collection '{collection_name}' already exists with correct dimensions. Setup skipped.")
        except Exception as e:
            print(f"Error checking collection: {e}. Skipping setup.")
            
    if not exists:
        print(f"Creating collection '{collection_name}'...")
        # 3. Create collection with vector config
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=768,  # 768 dimensions is the size for Gemini text-embedding-004
                distance=Distance.COSINE  # Cosine similarity is standard for text search
            )
        )
        print(f"Collection '{collection_name}' successfully created!")
        
    # Ensure payload indexes exist for metadata filtering (required by Qdrant Cloud)
    for field in ["experience_level", "distance_tier"]:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD
            )
            print(f"Verified or created payload index for '{field}'")
        except Exception as e:
            # Ignore if index is already present
            pass
            
    return client

if __name__ == "__main__":
    init_qdrant()
