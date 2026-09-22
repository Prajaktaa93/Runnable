import os

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def test_retrieval():
    load_dotenv()
    print("=" * 60)
    print("🔍 TESTING METADATA-FILTERED VECTOR RETRIEVAL")
    print("=" * 60)
    
    # 1. Setup embedding model
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        gemini_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        raise ValueError("❌ GEMINI_API_KEY not found in .env! Please set it to proceed.")
        
    embed_model = GeminiEmbedding(
        model_name="models/gemini-embedding-001",
        api_key=gemini_key
    )
    Settings.embed_model = embed_model
    Settings.llm = None
    
    # 2. Connect to Qdrant Server
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        client = QdrantClient(url=qdrant_url)

    vector_store = QdrantVectorStore(client=client, collection_name="running_knowledge")
    
    # 3. Load the index
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # 4. Define Mock User Profile
    mock_profile = {
        "experience_level": "beginner",
        "distance_tier": "marathon",
    }
    print(f"Mock User Profile: {mock_profile}")
    
    # 5. Define Query
    query = "What running injuries should I watch out for?"
    print(f"Query: '{query}'")
    
    # 6. Test 1: Retrieval WITHOUT Filters
    print("\n--- Test 1: Retrieval WITHOUT Filters ---")
    retriever_no_filter = index.as_retriever(similarity_top_k=3)
    nodes_no_filter = retriever_no_filter.retrieve(query)
    print(f"Retrieved {len(nodes_no_filter)} chunks:")
    for n in nodes_no_filter:
        print(f" - {n.metadata.get('file_name')} (Score: {n.score:.4f}) | Metadata: {n.metadata}")

    # 7. Test 2: Retrieval with EXACT MATCH Filter (experience_level == 'all')
    print("\n--- Test 2: Exact Match Filter (experience_level == 'all') ---")
    filters_exact = MetadataFilters(
        filters=[MetadataFilter(key="experience_level", value="all")]
    )
    retriever_exact = index.as_retriever(similarity_top_k=3, filters=filters_exact)
    nodes_exact = retriever_exact.retrieve(query)
    print(f"Retrieved {len(nodes_exact)} chunks:")
    for n in nodes_exact:
        print(f" - {n.metadata.get('file_name')} (Score: {n.score:.4f}) | Metadata: {n.metadata}")

    # 8. Test 3: Retrieval with ANY Filter (experience_level ANY ['beginner', 'all'])
    print("\n--- Test 3: ANY Filter (experience_level ANY ['beginner', 'all']) ---")
    filters_any = MetadataFilters(
        filters=[
            MetadataFilter(
                key="experience_level",
                value=["beginner", "all"],
                operator=FilterOperator.ANY
            )
        ]
    )
    retriever_any = index.as_retriever(similarity_top_k=3, filters=filters_any)
    nodes_any = retriever_any.retrieve(query)
    print(f"Retrieved {len(nodes_any)} chunks:")
    for n in nodes_any:
        print(f" - {n.metadata.get('file_name')} (Score: {n.score:.4f}) | Metadata: {n.metadata}")
    
    # 9. Test 4: Dual ANY Filters (experience_level ANY [user, all] AND distance_tier ANY [user, all])
    print("\n--- Test 4: Dual ANY Filters (experience_level AND distance_tier) ---")
    filters_dual = MetadataFilters(
        filters=[
            MetadataFilter(
                key="experience_level",
                value=[mock_profile["experience_level"], "all"],
                operator=FilterOperator.ANY
            ),
            MetadataFilter(
                key="distance_tier",
                value=[mock_profile["distance_tier"], "all"],
                operator=FilterOperator.ANY
            )
        ]
    )
    retriever_dual = index.as_retriever(similarity_top_k=3, filters=filters_dual)
    nodes_dual = retriever_dual.retrieve(query)
    print(f"Retrieved {len(nodes_dual)} chunks:")
    for n in nodes_dual:
        print(f" - {n.metadata.get('file_name')} (Score: {n.score:.4f}) | Metadata: {n.metadata}")

    print("=" * 60)

if __name__ == "__main__":
    test_retrieval()
