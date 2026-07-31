''' Steps happening in ingestion process
Data Loading: Gathers raw text, PDFs, web pages, or databases from your storage source.
Text Chunking: Splits large documents into smaller, bite-sized pieces so the system can read and process them easily.
Text Cleaning: Removes extra spaces, weird symbols, and hidden codes to keep the text clean.
Embedding Generation: Passes each text chunk through an AI model to turn words into a list of numbers that capture the true meaning.
Vector Storage: Saves the number lists and original text inside a vector database for future matching. '''

import os
import re
import yaml
from typing import List
from qdrant_client import QdrantClient
from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore


# 1. Custom reader to parse YAML frontmatter from markdown files
def custom_markdown_reader(file_path: str) -> List[Document]:
    """Reads a markdown file, parses its YAML frontmatter as metadata, 
    and returns a LlamaIndex Document with metadata attached and frontmatter removed."""
    print(f"Reading file: {os.path.basename(file_path)}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Match YAML frontmatter blocks enclosed in ---
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    
    metadata = {}
    text_content = content
    
    if frontmatter_match:
        frontmatter_text = frontmatter_match.group(1)
        try:
            metadata = yaml.safe_load(frontmatter_text) or {}
        except Exception as e:
            print(f"  ⚠️ Error parsing frontmatter YAML for {file_path}: {e}")
        
        # Remove frontmatter block from text content so it doesn't get embedded
        text_content = content[frontmatter_match.end():]
        
    # Standardize metadata fields (lowercase strings)
    clean_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, str):
            clean_metadata[k.strip().lower()] = v.strip()
        else:
            clean_metadata[k.strip().lower()] = v

    # Attach file_name as metadata
    clean_metadata["file_name"] = os.path.basename(file_path)
    
    # We want these key fields to be available for metadata filtering
    required_keys = ["category", "distance_tier", "experience_level", "topic", "source"]
    for key in required_keys:
        if key not in clean_metadata:
            clean_metadata[key] = "all"  # Default fallback

    return [Document(text=text_content, metadata=clean_metadata)]

def run_ingestion():
    print("=" * 60)
    print("🚀 STARTING LLAMAINDEX DOCUMENT INGESTION")
    print("=" * 60)
    
    # 2. Configure LlamaIndex Embeddings Model globally
    from dotenv import load_dotenv
    load_dotenv()
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        # Fallback to check if it's set as GOOGLE_API_KEY
        gemini_key = os.getenv("GOOGLE_API_KEY")
        
    if not gemini_key:
        raise ValueError("❌ GEMINI_API_KEY not found in .env! Please set it to proceed with cloud embeddings.")
        
    print("Loading Google Gemini Cloud Embedding Model...")
    embed_model = GeminiEmbedding(
        model_name="models/gemini-embedding-001",
        api_key=gemini_key
    )
    Settings.embed_model = embed_model
    Settings.llm = None  # Ingestion doesn't need an LLM
    
    # 3. Load Documents Manually
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    if not os.path.exists(data_dir):
        print(f"❌ Data directory does not exist: {data_dir}")
        return
        
    print(f"Loading documents manually from: {data_dir}")
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".md"):
            file_path = os.path.join(data_dir, filename)
            try:
                docs = custom_markdown_reader(file_path)
                documents.extend(docs)
            except Exception as e:
                print(f"  ❌ Error loading file {filename}: {e}")
                
    print(f"Successfully loaded {len(documents)} documents.")
    
    # 4. Parse documents into chunks (nodes) using SentenceSplitter
    print("\nChunking documents...")
    # chunk_size of 512 tokens with 50 overlap is standard for BGE
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks (nodes).")
    
    # Inspect a sample chunk
    if nodes:
        sample_node = nodes[0]
        print("\n--- Sample Node Verification ---")
        print(f"Node ID: {sample_node.node_id}")
        print(f"Metadata: {sample_node.metadata}")
        print(f"Content snippet (first 150 chars):\n{sample_node.get_content()[:150]}...")
        print("-" * 32)
        
    # 5. Connect to Qdrant Server
    print("\nConnecting to Qdrant Database...")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        client = QdrantClient(url=qdrant_url)

    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="running_knowledge"
    )
    
    # 6. Build the VectorStoreIndex
    print("Ingesting chunks and generating embeddings in Qdrant...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    print("\n" + "=" * 60)
    print("✅ INGESTION & INDEXING COMPLETED SUCCESSFULLY!")
    print(f"All {len(nodes)} chunks have been embedded and stored in Qdrant.")
    print("=" * 60)

if __name__ == "__main__":
    run_ingestion()
