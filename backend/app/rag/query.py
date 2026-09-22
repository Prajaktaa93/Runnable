import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler


# Load environment variables
load_dotenv()

# System prompt layout for personalized, grounded running coach replies
RAG_SYSTEM_PROMPT = (
    "You are an expert running coach and sports medicine assistant.\n"
    "The user's profile is as follows:\n"
    "- Experience Level: {experience_level}\n"
    "- Goal Distance: {distance_tier}\n"
    "\n"
    "Context information is below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Guidelines:\n"
    "1. Answer the query strictly using the provided context. If the context doesn't contain the answer, say: "
    "'I don't have enough information in my running database to answer this.'\n"
    "2. Personalize the answer to the user's experience level and distance goals where appropriate.\n"
    "3. Keep the tone encouraging, professional, and practical.\n"
    "4. Under NO circumstances should you make up facts or use external knowledge.\n"
    "5. Safety Guardrail: If the query mentions any active injuries, pain, or symptoms (e.g. shin splints, runner's knee, dizziness, joint pain), "
    "you MUST include a prominent medical disclaimer advising them to consult a medical doctor or sports physician before continuing.\n"
    "6. Citations: You MUST cite the source of your information by including bracketed file names of the files containing the info (e.g. [shin_splints_23.md]) "
    "at the end of sentences or paragraphs where that information is used.\n"
    "\n"
    "User Query: {query_str}\n"
    "Answer: "
)

from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms import LLMMetadata, CompletionResponse, CompletionResponseGen
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import Field
from openai import OpenAI

class GroqLLM(CustomLLM):
    model_name: str = "qwen/qwen3.8-27b"
    temperature: float = 0.0
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=131072,
            num_output=1024,
            is_chat_model=True,
            model_name="qwen/qwen3.8-27b"
        )
        
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.metadata.num_output
        )
        return CompletionResponse(text=response.choices[0].message.content)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        response = self.complete(prompt, **kwargs)
        yield response

def setup_rag_components():
    """Configure Embeddings and select LLM (Gemini with Groq fallback)."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        gemini_key = os.getenv("GOOGLE_API_KEY")
        
    if not gemini_key:
        raise ValueError("❌ GEMINI_API_KEY not found in .env! Please set it to proceed with cloud embeddings.")
        
    embed_model = GeminiEmbedding(
        model_name="models/gemini-embedding-001",
        api_key=gemini_key
    )
    Settings.embed_model = embed_model
    
    # 2. Select LLM
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if groq_key and groq_key.strip():
        print("🤖 Using Groq Llama 3 LLM...")
        llm = GroqLLM()
    elif gemini_key and gemini_key.strip():
        print("🤖 Using Google Gemini Flash LLM (Fallback)...")
        from llama_index.llms.gemini import Gemini
        llm = Gemini(model_name="models/gemini-flash-latest", api_key=gemini_key)
    else:
        raise ValueError("❌ No API Keys found! Please set GEMINI_API_KEY or GROQ_API_KEY in .env.")
        
    Settings.llm = llm

def query_rag(user_query: str, user_profile: dict) -> dict:
    """Runs a personalized query against the RAG knowledge base using profile filters."""
    # Initialize token counter at the very beginning so all components inherit it
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )
    Settings.callback_manager = CallbackManager([token_counter])

    # 1. Setup embedding and LLM
    setup_rag_components()
    
# 2. Connect to Qdrant Server
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
    
    # 3. Load Vector Store Index
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # 4. Construct Metadata Filters using Qdrant-supported ANY operator
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="experience_level",
                value=[user_profile.get("experience_level", "all"), "all"],
                operator=FilterOperator.ANY
            ),
            MetadataFilter(
                key="distance_tier",
                value=[user_profile.get("distance_tier", "all"), "all"],
                operator=FilterOperator.ANY
            )
        ]
    )
    
    # 5. Create retriever and custom prompt templates
    retriever = index.as_retriever(
        similarity_top_k=4,  # Retrieve top 4 most relevant chunks
        filters=filters
    )
    
    # Render system prompt with user profile tags
    formatted_prompt_str = RAG_SYSTEM_PROMPT.format(
        experience_level=user_profile.get("experience_level", "all"),
        distance_tier=user_profile.get("distance_tier", "all"),
        context_str="{context_str}",
        query_str="{query_str}"
    )
    
    qa_prompt = PromptTemplate(formatted_prompt_str)
    
    # 6. Build query engine with prompt templates
    response_synthesizer = CompactAndRefine(text_qa_template=qa_prompt)
    
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )
    # 7. Execute Query
    response = query_engine.query(user_query)
    
    # Extract the count
    tokens_used = token_counter.total_llm_token_count
    
    # 8. Extract citations (file names of source documents)
    citations = []
    for node in response.source_nodes:
        file_name = node.metadata.get("file_name")
        if file_name and file_name not in citations:
            citations.append(file_name)
            
    return {
        "answer": str(response),
        "citations": citations,
        "tokens_used": tokens_used,
        "source_nodes": [
            {
                "file_name": node.metadata.get("file_name"),
                "score": node.score,
                "text": node.text
            }
            for node in response.source_nodes
        ]
    }

if __name__ == "__main__":
    # Test execution
    mock_user = {
        "experience_level": "beginner",
        "distance_tier": "marathon"
    }
    
    test_queries = [
        "What are the warning signs of shin splints and how do I prevent them?",
        "What should I consume during a long marathon run?"
    ]
    
    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"❓ Query: {q}")
        print("=" * 80)
        try:
            result = query_rag(q, mock_user)
            print(f"\n💬 Answer:\n{result['answer']}")
            print(f"\n📎 Citations: {result['citations']}")
        except Exception as e:
            print(f"❌ Error querying RAG: {e}")
        print("=" * 80)
