import os
from typing import Any

import tiktoken
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# System prompt layout for personalized, grounded running coach replies
RAG_SYSTEM_PROMPT = (
    "You are Runnable, a friendly and expert running coach and sports medicine assistant.\n"
    "The user's profile is as follows:\n"
    "- Age: {age}\n"
    "- Gender: {gender}\n"
    "- Experience Level: {experience_level}\n"
    "- Goal Distance: {distance_tier}\n"
    "\n"
    "Context information retrieved from our running knowledge base is below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Guidelines:\n"
    "1. Use the context above as your primary source. Synthesize and build upon it to give a complete, helpful answer. "
    "If the context contains partial information, use it to construct a full answer.\n"
    "2. Only say you lack information if the context is completely unrelated to the user's question.\n"
    "3. Personalize the answer based on the user's age, gender, experience level, and goal distance where relevant.\n"
    "4. Keep the tone encouraging, warm, professional, and practical.\n"
    "5. Safety Guardrail: If the query mentions any active injuries, pain, or symptoms (e.g. shin splints, runner's knee, dizziness, joint pain), "
    "you MUST include a prominent medical disclaimer advising them to consult a medical doctor or sports physician before continuing.\n"
    "6. Citations: At the end of each paragraph or key piece of advice, cite the source file in brackets (e.g. [starting_a_running_routine_16.md]).\n"
    "\n"
    "User Query: {query_str}\n"
    "Answer: "
)

from llama_index.core.llms import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from openai import OpenAI


class GroqLLM(CustomLLM):
    model_name: str = "qwen/qwen3.8-27b"
    temperature: float = 0.0
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=131072,
            num_output=2048,
            is_chat_model=True,
            model_name="qwen/qwen3.8-27b"
        )
        
    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=2048
        )
        text_content = response.choices[0].message.content or ""
        return CompletionResponse(text=text_content)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        response = self.complete(prompt, formatted=formatted, **kwargs)
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
    
    # 4. Construct Metadata Filters — only apply if user has filled in their profile
    exp_level = user_profile.get("experience_level", "").strip()
    dist_tier = user_profile.get("distance_tier", "").strip()
    
    # Build filter list only for fields that are actually set by the user
    filter_list = []
    if exp_level and exp_level != "all":
        filter_list.append(
            MetadataFilter(
                key="experience_level",
                value=[exp_level, "all"],
                operator=FilterOperator.ANY
            )
        )
    if dist_tier and dist_tier != "all":
        filter_list.append(
            MetadataFilter(
                key="distance_tier",
                value=[dist_tier, "all"],
                operator=FilterOperator.ANY
            )
        )
    
    # Apply filters only if any were set; otherwise retrieve from entire knowledge base
    retriever_kwargs = {"similarity_top_k": 6}
    if filter_list:
        retriever_kwargs["filters"] = MetadataFilters(filters=filter_list)
    
    # 5. Create retriever and custom prompt templates
    retriever = index.as_retriever(**retriever_kwargs)
    
    # Render system prompt with user profile tags
    formatted_prompt_str = RAG_SYSTEM_PROMPT.format(
        age=user_profile.get("age", "not specified"),
        gender=user_profile.get("gender", "not specified"),
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
        except Exception as e:  # noqa: BLE001
            print(f"❌ Error querying RAG: {e}")
        print("=" * 80)
