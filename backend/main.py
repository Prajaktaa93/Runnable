from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
from fastapi import HTTPException
from app.rag.query import query_rag


# Schema for the incoming request from the Next.js frontend
class ChatRequest(BaseModel):
    message: str
    experience_level: str = "all"
    distance_tier: str = "all"

# Schema for the response we return to the frontend
class ChatResponse(BaseModel):
    answer: str
    citations: List[str]

# 1. Create a FastAPI app instance
app = FastAPI(title="RUN agent!")

# 2. Configure CORS (Cross-Origin Resource Sharing)
# This is crucial so our Next.js frontend (running on port 3000)
# is allowed to talk to our FastAPI backend (running on port 8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, we will specify our Next.js domain
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows all headers
)

# 3. Create a simple endpoint at "/api/health"
@app.get("/api/health")
def health_check():
    return {"status": "healthy", "message": "Runner RAG API is operational"}

@app.get("/api/debug")
def debug_check():
    import os
    groq = os.getenv("GROQ_API_KEY", "")
    gemini = os.getenv("GEMINI_API_KEY", "")
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_key = os.getenv("QDRANT_API_KEY", "")
    
    return {
        "groq_key_exists": len(groq) > 0,
        "groq_key_len": len(groq),
        "groq_key_val_hint": groq[:5] + "..." + groq[-5:] if len(groq) > 10 else groq,
        "gemini_key_exists": len(gemini) > 0,
        "gemini_key_len": len(gemini),
        "gemini_key_val_hint": gemini[:5] + "..." + gemini[-5:] if len(gemini) > 10 else gemini,
        "qdrant_url_exists": len(qdrant_url) > 0,
        "qdrant_url": qdrant_url[:15] + "..." if qdrant_url else "",
        "qdrant_key_exists": len(qdrant_key) > 0,
        "qdrant_key_len": len(qdrant_key),
        "qdrant_key_val_hint": qdrant_key[:5] + "..." + qdrant_key[-5:] if len(qdrant_key) > 10 else qdrant_key,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    # 1. Bind user parameters into a profile dictionary
    user_profile = {
        "experience_level": request.experience_level,
        "distance_tier": request.distance_tier
    }
    
    try:
        # 2. Call our RAG query engine
        result = query_rag(request.message, user_profile)
        
        # 3. Return the response structured as a ChatResponse
        return ChatResponse(
            answer=result["answer"],
            citations=result["citations"]
        )
    except Exception as e:
        # If something goes wrong, return a HTTP 500 Internal Server Error
        raise HTTPException(status_code=500, detail=str(e))


# ─── /api/plan: Parse a coach response into a structured checklist ───

class PlanRequest(BaseModel):
    coach_response: str       # The raw coach message text
    experience_level: str = "all"
    distance_tier: str = "all"

class PlanTask(BaseModel):
    label: str                # e.g. "Mon — Easy 2 mile run"
    type: str = "run"         # "run" | "rest" | "strength" | "hydration" | "nutrition"

class PlanResponse(BaseModel):
    title: str                # e.g. "10K Weekly Training Plan"
    icon: str                 # emoji icon: 🏃 💧 ⚡ 🥗
    tasks: List[PlanTask]

@app.post("/api/plan", response_model=PlanResponse)
def plan_endpoint(request: PlanRequest):
    """
    Takes a coach's text response and uses the Groq LLM to extract
    a structured checklist of trackable tasks (day-by-day or step-by-step).
    """
    import os, json, re
    from openai import OpenAI

    groq_client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = f"""You are a structured data extractor for a running coaching app.

A running coach gave the following response to a user:

---
{request.coach_response}
---

Your task is to extract a structured checklist from this response.
Return ONLY valid JSON (no markdown, no explanation) in exactly this format:
{{
  "title": "Short descriptive plan title (max 5 words)",
  "icon": "single emoji matching the plan type (🏃 for running/training, 💧 for hydration, ⚡ for strength, 🥗 for nutrition)",
  "tasks": [
    {{"label": "Short task description (max 8 words)", "type": "run|rest|strength|hydration|nutrition"}},
    ...
  ]
}}

Rules:
- Extract 4 to 7 tasks maximum.
- Each task label must be SHORT and ACTIONABLE (e.g. "Mon — Easy 2mi run", "Drink 500ml pre-run").
- If the response is not about a plan/schedule/routine, return: {{"title": "General Checklist", "icon": "✅", "tasks": [{{"label": "Review coach advice", "type": "rest"}}]}}
- Return ONLY the JSON object. No explanation. No markdown code blocks."""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        raw = completion.choices[0].message.content.strip()

        # Strip any accidental markdown code fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

        data = json.loads(raw)
        tasks = [PlanTask(label=t["label"], type=t.get("type", "run")) for t in data.get("tasks", [])]

        return PlanResponse(
            title=data.get("title", "Training Plan"),
            icon=data.get("icon", "🏃"),
            tasks=tasks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan parsing failed: {str(e)}")

