const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// TypeScript definition for our chat responses
export interface ChatResponse {
  answer: string;
  citations: string[];
  tokens_used?: number;
}

// TypeScript definition for the backend health check response
export interface HealthResponse {
  status: string;
  message: string;
}

/**
 * Checks if the FastAPI backend is operational
 */
export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/api/health`);
  if (!response.ok) {
    throw new Error("Failed to contact the backend server.");
  }
  return response.json();
}

/**
 * Sends a chat message + runner profile parameters to the RAG endpoint
 */
export async function sendChatMessage(
  message: string,
  experienceLevel: string = "all",
  distanceTier: string = "all",
  age: string = "",
  gender: string = "",
  signal?: AbortSignal
): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: message,
      experience_level: experienceLevel,
      distance_tier: distanceTier,
      age: age,
      gender: gender,
    }),
    signal,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to get response from the running coach.");
  }

  return response.json();
}

// ─── Plan / Checklist Structured Output Types ───

export interface PlanTask {
  label: string;
  type: "run" | "rest" | "strength" | "hydration" | "nutrition" | string;
}

export interface PlanResponse {
  title: string;
  icon: string;
  tasks: PlanTask[];
}

/**
 * Sends a coach response text to the backend which parses it into
 * a structured day-by-day checklist using the Groq LLM.
 */
export async function generatePlan(
  coachResponse: string,
  experienceLevel: string = "all",
  distanceTier: string = "all"
): Promise<PlanResponse> {
  const response = await fetch(`${API_URL}/api/plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      coach_response: coachResponse,
      experience_level: experienceLevel,
      distance_tier: distanceTier,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to generate plan.");
  }

  return response.json();
}
