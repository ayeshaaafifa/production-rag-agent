import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from backend.rag_pipeline import search_relevant_chunks

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# Pehla model busy ho to agla try hoga
MODELS = ["gemini-3.8-flash", "gemini-3.5-flash"]

SYSTEM_PROMPT = """You are an accurate, production-grade AI assistant.
Answer the user's question using ONLY the provided context snippets from the uploaded documents.
If the context does not contain the answer, politely state: "I cannot find this information in the uploaded document."
Always cite the source document name and page number when quoting or referring to facts."""


def generate_with_fallback(prompt: str):
    """Retries on 503 (high demand) and falls back to another model."""
    last_err = None
    for model in MODELS:
        for attempt in range(3):
            try:
                return client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=dict(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.2,
                    ),
                )
            except errors.ServerError as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
    raise last_err


def ask_rag_agent(query: str) -> dict:
    """Answers query using retrieved context."""
    if not client:
        return {
            "answer": "Error: GEMINI_API_KEY environment variable is missing. Please set it in your .env file.",
            "sources": []
        }

    # 1. Retrieve top chunks
    chunks = search_relevant_chunks(query, top_k=3)

    if not chunks:
        return {
            "answer": "No indexed documents found or no relevant matches. Please upload and index a PDF first.",
            "sources": []
        }

    # 2. Build context text
    context_sections = []
    for c in chunks:
        context_sections.append(
            f"[Source: {c['source']}, Page: {c['page']}]\n{c['text']}"
        )
    formatted_context = "\n\n---\n\n".join(context_sections)

    prompt = f"""Context:
{formatted_context}

User Question: {query}
"""

    # 3. Generate answer (with retry and fallback)
    try:
        response = generate_with_fallback(prompt)
        answer = response.text
    except errors.ServerError:
        answer = "Gemini is very busy right now. Please try again in a minute."

    return {
        "answer": answer,
        "sources": [f"{c['source']} (Page {c['page']})" for c in chunks]
    }