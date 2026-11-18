import os
from dotenv import load_dotenv
from google import genai
from backend.rag_pipeline import search_relevant_chunks

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

SYSTEM_PROMPT = """You are an accurate, production-grade AI assistant.
Answer the user's question using ONLY the provided context snippets from the uploaded documents.
If the context does not contain the answer, politely state: "I cannot find this information in the uploaded document."
Always cite the source document name and page number when quoting or referring to facts."""

def ask_rag_agent(query: str) -> dict:
    """Answers query using retrieved context."""
    if not client:
        return {
            "answer": "Error: GEMINI_API_KEY environment variable is missing. Please set it in your .env file.",
            "sources": []
        }

    # 1. Retrieve top chunks
    chunks = search_relevant_chunks(query, top_k=4)

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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=dict(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )

    return {
        "answer": response.text,
        "sources": [f"{c['source']} (Page {c['page']})" for c in chunks]
    }
