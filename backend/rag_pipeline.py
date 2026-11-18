import os
import uuid
from dotenv import load_dotenv
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from google import genai
from google.genai import types

load_dotenv()

# Setup Google GenAI Client
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# Qdrant: server mode in Docker, local folder otherwise
QDRANT_URL = os.getenv("QDRANT_URL")
if QDRANT_URL:
    qdrant = QdrantClient(url=QDRANT_URL)
else:
    QDRANT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qdrant_db")
    qdrant = QdrantClient(path=QDRANT_PATH)

COLLECTION_NAME = "documents"
EMBEDDING_DIM = 768
EMBED_MODEL = "gemini-embedding-001"


def embed_text(text: str, task_type: str) -> list[float]:
    res = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return res.embeddings[0].values


def init_collection():
    """Initializes Qdrant collection if not already created."""
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def extract_chunks_from_pdf(file_path: str, chunk_size: int = 400, overlap: int = 50) -> list[dict]:
    """Reads PDF and returns text chunks with metadata."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    reader = PdfReader(file_path)
    all_chunks = []
    chunk_id = 0

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        words = text.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            all_chunks.append({
                "chunk_id": chunk_id,
                "page": page_num + 1,
                "text": chunk_text,
                "source": os.path.basename(file_path),
            })
            chunk_id += 1
            if end == len(words):
                break
            start += chunk_size - overlap

    return all_chunks


def embed_and_upsert(chunks: list[dict]):
    """Embeds text chunks and upserts them into Qdrant."""
    if not client:
        raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")

    init_collection()
    points = []

    for item in chunks:
        vector = embed_text(item["text"], "RETRIEVAL_DOCUMENT")
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{item['source']}-{item['chunk_id']}")),
                vector=vector,
                payload={
                    "text": item["text"],
                    "page": item["page"],
                    "source": item["source"],
                },
            )
        )

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def search_relevant_chunks(query: str, top_k: int = 3) -> list[dict]:
    """Retrieves top_k relevant chunks for a user query."""
    if not client:
        raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")

    init_collection()
    query_vector = embed_text(query, "RETRIEVAL_QUERY")

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    return [
        {
            "text": hit.payload.get("text", ""),
            "page": hit.payload.get("page", 1),
            "source": hit.payload.get("source", "unknown"),
            "score": hit.score,
        }
        for hit in results
    ]