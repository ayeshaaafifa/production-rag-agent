# Production-Grade RAG AI Agent

Inspired by Tech With Tim's tutorial: **How to Build a Production-Ready RAG AI Agent in Python**.

This application implements a complete, reliable Retrieval-Augmented Generation (RAG) system with:
- **Durable Ingestion**: Background tasks handled by **Inngest** with automatic retries and step tracing.
- **Fast Vector Retrieval**: **Qdrant** for local persistent cosine similarity vector search.
- **Advanced LLM & Embeddings**: Google's `gemini-2.5-flash` and `text-embedding-004`.
- **Interactive UI**: **Streamlit** chat interface with source citations.

---

## 🚀 Quick Setup Guide

### 1. Set Up Environment & Install Dependencies
Open PowerShell or your terminal in this directory:
```powershell
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Your API Key
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Open `.env` and add your **GEMINI_API_KEY** (from [Google AI Studio](https://aistudio.google.com/app/api-keys)).

---

## 🏃 Running the Application

### Option A: Direct Streamlit UI (Fastest)
You can directly run the Streamlit frontend which has built-in embedding and querying:
```powershell
streamlit run frontend/app.py
```
Open `http://localhost:8501` in your browser.

---

### Option B: Full Production Setup with Inngest (Durable Background Jobs)

1. **Start the Inngest Dev Server** (in a separate terminal):
   ```powershell
   npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest
   ```
   *Dashboard opens at `http://localhost:8288` to inspect workflows and step traces.*

2. **Start the FastAPI Backend**:
   ```powershell
   uvicorn backend.main:app --reload --port 8000
   ```

3. **Start the Streamlit UI**:
   ```powershell
   streamlit run frontend/app.py
   ```
