import os
import sys
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.agent import ask_rag_agent
from backend.rag_pipeline import extract_chunks_from_pdf, embed_and_upsert

st.set_page_config(page_title="Production RAG AI Agent", page_icon="📄", layout="wide")

st.markdown("""
# 📄 Production-Grade RAG AI Agent
Chat with your documents powered by **Gemini**, **Qdrant Vector Database**, and **Inngest**.
""")

# Sidebar: File Upload & Indexing
with st.sidebar:
    st.header("📂 Document Ingestion")
    st.caption("Upload your PDFs to chunk, embed, and store in the Qdrant vector database.")

    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

    if uploaded_file is not None:
        data_dir = os.path.join(PROJECT_ROOT, "data")
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, uploaded_file.name)

        if st.button("⚡ Index Document into Vector DB"):
            with st.spinner("Extracting text and generating embeddings..."):
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Chunk and upsert
                    chunks = extract_chunks_from_pdf(file_path)
                    embed_and_upsert(chunks)
                    st.success(f"Indexed {len(chunks)} chunks from '{uploaded_file.name}'!")
                except Exception as e:
                    st.error(f"Error indexing document: {e}")

    st.divider()
    st.markdown("""
    **Architecture:**
    - 🧠 **LLM:** Gemini 2.5 Flash
    - ⚡ **Embeddings:** `text-embedding-004`
    - 🗄️ **Vector DB:** Qdrant
    - 🔄 **Orchestrator:** Inngest & FastAPI
    """)

# Chat Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload a PDF on the left and ask me any questions about it."}
    ]

# Render Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 Cited Sources"):
                for src in set(msg["sources"]):
                    st.markdown(f"- {src}")

# Chat Input & Response Generation
if user_prompt := st.chat_input("Ask a question about your uploaded documents..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Assistant message
    with st.chat_message("assistant"):
        with st.spinner("Searching vector index & synthesizing response..."):
            res = ask_rag_agent(user_prompt)
            answer = res.get("answer", "No response.")
            sources = res.get("sources", [])

            st.markdown(answer)
            if sources:
                with st.expander("🔍 Cited Sources"):
                    for src in set(sources):
                        st.markdown(f"- {src}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
