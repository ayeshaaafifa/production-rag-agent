import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import inngest.fast_api
from backend.inngest_client import inngest_client
from backend.rag_pipeline import extract_chunks_from_pdf, embed_and_upsert
from backend.agent import ask_rag_agent

app = FastAPI(title="Production RAG Agent API")

class QueryRequest(BaseModel):
    query: str

class IndexRequest(BaseModel):
    file_path: str

# 1. Inngest Durable Workflow for Background Document Indexing
@inngest_client.create_function(
    fn_id="process-uploaded-pdf",
    trigger=inngest.TriggerEvent(event="rag/pdf.uploaded"),
)
async def process_pdf_workflow(ctx: inngest.Context, step: inngest.Step):
    file_path = ctx.event.data["file_path"]

    # Step 1: Parse and Chunk Document
    chunks = await step.run("extract-chunks", lambda: extract_chunks_from_pdf(file_path))

    # Step 2: Embed and Upsert to Vector Store (auto-retried on rate-limits/failures)
    await step.run("embed-and-upsert", lambda: embed_and_upsert(chunks))

    return {
        "status": "success",
        "file": os.path.basename(file_path),
        "total_chunks": len(chunks)
    }

# 2. Mount Inngest handlers to FastAPI
inngest.fast_api.serve(app, inngest_client, [process_pdf_workflow])

# 3. Direct REST Endpoints (for Frontend or external callers)
@app.get("/")
def health_check():
    return {"status": "ok", "service": "Production RAG Agent"}

@app.post("/api/query")
def query_agent(req: QueryRequest):
    try:
        return ask_rag_agent(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/index")
async def trigger_indexing(req: IndexRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Send event to Inngest
    await inngest_client.send(
        inngest.Event(
            name="rag/pdf.uploaded",
            data={"file_path": req.file_path}
        )
    )
    return {"message": "Indexing job queued in Inngest successfully"}
