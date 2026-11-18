import inngest

# Initialize the Inngest client for durable background jobs & retries
inngest_client = inngest.Inngest(
    app_id="production-rag-agent"
)
