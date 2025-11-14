# LangChain FastAPI Chat Agent

Small LangChain-based chat agent with Pinecone vector store and FastAPI backend. Includes ingestion helper, prompt templates, and utilities for local development and Docker deployment.

## Features
- FastAPI server exposing a POST /ask endpoint
- Pinecone vector index integration for retrieval
- OpenAI (via langchain_openai) chat model for generation
- Ingest script to upsert documents into Pinecone
- Debug-friendly logging and VS Code launch configs
- Optional LangSmith tracing for model & agent observability

## Prerequisites
- Python 3.9+ installed
- Docker (optional for containerized runs)
- Pinecone account + API key
- OpenAI API key (or compatible model endpoint)

Environment variables (set before running)
- OPENAI_API_KEY — OpenAI API key
- PINECONE_API_KEY — Pinecone API key
- PINECONE_ENVIRONMENT — Pinecone environment (e.g. `us-west1-gcp`)
- PINECONE_INDEX — index name (default: `patient-record`)
- MODEL_NAME — model id (default: `gpt-4o-mini`)

Do NOT commit API keys to Git.

## LangSmith tracing (optional)
LangSmith provides traces and visualizations for LangChain runs (models, prompts, tools, agents).

- Install the LangSmith client (if not present):
```powershell
pip install langsmith
```

- Set LangSmith environment variables:
```powershell
$env:LANGSMITH_API_KEY = "ls_..."
$env:LANGSMITH_PROJECT = "my-project"
$env:LANGSMITH_TRACING = "true"       # enable tracing in LangChain if supported
$env:LANGSMITH_ENDPOINT = "https://api.langsmith.com"  # optional custom endpoint
```

- Enable tracing in your app (example pattern):
```python
import os
from langchain import tracing

if os.getenv("LANGSMITH_TRACING", "false").lower() in ("1","true","yes"):
    tracing.configure(project=os.getenv("LANGSMITH_PROJECT"), api_key=os.getenv("LANGSMITH_API_KEY"))
```

- After enabling, run your app and inspect traces at the LangSmith web UI. Use trace ids logged by the app to find runs.

## Quick local setup (recommended)
PowerShell:
```powershell
cd D:\ChatAgent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
# set env vars for this session
$env:OPENAI_API_KEY="sk_..."
$env:PINECONE_API_KEY="pc_..."
$env:PINECONE_ENVIRONMENT="us-west1-gcp"
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

If your FastAPI app is in a different file (e.g. `app_server.py`), replace `server:app` accordingly.

## Docker
Build image:
```powershell
cd D:\ChatAgent
docker build -t chatagentimg .
```
Run (pass secrets via env or a .env file):
```powershell
docker run --rm -p 8000:8000 -e OPENAI_API_KEY="sk_..." -e PINECONE_API_KEY="pc_..." chatagentimg
```

Common Docker error: `open //./pipe/dockerDesktopLinuxEngine` — start Docker Desktop or switch context.

## API
POST /ask  
Request JSON:
```json
{
  "patient_id": "patient123",
  "question": "Summarize medications",
  "user_id": "doctor1"
}
```
Example curl:
```powershell
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"patient_id\":\"p1\",\"question\":\"What meds?\",\"user_id\":\"doc1\"}"
```

Response: JSON with `trace_id` and `answer` (AI content only).

## Ingesting data
See `ingest_script.py` — ensure PINECONE_API_KEY set, then run:
```powershell
python ingest_script.py
```

## Debugging tips
- Use `python -m pip` if `pip` launcher is broken.
- Activate virtualenv before installs/runs.
- Check logs printed by FastAPI / uvicorn for trace_ids.
- Use VS Code launch configurations (provided) to run & debug.
- Ensure `clinical_prompt.txt` exists in working dir; server expects valid JSON output from model per schema.

## Security
- Remove any hard-coded API keys in source; rotate compromised keys immediately.
- Use secrets manager for production.

## Notes
- LangChain APIs change frequently; watch deprecation warnings and update imports (e.g., langchain_openai).
- LangSmith integration is optional but useful for debugging/observability.
- If you need help adapting to a different LLM backend or Pinecone SDK version, ask for guidance.

License: MIT (or choose appropriate license)