# api/main.py
import os
import time
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import requests

# ----- Configuration (env-driven) -----
N8N_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL",
    # Set your default here or require via environment var in production
    "https://pgangwar.app.n8n.cloud/webhook-test/e33d0db3-e6a3-4da4-b04b-064ffe596724",
)
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ----- FastAPI app with CORS -----
app = FastAPI(title="Commerce Concierge API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Payload schema -----
class QueryBody(BaseModel):
    inputText: Optional[str] = None     # user query text
    intent: Optional[str] = None        # "text_rec" | "image_rec" | "general_talk"
    file:   Optional[HttpUrl] = None    # image URL for image_rec

# ----- Health checks -----
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "commerce-concierge", "up": True}

@app.get("/ready")
def ready() -> Dict[str, Any]:
    return {"ok": True, "n8nConfigured": bool(N8N_WEBHOOK_URL), "webhook": N8N_WEBHOOK_URL}

# ----- Main endpoint -----
@app.post("/api/query")
def query(body: QueryBody, request: Request):
    if not N8N_WEBHOOK_URL:
        raise HTTPException(status_code=500, detail="N8N_WEBHOOK_URL is not configured")

    # Infer intent if not provided
    intent = body.intent or ("text_rec" if body.inputText else ("image_rec" if body.file else "general_talk"))

    payload = {
        "inputText": body.inputText,
        "intent": intent,
        "file": str(body.file) if body.file else None,
        # Optional: include client metadata
        "client": {
            "ip": request.client.host if request.client else None,
            "ts": int(time.time())
        }
    }

    try:
        resp = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=REQUEST_TIMEOUT)
        # Raise for HTTP 4xx/5xx from n8n
        resp.raise_for_status()
        # n8n returns JSON — pass it through
        return resp.json()
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Upstream (n8n) request timed out")
    except requests.HTTPError as e:
        # Surface upstream errors for easier debugging
        detail = {"status": resp.status_code, "body": None}
        try:
            detail["body"] = resp.json()
        except Exception:
            detail["body"] = resp.text
        raise HTTPException(status_code=502, detail={"upstream_error": str(e), **detail})
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Upstream (n8n) error: {e}")