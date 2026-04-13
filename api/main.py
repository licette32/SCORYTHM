"""
api/main.py
===========
FastAPI application for the Scorythm Agent.

Endpoints:
  POST /evaluate  — Evaluates a transaction for fraud
  GET  /health    — Health check

Run with:
    uvicorn api.main:app --reload --port 8000
  or from the fraud-agent/ directory:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
import os

# ─── Path setup: ensure project root is in sys.path ──────────────────────────
# This allows `from model.predict import ...` and `from agent.agent import ...`
# to work regardless of where uvicorn is launched from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import time
import json
import asyncio

from api.schemas import Transaction, EvaluationResult, HealthResponse
from agent.agent import evaluate_transaction, _get_bandit, evaluate_transaction_stream
from agent.explainer import explain_decision
from model.predict import get_model_metrics

# ─── App initialization ───────────────────────────────────────────────────────
app = FastAPI(
    title="Scorythm Agent API",
    description=(
        "An AI agent that detects financial fraud using calibrated uncertainty "
        "from XGBoost and autonomously purchases external signals via x402 "
        "micropayments on Stellar testnet when the model is uncertain."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Allow the frontend (opened as file://) and any localhost origin to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ─── Startup: pre-load the model ──────────────────────────────────────────────
_model_ready = False
_model_metrics: dict = {}


@app.on_event("startup")
async def startup_event():
    """Pre-loads the ML model on startup to avoid cold-start latency."""
    global _model_ready, _model_metrics
    try:
        print("[startup] Loading ML model...")
        _model_metrics = get_model_metrics()
        _model_ready = True
        auc = _model_metrics.get("roc_auc", "N/A")
        print(f"[startup] OK - Model loaded. ROC-AUC = {auc}")
    except Exception as exc:
        print(f"[startup] WARNING - Model not loaded: {exc}")
        print("[startup] The model will be trained on first request.")
        _model_ready = False


# ─── Middleware: request timing ───────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed = (time.monotonic() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    return response


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post(
    "/evaluate",
    response_model=EvaluationResult,
    summary="Evaluate a transaction for fraud",
    description=(
        "Runs the Scorythm Agent on a financial transaction. "
        "If the model is uncertain (prob_fraud between 0.35 and 0.65), "
        "the agent autonomously purchases external signals via x402 micropayments "
        "to reduce uncertainty before making a final decision."
    ),
    tags=["Agent"],
)
async def evaluate(transaction: Transaction) -> EvaluationResult:
    """
    Evaluate a financial transaction for fraud.

    The agent will:
    1. Run the calibrated XGBoost model
    2. If uncertain → purchase up to 2 external signals via x402
    3. Return the final decision with full reasoning trace
    """
    try:
        tx_dict = transaction.model_dump()
        result = evaluate_transaction(tx_dict)

        explanation = await explain_decision(result)
        if explanation:
            result["explanation"] = explanation

        # Convert signals_purchased list to SignalPurchase objects
        # (FastAPI will validate them against the schema)
        return EvaluationResult(**result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent evaluation failed: {str(exc)}",
        )


@app.post(
    "/evaluate-stream",
    summary="Evaluate a transaction for fraud (with live streaming)",
    description=(
        "Streaming version of /evaluate. Sends Server-Sent Events (SSE) "
        "to provide real-time updates as the agent processes the transaction, "
        "runs the model, assesses risk zones, and purchases signals."
    ),
    tags=["Agent"],
)
async def evaluate_stream(transaction: Transaction):
    """
    Evaluate a transaction with live streaming updates.
    Each step of the pipeline is streamed as it happens.
    """
    async def event_generator():
        tx_dict = transaction.model_dump()
        final_data = None

        try:
            async for event in evaluate_transaction_stream(tx_dict):
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "done":
                    break
                if event.get("type") == "step_complete" and event.get("step") == "decision":
                    final_data = event.get("data")

            explanation = await explain_decision(final_data) if final_data else None
            if explanation:
                yield f"data: {json.dumps({'type': 'explanation', 'text': explanation})}\n\n"

            yield f"data: [DONE]\n\n"
        except Exception as exc:
            error_event = {"type": "error", "error": str(exc)}
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health() -> HealthResponse:
    """Returns the API health status, model information, and bandit stats."""
    global _model_ready, _model_metrics

    if not _model_ready:
        try:
            _model_metrics = get_model_metrics()
            _model_ready = True
        except Exception:
            pass

    # Get bandit stats from the agent's global bandit instance
    bandit_state = {}
    try:
        bandit = _get_bandit()
        for signal_name in ["ip-reputation", "device-history", "tx-velocity"]:
            bandit_state[signal_name] = bandit.get_stats(signal_name)
    except Exception as e:
        print(f"[health] Could not load bandit stats: {e}")

    return HealthResponse(
        status="ok",
        service="Scorythm Agent API",
        model_loaded=_model_ready,
        model_metrics=_model_metrics if _model_metrics else None,
        bandit_state=bandit_state if bandit_state else None,
        version="1.0.0",
    )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Scorythm Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "evaluate": "POST /evaluate",
    }


# ─── Exception handlers ───────────────────────────────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )


# ─── Dev server entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
