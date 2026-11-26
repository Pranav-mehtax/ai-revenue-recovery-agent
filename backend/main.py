"""FastAPI entry point for the simulated Revenue Recovery Agent."""
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Query
from backend.api.demo_routes import router as demo_router
from backend.agent.recovery_pipeline import (
    get_payment, load_prioritized_payments, metrics, process_batch, process_payment_id, read_audit,
)

app = FastAPI(title="AI Revenue Recovery Agent", version="1.0.0")
app.include_router(demo_router)

@app.get("/health")
def health():
    return {"status": "ok", "mode": "simulation", "real_money_processing": False}

@app.get("/payments")
def payments(limit: int = Query(100, ge=1, le=500)):
    return {"items": load_prioritized_payments()[:limit]}

@app.get("/payments/{payment_id}")
def payment(payment_id: str):
    item = get_payment(payment_id)
    if item is None:
        raise HTTPException(404, "Payment not found")
    return item

@app.post("/payments/{payment_id}/recover")
def recover_payment(payment_id: str):
    try:
        return process_payment_id(payment_id)
    except KeyError:
        raise HTTPException(404, "Payment not found")
    except Exception as error:
        raise HTTPException(500, f"Recovery simulation failed safely: {error}")

@app.post("/recover/batch")
def recover_batch(limit: int = Query(10, ge=1, le=100)):
    try:
        items = process_batch(limit)
        return {"items": items, "metrics": metrics(items)}
    except Exception as error:
        raise HTTPException(500, f"Batch simulation failed safely: {error}")

@app.get("/metrics")
def get_metrics():
    return metrics()

@app.get("/audit")
def audit(limit: int = Query(200, ge=1, le=1000)):
    return {"items": read_audit()[-limit:]}
