"""FastAPI routes for the live demo.

Add to backend/main.py:
    from backend.api.demo_routes import router as demo_router
    app.include_router(demo_router)
"""
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.agent import demo_pipeline as demo

router = APIRouter(prefix="/demo", tags=["demo"])


class DemoRunRequest(BaseModel):
    failure_type: str
    amount: float = Field(gt=0, le=10_000_000)
    payment_method: Optional[str] = None
    retry_count: int = Field(default=0, ge=0, le=10)
    retry_outcome: Literal["sample", "success", "fail"] = "sample"
    red_team_retry: bool = False
    send_push: bool = True


@router.get("/scenarios")
def scenarios():
    return demo.list_scenarios()


@router.get("/status")
def status():
    return demo.status()


@router.post("/run")
def run(request: DemoRunRequest):
    try:
        return demo.run_demo_case(**request.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/test-push")
def test_push():
    return demo.send_test_push()


@router.get("/audit")
def audit(limit: int = 20):
    return {"records": demo.read_demo_audit(limit)}


@router.get("/audit/verify")
def verify():
    return demo.verify_demo_audit()