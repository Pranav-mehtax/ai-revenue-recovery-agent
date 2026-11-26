"""Batch recovery pipeline: one payment in, one audited result out.

The LLM only RECOMMENDS. Deterministic guardrails decide what is allowed, and only
then does the (simulated) executor act.

    payment
      1. RAG        _rag_context()           find relevant recovery policies
      2. LLM        _llm_recommendation()    Gemini suggests an action
      3. Guardrails _guardrails()            hard rules allow / override that action
      4. Executor   execute_action()         simulate it (+ optional ntfy push)
      5. Audit      _append_audit()          one row in data/recovery_audit_log.csv

Entry points: process_payment, process_payment_id, process_batch, read_audit, metrics.
(demo_pipeline.py reuses the helpers in this file.)
"""
from __future__ import annotations

import csv
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from backend.actions import action_executor

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PRIORITIZED_PAYMENTS_PATH = DATA_DIR / "prioritized_payments.csv"
AUDIT_LOG_PATH = DATA_DIR / "recovery_audit_log.csv"

AUDIT_FIELDS = [
    "timestamp", "payment_id", "amount", "failure_type", "recovery_probability",
    "expected_recovery_value", "priority", "rag_sources", "rag_context",
    "llm_recommended_action", "llm_explanation", "llm_status", "llm_error",
    "guardrail_allowed", "final_action", "guardrail_reason", "execution_status",
    "execution_message", "recovered_amount",
]


# ======================================================================
# Small helpers
# ======================================================================

def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalise_payment(row: Dict[str, Any]) -> Dict[str, Any]:
    """CSV rows are all strings; turn the numeric fields into numbers and make sure priority exists."""
    payment = dict(row)
    for field in ("amount", "customer_success_rate", "merchant_success_rate",
                  "merchant_recovery_rate", "recovery_probability", "expected_recovery_value"):
        payment[field] = _number(payment.get(field))
    payment["retry_count"] = _int(payment.get("retry_count"))
    payment["priority"] = payment.get("priority") or payment.get("priority_level") or "LOW"
    payment["priority_level"] = payment["priority"]
    return payment


def _add_rag_to_path() -> None:
    """The rag/ folder is imported as plain modules (rag_engine, llm_agent, guardrails)."""
    rag_dir = str(ROOT / "rag")
    if rag_dir not in sys.path:
        sys.path.insert(0, rag_dir)


def _guardrails():
    """Import the guardrails module (works whether rag/ is a package or just a folder)."""
    try:
        return importlib.import_module("rag.guardrails")
    except ModuleNotFoundError:
        _add_rag_to_path()
        return importlib.import_module("guardrails")


# ======================================================================
# Loading payments
# ======================================================================

def load_prioritized_payments() -> List[Dict[str, Any]]:
    if not PRIORITIZED_PAYMENTS_PATH.exists():
        raise FileNotFoundError(f"Priority data not found: {PRIORITIZED_PAYMENTS_PATH}")
    with PRIORITIZED_PAYMENTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        return [_normalise_payment(row) for row in csv.DictReader(handle)]


def get_payment(payment_id: str) -> Optional[Dict[str, Any]]:
    target = payment_id.strip().upper()
    return next((p for p in load_prioritized_payments() if str(p["payment_id"]).upper() == target), None)


# ======================================================================
# Pipeline stages 1 and 2: RAG and LLM
# ======================================================================

def _rag_context(payment: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Retrieve the top 3 policy documents. Returns (documents, error). Failure is not fatal."""
    try:
        _add_rag_to_path()
        retrieve = importlib.import_module("rag_engine").retrieve
        query = (
            f"failure type {payment['failure_type']} payment method {payment.get('payment_method', '')} "
            f"retry count {payment['retry_count']} merchant recovery policy"
        )
        return retrieve(query, top_k=3), None
    except Exception as error:  # retrieval is advisory; guardrails still protect actioning
        return [], str(error)


def _fallback_llm(error: str) -> Dict[str, str]:
    """Safe answer used when Gemini is unavailable: send the payment to a human."""
    return {
        "response": (
            "FAILURE_EXPLANATION:\nAI recommendation unavailable.\n\n"
            "RECOMMENDED_ACTION:\nMANUAL_REVIEW\n\n"
            "REASONING:\nSafe fallback because Gemini could not provide a recommendation.\n\n"
            "RISK:\nDo not automate while AI reasoning is unavailable.\n\nAUTOMATIC_ACTION:\nNO"
        ),
        "action": "MANUAL_REVIEW",
        "status": "FALLBACK",
        "error": error,
    }


def _llm_recommendation(payment: Dict[str, Any]) -> Dict[str, str]:
    """Ask Gemini for a recommendation. Returns {response, action, status, error}."""
    try:
        _add_rag_to_path()
        agent = importlib.import_module("llm_agent")
        response = str(agent.generate_recovery_recommendation(payment)["response"])
        action = _guardrails().extract_action(response) or "MANUAL_REVIEW"
        return {"response": response, "action": action, "status": "SUCCESS", "error": ""}
    except Exception as error:
        return _fallback_llm(str(error))


# ======================================================================
# Stage 5: audit log (CSV)
# ======================================================================

def _append_audit(result: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    row = {field: result.get(field, "") for field in AUDIT_FIELDS}
    for field in ("rag_sources", "rag_context"):  # lists are stored as JSON text
        row[field] = json.dumps(row[field], ensure_ascii=False)
    write_header = not AUDIT_LOG_PATH.exists() or AUDIT_LOG_PATH.stat().st_size == 0
    with AUDIT_LOG_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_audit() -> List[Dict[str, Any]]:
    if not AUDIT_LOG_PATH.exists():
        return []
    with AUDIT_LOG_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ("amount", "recovery_probability", "expected_recovery_value", "recovered_amount"):
            row[field] = _number(row.get(field))
        row["guardrail_allowed"] = str(row.get("guardrail_allowed")).lower() == "true"
        for field in ("rag_sources", "rag_context"):
            try:
                row[field] = json.loads(row.get(field) or "[]")
            except json.JSONDecodeError:
                row[field] = []
    return rows


# ======================================================================
# Public entry points
# ======================================================================

def process_payment(payment: Dict[str, Any], persist: bool = True) -> Dict[str, Any]:
    """Run one payment through the whole pipeline and return the audited result."""
    payment = _normalise_payment(payment)

    documents, rag_error = _rag_context(payment)                                      # 1. RAG
    llm = _llm_recommendation(payment)                                                # 2. LLM
    guardrail = _guardrails().evaluate_action(payment, llm["response"])               # 3. Guardrails
    execution = action_executor.execute_action(payment, guardrail["final_action"])    # 4. Executor

    result: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payment_id": payment["payment_id"], "amount": payment["amount"],
        "payment_method": payment.get("payment_method", ""), "failure_type": payment["failure_type"],
        "retry_count": payment["retry_count"], "recovery_probability": payment["recovery_probability"],
        "expected_recovery_value": payment["expected_recovery_value"], "priority": payment["priority"],
        "rag_sources": [doc.get("source", "") for doc in documents], "rag_context": documents,
        "rag_error": rag_error, "llm_response": llm["response"], "llm_explanation": llm["response"],
        "llm_recommended_action": llm["action"], "llm_status": llm["status"], "llm_error": llm["error"],
        "guardrail_allowed": guardrail["allowed"], "final_action": guardrail["final_action"],
        "guardrail_reason": guardrail["reason"], "execution_status": execution["status"],
        "execution_message": execution["message"], "recovered_amount": _number(execution["recovered_amount"]),
        "simulation_mode": True,
    }
    if persist:
        _append_audit(result)                                                         # 5. Audit
    return result


def process_payment_id(payment_id: str) -> Dict[str, Any]:
    payment = get_payment(payment_id)
    if payment is None:
        raise KeyError(f"Payment not found: {payment_id}")
    return process_payment(payment)


def process_batch(limit: int = 10) -> List[Dict[str, Any]]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    return [process_payment(payment) for payment in load_prioritized_payments()[:limit]]


def metrics(records: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Dashboard numbers computed from the audit log."""
    records = list(read_audit() if records is None else records)

    def total(field: str) -> float:
        return sum(_number(r.get(field)) for r in records)

    def count(field: str, value: str) -> int:
        return sum(1 for r in records if r.get(field) == value)

    failed_value, recovered = total("amount"), total("recovered_amount")
    return {
        "total_failed_value": round(failed_value, 2),
        "total_expected_recovery": round(total("expected_recovery_value"), 2),
        "total_recovered_amount": round(recovered, 2),
        "total_unrecovered_amount": round(failed_value - recovered, 2),
        "recovery_rate": round(recovered / failed_value, 4) if failed_value else 0.0,
        "number_processed": len(records),
        "number_recovered": count("execution_status", "RECOVERED"),
        "number_failed": count("execution_status", "FAILED"),
        "number_escalated": count("final_action", "ESCALATE"),
        "number_manual_review": count("final_action", "MANUAL_REVIEW"),
        "number_no_action": count("final_action", "NO_ACTION"),
    }