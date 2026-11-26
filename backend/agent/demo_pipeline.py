"""Live demo pipeline: intentionally fail a payment, then run the full guarded recovery flow.

run_demo_case() walks through 6 timed stages. Each stage is saved in the audit record:

    1. gateway     the payment fails on purpose (simulated)
    2. ml          recovery probability + priority
    3. rag         retrieve recovery policies
    4. llm         Gemini recommends an action
    5. guardrails  hard rules allow or override it
    6. executor    simulate the action + send a real ntfy push

The audit log (data/demo_audit_log.jsonl) is append-only and hash-chained: each record
stores the hash of the previous one, so edits or deletions show up in verify_demo_audit().

Sections of this file:  scenarios -> ML scoring -> audit log -> stages -> public API
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.actions import action_executor
from backend.agent import recovery_pipeline as rp

ROOT = rp.ROOT
DEMO_AUDIT_PATH = rp.DATA_DIR / "demo_audit_log.jsonl"
GENESIS_HASH = "0" * 64
RETRY_OUTCOMES = ("sample", "success", "fail")

# ======================================================================
# Demo scenarios
# ======================================================================

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "NETWORK_ERROR": {
        "label": "Network timeout at the issuer bank",
        "amount": 60063.19,
        "gateway_message": "Gateway timeout while contacting the issuer bank.",
        "expectation": "Temporary fault: a retry should be recommended and allowed.",
    },
    "INSUFFICIENT_FUNDS": {
        "label": "Insufficient funds",
        "amount": 25000.00,
        "gateway_message": "Issuer declined: insufficient funds.",
        "expectation": "Retrying will not help: expect a customer notification or payment link.",
    },
    "EXPIRED_CARD": {
        "label": "Expired card",
        "amount": 12999.00,
        "gateway_message": "Issuer declined: card expired.",
        "expectation": "Expect a payment-method update request.",
    },
    "DUPLICATE": {
        "label": "Duplicate payment",
        "amount": 8499.00,
        "gateway_message": "Rejected: duplicate transaction reference.",
        "expectation": "Must never be retried: expect no action or manual review.",
    },
    "SUSPECTED_FRAUD": {
        "label": "Suspected fraud",
        "amount": 149999.00,
        "gateway_message": "Declined: risk engine flagged the transaction.",
        "expectation": "Must be escalated or sent to manual review, never retried.",
    },
}

# Fake LLM answer used by the "red team" switch to prove the guardrails catch a bad RETRY.
RED_TEAM_RESPONSE = (
    "FAILURE_EXPLANATION:\n[RED-TEAM] Recommendation injected to test the guardrails.\n\n"
    "RECOMMENDED_ACTION:\nRETRY\n\n"
    "REASONING:\nInjected for testing: pretend the LLM wants an automatic retry whatever the failure type.\n\n"
    "RISK:\nLow\n\nAUTOMATIC_ACTION:\nYES"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique(items) -> List[str]:
    return list(dict.fromkeys(str(i) for i in items if i))


# ======================================================================
# ML scoring: trained model if available, otherwise historical average
# ======================================================================

def _model_candidates() -> List[Path]:
    """Model files to try: ML_MODEL_PATH from .env first, then anything in ml/."""
    pinned = os.getenv("ML_MODEL_PATH", "").strip()
    paths = [Path(pinned)] if pinned else []
    ml_dir = ROOT / "ml"
    if ml_dir.exists():
        for pattern in ("*.joblib", "*.pkl"):
            paths += sorted(ml_dir.rglob(pattern))
    return paths


def _coerce(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _predict_with_model(payment: Dict[str, Any]) -> Tuple[float, str, List[str]]:
    """Return (probability, model_name, features_we_had_to_leave_empty). Raises if no model works."""
    import joblib
    import pandas as pd

    errors: List[str] = []
    for path in _model_candidates():
        try:
            loaded = joblib.load(path)
            estimator = loaded.get("model") if isinstance(loaded, dict) else loaded
            columns = None
            if isinstance(loaded, dict):
                columns = loaded.get("features") or loaded.get("feature_names")
            if columns is None:
                names = getattr(estimator, "feature_names_in_", None)
                columns = list(names) if names is not None else []
            if not columns or not hasattr(estimator, "predict_proba"):
                continue
            imputed = [c for c in columns if payment.get(c) in (None, "")]
            frame = pd.DataFrame([{c: _coerce(payment.get(c)) for c in columns}], columns=list(columns))
            proba = estimator.predict_proba(frame)[0]
            classes = list(getattr(estimator, "classes_", [0, 1]))
            index = classes.index(1) if 1 in classes else -1
            return float(proba[index]), f"trained_model:{path.name}", imputed
        except Exception as error:
            errors.append(f"{path.name}: {error}")
    raise RuntimeError("; ".join(errors) or "no usable model artifact found in ml/")


def _historical_probability(failure_type: str) -> Tuple[float, str]:
    """Fallback: average recovery probability of past payments with the same failure type."""
    rows = rp.load_prioritized_payments()
    same = [r["recovery_probability"] for r in rows if str(r.get("failure_type")).upper() == failure_type]
    pool = same or [r["recovery_probability"] for r in rows]
    if not pool:
        raise RuntimeError("prioritized_payments.csv is empty")
    return sum(pool) / len(pool), f"historical_average:{failure_type}:{len(same)}_payments"


def _priority(expected_value: float) -> Tuple[str, float]:
    """Same rule as ml/priority_engine.py: score = expected value / largest expected value x 100.
    HIGH at 66 and above, MEDIUM at 33 and above, otherwise LOW."""
    rows = rp.load_prioritized_payments()
    top = max((r["expected_recovery_value"] for r in rows), default=0.0)
    score = expected_value / top * 100 if top > 0 else 0.0
    return ("HIGH" if score >= 66 else "MEDIUM" if score >= 33 else "LOW"), round(score, 2)


def score_payment(payment: Dict[str, Any]) -> Dict[str, Any]:
    note, imputed = "", []
    try:
        probability, source, imputed = _predict_with_model(payment)
    except Exception as error:
        probability, source = _historical_probability(str(payment["failure_type"]).upper())
        note = f"Trained model unavailable ({error}); used the historical average for this failure type."
    probability = min(max(probability, 0.0), 1.0)
    expected = round(float(payment["amount"]) * probability, 2)
    priority, priority_score = _priority(expected)
    return {
        "recovery_probability": round(probability, 4),
        "expected_recovery_value": expected,
        "priority": priority,
        "priority_score": priority_score,
        "source": source,
        "imputed_features": imputed,
        "note": note,
    }


# ======================================================================
# Hash-chained audit log
# ======================================================================

def _digest(body: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _read_records() -> List[Dict[str, Any]]:
    if not DEMO_AUDIT_PATH.exists():
        return []
    records = []
    with DEMO_AUDIT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    records.append({"corrupt": True})
    return records


def _last_hash() -> str:
    for record in reversed(_read_records()):
        if record.get("hash"):
            return record["hash"]
    return GENESIS_HASH


def append_demo_audit(record: Dict[str, Any]) -> Dict[str, Any]:
    """Add a record to the chain: store the previous hash, then hash this record."""
    body = json.loads(json.dumps(record, default=str))  # normalise so re-reading hashes identically
    body["prev_hash"] = _last_hash()
    body["hash"] = _digest(body)
    rp.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with DEMO_AUDIT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(body, ensure_ascii=False) + "\n")
    return body


def read_demo_audit(limit: int = 50) -> List[Dict[str, Any]]:
    """Newest records first."""
    return _read_records()[::-1][: max(1, int(limit))]


def verify_demo_audit() -> Dict[str, Any]:
    """Re-check every hash in order. broken_at is the first record that doesn't match."""
    previous = GENESIS_HASH
    records = _read_records()
    for index, record in enumerate(records):
        stored = record.get("hash")
        body = {k: v for k, v in record.items() if k != "hash"}
        if record.get("corrupt") or record.get("prev_hash") != previous or stored != _digest(body):
            return {"ok": False, "records": len(records), "broken_at": index}
        previous = stored
    return {"ok": True, "records": len(records), "broken_at": None}


# ======================================================================
# The 6 stages. Each returns (status, output) and is timed by _timed().
# ======================================================================

def _timed(stages: List[Dict[str, Any]], name: str, work: Callable[[], Tuple[str, Dict[str, Any]]]):
    """Run one stage, record its status/output/latency in `stages`, and return (status, output)."""
    started, t0 = _now(), time.perf_counter()
    status, output = work()
    stages.append({
        "stage": name, "status": status, "started_at": started,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "output": output,
    })
    return status, output


def _template(failure_type: str) -> Dict[str, Any]:
    """Borrow a real historical payment's profile so the model and later stages get every column.
    Prefers data/ml_dataset.csv (all model features), falls back to prioritized_payments.csv."""
    ml_path = rp.DATA_DIR / "ml_dataset.csv"
    if ml_path.exists():
        with ml_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        same = [r for r in rows if str(r.get("failure_type")).upper() == failure_type]
        if same:
            row = dict(same[0])
            row.pop("recovered", None)  # ground truth must never reach the LLM
            return row
    rows = rp.load_prioritized_payments()
    if not rows:
        raise RuntimeError("prioritized_payments.csv is empty; cannot build a demo payment")
    same = [r for r in rows if str(r.get("failure_type")).upper() == failure_type]
    return dict((same or rows)[0])


def _build_failed_payment(run_id, failure_type, amount, payment_method, retry_count):
    """Stage 1 input: the payment that 'fails at the gateway'. Returns (payment, borrowed_from_id)."""
    template = _template(failure_type)
    payment = dict(template)
    payment.pop("recommended_action", None)  # that column belongs to the historical payment
    payment.update({
        "payment_id": f"DEMO_{run_id[4:]}", "amount": amount, "failure_type": failure_type,
        "retry_count": int(retry_count),
        "payment_method": payment_method or template.get("payment_method", ""),
    })
    return payment, template.get("payment_id")


def _rag_stage(payment):
    documents, error = rp._rag_context(payment)
    return ("ERROR" if error else "OK"), {
        "documents": documents,
        "sources": _unique(d.get("source", "") for d in documents),
        "error": error or "",
        "note": "llm_agent may run its own retrieval; compare with its prompt if sources look different.",
    }


def parse_sections(text: str) -> Dict[str, str]:
    """Split the LLM answer into {'RECOMMENDED_ACTION': '...', 'REASONING': '...', ...}."""
    sections: Dict[str, str] = {}
    key = None
    for line in str(text).splitlines():
        match = re.match(r"^\s*\**([A-Z_]{4,})\**\s*:\s*(.*)$", line)
        if match:
            key = match.group(1)
            sections[key] = match.group(2).strip("* ").strip()
        elif key:
            sections[key] = (sections[key] + "\n" + line).strip()
    return sections


def _llm_stage(payment, red_team):
    if red_team:
        llm = {"response": RED_TEAM_RESPONSE, "action": "RETRY", "status": "INJECTED_REDTEAM", "error": ""}
    else:
        llm = rp._llm_recommendation(payment)
    return llm["status"], {
        "response": llm["response"], "sections": parse_sections(llm["response"]),
        "recommended_action": llm["action"], "error": llm["error"],
    }


def _guardrail_stage(payment, llm_output):
    try:
        verdict = rp._guardrails().evaluate_action(payment, llm_output["response"])
        status = "OK"
    except Exception as error:  # fail closed: when in doubt, a human decides
        verdict = {"allowed": False, "final_action": "MANUAL_REVIEW",
                   "reason": f"Guardrail error, failing closed: {error}"}
        status = "ERROR"
    return status, {
        "allowed": verdict.get("allowed"), "llm_action": llm_output["recommended_action"],
        "final_action": verdict.get("final_action"), "reason": verdict.get("reason", ""), "raw": verdict,
    }


def _executor_stage(payment, final_action):
    try:
        return "OK", action_executor.execute_action(payment, final_action)
    except Exception as error:
        return "ERROR", {
            "payment_id": payment["payment_id"], "action": final_action, "status": "EXECUTION_ERROR",
            "recovered_amount": 0, "message": str(error), "notification": {"sent": False, "detail": str(error)},
        }


def _explain(s: Dict[str, Any], gateway_message: str, reasoning: str, guardrail_reason: str, push: Dict[str, Any]) -> List[str]:
    """Plain-English story of the run, one line per stage (shown on the Live Demo page)."""
    verdict = (
        f"changed it to {s['final_action']}" if s["llm_action"] != s["final_action"]
        else f"allowed it, final action {s['final_action']}"
    )
    return [
        f"1. Gateway: payment of Rs {s['amount']:,.2f} failed with {s['failure_type']} ({gateway_message}).",
        f"2. ML ({s['probability_source']}): recovery probability {s['recovery_probability']:.1%}, "
        f"expected recovery Rs {s['expected_recovery_value']:,.2f}, priority {s['priority']}.",
        f"3. RAG: retrieved {len(s['rag_sources'])} source(s): {', '.join(s['rag_sources']) or 'none'}.",
        f"4. LLM ({s['llm_status']}) recommended {s['llm_action']}. Reasoning: {reasoning or 'not provided'}",
        f"5. Guardrails {verdict}. Reason: {guardrail_reason or 'not provided'}",
        f"6. Executor: {s['execution_status']}. Push to phone: "
        f"{'delivered' if push.get('sent') else 'not delivered'} ({push.get('detail', 'n/a')}).",
    ]


# ======================================================================
# Public API used by FastAPI and Streamlit
# ======================================================================

def list_scenarios() -> Dict[str, Any]:
    methods: List[str] = []
    try:
        methods = sorted({str(r["payment_method"]) for r in rp.load_prioritized_payments() if r.get("payment_method")})
    except Exception:
        pass
    return {"scenarios": SCENARIOS, "payment_methods": methods, "retry_outcomes": list(RETRY_OUTCOMES)}


def status() -> Dict[str, Any]:
    return {
        "ntfy_configured": bool(os.getenv("NTFY_TOPIC", "").strip()),
        "ntfy_destination": action_executor.ntfy_destination(),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "model_artifacts": [p.name for p in _model_candidates()],
        "audit_chain": verify_demo_audit(),
    }


def send_test_push() -> Dict[str, Any]:
    return action_executor.send_ntfy(
        "Recovery agent: test",
        "If you can read this on your phone, ntfy is wired up correctly.\nSIMULATED DEMO - no real money moved.",
        "default",
        ["white_check_mark"],
    )


def run_demo_case(
    failure_type: str,
    amount: float,
    payment_method: Optional[str] = None,
    retry_count: int = 0,
    retry_outcome: str = "sample",
    red_team_retry: bool = False,
    send_push: bool = True,
) -> Dict[str, Any]:
    """Run one demo payment through all 6 stages and append the result to the audit chain."""
    # --- validate input ---------------------------------------------------
    failure_type = str(failure_type).upper()
    if failure_type not in SCENARIOS:
        raise ValueError(f"Unknown failure_type {failure_type!r}; choose one of {sorted(SCENARIOS)}")
    amount = float(amount)
    if not 0 < amount <= 10_000_000:
        raise ValueError("amount must be between 0 and 10,000,000")
    if retry_outcome not in RETRY_OUTCOMES:
        raise ValueError(f"retry_outcome must be one of {RETRY_OUTCOMES}")

    run_id = "RUN-" + uuid.uuid4().hex[:8].upper()
    run_started = time.perf_counter()
    stages: List[Dict[str, Any]] = []

    # --- 1. gateway: the payment fails on purpose ---------------------------
    payment, borrowed_from = _build_failed_payment(run_id, failure_type, amount, payment_method, retry_count)
    gateway_message = SCENARIOS[failure_type]["gateway_message"]
    _timed(stages, "gateway", lambda: ("FAILED", {
        "payment_id": payment["payment_id"], "gateway_status": "FAILED", "failure_code": failure_type,
        "message": gateway_message, "simulated": True, "profile_borrowed_from": borrowed_from,
    }))

    # --- 2. ML: recovery probability + priority -----------------------------
    def ml_stage():
        score = score_payment(payment)
        return ("FALLBACK" if score["note"] else "OK"), score

    _, score = _timed(stages, "ml", ml_stage)
    payment.update({
        "recovery_probability": score["recovery_probability"],
        "expected_recovery_value": score["expected_recovery_value"],
        "priority": score["priority"], "priority_level": score["priority"],
    })
    payment = rp._normalise_payment(payment)

    # --- 3. RAG: retrieve policies -------------------------------------------
    _, rag = _timed(stages, "rag", lambda: _rag_stage(payment))

    # --- 4. LLM: Gemini recommends an action ---------------------------------
    llm_status, llm = _timed(stages, "llm", lambda: _llm_stage(payment, red_team_retry))

    # --- 5. Guardrails: allow or override ------------------------------------
    _, guard = _timed(stages, "guardrails", lambda: _guardrail_stage(payment, llm))
    final_action = guard["final_action"]

    # --- 6. Executor: simulate the action + real ntfy push -------------------
    why = f"LLM suggested {llm['recommended_action']}; guardrails -> {final_action}. {guard['reason']}"
    payment.update({
        "send_push": bool(send_push), "simulate_retry_outcome": retry_outcome,
        "notify_context": {"run_id": run_id, "why": why[:240]},
    })
    _, execution = _timed(stages, "executor", lambda: _executor_stage(payment, final_action))
    push = execution.get("notification", {"sent": False, "detail": "n/a"})

    # --- summary, explanation, audit record ---------------------------------
    summary = {
        "failure_type": failure_type, "amount": amount,
        "recovery_probability": score["recovery_probability"], "probability_source": score["source"],
        "expected_recovery_value": score["expected_recovery_value"], "priority": score["priority"],
        "rag_sources": rag["sources"],
        "llm_action": llm["recommended_action"], "llm_status": llm_status,
        "guardrail_allowed": guard["allowed"], "guardrail_overrode_llm": llm["recommended_action"] != final_action,
        "final_action": final_action, "execution_status": execution.get("status"),
        "recovered_amount": rp._number(execution.get("recovered_amount")),
        "push_sent": bool(push.get("sent")), "push_detail": push.get("detail", ""),
    }
    explanation = _explain(summary, gateway_message, llm["sections"].get("REASONING", ""), guard["reason"], push)

    return append_demo_audit({
        "run_id": run_id, "timestamp": _now(), "simulation_mode": True,
        "inputs": {"failure_type": failure_type, "amount": amount, "payment_method": payment.get("payment_method"),
                   "retry_count": int(retry_count), "retry_outcome": retry_outcome,
                   "red_team_retry": bool(red_team_retry), "send_push": bool(send_push)},
        "payment": {k: v for k, v in payment.items() if k != "notify_context"},
        "stages": stages, "summary": summary, "explanation": explanation,
        "total_latency_ms": round((time.perf_counter() - run_started) * 1000, 1),
    })