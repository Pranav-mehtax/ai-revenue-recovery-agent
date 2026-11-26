"""Live Demo: intentionally fail a payment and watch the guarded recovery pipeline end to end."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Live Demo", page_icon="₹", layout="wide")
st.markdown("""
<style>
  .stApp { background: #07141f; color: #eaf2f7; }
  [data-testid="stMetric"] { background: #102535; border: 1px solid #1d4055; padding: 14px; border-radius: 14px; }
  [data-testid="stMetricLabel"] { color: #9bb3c3; }
  .eyebrow { color: #57d4b5; font-weight: 700; letter-spacing: .12em; font-size: .75rem; }
  .notice { background: #123247; border-left: 4px solid #57d4b5; padding: 12px 16px; border-radius: 6px; }
  .why { white-space: pre-wrap; overflow-wrap: anywhere; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Backend access: FastAPI first, in-process fallback (always labelled)
# ---------------------------------------------------------------------

def _local(method: str, path: str, payload: dict | None, params: dict | None):
    from backend.agent import demo_pipeline as demo

    if (method, path) == ("GET", "/demo/scenarios"):
        return demo.list_scenarios()
    if (method, path) == ("GET", "/demo/status"):
        return demo.status()
    if (method, path) == ("POST", "/demo/run"):
        return demo.run_demo_case(**(payload or {}))
    if (method, path) == ("POST", "/demo/test-push"):
        return demo.send_test_push()
    if (method, path) == ("GET", "/demo/audit"):
        return {"records": demo.read_demo_audit((params or {}).get("limit", 20))}
    if (method, path) == ("GET", "/demo/audit/verify"):
        return demo.verify_demo_audit()
    raise ValueError(f"Unsupported call {method} {path}")


def call(method: str, path: str, payload: dict | None = None, params: dict | None = None):
    """Returns (data, route). Raises RuntimeError with a readable message on failure."""
    if use_api:
        try:
            response = requests.request(method, api_url.rstrip("/") + path, json=payload, params=params, timeout=180)
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                raise RuntimeError(f"Backend returned {response.status_code}: {detail}")
            return response.json(), "FastAPI backend"
        except requests.ConnectionError:
            st.session_state["api_down"] = True
    try:
        return _local(method, path, payload, params), "in-process (backend not used)"
    except Exception as error:
        raise RuntimeError(str(error))


with st.sidebar:
    st.header("Backend")
    use_api = st.toggle("Call FastAPI backend", value=True, help="Off = run the same code in this Streamlit process.")
    api_url = st.text_input("API URL", os.getenv("DEMO_API_URL", "http://127.0.0.1:8000"))
    st.caption("Start it with: uvicorn backend.main:app --reload")

st.markdown('<div class="eyebrow">RAZORPAY AI BUILDATHON · LIVE DEMO · TEST MODE</div>', unsafe_allow_html=True)
st.title("Fail a payment, watch the agent recover it")
st.markdown(
    '<div class="notice">Everything is simulated except one thing: the push notification. '
    "It is a real message sent through ntfy to your phone. No money moves.</div>",
    unsafe_allow_html=True,
)

st.session_state["api_down"] = False
try:
    catalog, _ = call("GET", "/demo/scenarios")
    system, route = call("GET", "/demo/status")
except RuntimeError as error:
    st.error(f"Cannot load the demo: {error}")
    st.stop()

if use_api and st.session_state.get("api_down"):
    st.warning(f"FastAPI backend not reachable at {api_url}. Running in-process instead. "
               "Start it with `uvicorn backend.main:app --reload` to demo the real API path.")
else:
    st.caption(f"Route: {route}")

c1, c2, c3 = st.columns(3)
c1.metric("Phone push (ntfy)", "Configured" if system["ntfy_configured"] else "NOT configured",
          help=system.get("ntfy_destination") or "Set NTFY_TOPIC in .env")
c2.metric("Gemini key", "Found" if system["gemini_configured"] else "Missing")
chain = system["audit_chain"]
c3.metric("Audit chain", "Intact" if chain["ok"] else "TAMPERED", f"{chain['records']} runs")
if not system["ntfy_configured"]:
    st.warning("NTFY_TOPIC is not set, so no notification will reach your phone. See the setup steps.")

# ---------------------------------------------------------------------
# Scenario controls
# ---------------------------------------------------------------------
st.subheader("1 · Choose how the payment fails")
scenarios = catalog["scenarios"]
failure_type = st.radio(
    "Failure scenario", list(scenarios),
    format_func=lambda key: f"{scenarios[key]['label']}  ({key})", horizontal=False,
)
st.caption("Expected: " + scenarios[failure_type]["expectation"])

col_a, col_b, col_c = st.columns(3)
amount = col_a.number_input("Amount (₹)", min_value=1.0, max_value=10_000_000.0,
                            value=float(scenarios[failure_type]["amount"]), step=500.0,
                            key=f"amount_{failure_type}")
methods = catalog.get("payment_methods") or []
method = col_b.selectbox("Payment method", methods) if methods else None
retry_count = col_c.number_input("Retries already made", min_value=0, max_value=10, value=0)

col_d, col_e, col_f = st.columns(3)
retry_outcome = col_d.selectbox(
    "If a retry happens, outcome is", catalog["retry_outcomes"],
    format_func={"sample": "Sampled from ML probability", "success": "Forced success (demo)", "fail": "Forced failure (demo)"}.get,
)
red_team = col_e.checkbox("Red-team: inject an unsafe RETRY recommendation",
                          help="Skips Gemini and pretends it recommended RETRY, to prove the guardrails hold.")
send_push = col_f.checkbox("Send real push to my phone", value=True)

b1, b2 = st.columns([3, 1])
run_clicked = b1.button("Fail this payment and run the recovery agent", type="primary", use_container_width=True)
if b2.button("Send test push", use_container_width=True):
    try:
        result, _ = call("POST", "/demo/test-push")
        (st.success if result.get("sent") else st.error)(f"ntfy: {result.get('detail')}")
    except RuntimeError as error:
        st.error(str(error))

if run_clicked:
    payload = {"failure_type": failure_type, "amount": amount, "payment_method": method,
               "retry_count": int(retry_count), "retry_outcome": retry_outcome,
               "red_team_retry": red_team, "send_push": send_push}
    with st.spinner("Gateway → ML → RAG → Gemini → guardrails → executor…"):
        try:
            record, route = call("POST", "/demo/run", payload)
            st.session_state["last_run"] = record
            st.session_state["last_route"] = route
        except RuntimeError as error:
            st.error(f"Run failed: {error}")


# ---------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------

def stage_of(record: dict, name: str) -> dict:
    return next(s for s in record["stages"] if s["stage"] == name)


def render_run(record: dict, heading_prefix: str = "") -> None:
    summary = record["summary"]
    if summary["push_sent"]:
        st.success(f"Push delivered to your phone · final action {summary['final_action']} · "
                   f"status {summary['execution_status']}")
    else:
        st.warning(f"No push delivered ({summary['push_detail']}) · final action {summary['final_action']} · "
                   f"status {summary['execution_status']}")
    if summary["guardrail_overrode_llm"]:
        st.info(f"Guardrails overrode the LLM: it recommended {summary['llm_action']}, "
                f"the final action is {summary['final_action']}.")
    st.caption(f"Run {record['run_id']} · {record['total_latency_ms']:,.0f} ms total · "
               f"ran via {st.session_state.get('last_route', 'n/a')}")

    gw, ml, rag, llm, grd, ex = (stage_of(record, n) for n in ("gateway", "ml", "rag", "llm", "guardrails", "executor"))

    with st.container(border=True):
        st.markdown(f"**Stage 1 · Payment gateway** · {gw['latency_ms']} ms")
        st.error(f"{gw['output']['failure_code']}: {gw['output']['message']}")
        st.caption(f"Payment {gw['output']['payment_id']} (simulated). "
                   f"Customer/merchant profile borrowed from historical payment {gw['output']['profile_borrowed_from']}.")

    with st.container(border=True):
        st.markdown(f"**Stage 2 · ML prediction** · {ml['latency_ms']} ms")
        m1, m2, m3 = st.columns(3)
        m1.metric("Recovery probability", f"{ml['output']['recovery_probability']:.1%}")
        m2.metric("Expected recovery", f"₹{ml['output']['expected_recovery_value']:,.2f}")
        m3.metric("Priority", ml["output"]["priority"])
        st.caption(f"Source: {ml['output']['source']}")
        if ml["output"].get("imputed_features"):
            st.caption("Not available for this demo payment, so the model used training-set defaults for: "
                       + ", ".join(ml["output"]["imputed_features"]))
        if ml["output"]["note"]:
            st.warning(ml["output"]["note"])

    with st.container(border=True):
        st.markdown(f"**Stage 3 · RAG retrieval** · {rag['latency_ms']} ms")
        if rag["output"]["error"]:
            st.warning("Retrieval error: " + rag["output"]["error"])
        for index, doc in enumerate(rag["output"]["documents"], 1):
            with st.expander(f"{index}. {doc.get('source', 'unknown source')}", expanded=index == 1):
                text = next((doc[k] for k in ("text", "content", "chunk", "page_content") if doc.get(k)), None)
                st.write(text) if text else st.json(doc)
        st.caption(rag["output"]["note"])

    with st.container(border=True):
        st.markdown(f"**Stage 4 · LLM decision** · {llm['latency_ms']} ms · status {llm['status']}")
        if llm["status"] == "INJECTED_REDTEAM":
            st.info("This recommendation was injected for the guardrail test. Gemini was not called.")
        if llm["status"] == "FALLBACK":
            st.warning("Gemini failed, so the safe fallback MANUAL_REVIEW was used: " + llm["output"]["error"])
        sections = llm["output"]["sections"]
        for key in ("FAILURE_EXPLANATION", "RECOMMENDED_ACTION", "REASONING", "RISK", "AUTOMATIC_ACTION"):
            if sections.get(key):
                st.markdown(f"**{key.replace('_', ' ').title()}**")
                st.markdown(sections[key])
        if not sections:
            st.markdown(llm["output"]["response"])

    with st.container(border=True):
        st.markdown(f"**Stage 5 · Guardrails** · {grd['latency_ms']} ms")
        g1, g2, g3 = st.columns(3)
        g1.metric("LLM recommended", str(grd["output"]["llm_action"]))
        g2.metric("Allowed", "Yes" if grd["output"]["allowed"] else "No")
        g3.metric("Final action", str(grd["output"]["final_action"]))
        st.write("**Reason:** " + str(grd["output"]["reason"]))
        with st.expander("Raw guardrail output"):
            st.json(grd["output"]["raw"])

    with st.container(border=True):
        st.markdown(f"**Stage 6 · Executor and phone notification** · {ex['latency_ms']} ms")
        e1, e2 = st.columns(2)
        e1.metric("Execution status", ex["output"].get("status", "n/a"))
        e2.metric("Recovered (simulated)", f"₹{float(ex['output'].get('recovered_amount', 0)):,.2f}")
        st.write(ex["output"].get("message", ""))
        if ex["output"].get("outcome_source"):
            st.caption("Retry outcome source: " + ex["output"]["outcome_source"])
        push = ex["output"].get("notification", {})
        (st.success if push.get("sent") else st.warning)(
            f"ntfy: {push.get('detail', 'n/a')}" + (f" · message id {push['message_id']}" if push.get("message_id") else ""))

    st.markdown("### Why did the agent do this?")
    st.markdown("\n\n".join(record["explanation"]))
    st.caption("This narrative is assembled from the stage records above, not written by the LLM.")
    with st.expander("Audit record (tamper-evident)"):
        st.code(f"hash      {record['hash']}\nprev_hash {record['prev_hash']}")
        st.json(record["inputs"])
    st.download_button("Download full audit record (JSON)", json.dumps(record, ensure_ascii=False, indent=2),
                       file_name=f"{record['run_id']}.json", mime="application/json", key=f"dl_{heading_prefix}{record['run_id']}")


if "last_run" in st.session_state:
    st.subheader("2 · Result")
    render_run(st.session_state["last_run"])

# ---------------------------------------------------------------------
# Audit history
# ---------------------------------------------------------------------
st.divider()
st.subheader("3 · Audit trail")
try:
    history, _ = call("GET", "/demo/audit", params={"limit": 30})
    verify, _ = call("GET", "/demo/audit/verify")
except RuntimeError as error:
    st.error(str(error))
    st.stop()

if verify["ok"]:
    st.success(f"Hash chain verified: {verify['records']} record(s), no edits or deletions detected.")
else:
    st.error(f"Audit log was modified: chain breaks at record #{verify['broken_at']}.")

records = history["records"]
if not records:
    st.info("No demo runs yet.")
else:
    st.dataframe(pd.DataFrame([{
        "run": r["run_id"], "time (UTC)": r["timestamp"][:19].replace("T", " "), "failure": r["summary"]["failure_type"],
        "amount": r["summary"]["amount"], "LLM said": r["summary"]["llm_action"], "final": r["summary"]["final_action"],
        "status": r["summary"]["execution_status"], "push": r["summary"]["push_sent"], "hash": r["hash"][:12],
    } for r in records]), use_container_width=True, hide_index=True)
    chosen = st.selectbox("Inspect a past run", [r["run_id"] for r in records])
    picked = next(r for r in records if r["run_id"] == chosen)
    st.markdown("\n\n".join(picked["explanation"]))
    with st.expander("Full stage-by-stage record"):
        st.json(picked["stages"])