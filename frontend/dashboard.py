"""Streamlit control room for the simulated AI Revenue Recovery Agent."""
from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agent.recovery_pipeline import (  # noqa: E402
    load_prioritized_payments, metrics, process_batch, process_payment_id, read_audit,
)

st.set_page_config(page_title="Recovery Control Room", page_icon="₹", layout="wide")
st.markdown("""
<style>
  .stApp { background: #07141f; color: #eaf2f7; }
  [data-testid="stMetric"] { background: #102535; border: 1px solid #1d4055; padding: 18px; border-radius: 14px; }
  [data-testid="stMetricLabel"] { color: #9bb3c3; }
  .eyebrow { color: #57d4b5; font-weight: 700; letter-spacing: .12em; font-size: .75rem; }
  .notice { background: #123247; border-left: 4px solid #57d4b5; padding: 12px 16px; border-radius: 6px; }
  div[data-testid="stDataFrame"] { border: 1px solid #1d4055; border-radius: 12px; overflow: hidden; }
  .llm-reasoning { white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; }
</style>
""", unsafe_allow_html=True)

def rupees(value: float) -> str:
    return f"₹{float(value):,.2f}"

def display_metrics(summary: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Failed payment value", rupees(summary["total_failed_value"]))
    cols[1].metric("Expected recovery", rupees(summary["total_expected_recovery"]), help="Economic expectation, not a guaranteed payment outcome.")
    cols[2].metric("Recovered (simulated)", rupees(summary["total_recovered_amount"]))
    cols[3].metric("Recovery rate", f"{summary['recovery_rate']:.1%}")
    cols[4].metric("Payments processed", summary["number_processed"])

st.markdown('<div class="eyebrow">RAZORPAY AI BUILDATHON · TEST MODE</div>', unsafe_allow_html=True)
st.title("AI Revenue Recovery Control Room")
st.markdown('<div class="notice">All actions are simulated. Expected recovery is a prioritization estimate; recovered amount only reflects simulated execution.</div>', unsafe_allow_html=True)

try:
    payments = load_prioritized_payments()
except Exception as error:
    st.error(f"Unable to load prioritized payments: {error}")
    st.stop()

audit = read_audit()
display_metrics(metrics())

left, right = st.columns([2, 1])
with left:
    st.subheader("Run recovery batch")
    count = st.slider("Highest-priority opportunities", min_value=1, max_value=25, value=10)
with right:
    st.write("")
    st.write("")
    if st.button("Run Recovery Batch", type="primary", use_container_width=True):
        with st.spinner("Running ML-selected payments through RAG, Gemini, guardrails, and simulation…"):
            try:
                results = process_batch(count)
                st.success(f"Processed {len(results)} payments. Audit log updated.")
                st.rerun()
            except Exception as error:
                st.error(f"Batch could not complete: {error}")

st.subheader("Recovery opportunities")
audit_by_id = {row["payment_id"]: row for row in audit}
rows = []
for payment in payments[:100]:
    latest = audit_by_id.get(payment["payment_id"], {})
    rows.append({
        "Payment ID": payment["payment_id"], "Amount": rupees(payment["amount"]),
        "Failure reason": payment["failure_type"], "Recovery probability": f"{payment['recovery_probability']:.1%}",
        "Expected recovery": rupees(payment["expected_recovery_value"]), "Priority": payment["priority"],
        "Recommended": payment.get("recommended_action", ""), "Final action": latest.get("final_action", "—"),
        "Status": latest.get("execution_status", "Not processed"),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=360)

st.subheader("Recovery breakdown")
summary = metrics()
breakdown = pd.DataFrame([
    {"Outcome": "Recovered", "Count": summary["number_recovered"]},
    {"Outcome": "Retry failed", "Count": summary["number_failed"]},
    {"Outcome": "Escalated", "Count": summary["number_escalated"]},
    {"Outcome": "Manual review", "Count": summary["number_manual_review"]},
    {"Outcome": "No action", "Count": summary["number_no_action"]},
])
st.bar_chart(breakdown.set_index("Outcome"))

st.subheader("Payment decision detail")
ids = [payment["payment_id"] for payment in payments]
selected_id = st.selectbox("Select a payment", ids)
payment = next(row for row in payments if row["payment_id"] == selected_id)
latest = audit_by_id.get(selected_id)

if st.button("Run selected payment", use_container_width=False):
    with st.spinner("Running the guarded recovery pipeline…"):
        try:
            process_payment_id(selected_id)
            st.rerun()
        except Exception as error:
            st.error(f"Payment could not be processed: {error}")

detail_left, detail_right = st.columns(2)
with detail_left:
    st.markdown("#### Payment & ML")
    st.json({key: payment.get(key) for key in ("payment_id", "amount", "failure_type", "retry_count", "recovery_probability", "expected_recovery_value", "priority")})
with detail_right:
    st.markdown("#### Guardrail & execution")
    if latest:
        st.json({key: latest.get(key) for key in ("llm_recommended_action", "llm_status", "guardrail_allowed", "final_action", "guardrail_reason", "execution_status", "recovered_amount")})
    else:
        st.info("Run this payment to view its RAG, LLM, guardrail, and simulated execution trail.")

if latest:
    with st.expander("RAG context and LLM reasoning", expanded=False):
        # The audit record retains every retrieved chunk. Deduplication is display-only.
        displayed_sources = list(dict.fromkeys(source for source in latest.get("rag_sources", []) if source))
        st.markdown("**RAG sources:** " + ", ".join(displayed_sources))
        reasoning = latest.get("llm_response", latest.get("llm_explanation", ""))
        st.markdown(f'<div class="llm-reasoning">{html.escape(reasoning)}</div>', unsafe_allow_html=True)
        if latest.get("llm_error"):
            st.warning("Gemini fallback was used: " + latest["llm_error"])

with st.expander("Audit trail", expanded=False):
    st.dataframe(pd.DataFrame(audit[::-1]), use_container_width=True, hide_index=True)
