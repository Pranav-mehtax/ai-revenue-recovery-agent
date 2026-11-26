"""Model Insights: why this model, which features matter, how priority is set, and what to double-check.

Reads the files produced by:  python ml/evaluate_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "evaluation"
FIG = EVAL / "figures"

st.set_page_config(page_title="Model Insights", page_icon="₹", layout="wide")
st.markdown("""
<style>
  .stApp { background: #07141f; color: #eaf2f7; }
  [data-testid="stMetric"] { background: #102535; border: 1px solid #1d4055; padding: 14px; border-radius: 14px; }
  [data-testid="stMetricLabel"] { color: #9bb3c3; }
  .eyebrow { color: #57d4b5; font-weight: 700; letter-spacing: .12em; font-size: .75rem; }
  .notice { background: #123247; border-left: 4px solid #57d4b5; padding: 12px 16px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">RAZORPAY AI BUILDATHON · MODEL INSIGHTS</div>', unsafe_allow_html=True)
st.title("Why the model, the features and the priority rule")

report_path = EVAL / "model_report.json"
if not report_path.exists():
    st.warning("No evaluation report yet. Run this once from the project root, then refresh:")
    st.code("python ml/evaluate_model.py", language="bash")
    st.stop()

report = json.loads(report_path.read_text(encoding="utf-8"))
data, prod = report["dataset"], report["production_model"]
cal, cap = report["calibration_in_the_large"], report["priority_capture_top20"]
leak, tiers = report["leakage_check_merchant_recovery_rate"], report["current_tier_counts"]

st.markdown(
    f'<div class="notice">Every number below is measured on <b>{data["test"]:,} held-out payments from customers the model '
    f'never saw</b> (customer-level split, {data["failed_payments"]:,} failed payments in total). '
    "The data is synthetic.</div>", unsafe_allow_html=True)


def show(name: str, caption: str | None = None) -> None:
    path = FIG / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"{name} not found. Re-run python ml/evaluate_model.py")


def table(name: str, **kwargs) -> None:
    path = EVAL / name
    if path.exists():
        st.dataframe(pd.read_csv(path), use_container_width=True, hide_index=True, **kwargs)


m1, m2, m3, m4 = st.columns(4)
m1.metric("Production model", prod["name"])
m2.metric("Test ROC-AUC", f"{prod['test']['auc']:.3f}", help="0.5 is a coin flip, 1.0 is perfect ranking.")
m3.metric("Better than guessing the average", f"{prod['test']['brier_skill']:.1%}",
          help="Brier skill score: reduction in squared probability error vs always predicting the overall recovery rate.")
m4.metric("Expected / actual recovered ₹", f"{cal['ratio_expected_over_actual']:.2f}",
          help="1.00 means the model's expected recovery matches what was really recovered.")

tab_model, tab_feat, tab_prio, tab_checks = st.tabs(
    ["Why this model", "Which features matter", "How priority is set", "Data checks"])

with tab_model:
    st.subheader("Model comparison")
    show("01_model_comparison.png")
    st.markdown(
        "Expected recovery is **amount × probability**, so the probabilities themselves must be trustworthy, "
        "not just their ranking. That is why log loss, Brier score and the calibration plot count as much as AUC. "
        "The two grey bars are simple baselines a model has to beat.")
    sel = report["selection"]
    if sel["best_by_validation_log_loss"] != sel["best_by_test_auc"]:
        st.warning(f"Ranking by validation log loss picks {sel['best_by_validation_log_loss']}, but train_model.py picks "
                   f"{sel['best_by_test_auc']} using the test set. Select on validation and report test once.")
    left, right = st.columns(2)
    with left:
        show("02_roc_curves.png")
    with right:
        show("03_calibration.png")
    st.caption("On the calibration plot, points on the dotted diagonal mean a predicted 60% really recovers about 60% of the time.")
    st.subheader("All metrics")
    table("model_metrics_full.csv")

with tab_feat:
    st.subheader("Which features matter most")
    show("04_feature_importance.png")
    st.markdown("Permutation importance: shuffle one column and see how much the test AUC drops. Bigger drop, more important.")
    st.subheader("By feature group, and what happens if we remove it")
    show("05_feature_groups_and_ablation.png")
    st.markdown("Related columns (for example total, successful and failed transactions) share information, so single-feature "
                "scores understate them. The group view and the retrain-without view are the fairer comparison.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Retrain without the feature(s)**")
        table("ablation.csv")
    with right:
        st.markdown("**Does the model follow reality by segment?**")
        show("06_actual_vs_predicted_by_segment.png")

with tab_prio:
    st.subheader("Is amount × probability a good way to rank?")
    show("07_priority_capture_curve.png")
    exp = cap.get("Expected value: amount x probability (current)", 0)
    amt, prb = cap.get("Amount only", 0), cap.get("Probability only", 0)
    st.markdown(
        f"Working only the top 20% of payments in this order captures **{exp:.0%}** of all recoverable rupees, versus "
        f"{amt:.0%} ranking by amount alone and {prb:.0%} ranking by probability alone (random order captures 20%). "
        "A high amount with a hopeless chance, or a sure thing worth almost nothing, are both poor use of effort.")
    table("priority_capture.csv")
    st.subheader("How the HIGH / MEDIUM / LOW tiers behave")
    high_share = tiers["HIGH"] / max(sum(tiers.values()), 1)
    if high_share < 0.01:
        st.error(f"With the current rule only {tiers['HIGH']} of {sum(tiers.values()):,} test payments are HIGH "
                 f"({high_share:.1%}). priority_engine.py divides by the single largest expected value, so one large payment "
                 "pushes almost everything into LOW.")
    show("08_priority_tiers.png")
    table("priority_tiering.csv")

with tab_checks:
    st.subheader("Leakage check: merchant_recovery_rate")
    st.write(f"{leak['merchants']} merchants · mean gap to the observed recovery rate of the same payments: "
             f"{leak['mean_abs_diff_vs_observed']:.4f} · correlation {leak['correlation']:.3f} · "
             f"gap expected from sampling noise alone: about {leak['expected_sampling_noise']:.4f}")
    if leak["looks_computed_from_these_outcomes"]:
        st.error("This column matches the outcomes far more closely than chance, so it was most likely computed from the "
                 "same recoveries the model predicts. Recompute it from training payments only, then retrain.")
    else:
        st.success("Within normal sampling noise: no sign it was computed from these outcomes.")
    st.subheader("Other things to know")
    st.markdown(
        "- Customer history columns look like whole-period totals, which can include later transactions. "
        "A time-based split would be a stricter test.\n"
        "- Outcomes are single labelled events, so the real uplift from acting is not measured.\n"
        "- Simulated recoveries in the dashboard come from the model's own probability, so they are not evidence.")
    md = EVAL / "model_report.md"
    if md.exists():
        st.download_button("Download full report (Markdown)", md.read_text(encoding="utf-8"),
                           file_name="model_report.md", mime="text/markdown")