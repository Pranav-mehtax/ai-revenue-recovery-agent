"""Model evaluation report for the AI Revenue Recovery Agent.

Run from anywhere:   python ml/evaluate_model.py

It rebuilds the SAME customer-level train / validation / test split as
ml/train_model.py, evaluates the saved production model (ml/recovery_model.joblib)
against the other candidate models and simple baselines, and answers four questions:

  1. Why this model?                    -> model comparison, calibration
  2. Which features matter most?        -> permutation importance, feature-group
                                           importance, retrain-without ablations
  3. Is "amount x probability" a good   -> value captured when payments are worked
     way to set priority?                  in different orders, tier analysis
  4. Is anything leaking?               -> merchant_recovery_rate diagnostic

Outputs (all under evaluation/):
  figures/*.png, *.csv tables, model_report.md, model_report.json
Every number is computed on held-out customers the model never saw in training.
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA, EVAL = ROOT / "data", ROOT / "evaluation"
FIG = EVAL / "figures"
MODEL_PATH = ROOT / "ml" / "recovery_model.joblib"
RANDOM_STATE = 42
TARGET = "recovered"

CATEGORICAL = ["payment_method", "failure_type", "merchant_category"]
NUMERIC = ["amount", "retry_count", "total_transactions", "successful_transactions",
           "failed_transactions", "avg_transaction_amount", "customer_success_rate",
           "merchant_success_rate", "merchant_recovery_rate", "merchant_avg_order_value",
           "hour_of_day", "day_of_week", "is_weekend"]
FEATURES = ["amount", "payment_method", "failure_type", "retry_count", "total_transactions",
            "successful_transactions", "failed_transactions", "avg_transaction_amount",
            "customer_success_rate", "merchant_category", "merchant_success_rate",
            "merchant_recovery_rate", "merchant_avg_order_value", "hour_of_day",
            "day_of_week", "is_weekend"]
GROUPS = {
    "Payment details": ["amount", "payment_method", "failure_type", "retry_count"],
    "Customer history": ["total_transactions", "successful_transactions", "failed_transactions",
                         "avg_transaction_amount", "customer_success_rate"],
    "Merchant profile": ["merchant_category", "merchant_success_rate", "merchant_recovery_rate",
                         "merchant_avg_order_value"],
    "Time of day and week": ["hour_of_day", "day_of_week", "is_weekend"],
}
# Priority rule used by ml/priority_engine.py
HIGH_CUT, MEDIUM_CUT = 66, 33

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
PALETTE = {"Logistic Regression": "#4c78a8", "Random Forest": "#f58518", "XGBoost": "#54a24b",
           "Baseline: failure-type average": "#9d9d9d", "Baseline: overall average": "#cfcfcf"}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------
# Data + the same splits as train_model.py
# ---------------------------------------------------------------------
def load():
    df = pd.read_csv(DATA / "ml_dataset.csv")
    ids = pd.read_csv(DATA / "payments.csv")[["payment_id", "customer_id", "merchant_id"]]
    df = df.merge(ids, on="payment_id", how="left", validate="one_to_one")
    if df[TARGET].isna().any():
        raise SystemExit("ml_dataset.csv has missing 'recovered' labels; fix feature_engineering first.")
    return df


def split(df):
    X, y, g = df[FEATURES], df[TARGET], df["customer_id"]
    tv, te = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE).split(X, y, groups=g))
    tr_rel, va_rel = next(GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=RANDOM_STATE)
                          .split(X.iloc[tv], y.iloc[tv], groups=g.iloc[tv]))
    return tv[tr_rel], tv[va_rel], te


def make_pipeline(estimator, numeric=NUMERIC, categorical=CATEGORICAL):
    parts = []
    if numeric:
        parts.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), list(numeric)))
    if categorical:
        parts.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                               ("onehot", OneHotEncoder(handle_unknown="ignore"))]), list(categorical)))
    return Pipeline([("preprocessor", ColumnTransformer(parts)), ("model", estimator)])


def candidates():  # mirrors ml/train_model.py
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=10, reg_lambda=1.0,
                                 objective="binary:logistic", eval_metric="logloss",
                                 random_state=RANDOM_STATE, n_jobs=4),
    }


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def score(y, p, prior):
    y = np.asarray(y)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    brier = brier_score_loss(y, p)
    return {"auc": roc_auc_score(y, p), "log_loss": log_loss(y, p), "brier": brier,
            "brier_skill": 1 - brier / np.mean((y - prior) ** 2), "ece": ece(y, p),
            "precision@0.5": precision_score(y, pred, zero_division=0),
            "recall@0.5": recall_score(y, pred, zero_division=0),
            "f1@0.5": f1_score(y, pred, zero_division=0), "accuracy@0.5": accuracy_score(y, pred)}


def pct(x):
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------
def main():
    EVAL.mkdir(exist_ok=True)
    df = load()
    tr, va, te = split(df)
    X, y = df[FEATURES], df[TARGET].to_numpy()
    Xtr, Xva, Xte = X.iloc[tr], X.iloc[va], X.iloc[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    prior = float(ytr.mean())
    print(f"rows={len(df)} train={len(tr)} val={len(va)} test={len(te)} base recovery rate={prior:.3f}")

    # ---- 1. models + baselines ---------------------------------------
    probs = {"val": {}, "test": {}}
    fitted = {}
    for name, est in candidates().items():
        pipe = make_pipeline(est).fit(Xtr, ytr)
        fitted[name] = pipe
        probs["val"][name] = pipe.predict_proba(Xva)[:, 1]
        probs["test"][name] = pipe.predict_proba(Xte)[:, 1]
        print("trained", name)
    rates = pd.Series(ytr).groupby(Xtr["failure_type"].to_numpy()).mean()
    for split_name, Xs in (("val", Xva), ("test", Xte)):
        probs[split_name]["Baseline: failure-type average"] = Xs["failure_type"].map(rates).fillna(prior).to_numpy()
        probs[split_name]["Baseline: overall average"] = np.full(len(Xs), prior)

    rows = []
    for split_name, ys in (("validation", yva), ("test", yte)):
        key = "val" if split_name == "validation" else "test"
        for name, p in probs[key].items():
            rows.append({"model": name, "split": split_name, **score(ys, p, prior)})
    metrics_df = pd.DataFrame(rows)

    prod, prod_name, prod_source = None, None, "retrained (no saved model found)"
    if MODEL_PATH.exists():
        try:
            prod = joblib.load(MODEL_PATH)
            cls = type(prod.named_steps["model"]).__name__
            prod_name = {"LogisticRegression": "Logistic Regression", "RandomForestClassifier": "Random Forest",
                         "XGBClassifier": "XGBoost"}.get(cls, cls)
            prod_source = "ml/recovery_model.joblib"
        except Exception as error:
            print("could not load saved model:", error)
    model_rows = metrics_df[(metrics_df.split == "validation") & metrics_df.model.isin(candidates())]
    best_by_val = model_rows.sort_values("log_loss").iloc[0]["model"]
    best_by_test_auc = metrics_df[(metrics_df.split == "test") & metrics_df.model.isin(candidates())] \
        .sort_values("auc", ascending=False).iloc[0]["model"]
    if prod is None:
        prod_name, prod = best_by_val, fitted[best_by_val]
    p_prod = prod.predict_proba(Xte)[:, 1]
    prod_test = score(yte, p_prod, prior)
    metrics_df.to_csv(EVAL / "model_metrics_full.csv", index=False)

    # Figure: comparison
    test_df = metrics_df[metrics_df.split == "test"].set_index("model")
    order = list(test_df.index)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (col, title) in zip(axes, [("auc", "ROC-AUC (higher is better)"),
                                       ("log_loss", "Log loss (lower is better)"),
                                       ("brier", "Brier score (lower is better)")]):
        vals = test_df[col].reindex(order)
        bars = ax.barh(range(len(order)), vals.values, color=[PALETTE.get(m, "#888") for m in order])
        for i, (m, v) in enumerate(vals.items()):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=9, fontweight="bold" if m == prod_name else None)
            if m == prod_name:
                bars[i].set_edgecolor("black"); bars[i].set_linewidth(2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order if ax is axes[0] else [""] * len(order))
        ax.invert_yaxis(); ax.set_title(title, fontsize=10)
        ax.set_xlim(0, vals.max() * 1.18)  # bars always start at zero so gaps are not exaggerated
    fig.suptitle(f"Held-out test customers. Outlined bar = model in production ({prod_name})", y=1.02)
    save(fig, "01_model_comparison.png")

    # Figure: ROC
    fig, ax = plt.subplots(figsize=(5.8, 5))
    for name, p in probs["test"].items():
        if name == "Baseline: overall average":
            continue
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, color=PALETTE.get(name), lw=2.4 if name == prod_name else 1.4,
                label=f"{name} (AUC {test_df.loc[name, 'auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test)")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "02_roc_curves.png")

    # Figure: calibration
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for name in candidates():
        frac, mean_p = calibration_curve(yte, probs["test"][name], n_bins=10, strategy="quantile")
        a1.plot(mean_p, frac, "o-", color=PALETTE[name], lw=2.4 if name == prod_name else 1.2,
                label=f"{name} (ECE {test_df.loc[name, 'ece']:.3f})")
    a1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    a1.set(xlabel="Predicted recovery probability", ylabel="Actual recovery rate",
           title="Calibration: does 60% mean 60%? (test)")
    a1.legend(fontsize=8)
    a2.hist(p_prod, bins=30, color=PALETTE.get(prod_name, "#54a24b"), alpha=0.85)
    a2.axvline(p_prod.mean(), color="black", ls="--", label=f"mean prediction {p_prod.mean():.2f}")
    a2.axvline(yte.mean(), color="red", ls=":", label=f"actual rate {yte.mean():.2f}")
    a2.set(xlabel="Predicted probability", ylabel="Payments", title=f"Predictions of {prod_name} (test)")
    a2.legend(fontsize=8)
    save(fig, "03_calibration.png")

    # ---- 2. feature importance ---------------------------------------
    perm = permutation_importance(prod, Xte, yte, scoring="roc_auc", n_repeats=10,
                                  random_state=RANDOM_STATE, n_jobs=1)
    imp = pd.DataFrame({"feature": FEATURES, "auc_drop": perm.importances_mean, "std": perm.importances_std}) \
        .sort_values("auc_drop", ascending=False)
    imp.to_csv(EVAL / "feature_importance.csv", index=False)
    top = imp.head(16).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.barh(top.feature, top.auc_drop, xerr=top["std"], color="#4c78a8")
    ax.set(xlabel="Drop in test ROC-AUC when the feature is shuffled",
           title=f"Which features matter most? ({prod_name}, permutation importance)")
    save(fig, "04_feature_importance.png")

    base_auc = roc_auc_score(yte, p_prod)
    rng = np.random.default_rng(RANDOM_STATE)
    group_rows = []
    for gname, cols in GROUPS.items():
        drops = []
        for _ in range(10):
            Xp, perm_idx = Xte.copy(), rng.permutation(len(Xte))
            for c in cols:
                Xp[c] = Xte[c].to_numpy()[perm_idx]
            drops.append(base_auc - roc_auc_score(yte, prod.predict_proba(Xp)[:, 1]))
        group_rows.append({"group": gname, "auc_drop": np.mean(drops), "std": np.std(drops)})
    groups_df = pd.DataFrame(group_rows).sort_values("auc_drop", ascending=False)

    # ablation: retrain without a feature group
    est = clone(prod.named_steps["model"])
    ablate = {"All features (production)": []}
    ablate.update({f"Without: {g}": cols for g, cols in GROUPS.items()})
    ablate["Without merchant_recovery_rate only"] = ["merchant_recovery_rate"]
    ablate["Without both merchant rates"] = ["merchant_success_rate", "merchant_recovery_rate"]
    abl_rows = []
    for label, drop in ablate.items():
        num = [c for c in NUMERIC if c not in drop]
        cat = [c for c in CATEGORICAL if c not in drop]
        pipe = make_pipeline(clone(est), num, cat).fit(Xtr[num + cat], ytr)
        pp = pipe.predict_proba(Xte[num + cat])[:, 1]
        s = score(yte, pp, prior)
        abl_rows.append({"variant": label, "test_auc": s["auc"], "log_loss": s["log_loss"], "brier": s["brier"]})
    abl = pd.DataFrame(abl_rows)
    abl["auc_change"] = abl.test_auc - abl.test_auc.iloc[0]
    abl.to_csv(EVAL / "ablation.csv", index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6))
    g = groups_df.iloc[::-1]
    a1.barh(g.group, g.auc_drop, xerr=g["std"], color="#f58518")
    a1.set(xlabel="AUC drop when the whole group is shuffled", title="Feature groups: shuffle test")
    a = abl.iloc[1:].iloc[::-1]
    a2.barh(a.variant, a.auc_change, color=["#e45756" if v < 0 else "#54a24b" for v in a.auc_change])
    a2.axvline(0, color="black", lw=1)
    a2.set(xlabel="Change in test AUC vs the full model", title="Retrain without the feature(s)")
    save(fig, "05_feature_groups_and_ablation.png")

    # leakage diagnostic for merchant_recovery_rate
    obs = df.groupby("merchant_id")[TARGET].agg(["mean", "count"])
    col = df.groupby("merchant_id")["merchant_recovery_rate"].first()
    diff = (col - obs["mean"]).abs()
    noise = float(np.sqrt(obs["mean"] * (1 - obs["mean"]) / obs["count"]).mean())
    leak = {"merchants": int(len(col)), "mean_abs_diff_vs_observed": float(diff.mean()),
            "correlation": float(col.corr(obs["mean"])), "expected_sampling_noise": noise,
            "looks_computed_from_these_outcomes": bool(diff.mean() < 0.25 * noise)}

    # ---- calibration in the large + by group --------------------------
    def grp(col_name):
        t = pd.DataFrame({"g": Xte[col_name].to_numpy(), "y": yte, "p": p_prod})
        r = t.groupby("g").agg(payments=("y", "size"), actual_rate=("y", "mean"), predicted_rate=("p", "mean"))
        return r.reset_index().rename(columns={"g": col_name})
    by_type, by_retry = grp("failure_type"), grp("retry_count")
    by_type.to_csv(EVAL / "by_failure_type.csv", index=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    for ax, t, c in ((a1, by_type.sort_values("actual_rate"), "failure_type"), (a2, by_retry, "retry_count")):
        pos = np.arange(len(t))
        ax.bar(pos - 0.2, t.actual_rate, 0.4, label="Actual", color="#4c78a8")
        ax.bar(pos + 0.2, t.predicted_rate, 0.4, label="Model", color="#f58518")
        ax.set_xticks(pos); ax.set_xticklabels(t[c].astype(str), rotation=30 if c == "failure_type" else 0, ha="right" if c == "failure_type" else "center")
        ax.set(ylabel="Recovery rate", title=f"By {c.replace('_', ' ')} (test)"); ax.legend(fontsize=8)
    save(fig, "06_actual_vs_predicted_by_segment.png")

    # ---- 3. priority validation --------------------------------------
    amount = Xte["amount"].to_numpy(float)
    ev = amount * p_prod
    actual_value = amount * yte
    rankings = {"Expected value: amount x probability (current)": ev,
                "Probability only": p_prod, "Amount only": amount, "Oracle (knows the outcome)": actual_value}
    n, total_actual = len(yte), actual_value.sum()
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    cap_rows, grid = [], np.arange(1, n + 1) / n
    styles = {"Expected value: amount x probability (current)": ("#54a24b", 2.8), "Probability only": ("#f58518", 1.6),
              "Amount only": ("#4c78a8", 1.6), "Oracle (knows the outcome)": ("#222222", 1.0)}
    for name, s in rankings.items():
        cum = np.cumsum(actual_value[np.argsort(-s, kind="stable")]) / total_actual
        ax.plot(np.r_[0, grid], np.r_[0, cum], color=styles[name][0], lw=styles[name][1],
                ls="--" if name.startswith("Oracle") else "-", label=name)
        cap_rows.append({"ranking": name, **{f"top_{int(f * 100)}%": float(cum[max(int(f * n) - 1, 0)]) for f in (0.1, 0.2, 0.3, 0.5)}})
    ax.plot([0, 1], [0, 1], ":", color="gray", label="Random order")
    cap_rows.append({"ranking": "Random order", **{f"top_{int(f * 100)}%": f for f in (0.1, 0.2, 0.3, 0.5)}})
    ax.set(xlabel="Share of failed payments worked (highest priority first)",
           ylabel="Share of recoverable rupees captured", title="Does the priority order find the money? (test)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "07_priority_capture_curve.png")
    capture = pd.DataFrame(cap_rows)
    capture.to_csv(EVAL / "priority_capture.csv", index=False)

    score_max = ev / ev.max() * 100
    cur = np.where(score_max >= HIGH_CUT, "HIGH", np.where(score_max >= MEDIUM_CUT, "MEDIUM", "LOW"))
    score_p99 = np.clip(ev / np.percentile(ev, 99), 0, 1) * 100
    p99 = np.where(score_p99 >= HIGH_CUT, "HIGH", np.where(score_p99 >= MEDIUM_CUT, "MEDIUM", "LOW"))
    rank = pd.Series(ev).rank(ascending=False, pct=True).to_numpy()
    rk = np.where(rank <= 0.10, "HIGH", np.where(rank <= 0.40, "MEDIUM", "LOW"))
    schemes = {"Current: divide by the maximum (66/33)": cur, "Divide by 99th percentile (66/33)": p99,
               "Rank-based: top 10% / next 30%": rk}
    tier_rows = []
    for sname, lv in schemes.items():
        for lvl in ("HIGH", "MEDIUM", "LOW"):
            m = lv == lvl
            tier_rows.append({"scheme": sname, "level": lvl, "payments": int(m.sum()), "share_of_payments": m.mean(),
                              "failed_rupees": amount[m].sum(), "expected_recovery": ev[m].sum(),
                              "actual_recovered": actual_value[m].sum(),
                              "share_of_actual_recovered": actual_value[m].sum() / total_actual,
                              "avg_actual_per_payment": actual_value[m].mean() if m.any() else 0.0,
                              "avg_expected_per_payment": ev[m].mean() if m.any() else 0.0})
    tier_df = pd.DataFrame(tier_rows)
    tier_df.to_csv(EVAL / "priority_tiering.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    sample = np.random.default_rng(1).choice(n, size=min(n, 3000), replace=False)
    for lvl, colr in (("LOW", "#9d9d9d"), ("MEDIUM", "#f58518"), ("HIGH", "#e45756")):
        m = sample[cur[sample] == lvl]
        axes[0].scatter(amount[m], p_prod[m], s=9, alpha=0.5, color=colr, label=f"{lvl} ({(cur == lvl).sum()})")
    axes[0].set_xscale("log"); axes[0].legend(fontsize=8, title="Current tiers")
    axes[0].set(xlabel="Payment amount (Rs, log scale)", ylabel="Recovery probability", title="Where tiers fall")
    cur_t = tier_df[tier_df.scheme == list(schemes)[0]].set_index("level").reindex(["HIGH", "MEDIUM", "LOW"])
    axes[1].bar(cur_t.index, cur_t.share_of_payments * 100, color=["#e45756", "#f58518", "#9d9d9d"])
    for i, v in enumerate(cur_t.share_of_payments * 100):
        axes[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set(ylabel="% of failed payments", title="Current rule: how payments are split")
    rk_t = tier_df[tier_df.scheme == list(schemes)[2]].set_index("level").reindex(["HIGH", "MEDIUM", "LOW"])
    pos = np.arange(3)
    axes[2].bar(pos - 0.2, rk_t.avg_expected_per_payment, 0.4, color="#f58518", label="Expected (model)")
    axes[2].bar(pos + 0.2, rk_t.avg_actual_per_payment, 0.4, color="#4c78a8", label="Actual")
    axes[2].set_xticks(pos); axes[2].set_xticklabels(rk_t.index); axes[2].legend(fontsize=8)
    axes[2].set(ylabel="Rs recovered per payment", title="Rank-based tiers: expected vs actual")
    save(fig, "08_priority_tiers.png")

    # ---- summary -------------------------------------------------------
    cap20 = capture.set_index("ranking")["top_20%"]
    cur_tiers = tier_df[tier_df.scheme == list(schemes)[0]].set_index("level")
    summary = {
        "dataset": {"failed_payments": int(len(df)), "train": int(len(tr)), "validation": int(len(va)),
                    "test": int(len(te)), "customers": int(df.customer_id.nunique()),
                    "base_recovery_rate": prior, "split": "by customer (GroupShuffleSplit), 70/15/15"},
        "production_model": {"name": prod_name, "source": prod_source, "test": prod_test},
        "selection": {"best_by_validation_log_loss": best_by_val, "best_by_test_auc": best_by_test_auc,
                      "train_model_py_rule": "highest test ROC-AUC"},
        "calibration_in_the_large": {"expected_recovered_rupees": float(ev.sum()),
                                     "actual_recovered_rupees": float(total_actual),
                                     "ratio_expected_over_actual": float(ev.sum() / total_actual)},
        "top_features": imp.head(5)[["feature", "auc_drop"]].to_dict("records"),
        "group_importance": groups_df.to_dict("records"),
        "ablation": abl.to_dict("records"),
        "leakage_check_merchant_recovery_rate": leak,
        "priority_capture_top20": {k: float(v) for k, v in cap20.items()},
        "current_tier_counts": {lvl: int(cur_tiers.loc[lvl, "payments"]) for lvl in ("HIGH", "MEDIUM", "LOW")},
        "figures": sorted(p.name for p in FIG.glob("*.png")),
    }
    (EVAL / "model_report.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    write_markdown(summary, metrics_df, imp, groups_df, abl, capture, tier_df, schemes, by_type)
    print("\nDone. Open evaluation/model_report.md and evaluation/figures/")
    return summary


def write_markdown(s, metrics_df, imp, groups_df, abl, capture, tier_df, schemes, by_type):
    t = metrics_df[metrics_df.split == "test"].set_index("model")
    v = metrics_df[metrics_df.split == "validation"].set_index("model")
    prod = s["production_model"]
    d = s["dataset"]
    cal = s["calibration_in_the_large"]
    cap = capture.set_index("ranking")
    leak = s["leakage_check_merchant_recovery_rate"]
    lines = [
        "# Model evaluation report", "",
        f"Data: {d['failed_payments']:,} failed payments from {d['customers']:,} customers, split **by customer** "
        f"({d['train']:,} train / {d['validation']:,} validation / {d['test']:,} test) so the model is always scored on "
        f"customers it never saw. Overall recovery rate: {pct(d['base_recovery_rate'])}.", "",
        "## 1. Why this model", "",
        "| Model | Val AUC | Test AUC | Test log loss | Test Brier | Calibration error (ECE) |", "|---|---|---|---|---|---|"]
    for m in t.index:
        lines.append(f"| {m} | {v.loc[m, 'auc']:.3f} | {t.loc[m, 'auc']:.3f} | {t.loc[m, 'log_loss']:.3f} | "
                     f"{t.loc[m, 'brier']:.3f} | {t.loc[m, 'ece']:.3f} |")
    lines += ["",
              f"Production model: **{prod['name']}** (`{prod['source']}`), test AUC {prod['test']['auc']:.3f}, "
              f"Brier skill {pct(prod['test']['brier_skill'])} better than guessing the average rate.", "",
              "Probabilities, not just rankings, feed the money estimate (expected recovery = amount x probability), "
              "so log loss, Brier score and calibration matter as much as AUC.", ""]
    sel = s["selection"]
    if sel["best_by_validation_log_loss"] != sel["best_by_test_auc"]:
        lines += [f"> Note: `train_model.py` picks the highest **test** AUC ({sel['best_by_test_auc']}), but ranking by "
                  f"**validation** log loss picks {sel['best_by_validation_log_loss']}. Choosing on the test set makes the "
                  "reported test score slightly optimistic; select on validation and report test once.", ""]
    else:
        lines += [f"Selecting on validation log loss and on test AUC agree: {sel['best_by_test_auc']}. "
                  "(`train_model.py` selects on the test set; selecting on validation is the safer habit.)", ""]
    lines += ["## 2. Which features matter most", "", "Top features by permutation importance (drop in test AUC when shuffled):", ""]
    lines += [f"{i}. `{r['feature']}`: {r['auc_drop']:.4f}" for i, r in enumerate(imp.head(6).to_dict('records'), 1)]
    lines += ["", "By feature group (shuffle test, and AUC change when retrained without the group):", "",
              "| Group | Columns | Shuffle AUC drop |", "|---|---|---|"]
    lines += [f"| {r['group']} | {', '.join(GROUPS[r['group']])} | {r['auc_drop']:.4f} |" for r in groups_df.to_dict("records")]
    lines += ["", "| Retrained without | Test AUC | Change |", "|---|---|---|"]
    lines += [f"| {r['variant']} | {r['test_auc']:.3f} | {r['auc_change']:+.3f} |" for r in abl.to_dict("records")]
    lines += ["", "## 3. Why priority = amount x probability", "",
              f"Share of all recoverable rupees captured by working only the top 20% of payments (test set):", "",
              "| Ordering | Top 10% | Top 20% | Top 30% |", "|---|---|---|---|"]
    lines += [f"| {k} | {pct(r['top_10%'])} | {pct(r['top_20%'])} | {pct(r['top_30%'])} |" for k, r in cap.iterrows()]
    lines += ["", f"Expected vs actual recovered rupees on the test set: Rs {cal['expected_recovered_rupees']:,.0f} expected, "
              f"Rs {cal['actual_recovered_rupees']:,.0f} actual (ratio {cal['ratio_expected_over_actual']:.2f}; 1.00 is perfect).", "",
              "### How the HIGH / MEDIUM / LOW tiers behave", "",
              "| Scheme | HIGH | MEDIUM | LOW | Share of recovered Rs in HIGH |", "|---|---|---|---|---|"]
    for sname in schemes:
        sub = tier_df[tier_df.scheme == sname].set_index("level")
        lines.append(f"| {sname} | {sub.loc['HIGH', 'payments']:,} | {sub.loc['MEDIUM', 'payments']:,} | "
                     f"{sub.loc['LOW', 'payments']:,} | {pct(sub.loc['HIGH', 'share_of_actual_recovered'])} |")
    lines += ["", "`priority_engine.py` divides every expected value by the single largest one. Because payment amounts "
              "are heavy-tailed, one large payment can push almost everyone else into LOW; the table shows whether that "
              "happens on your data and what the alternatives would give.", "",
              "## 4. Leakage check: `merchant_recovery_rate`", "",
              f"Across {leak['merchants']} merchants, the column differs from the observed recovery rate of the same "
              f"payments by {leak['mean_abs_diff_vs_observed']:.4f} on average (correlation {leak['correlation']:.3f}); "
              f"pure sampling noise alone would give about {leak['expected_sampling_noise']:.4f}.", ""]
    lines.append("**Warning: the column matches these outcomes far more closely than chance, so it was most likely "
                 "computed from the same recoveries the model predicts. Compare the ablation row above; if the AUC falls a lot "
                 "without it, recompute it from training data only.**" if leak["looks_computed_from_these_outcomes"]
                 else "The match is within what a genuine per-merchant statistic would show, so there is no sign it was "
                      "computed from these outcomes.")
    lines += ["", "## Limitations", "",
              "- The data is synthetic; these numbers describe the system, not real Razorpay traffic.",
              "- Customer history columns look like whole-period aggregates, which can include later transactions. "
              "A time-based split would be a stricter test.",
              "- Recovery outcomes are single labelled events, so uplift from actually acting is not measured.", ""]
    (EVAL / "model_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()