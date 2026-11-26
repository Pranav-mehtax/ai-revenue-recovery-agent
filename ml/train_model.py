import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    log_loss,
    brier_score_loss,
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42


# ============================================================
# 1. LOAD ML DATASET
# ============================================================

print("Loading ML dataset...")

df = pd.read_csv("data/ml_dataset.csv")

print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. GET CUSTOMER IDs FOR GROUP SPLITTING
# ============================================================

# payment_id is NOT an ML feature.
#
# We use it only to find the customer associated with
# each payment so that we can split the data by customer.

payments = pd.read_csv("data/payments.csv")[
    [
        "payment_id",
        "customer_id",
    ]
]

df = df.merge(
    payments,
    on="payment_id",
    how="left",
    validate="one_to_one",
)


# ============================================================
# 3. DEFINE FEATURES
# ============================================================

feature_columns = [
    # Payment
    "amount",
    "payment_method",
    "failure_type",
    "retry_count",

    # Customer
    "total_transactions",
    "successful_transactions",
    "failed_transactions",
    "avg_transaction_amount",
    "customer_success_rate",

    # Merchant
    "merchant_category",
    "merchant_success_rate",
    "merchant_recovery_rate",
    "merchant_avg_order_value",

    # Time
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

TARGET = "recovered"


# ============================================================
# 4. CREATE X, y AND GROUPS
# ============================================================

X = df[feature_columns].copy()
y = df[TARGET].copy()
groups = df["customer_id"].copy()


# ============================================================
# 5. CHECK FOR MISSING DATA
# ============================================================

print("\nMissing values in features:")
print(X.isnull().sum())

print("\nMissing target values:")
print(y.isnull().sum())


# ============================================================
# 6. CUSTOMER-LEVEL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nCreating customer-level splits...")

# ------------------------------------------------------------
# FIRST SPLIT
#
# 85% development
# 15% completely unseen test customers
# ------------------------------------------------------------

gss_test = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=RANDOM_STATE,
)

train_val_idx, test_idx = next(
    gss_test.split(
        X,
        y,
        groups=groups,
    )
)

X_train_val = X.iloc[train_val_idx]
y_train_val = y.iloc[train_val_idx]
groups_train_val = groups.iloc[train_val_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# ------------------------------------------------------------
# SECOND SPLIT
#
# Approximately:
#
# 70% total → train
# 15% total → validation
# 15% total → test
# ------------------------------------------------------------

gss_validation = GroupShuffleSplit(
    n_splits=1,
    test_size=0.1765,
    random_state=RANDOM_STATE,
)

train_idx, val_idx = next(
    gss_validation.split(
        X_train_val,
        y_train_val,
        groups=groups_train_val,
    )
)

X_train = X_train_val.iloc[train_idx]
y_train = y_train_val.iloc[train_idx]

X_val = X_train_val.iloc[val_idx]
y_val = y_train_val.iloc[val_idx]


# ============================================================
# SPLIT SUMMARY
# ============================================================

print("\nSplit sizes:")
print(f"Training rows:   {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows:       {len(X_test)}")

print("\nTarget distribution:")
print("Training:", y_train.mean())
print("Validation:", y_val.mean())
print("Test:", y_test.mean())


# ============================================================
# 7. FEATURE TYPES
# ============================================================

categorical_features = [
    "payment_method",
    "failure_type",
    "merchant_category",
]

numeric_features = [
    "amount",
    "retry_count",
    "total_transactions",
    "successful_transactions",
    "failed_transactions",
    "avg_transaction_amount",
    "customer_success_rate",
    "merchant_success_rate",
    "merchant_recovery_rate",
    "merchant_avg_order_value",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]


# ============================================================
# 8. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features,
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)


# ============================================================
# 9. DEFINE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=1.0,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=4,
    ),
}


# ============================================================
# 10. TRAIN MODELS
# ============================================================

results = []
trained_models = {}

for name, model in models.items():
    print("\n")
    print("=" * 70)
    print(f"TRAINING: {name}")
    print("=" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # PROBABILITY PREDICTIONS
    # --------------------------------------------------------

    train_prob = pipeline.predict_proba(X_train)[:, 1]
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    test_prob = pipeline.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # CLASS PREDICTIONS
    # --------------------------------------------------------

    test_pred = (test_prob >= 0.50).astype(int)

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    train_auc = roc_auc_score(y_train, train_prob)
    validation_auc = roc_auc_score(y_val, val_prob)
    test_auc = roc_auc_score(y_test, test_prob)

    # --------------------------------------------------------
    # CLASSIFICATION METRICS
    # --------------------------------------------------------

    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    accuracy = accuracy_score(y_test, test_pred)

    # --------------------------------------------------------
    # PROBABILITY QUALITY
    # --------------------------------------------------------

    test_logloss = log_loss(y_test, test_prob)
    test_brier = brier_score_loss(y_test, test_prob)

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(f"Train ROC-AUC:       {train_auc:.4f}")
    print(f"Validation ROC-AUC:  {validation_auc:.4f}")
    print(f"Test ROC-AUC:        {test_auc:.4f}")
    print(f"Test Precision:      {precision:.4f}")
    print(f"Test Recall:         {recall:.4f}")
    print(f"Test F1:              {f1:.4f}")
    print(f"Test Accuracy:       {accuracy:.4f}")
    print(f"Test Log Loss:       {test_logloss:.4f}")
    print(f"Test Brier Score:    {test_brier:.4f}")

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({
        "model": name,
        "train_auc": train_auc,
        "validation_auc": validation_auc,
        "test_auc": test_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "log_loss": test_logloss,
        "brier_score": test_brier,
    })

    trained_models[name] = pipeline


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="test_auc",
    ascending=False,
)

print("\n")
print("=" * 90)
print("MODEL COMPARISON")
print("=" * 90)
print(results_df.to_string(index=False))


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

print("\n")
print(f"Best model based on test ROC-AUC:")
print(best_model_name)


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "ml/recovery_model.joblib",
)

print("\nSaved best model to:")
print("ml/recovery_model.joblib")


# ============================================================
# 14. SAVE ALL RESULTS
# ============================================================

results_df.to_csv(
    "data/model_comparison.csv",
    index=False,
)

print("\nSaved model comparison to:")
print("data/model_comparison.csv")

print("\n======================================")
print("MODEL TRAINING COMPLETE")
print("======================================")
