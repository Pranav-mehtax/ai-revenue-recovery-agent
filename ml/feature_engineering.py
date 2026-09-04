import pandas as pd


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

payments = pd.read_csv(
    "data/payments.csv"
)

customers = pd.read_csv(
    "data/customers.csv"
)

merchants = pd.read_csv(
    "data/merchants.csv"
)

recovery = pd.read_csv(
    "data/recovery_outcomes.csv"
)


# ============================================================
# KEEP ONLY FAILED PAYMENTS
# ============================================================

failed_payments = payments[
    payments["status"] == "FAILED"
].copy()

print(
    f"Failed payments: {len(failed_payments)}"
)


# ============================================================
# CUSTOMER FEATURES
# ============================================================

customers["customer_success_rate"] = (
    customers["successful_transactions"]
    /
    customers["total_transactions"]
)

customer_features = customers[
    [
        "customer_id",
        "total_transactions",
        "successful_transactions",
        "failed_transactions",
        "avg_transaction_amount",
        "customer_success_rate"
    ]
].copy()


# ============================================================
# MERCHANT FEATURES
# ============================================================

merchant_features = merchants[
    [
        "merchant_id",
        "merchant_category",
        "merchant_success_rate",
        "merchant_recovery_rate",
        "merchant_avg_order_value"
    ]
].copy()


# ============================================================
# MERGE PAYMENT + CUSTOMER
# ============================================================

df = failed_payments.merge(
    customer_features,
    on="customer_id",
    how="left"
)


# ============================================================
# MERGE PAYMENT + MERCHANT
# ============================================================

df = df.merge(
    merchant_features,
    on="merchant_id",
    how="left"
)


# ============================================================
# MERGE RECOVERY OUTCOME
# ============================================================

df = df.merge(
    recovery[
        [
            "payment_id",
            "recovered"
        ]
    ],
    on="payment_id",
    how="left"
)


# ============================================================
# TIME FEATURES
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df["hour_of_day"] = (
    df["timestamp"].dt.hour
)

df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ============================================================
# ML FEATURES
# ============================================================

features = [

    # -------------------------
    # Payment features
    # -------------------------

    "amount",

    "payment_method",

    "failure_type",

    "retry_count",


    # -------------------------
    # Customer features
    # -------------------------

    "total_transactions",

    "successful_transactions",

    "failed_transactions",

    "avg_transaction_amount",

    "customer_success_rate",


    # -------------------------
    # Merchant features
    # -------------------------

    "merchant_category",

    "merchant_success_rate",

    "merchant_recovery_rate",

    "merchant_avg_order_value",


    # -------------------------
    # Time features
    # -------------------------

    "hour_of_day",

    "day_of_week",

    "is_weekend"

]


target = "recovered"


# ============================================================
# CREATE ML DATASET
# ============================================================

# payment_id is retained ONLY so that the training pipeline
# can determine which customer each payment belongs to.
#
# payment_id will NOT be used as an ML feature.

ml_data = df[
    ["payment_id"] + features + [target]
].copy()


# ============================================================
# VALIDATION
# ============================================================

print("\n======================================")
print("FEATURE ENGINEERING COMPLETE")
print("======================================")

print(
    f"Rows: {len(ml_data)}"
)

print(
    f"Features: {len(features)}"
)

print("\nFeature columns:")

for feature in features:

    print(
        f"  - {feature}"
    )


print("\nMissing values:")

print(
    ml_data[
        features + [target]
    ].isnull().sum()
)


print("\nTarget distribution:")

print(
    ml_data["recovered"]
    .value_counts(
        normalize=True
    )
)


# ============================================================
# SAVE DATASET
# ============================================================

ml_data.to_csv(
    "data/ml_dataset.csv",
    index=False
)

print("\nSaved:")
print("data/ml_dataset.csv")