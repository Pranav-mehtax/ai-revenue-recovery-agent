import pandas as pd
import numpy as np
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "ml/recovery_model.joblib"

INPUT_PATH = "data/ml_dataset.csv"

OUTPUT_PATH = "data/prioritized_payments.csv"


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("Loading recovery model...")

model = joblib.load(
    MODEL_PATH
)

print("Recovery model loaded.")


# ============================================================
# 2. LOAD FAILED PAYMENT DATA
# ============================================================

print("\nLoading failed payment dataset...")

df = pd.read_csv(
    INPUT_PATH
)

print(
    f"Payments loaded: {len(df)}"
)


# ============================================================
# 3. DEFINE ML FEATURES
# ============================================================

feature_columns = [

    "amount",

    "payment_method",

    "failure_type",

    "retry_count",

    "total_transactions",

    "successful_transactions",

    "failed_transactions",

    "avg_transaction_amount",

    "customer_success_rate",

    "merchant_category",

    "merchant_success_rate",

    "merchant_recovery_rate",

    "merchant_avg_order_value",

    "hour_of_day",

    "day_of_week",

    "is_weekend"

]


X = df[
    feature_columns
]


# ============================================================
# 4. PREDICT RECOVERY PROBABILITY
# ============================================================

print("\nPredicting recovery probabilities...")

df["recovery_probability"] = (

    model.predict_proba(X)[:, 1]

)


# ============================================================
# 5. EXPECTED RECOVERY VALUE
# ============================================================

# Expected recovery is:
#
# Amount × Probability of recovery

df["expected_recovery_value"] = (

    df["amount"]

    *

    df["recovery_probability"]

)


# ============================================================
# 6. PRIORITY SCORE
# ============================================================

# For the first version, expected recovery value
# is our primary economic signal.
#
# We normalize it to 0-100 so that it is easier
# to display in the dashboard.

max_expected_value = (

    df["expected_recovery_value"].max()

)


if max_expected_value > 0:

    df["priority_score"] = (

        df["expected_recovery_value"]

        /

        max_expected_value

        *

        100

    )

else:

    df["priority_score"] = 0


# ============================================================
# 7. PRIORITY LEVEL
# ============================================================

def assign_priority(score):

    if score >= 66:

        return "HIGH"

    elif score >= 33:

        return "MEDIUM"

    else:

        return "LOW"


df["priority_level"] = (

    df["priority_score"]

    .apply(assign_priority)

)


# ============================================================
# 8. RECOMMENDED ACTION
# ============================================================

def recommend_action(row):

    failure = row["failure_type"]

    probability = row["recovery_probability"]


    # Very low recovery probability
    # should generally not receive aggressive recovery actions.

    if probability < 0.15:

        return "NO_ACTION"


    # Network problems are usually transient.

    if failure in [

        "NETWORK_ERROR",

        "TIMEOUT"

    ]:

        return "RETRY"


    # Expired card needs a new payment method.

    if failure == "EXPIRED_CARD":

        return "PAYMENT_LINK"


    # Authentication issues can be resolved
    # by asking the customer to complete payment again.

    if failure == "AUTHENTICATION_FAILURE":

        return "PAYMENT_LINK"


    # Insufficient funds should generally be handled
    # through a customer notification rather than
    # repeated immediate retries.

    if failure == "INSUFFICIENT_FUNDS":

        return "NOTIFICATION"


    # Duplicate payments should not be retried blindly.

    if failure == "DUPLICATE":

        return "NO_ACTION"


    # Suspected fraud should not be automatically retried.

    if failure == "SUSPECTED_FRAUD":

        return "ESCALATE"


    return "RETRY"


df["recommended_action"] = (

    df.apply(
        recommend_action,
        axis=1
    )

)


# ============================================================
# 9. SORT BY PRIORITY
# ============================================================

df = df.sort_values(

    by="expected_recovery_value",

    ascending=False

).reset_index(
    drop=True
)


# ============================================================
# 10. ADD RANK
# ============================================================

df["priority_rank"] = (

    df.index + 1

)


# ============================================================
# 11. SELECT OUTPUT COLUMNS
# ============================================================

output_columns = [

    "payment_id",

    "amount",

    "payment_method",

    "failure_type",

    "retry_count",

    "customer_success_rate",

    "merchant_success_rate",

    "merchant_recovery_rate",

    "recovery_probability",

    "expected_recovery_value",

    "priority_score",

    "priority_level",

    "recommended_action"

]


result = df[
    output_columns
].copy()


# ============================================================
# 12. ROUND VALUES
# ============================================================

result["recovery_probability"] = (

    result[
        "recovery_probability"
    ].round(4)

)


result["expected_recovery_value"] = (

    result[
        "expected_recovery_value"
    ].round(2)

)


result["priority_score"] = (

    result[
        "priority_score"
    ].round(2)

)


# ============================================================
# 13. SAVE
# ============================================================

result.to_csv(

    OUTPUT_PATH,

    index=False

)


# ============================================================
# 14. SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("PRIORITY ENGINE COMPLETE")
print("=" * 70)


print(
    f"Total payments: {len(result)}"
)


print("\nPriority distribution:")

print(
    result[
        "priority_level"
    ].value_counts()
)


print("\nRecommended actions:")

print(
    result[
        "recommended_action"
    ].value_counts()
)


print("\nTotal payment value:")

print(
    f"₹{result['amount'].sum():,.2f}"
)


print("\nTotal expected recovery value:")

print(

    f"₹{result['expected_recovery_value'].sum():,.2f}"

)


print("\nTop 10 recovery opportunities:")

print(

    result[
        [
            "payment_id",
            "amount",
            "recovery_probability",
            "expected_recovery_value",
            "priority_level",
            "recommended_action"
        ]
    ].head(10).to_string(index=False)

)


print("\nSaved:")

print(
    OUTPUT_PATH
)