import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CUSTOMERS = 5000
NUM_MERCHANTS = 100
NUM_PAYMENTS = 50000

SEED = 42

np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# 1. GENERATE CUSTOMERS
# ============================================================

print("Generating customers...")

customers = []

for i in range(NUM_CUSTOMERS):

    customer_id = f"CUST_{i + 1:05d}"

    # Customer behavioural profile
    customer_type = random.choices(
        ["reliable", "normal", "risky"],
        weights=[0.50, 0.35, 0.15]
    )[0]

    if customer_type == "reliable":

        total_transactions = random.randint(15, 60)

        success_rate = np.random.uniform(
            0.90, 0.99
        )

    elif customer_type == "normal":

        total_transactions = random.randint(8, 40)

        success_rate = np.random.uniform(
            0.70, 0.90
        )

    else:

        total_transactions = random.randint(3, 30)

        success_rate = np.random.uniform(
            0.40, 0.70
        )

    successful_transactions = int(
        total_transactions * success_rate
    )

    failed_transactions = (
        total_transactions
        - successful_transactions
    )

    avg_transaction_amount = round(
        np.random.lognormal(
            mean=np.log(2500),
            sigma=0.8
        ),
        2
    )

    signup_date = (
        datetime(2024, 1, 1)
        + timedelta(
            days=random.randint(0, 700)
        )
    )

    customers.append({

        "customer_id": customer_id,

        "signup_date": signup_date.date(),

        "total_transactions": total_transactions,

        "successful_transactions":
            successful_transactions,

        "failed_transactions":
            failed_transactions,

        "avg_transaction_amount":
            avg_transaction_amount

    })


customers_df = pd.DataFrame(customers)

customers_df.to_csv(
    "data/customers.csv",
    index=False
)

print(
    f"Created {len(customers_df)} customers."
)


# ============================================================
# 2. GENERATE MERCHANTS
# ============================================================

print("Generating merchants...")

merchants = []

merchant_categories = [

    "E_COMMERCE",
    "SAAS",
    "EDUCATION",
    "TRAVEL",
    "FOOD",
    "DIGITAL_SERVICES"

]


for i in range(NUM_MERCHANTS):

    merchant_id = f"MERCH_{i + 1:03d}"

    merchant_category = random.choice(
        merchant_categories
    )

    # Historical merchant payment success rate
    merchant_success_rate = np.random.uniform(
        0.75,
        0.98
    )

    # Historical recovery performance
    merchant_recovery_rate = np.random.uniform(
        0.35,
        0.80
    )

    merchant_avg_order_value = round(
        np.random.lognormal(
            mean=np.log(3000),
            sigma=0.7
        ),
        2
    )

    merchants.append({

        "merchant_id":
            merchant_id,

        "merchant_category":
            merchant_category,

        "merchant_success_rate":
            round(
                merchant_success_rate,
                3
            ),

        "merchant_recovery_rate":
            round(
                merchant_recovery_rate,
                3
            ),

        "merchant_avg_order_value":
            merchant_avg_order_value

    })


merchants_df = pd.DataFrame(merchants)

merchants_df.to_csv(
    "data/merchants.csv",
    index=False
)

print(
    f"Created {len(merchants_df)} merchants."
)


# ============================================================
# 3. GENERATE PAYMENTS
# ============================================================

print("Generating payments...")

payment_methods = [

    "UPI",
    "CARD",
    "NETBANKING",
    "WALLET"

]


# More realistic failure relationships
failure_by_method = {

    "UPI": [
        "NETWORK_ERROR",
        "TIMEOUT",
        "INSUFFICIENT_FUNDS",
        "AUTHENTICATION_FAILURE"
    ],

    "CARD": [
        "NETWORK_ERROR",
        "TIMEOUT",
        "INSUFFICIENT_FUNDS",
        "EXPIRED_CARD",
        "AUTHENTICATION_FAILURE",
        "DUPLICATE",
        "SUSPECTED_FRAUD"
    ],

    "NETBANKING": [
        "NETWORK_ERROR",
        "TIMEOUT",
        "INSUFFICIENT_FUNDS",
        "AUTHENTICATION_FAILURE"
    ],

    "WALLET": [
        "NETWORK_ERROR",
        "TIMEOUT",
        "INSUFFICIENT_FUNDS",
        "DUPLICATE"
    ]

}


failure_weights = {

    "NETWORK_ERROR": 0.25,

    "TIMEOUT": 0.20,

    "INSUFFICIENT_FUNDS": 0.20,

    "EXPIRED_CARD": 0.12,

    "AUTHENTICATION_FAILURE": 0.10,

    "DUPLICATE": 0.05,

    "SUSPECTED_FRAUD": 0.03

}


payments = []


for i in range(NUM_PAYMENTS):

    payment_id = (
        f"PAY_{i + 1:06d}"
    )

    order_id = (
        f"ORD_{i + 1:06d}"
    )


    # Random customer
    customer = customers_df.sample(
        1
    ).iloc[0]


    # Random merchant
    merchant = merchants_df.sample(
        1
    ).iloc[0]


    customer_id = customer[
        "customer_id"
    ]

    merchant_id = merchant[
        "merchant_id"
    ]


    # Transaction amount
    amount = round(

        np.random.lognormal(
            mean=np.log(2500),
            sigma=0.9
        ),

        2
    )


    # Prevent extreme synthetic transactions
    amount = min(
        amount,
        100000
    )


    payment_method = random.choice(
        payment_methods
    )


    # Around 30% of payments fail
    is_failed = (
        np.random.random()
        < 0.30
    )


    if is_failed:

        possible_failures = (
            failure_by_method[
                payment_method
            ]
        )

        failure_type = random.choices(

            possible_failures,

            weights=[
                failure_weights[x]
                for x in possible_failures
            ]

        )[0]

        status = "FAILED"


        # Some failures happen after previous attempts
        retry_count = random.choices(

            [0, 1, 2],

            weights=[
                0.55,
                0.30,
                0.15
            ]

        )[0]


    else:

        failure_type = None

        status = "SUCCESS"

        retry_count = 0


    timestamp = (

        datetime(2026, 1, 1)

        + timedelta(

            minutes=random.randint(
                0,
                60 * 24 * 240
            )

        )

    )


    payments.append({

        "payment_id":
            payment_id,

        "customer_id":
            customer_id,

        "order_id":
            order_id,

        "merchant_id":
            merchant_id,

        "amount":
            amount,

        "payment_method":
            payment_method,

        "failure_type":
            failure_type,

        "retry_count":
            retry_count,

        "status":
            status,

        "timestamp":
            timestamp

    })


payments_df = pd.DataFrame(
    payments
)


payments_df.to_csv(

    "data/payments.csv",

    index=False

)


print(
    f"Created {len(payments_df)} payments."
)


# ============================================================
# 4. GENERATE RECOVERY OUTCOMES
# ============================================================

print("Generating recovery outcomes...")


failed_payments = payments_df[
    payments_df["status"] == "FAILED"
].copy()


recovery_results = []


for _, payment in failed_payments.iterrows():

    failure_type = payment[
        "failure_type"
    ]


    # --------------------------------------------------------
    # Get customer information
    # --------------------------------------------------------

    customer = customers_df[
        customers_df["customer_id"]
        ==
        payment["customer_id"]
    ].iloc[0]


    customer_success_rate = (

        customer[
            "successful_transactions"
        ]

        /

        max(
            customer[
                "total_transactions"
            ],
            1
        )

    )


    # --------------------------------------------------------
    # Get merchant information
    # --------------------------------------------------------

    merchant = merchants_df[
        merchants_df["merchant_id"]
        ==
        payment["merchant_id"]
    ].iloc[0]


    merchant_recovery_rate = (
        merchant[
            "merchant_recovery_rate"
        ]
    )


    merchant_success_rate = (
        merchant[
            "merchant_success_rate"
        ]
    )


    # --------------------------------------------------------
    # BASE RECOVERY PROBABILITY
    # --------------------------------------------------------

    if failure_type == "NETWORK_ERROR":

        base_probability = 0.75


    elif failure_type == "TIMEOUT":

        base_probability = 0.70


    elif failure_type == "INSUFFICIENT_FUNDS":

        base_probability = 0.45


    elif failure_type == "EXPIRED_CARD":

        base_probability = 0.25


    elif failure_type == "AUTHENTICATION_FAILURE":

        base_probability = 0.45


    elif failure_type == "DUPLICATE":

        base_probability = 0.05


    elif failure_type == "SUSPECTED_FRAUD":

        base_probability = 0.01


    else:

        base_probability = 0.40


    # --------------------------------------------------------
    # CUSTOMER BEHAVIOUR
    # --------------------------------------------------------

    probability = (

        base_probability

        +

        (
            customer_success_rate
            - 0.70
        )
        * 0.50

    )


    # --------------------------------------------------------
    # MERCHANT BEHAVIOUR
    # --------------------------------------------------------

    # Better merchant recovery history
    # slightly increases recoverability

    probability += (

        merchant_recovery_rate
        - 0.55

    ) * 0.30


    # Better merchant payment performance
    # also contributes

    probability += (

        merchant_success_rate
        - 0.85

    ) * 0.20


    # --------------------------------------------------------
    # RETRY PENALTY
    # --------------------------------------------------------

    probability -= (

        payment["retry_count"]
        * 0.10

    )


    # --------------------------------------------------------
    # PAYMENT METHOD EFFECT
    # --------------------------------------------------------

    if payment["payment_method"] == "UPI":

        probability += 0.02


    elif payment["payment_method"] == "CARD":

        probability += 0.01


    elif payment["payment_method"] == "NETBANKING":

        probability -= 0.01


    elif payment["payment_method"] == "WALLET":

        probability -= 0.01


    # --------------------------------------------------------
    # RANDOM NOISE
    # --------------------------------------------------------

    # Real-world behaviour isn't perfectly predictable.

    probability += np.random.normal(
        0,
        0.05
    )


    # --------------------------------------------------------
    # KEEP PROBABILITY IN VALID RANGE
    # --------------------------------------------------------

    probability = np.clip(

        probability,

        0.01,

        0.95

    )


    # --------------------------------------------------------
    # GENERATE ACTUAL OUTCOME
    # --------------------------------------------------------

    recovered = (

        np.random.random()
        <
        probability

    )


    # --------------------------------------------------------
    # SIMULATED RECOVERY ACTION
    # --------------------------------------------------------

    if recovered:

        recovered_amount = (
            payment["amount"]
        )


        if failure_type in [

            "NETWORK_ERROR",
            "TIMEOUT"

        ]:

            action = "RETRY"


        elif failure_type == "EXPIRED_CARD":

            action = "PAYMENT_LINK"


        elif failure_type == "INSUFFICIENT_FUNDS":

            action = "NOTIFICATION"


        elif failure_type == "AUTHENTICATION_FAILURE":

            action = "PAYMENT_LINK"


        else:

            action = "RETRY"


    else:

        recovered_amount = 0

        action = "FAILED_RECOVERY"


    recovery_results.append({

        "payment_id":
            payment["payment_id"],

        "action_taken":
            action,

        "recovered":
            int(recovered),

        "recovered_amount":
            round(
                recovered_amount,
                2
            )

    })


recovery_df = pd.DataFrame(
    recovery_results
)


recovery_df.to_csv(

    "data/recovery_outcomes.csv",

    index=False

)


print(

    f"Created "
    f"{len(recovery_df)} "
    f"recovery outcomes."

)


# ============================================================
# 5. DATA SUMMARY
# ============================================================

print("\n")
print("======================================")
print("       DATA GENERATION COMPLETE")
print("======================================")


print(
    f"Customers: "
    f"{len(customers_df)}"
)


print(
    f"Merchants: "
    f"{len(merchants_df)}"
)


print(
    f"Payments: "
    f"{len(payments_df)}"
)


print(
    f"Failed payments: "
    f"{len(failed_payments)}"
)


print(
    f"Recovery cases: "
    f"{len(recovery_df)}"
)


print("\nFiles created:")

print(
    "data/customers.csv"
)

print(
    "data/merchants.csv"
)

print(
    "data/payments.csv"
)

print(
    "data/recovery_outcomes.csv"
)


# ============================================================
# 6. RECOVERY SUMMARY
# ============================================================

print("\nRecovery outcome distribution:")

print(
    recovery_df[
        "recovered"
    ].value_counts(
        normalize=True
    )
)


print("\nFailure distribution:")

print(
    payments_df[
        "failure_type"
    ].value_counts(
        dropna=True
    )
)
