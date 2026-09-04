import os
import sys
import time

import pandas as pd

from dotenv import load_dotenv
from google import genai


# ============================================================
# PATH CONFIGURATION
# ============================================================

RAG_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    RAG_DIR
)


# ============================================================
# IMPORT RAG + GUARDRAILS
# ============================================================

sys.path.insert(
    0,
    RAG_DIR
)

from rag_engine import retrieve
from guardrails import evaluate_action


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

ENV_PATH = os.path.join(
    PROJECT_ROOT,
    ".env"
)

load_dotenv(
    ENV_PATH
)


# ============================================================
# GEMINI API
# ============================================================

api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not api_key:

    raise ValueError(
        "\nGEMINI_API_KEY was not found.\n\n"
        f"Expected .env at:\n{ENV_PATH}\n"
    )


client = genai.Client(
    api_key=api_key
)


# ============================================================
# MODELS
# ============================================================

MODELS_TO_TRY = [

    "gemini-2.5-flash",

    "gemini-3.7-flash"

]


# ============================================================
# AI RECOVERY AGENT
# ============================================================

def generate_recovery_recommendation(
    payment_context
):

    # ========================================================
    # 1. CREATE RETRIEVAL QUERY
    # ========================================================

    query = f"""
Payment failure type:
{payment_context["failure_type"]}

Payment method:
{payment_context["payment_method"]}

Retry count:
{payment_context["retry_count"]}

What recovery action should be considered
for this failed payment?
"""


    # ========================================================
    # 2. RETRIEVE KNOWLEDGE
    # ========================================================

    print(
        "\nRetrieving relevant knowledge..."
    )

    retrieved_documents = retrieve(
        query,
        top_k=3
    )


    # ========================================================
    # 3. BUILD KNOWLEDGE CONTEXT
    # ========================================================

    knowledge_parts = []

    for document in retrieved_documents:

        knowledge_parts.append(

            f"""
SOURCE: {document["source"]}

SIMILARITY:
{document["score"]:.4f}

CONTENT:
{document["text"]}
"""

        )


    knowledge_context = "\n".join(
        knowledge_parts
    )


    # ========================================================
    # 4. BUILD PROMPT
    # ========================================================

    prompt = f"""
You are an AI Revenue Recovery Agent for a
payment processing platform.

Your job is to analyze a failed payment and recommend
a safe recovery action.

The ML model has already calculated the recovery
probability.

DO NOT change or recalculate the ML probability.

The retrieved knowledge contains the recovery strategies
and policies available to you.


============================================================
IMPORTANT RULES
============================================================

1. Use retrieved knowledge as the primary source
   for recovery policies.

2. Do not invent policies.

3. Do not change the ML recovery probability.

4. Explain why the recommended action was selected.

5. NEVER recommend automatic retry for SUSPECTED_FRAUD.

6. NEVER recommend automatic retry for DUPLICATE payments.

7. If information is insufficient, recommend MANUAL_REVIEW.

8. Consider payment amount and recovery probability.

9. Expected recovery value has already been calculated.

10. The final automatic-action decision will be made
    by a separate guardrail system.

11. Do not claim that an action has actually been
    executed.


============================================================
PAYMENT INFORMATION
============================================================

Payment ID:
{payment_context["payment_id"]}

Amount:
₹{float(payment_context["amount"]):,.2f}

Payment method:
{payment_context["payment_method"]}

Failure type:
{payment_context["failure_type"]}

Retry count:
{payment_context["retry_count"]}

Customer historical success rate:
{float(payment_context["customer_success_rate"]):.2%}

Merchant historical success rate:
{float(payment_context["merchant_success_rate"]):.2%}

Merchant historical recovery rate:
{float(payment_context["merchant_recovery_rate"]):.2%}

ML recovery probability:
{float(payment_context["recovery_probability"]):.2%}

Expected recovery value:
₹{float(payment_context["expected_recovery_value"]):,.2f}


============================================================
RETRIEVED KNOWLEDGE
============================================================

{knowledge_context}


============================================================
TASK
============================================================

Analyze the payment.

Return exactly:

FAILURE_EXPLANATION:
<short explanation>

RECOMMENDED_ACTION:
<one of RETRY, PAYMENT_LINK, NOTIFICATION, ESCALATE,
NO_ACTION, MANUAL_REVIEW>

REASONING:
<why this action is appropriate>

RISK:
<short risk assessment>

AUTOMATIC_ACTION:
<YES or NO>

Remember:

The guardrail system makes the final decision about
whether the recommended action is actually allowed.
"""


    # ========================================================
    # 5. CALL GEMINI
    # ========================================================

    response = None

    last_error = None


    for model_name in MODELS_TO_TRY:

        print(
            f"\nTrying model: {model_name}"
        )


        for attempt in range(3):

            try:

                print(
                    f"Attempt {attempt + 1}/3..."
                )


                response = client.models.generate_content(

                    model=model_name,

                    contents=prompt

                )


                if response is not None:

                    print(
                        f"Successfully generated response "
                        f"using {model_name}."
                    )

                    break


            except Exception as error:

                last_error = error

                print(
                    f"Attempt {attempt + 1} failed."
                )

                print(
                    f"Error: {error}"
                )


                if attempt < 2:

                    print(
                        "Waiting 2 seconds before retry..."
                    )

                    time.sleep(2)


        if response is not None:

            break


    # ========================================================
    # 6. HANDLE LLM FAILURE
    # ========================================================

    if response is None:

        raise RuntimeError(

            "\nAll Gemini model attempts failed.\n\n"
            f"Last error:\n{last_error}"

        )


    # ========================================================
    # 7. RETURN LLM RESULT
    # ========================================================

    return {

        "response":
            response.text,

        "retrieved_documents":
            retrieved_documents

    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # LOAD PRIORITIZED PAYMENTS
    # ========================================================

    DATA_PATH = os.path.join(

        PROJECT_ROOT,

        "data",

        "prioritized_payments.csv"

    )


    print(
        "\nLoading prioritized payments..."
    )


    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(

            f"\nCould not find:\n{DATA_PATH}\n\n"
            "Run:\n"
            "python ml\\priority_engine.py"

        )


    df = pd.read_csv(
        DATA_PATH
    )


    if df.empty:

        raise ValueError(
            "prioritized_payments.csv is empty."
        )


    # ========================================================
    # SELECT TOP PAYMENT
    # ========================================================

    payment = df.iloc[0]


    # ========================================================
    # CREATE PAYMENT CONTEXT
    # ========================================================

    payment_context = {

        "payment_id":
            payment["payment_id"],

        "amount":
            payment["amount"],

        "payment_method":
            payment["payment_method"],

        "failure_type":
            payment["failure_type"],

        "retry_count":
            payment["retry_count"],

        "customer_success_rate":
            payment["customer_success_rate"],

        "merchant_success_rate":
            payment["merchant_success_rate"],

        "merchant_recovery_rate":
            payment["merchant_recovery_rate"],

        "recovery_probability":
            payment["recovery_probability"],

        "expected_recovery_value":
            payment["expected_recovery_value"]

    }


    # ========================================================
    # DISPLAY PAYMENT
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "AI REVENUE RECOVERY AGENT"
    )

    print("=" * 70)


    print("\nSelected payment:")

    print(
        f"Payment ID: "
        f"{payment_context['payment_id']}"
    )

    print(
        f"Amount: "
        f"₹{float(payment_context['amount']):,.2f}"
    )

    print(
        f"Payment method: "
        f"{payment_context['payment_method']}"
    )

    print(
        f"Failure: "
        f"{payment_context['failure_type']}"
    )

    print(
        f"Retry count: "
        f"{payment_context['retry_count']}"
    )

    print(
        f"Recovery probability: "
        f"{float(payment_context['recovery_probability']):.2%}"
    )

    print(
        f"Expected recovery: "
        f"₹{float(payment_context['expected_recovery_value']):,.2f}"
    )


    # ========================================================
    # RUN LLM AGENT
    # ========================================================

    result = generate_recovery_recommendation(

        payment_context

    )


    llm_response = result[
        "response"
    ]


    # ========================================================
    # DISPLAY LLM RESPONSE
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "AI RECOMMENDATION"
    )

    print("=" * 70)


    print(
        llm_response
    )


    # ========================================================
    # RUN GUARDRAILS
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "GUARDRAIL EVALUATION"
    )

    print("=" * 70)


    guardrail_result = evaluate_action(

        payment_context,

        llm_response

    )


    # ========================================================
    # DISPLAY GUARDRAIL RESULT
    # ========================================================

    llm_action = (
        guardrail_result.get(
            "final_action"
        )
    )


    print(
        f"\nLLM RECOMMENDED ACTION:"
    )

    print(
        f"{llm_action}"
    )


    print(
        f"\nGUARDRAIL STATUS:"
    )


    if guardrail_result["allowed"]:

        print(
            "PASSED"
        )

        status = "APPROVED"

    else:

        print(
            "BLOCKED"
        )

        status = "REJECTED"


    print(
        f"\nFINAL ACTION:"
    )

    print(
        guardrail_result[
            "final_action"
        ]
    )


    print(
        f"\nSTATUS:"
    )

    print(
        status
    )


    print(
        f"\nGUARDRAIL REASON:"
    )

    print(
        guardrail_result[
            "reason"
        ]
    )


    # ========================================================
    # DISPLAY RETRIEVED SOURCES
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "RETRIEVED KNOWLEDGE"
    )

    print("=" * 70)


    for i, document in enumerate(

        result["retrieved_documents"],

        start=1

    ):

        print(

            f"\n{i}. "
            f"{document['source']} "
            f"(similarity="
            f"{document['score']:.4f})"

        )