import re


# ============================================================
# GUARDRAIL CONFIGURATION
# ============================================================

MAX_AUTOMATIC_RETRIES = 2

MIN_RECOVERY_PROBABILITY = 0.15

HIGH_VALUE_THRESHOLD = 50000


# ============================================================
# EXTRACT ACTION FROM LLM RESPONSE
# ============================================================

def extract_action(response):

    match = re.search(
        r"RECOMMENDED_ACTION:\s*([A-Z_]+)",
        response,
        re.IGNORECASE
    )

    if match:

        return match.group(1).upper()

    return None


# ============================================================
# GUARDRAIL ENGINE
# ============================================================

def evaluate_action(
    payment,
    llm_response
):

    action = extract_action(
        llm_response
    )


    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    allowed_actions = {

        "RETRY",

        "PAYMENT_LINK",

        "NOTIFICATION",

        "ESCALATE",

        "NO_ACTION",

        "MANUAL_REVIEW"

    }


    if action not in allowed_actions:

        return {

            "allowed": False,

            "final_action": "MANUAL_REVIEW",

            "reason":
                "LLM returned an invalid action."

        }


    failure_type = str(
        payment["failure_type"]
    ).upper()


    retry_count = int(
        payment["retry_count"]
    )


    probability = float(
        payment["recovery_probability"]
    )


    amount = float(
        payment["amount"]
    )


    # ========================================================
    # RULE 1 — FRAUD PROTECTION
    # ========================================================

    if failure_type == "SUSPECTED_FRAUD":

        return {

            "allowed": False,

            "final_action": "ESCALATE",

            "reason":
                "Automatic recovery is prohibited "
                "for suspected fraud."

        }


    # ========================================================
    # RULE 2 — DUPLICATE PROTECTION
    # ========================================================

    if failure_type == "DUPLICATE":

        return {

            "allowed": False,

            "final_action": "NO_ACTION",

            "reason":
                "Duplicate payments must not be "
                "automatically retried."

        }


    # ========================================================
    # RULE 3 — LOW RECOVERY PROBABILITY
    # ========================================================

    if probability < MIN_RECOVERY_PROBABILITY:

        return {

            "allowed": False,

            "final_action": "NO_ACTION",

            "reason":
                "Recovery probability is below "
                "the automatic recovery threshold."

        }


    # ========================================================
    # RULE 4 — RETRY LIMIT
    # ========================================================

    if action == "RETRY":

        if retry_count >= MAX_AUTOMATIC_RETRIES:

            return {

                "allowed": False,

                "final_action":
                    "MANUAL_REVIEW",

                "reason":
                    "Maximum automatic retry "
                    "limit has been reached."

            }


    # ========================================================
    # RULE 5 — HIGH-VALUE PAYMENT
    # ========================================================

    if amount >= HIGH_VALUE_THRESHOLD:

        if action == "RETRY" and retry_count >= 1:

            return {

                "allowed": False,

                "final_action":
                    "MANUAL_REVIEW",

                "reason":
                    "High-value payment has already "
                    "received a retry and requires "
                    "additional review."

            }


    # ========================================================
    # ACTION-SPECIFIC VALIDATION
    # ========================================================

    if failure_type == "NETWORK_ERROR":

        if action not in {

            "RETRY",

            "NOTIFICATION"

        }:

            return {

                "allowed": False,

                "final_action": "MANUAL_REVIEW",

                "reason":
                    "Invalid action for network failure."

            }


    if failure_type == "TIMEOUT":

        if action not in {

            "RETRY",

            "NOTIFICATION"

        }:

            return {

                "allowed": False,

                "final_action": "MANUAL_REVIEW",

                "reason":
                    "Invalid action for timeout failure."

            }


    if failure_type == "EXPIRED_CARD":

        if action not in {

            "PAYMENT_LINK",

            "NOTIFICATION"

        }:

            return {

                "allowed": False,

                "final_action": "MANUAL_REVIEW",

                "reason":
                    "Expired card requires payment "
                    "method update."

            }


    if failure_type == "INSUFFICIENT_FUNDS":

        if action not in {

            "NOTIFICATION",

            "PAYMENT_LINK"

        }:

            return {

                "allowed": False,

                "final_action": "MANUAL_REVIEW",

                "reason":
                    "Insufficient funds should not "
                    "trigger an immediate retry."

            }


    # ========================================================
    # ALL CHECKS PASSED
    # ========================================================

    return {

        "allowed": True,

        "final_action": action,

        "reason":
            "Action passed all guardrail checks."

    }


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    print(
        "Guardrail module loaded successfully."
    )