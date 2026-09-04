import random


def _log(message):
    """Console logging must not break a simulated recovery on Windows cp1252 consoles."""
    try:
        print(message)
    except UnicodeEncodeError:
        print(str(message).encode("ascii", "replace").decode("ascii"))


def execute_action(payment, final_action):
    """
    Execute the final action approved by the guardrail layer.

    This is a simulation for the hackathon prototype.
    No real payment is processed.
    """

    payment_id = payment["payment_id"]
    amount = float(payment["amount"])
    recovery_probability = float(payment["recovery_probability"])

    _log("=" * 70)
    _log("ACTION EXECUTOR")
    _log("=" * 70)

    _log(f"Payment ID           : {payment_id}")
    _log(f"Amount               : ₹{amount:,.2f}")
    _log(f"Final Action         : {final_action}")
    _log(f"Recovery Probability : {recovery_probability:.2%}")

    # ---------------------------------------------------------
    # RETRY
    # ---------------------------------------------------------

    if final_action == "RETRY":

        random_value = random.random()

        if random_value <= recovery_probability:

            return {
                "payment_id": payment_id,
                "action": "RETRY",
                "status": "RECOVERED",
                "recovered_amount": amount,
                "message": "Payment successfully recovered after retry."
            }

        else:

            return {
                "payment_id": payment_id,
                "action": "RETRY",
                "status": "FAILED",
                "recovered_amount": 0,
                "message": "Payment retry failed."
            }

    # ---------------------------------------------------------
    # PAYMENT LINK
    # ---------------------------------------------------------

    elif final_action == "PAYMENT_LINK":

        payment_link = f"https://pay.example.com/{payment_id}"

        return {
            "payment_id": payment_id,
            "action": "PAYMENT_LINK",
            "status": "LINK_GENERATED",
            "recovered_amount": 0,
            "payment_link": payment_link,
            "message": "Payment link generated successfully."
        }

    # ---------------------------------------------------------
    # NOTIFICATION
    # ---------------------------------------------------------

    elif final_action == "NOTIFICATION":

        return {
            "payment_id": payment_id,
            "action": "NOTIFICATION",
            "status": "NOTIFICATION_SENT",
            "recovered_amount": 0,
            "message": "Recovery notification sent to customer."
        }

    # ---------------------------------------------------------
    # ESCALATE
    # ---------------------------------------------------------

    elif final_action == "ESCALATE":

        return {
            "payment_id": payment_id,
            "action": "ESCALATE",
            "status": "ESCALATED",
            "recovered_amount": 0,
            "message": "Payment escalated for manual investigation."
        }

    # ---------------------------------------------------------
    # NO ACTION
    # ---------------------------------------------------------

    elif final_action == "NO_ACTION":

        return {
            "payment_id": payment_id,
            "action": "NO_ACTION",
            "status": "NO_ACTION",
            "recovered_amount": 0,
            "message": "No recovery action taken."
        }

    # ---------------------------------------------------------
    # MANUAL REVIEW
    # ---------------------------------------------------------

    elif final_action == "MANUAL_REVIEW":

        return {
            "payment_id": payment_id,
            "action": "MANUAL_REVIEW",
            "status": "MANUAL_REVIEW",
            "recovered_amount": 0,
            "message": "Payment requires manual review."
        }

    # ---------------------------------------------------------
    # UNKNOWN ACTION
    # ---------------------------------------------------------

    else:

        return {
            "payment_id": payment_id,
            "action": final_action,
            "status": "REJECTED",
            "recovered_amount": 0,
            "message": "Unknown action rejected by executor."
        }


# -------------------------------------------------------------
# TEST MODE
# -------------------------------------------------------------

if __name__ == "__main__":

    example_payment = {
        "payment_id": "PAY_TEST",
        "amount": 50000,
        "failure_type": "NETWORK_ERROR",
        "recovery_probability": 0.75
    }

    result = execute_action(
        example_payment,
        "RETRY"
    )

    _log("\nEXECUTION RESULT")
    _log("=" * 70)

    for key, value in result.items():
        _log(f"{key}: {value}")
