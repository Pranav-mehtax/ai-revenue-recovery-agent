"""Action executor - the LAST step of the recovery pipeline.

    guardrails pick the final action  ->  execute_action()  ->  result dict

SIMULATION ONLY: no real payment is ever processed. The one real side effect is an
optional push notification to your phone through ntfy.

What execute_action(payment, final_action) does:
    1. Safety net : refuse RETRY for fraud / duplicate payments (second layer after guardrails)
    2. Do the action (everything is simulated)
    3. _finish()  : send the ntfy push if enabled, attach it as result["notification"]

.env settings:
    NTFY_TOPIC     required for push (use a long random name, ntfy.sh topics are public)
    NTFY_SERVER    optional, default https://ntfy.sh
    NTFY_TOKEN     optional, bearer token for protected topics
    NTFY_PUSH_ALL  optional, "true" = push for every executed payment (batch runs too)
"""
import json
import os
import random
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:  # .env is optional; real environment variables always win
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:
    pass

# Failure types that must never be retried automatically (backup for rag/guardrails.py).
NEVER_RETRY = {"SUSPECTED_FRAUD", "DUPLICATE"}

# Actions with nothing special to simulate:  action -> (status, message)
SIMPLE_ACTIONS = {
    "PAYMENT_METHOD_UPDATE": ("PAYMENT_METHOD_UPDATE_REQUESTED", "Customer asked to update their payment method."),
    "NOTIFICATION": ("NOTIFICATION_SENT", "Recovery notification sent to customer."),
    "ESCALATE": ("ESCALATED", "Payment escalated for manual investigation."),
    "NO_ACTION": ("NO_ACTION", "No recovery action taken."),
    "MANUAL_REVIEW": ("MANUAL_REVIEW", "Payment requires manual review."),
}

# status -> (push title, ntfy priority, emoji tags)
_PUSH_STYLE = {
    "RECOVERED": ("Payment recovered", "default", ["white_check_mark", "moneybag"]),
    "FAILED": ("Retry failed", "high", ["x"]),
    "NOTIFICATION_SENT": ("Customer notification", "default", ["bell"]),
    "NOTIFICATION_FAILED": ("Customer notification", "default", ["bell"]),
    "PAYMENT_METHOD_UPDATE_REQUESTED": ("Update payment method", "default", ["credit_card"]),
    "LINK_GENERATED": ("Payment link generated", "default", ["link"]),
    "ESCALATED": ("Escalated for investigation", "high", ["rotating_light"]),
    "MANUAL_REVIEW": ("Manual review needed", "high", ["mag"]),
    "NO_ACTION": ("No action taken", "low", ["zzz"]),
    "BLOCKED_BY_EXECUTOR": ("Unsafe action blocked", "urgent", ["no_entry"]),
    "REJECTED": ("Unknown action rejected", "high", ["warning"]),
}


def _log(message):
    """print() that never crashes on Windows consoles that can't show symbols like the rupee sign."""
    try:
        print(message)
    except UnicodeEncodeError:
        print(str(message).encode("ascii", "replace").decode("ascii"))


# ======================================================================
# PART 1 - ntfy push notifications
# ======================================================================

def _ascii(text):
    """HTTP header values must be ASCII; the message body can stay UTF-8."""
    return str(text).encode("ascii", "replace").decode("ascii")


def ntfy_destination():
    """Masked destination for logs/audit (the topic name is effectively a password)."""
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        return ""
    server = (os.getenv("NTFY_SERVER") or "https://ntfy.sh").rstrip("/")
    return f"{server}/...{topic[-4:]}"


def send_ntfy(title, message, priority="default", tags=None):
    """Publish one notification. Never raises; returns a dict describing what happened."""
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        return {"channel": "ntfy", "sent": False, "detail": "NTFY_TOPIC is not set in .env"}

    server = (os.getenv("NTFY_SERVER") or "https://ntfy.sh").rstrip("/")
    url = f"{server}/{urllib.parse.quote(topic, safe='')}"
    headers = {"Title": _ascii(title)[:120], "Priority": str(priority)}
    if tags:
        headers["Tags"] = ",".join(tags)
    token = os.getenv("NTFY_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(
        url, data=str(message)[:3500].encode("utf-8"), headers=headers, method="POST"
    )
    info = {"channel": "ntfy", "destination": ntfy_destination(), "sent": False}
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8", "replace")
            info["http_status"] = response.status
            info["sent"] = 200 <= response.status < 300
        try:
            info["message_id"] = json.loads(body).get("id")
        except ValueError:
            info["message_id"] = None
        info["detail"] = "Delivered to ntfy server" if info["sent"] else f"HTTP {info['http_status']}"
    except urllib.error.HTTPError as error:
        info["http_status"] = error.code
        info["detail"] = f"ntfy rejected the request (HTTP {error.code})"
    except Exception as error:  # network down, DNS, timeout...
        info["detail"] = f"Could not reach ntfy: {error}"
    return info


def _compose_push(payment, result):
    """Build (title, message, priority, tags) for the phone notification."""
    title, priority, tags = _PUSH_STYLE.get(result["status"], ("Recovery update", "default", []))
    context = payment.get("notify_context") or {}
    lines = [
        f"{payment['payment_id']} | Rs {float(payment['amount']):,.2f} | {payment.get('failure_type', 'UNKNOWN')}",
        result["message"],
    ]
    if result.get("payment_link"):
        lines.append(f"Link: {result['payment_link']} (simulated)")
    if context.get("why"):
        lines.append(f"Why: {context['why']}")
    if context.get("run_id"):
        lines.append(f"Audit run: {context['run_id']}")
    lines.append("SIMULATED DEMO - no real money moved.")
    return f"Recovery agent: {title}", "\n".join(lines), priority, tags


def _wants_push(payment):
    return bool(payment.get("send_push")) or os.getenv("NTFY_PUSH_ALL", "").lower() == "true"


def _finish(payment, result):
    """Send the push (if requested) and keep the status honest about whether it was delivered."""
    if _wants_push(payment):
        title, message, priority, tags = _compose_push(payment, result)
        push = send_ntfy(title, message, priority, tags)
        result["notification"] = push
        if not push["sent"]:
            result["message"] += f" (push NOT delivered: {push['detail']})"
            if result["status"] == "NOTIFICATION_SENT":
                result["status"] = "NOTIFICATION_FAILED"
    else:
        result["notification"] = {"channel": "ntfy", "sent": False, "detail": "Push disabled for this payment"}
        if result["status"] == "NOTIFICATION_SENT":
            result["message"] = "Recovery notification simulated (push disabled)."
    return result


# ======================================================================
# PART 2 - the executor itself
# ======================================================================

def _simulate_retry(payment):
    """Decide whether a simulated retry succeeds. Returns (recovered, how_we_decided)."""
    mode = str(payment.get("simulate_retry_outcome", "sample")).lower()
    if mode == "success":
        return True, "forced_success_for_demo"
    if mode == "fail":
        return False, "forced_failure_for_demo"
    probability = float(payment.get("recovery_probability", 0))
    return random.random() <= probability, "sampled_from_ml_probability"


def execute_action(payment, final_action):
    """Carry out the action the guardrails approved. Simulation only - no real payment."""
    payment_id = payment["payment_id"]
    amount = float(payment["amount"])
    failure_type = str(payment.get("failure_type", "")).upper()

    _log(f"[executor] {payment_id} | Rs {amount:,.2f} | action={final_action}")

    def done(status, message, recovered=0, **extra):
        result = {
            "payment_id": payment_id,
            "action": final_action,
            "status": status,
            "recovered_amount": recovered,
            "message": message,
            **extra,
        }
        return _finish(payment, result)

    # Safety net: never retry fraud/duplicates, even if the guardrails let it through.
    if final_action == "RETRY" and failure_type in NEVER_RETRY:
        return done("BLOCKED_BY_EXECUTOR",
                    f"Executor refused RETRY for {failure_type}: automatic retries are never allowed.")

    if final_action == "RETRY":
        recovered, source = _simulate_retry(payment)
        if recovered:
            return done("RECOVERED", "Payment successfully recovered after retry.", amount, outcome_source=source)
        return done("FAILED", "Payment retry failed.", outcome_source=source)

    if final_action == "PAYMENT_LINK":
        return done("LINK_GENERATED", "Payment link generated successfully.",
                    payment_link=f"https://pay.example.com/{payment_id}")

    if final_action in SIMPLE_ACTIONS:
        status, message = SIMPLE_ACTIONS[final_action]
        return done(status, message)

    return done("REJECTED", "Unknown action rejected by executor.")


# Quick manual test:  python -m backend.actions.action_executor
if __name__ == "__main__":
    example = {
        "payment_id": "PAY_TEST",
        "amount": 50000,
        "failure_type": "NETWORK_ERROR",
        "recovery_probability": 0.75,
        "send_push": True,  # sends a real ntfy push if NTFY_TOPIC is set
        "simulate_retry_outcome": "success",
    }
    for key, value in execute_action(example, "RETRY").items():
        _log(f"{key}: {value}")