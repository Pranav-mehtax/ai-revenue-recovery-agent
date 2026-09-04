# Recovery Strategies

## RETRY

Retry the failed payment.

Best suited for:
- NETWORK_ERROR
- TIMEOUT

Avoid when:
- SUSPECTED_FRAUD
- DUPLICATE

---

## PAYMENT_LINK

Generate or send a payment link so the customer can
complete the payment again.

Best suited for:
- EXPIRED_CARD
- AUTHENTICATION_FAILURE

---

## NOTIFICATION

Notify the customer that their payment failed and
provide instructions or an alternative payment method.

Best suited for:
- INSUFFICIENT_FUNDS

---

## ESCALATE

Send the payment for manual review.

Best suited for:
- SUSPECTED_FRAUD

---

## NO_ACTION

Do not attempt automatic recovery.

Best suited for:
- DUPLICATE
- Very low recovery probability