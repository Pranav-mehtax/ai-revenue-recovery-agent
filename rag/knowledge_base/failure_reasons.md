# Payment Failure Reasons

## NETWORK_ERROR

The payment could not be completed because of a temporary
network or connectivity issue.

Recommended approach:
Retry the payment after a short delay.

Recovery potential:
High.

---

## TIMEOUT

The payment request exceeded the allowed processing time.

Recommended approach:
Retry after a short delay.

Recovery potential:
High.

---

## INSUFFICIENT_FUNDS

The customer's account does not currently have sufficient funds.

Recommended approach:
Notify the customer and provide an alternative payment option.

Do not repeatedly retry immediately.

---

## EXPIRED_CARD

The customer's card has expired.

Recommended approach:
Ask the customer to update their payment method or provide
a payment link.

---

## AUTHENTICATION_FAILURE

The authentication step required to complete the payment failed.

Recommended approach:
Ask the customer to retry authentication or use a payment link.

---

## DUPLICATE

The transaction appears to duplicate another payment attempt.

Recommended approach:
Do not automatically retry.

Investigate the existing transaction first.

---

## SUSPECTED_FRAUD

The payment has been flagged as potentially fraudulent.

Recommended approach:
Do not automatically retry.

Escalate for review.