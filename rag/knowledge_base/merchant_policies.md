# Merchant Recovery Policies

## DEFAULT POLICY

Maximum automatic retries:
2

Allowed actions:
- RETRY
- PAYMENT_LINK
- NOTIFICATION

Fraud payments:
Manual review required

Duplicate payments:
No automatic retry


## HIGH_VALUE POLICY

Payments above ₹50,000 require additional caution.

Maximum automatic retries:
1

Fraud payments:
Manual review required

Duplicate payments:
Manual review required