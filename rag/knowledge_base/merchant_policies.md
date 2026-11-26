# Merchant Recovery Policies and Governance

## RECOVERY POLICY FRAMEWORK AND PRINCIPLES

The AI Revenue Recovery Agent operates under a strict policy framework designed to optimize recovered revenue while rigorously safeguarding customer experience, merchant reputation, and regulatory compliance.

Core Operating Principles:
1. Safety First: Deterministic guardrails always supersede probabilistic machine learning or LLM recommendations.
2. Value Preservation: High-value transactions receive heightened scrutiny to prevent costly double debits or unauthorized retries.
3. Customer Trust: Recovery communications must be transparent, non-intrusive, and respect frequency capping limits.
4. Compliance and Fraud Protection: Zero tolerance for automated actions on flagged, fraudulent, or duplicate transactions.

---

## DEFAULT RECOVERY POLICY

The Default Policy governs all standard transactions below the high-value threshold (under ₹50,000) that are not flagged for security anomalies.

### Retry Limits and Thresholds
- Maximum Automatic Retries: 2 retry attempts (`MAX_AUTOMATIC_RETRIES = 2`). Any transaction having reached 2 retries cannot be automatically retried again.
- Minimum Recovery Probability Threshold: 15% (`MIN_RECOVERY_PROBABILITY = 0.15`). Transactions with ML recovery probabilities below 15% are deemed unviable for automated retry; they must be assigned NO_ACTION or low-touch NOTIFICATION.

### Allowed Recovery Actions Catalog
The authorized recovery action space consists of:
- RETRY: Automated programmatic re-submission of the payment.
- PAYMENT_LINK: Dynamic checkout link delivered to the customer.
- NOTIFICATION: Informational customer communication without direct debit.
- ESCALATE: Immediate referral to risk, fraud, or compliance teams.
- NO_ACTION: Intentional termination of recovery processing.
- MANUAL_REVIEW: Queuing for human operations evaluation.

### Policy Action Matrix by Failure Type
- NETWORK_ERROR: Eligible for RETRY (if retry count < 2) or NOTIFICATION.
- TIMEOUT: Eligible for RETRY (if retry count < 2) or NOTIFICATION.
- INSUFFICIENT_FUNDS: Eligible for NOTIFICATION or PAYMENT_LINK. Automated RETRY is strictly prohibited.
- EXPIRED_CARD: Eligible for PAYMENT_LINK or NOTIFICATION. Automated RETRY is strictly prohibited.
- AUTHENTICATION_FAILURE: Eligible for PAYMENT_LINK or NOTIFICATION. Automated RETRY is strictly prohibited.
- DUPLICATE: Exclusively NO_ACTION. Automated RETRY is strictly prohibited.
- SUSPECTED_FRAUD: Exclusively ESCALATE. Automated RETRY and direct customer notifications are strictly prohibited.

---

## HIGH_VALUE TRANSACTION POLICY

### Threshold Definition
Transactions with an amount equal to or greater than ₹50,000 (`HIGH_VALUE_THRESHOLD = 50000`) fall under the High-Value Transaction Policy.

### Stricter Retry Constraints
- Maximum Automatic Retries: Limited to exactly 1 retry attempt.
- Subsequent Action Requirement: If a high-value payment has already undergone 1 retry (`retry_count >= 1`), further automatic retries are blocked by guardrails and must be routed to MANUAL_REVIEW.
- Expected Value Considerations: High-value transactions often represent substantial expected recovery value (`amount * recovery_probability`), making human oversight economically justified over automated risks.

### Mandatory Manual Review Scenarios
- High-value payments experiencing secondary failures.
- Duplicate or ambiguous payment statuses on amounts >= ₹50,000.
- Unusual merchant or customer failure spikes involving high-value orders.

---

## FRAUD PREVENTION AND COMPLIANCE POLICY

### Fraud Detection Protocol
Any transaction flagged with `SUSPECTED_FRAUD` must bypass standard automated recovery channels:
- Automatic Retry Prohibition: Under no circumstances may an automated retry be scheduled or executed for a suspected fraud payment.
- Action Routing: The mandatory and exclusive action is `ESCALATE`.
- Customer Communication Freeze: Automated notifications or payment links must not be transmitted to suspected fraudulent parties, as this can alert bad actors or compromise investigative tracking.
- Regulatory Alignment: In compliance with RBI cyber security and anti-money laundering (AML) guidelines, all escalated cases are logged in the security audit trail for compliance review.

---

## DUPLICATE TRANSACTION PREVENTION POLICY

### Idempotency and Ledger Safeguards
Duplicate transactions (`DUPLICATE`) indicate repeated submissions of the same underlying order or conflicting idempotency keys:
- Immediate Rejection of Retries: Automatic retries are strictly prohibited to prevent duplicate debit from the customer's account.
- Mandatory Action: The required action is `NO_ACTION` to gracefully drop redundant requests.
- Investigation Protocol: If the primary transaction is unresolved or marked pending, merchant operations must inspect payment settlement records rather than retrying the duplicate record.
- Chargeback Mitigation: Double debits carry high merchant dispute costs, mandatory refund fees, and card network compliance penalties.

---

## CUSTOMER EXPERIENCE AND FREQUENCY CAPPING POLICY

### Frequency Capping Rules
To maintain customer satisfaction and avoid brand damage:
- Notification Limit: A customer may receive a maximum of 2 recovery communications per 24 hours per order.
- Cooling-off Period: A minimum 15-minute interval must elapse between consecutive recovery touchpoints.
- Preferred Channels: High-urgency cart recovery should prioritize WhatsApp and SMS; subscription renewals may utilize email and in-app notifications.

### Customer Historical Profile Weighting
- High-Trust Customers (`customer_success_rate >= 0.80`): Prioritize frictionless PAYMENT_LINK or quiet background RETRY for transient network errors.
- Low-Trust / Inactive Customers (`customer_success_rate < 0.40`): Restrict aggressive retries; require customer authentication confirmation via PAYMENT_LINK.

---

## GUARDRAIL ENFORCEMENT AND DETERMINISTIC OVERRIDES

### Primacy of Guardrails
The LLM serves as an intelligent advisory agent that synthesizes contextual reasoning. However, the deterministic guardrail module (`evaluate_action`) has final sovereignty over all decisions:
- If the LLM recommends `RETRY` for a `SUSPECTED_FRAUD` payment, the guardrail rejects the recommendation and forces `ESCALATE`.
- If the LLM recommends `RETRY` for a `DUPLICATE` payment, the guardrail rejects the recommendation and forces `NO_ACTION`.
- If the LLM recommends `RETRY` when `retry_count >= 2`, the guardrail rejects the recommendation and forces `MANUAL_REVIEW`.
- If the LLM recommends `RETRY` on a payment >= ₹50,000 with `retry_count >= 1`, the guardrail forces `MANUAL_REVIEW`.
- If the recovery probability is below 15% (`recovery_probability < 0.15`), the guardrail blocks automated action and forces `NO_ACTION`.
- If the LLM returns an unapproved or hallucinated action, the guardrail forces `MANUAL_REVIEW`.

All guardrail interventions are recorded in the audit log with explicit justification.