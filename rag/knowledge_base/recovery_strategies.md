# Payment Recovery Strategies and Operational Playbooks

## RECOVERY STRATEGY TAXONOMY AND DECISION MATRIX

The recovery strategy layer defines the specific operational interventions executed when a payment attempt fails. The platform supports six distinct recovery strategies:
1. `RETRY`: Automated system-level re-submission of the authorization request.
2. `PAYMENT_LINK`: Generating and dispatching an omnichannel checkout URL.
3. `NOTIFICATION`: Informative customer messaging advising of payment status.
4. `ESCALATE`: Security and fraud routing for specialized investigation.
5. `MANUAL_REVIEW`: Human operational review for high-value or edge-case transactions.
6. `NO_ACTION`: Deliberate termination of recovery workflows.

### Strategic Decision Matrix

| Strategy | Primary Failure Types | Recovery Probability Tier | Key Pre-conditions |
| :--- | :--- | :--- | :--- |
| **RETRY** | NETWORK_ERROR, TIMEOUT | High (>= 40%) | `retry_count < 2` (< 1 if >= ₹50k), no fraud/duplicate flag |
| **PAYMENT_LINK** | AUTHENTICATION_FAILURE, EXPIRED_CARD, INSUFFICIENT_FUNDS | Moderate to High (>= 30%) | Customer contact info valid, link expires in 30 mins |
| **NOTIFICATION** | INSUFFICIENT_FUNDS, TIMEOUT (exhausted retry) | Moderate (>= 20%) | Valid SMS/WhatsApp/email, frequency cap respected |
| **ESCALATE** | SUSPECTED_FRAUD | Any | Immediate freeze; automated actions prohibited |
| **MANUAL_REVIEW**| High-Value (>= ₹50k) with prior retry, Ambiguous states | Moderate to High | Operations team assignment; human verification |
| **NO_ACTION** | DUPLICATE, Low Probability (< 15%) | Sub-threshold (< 15%) | Prevent double debit or wasted communication cost |

---

## STRATEGY: RETRY

### Purpose and Execution
The RETRY strategy re-submits the exact payment payload to the acquiring gateway without requiring customer intervention. It leverages the transient nature of network, gateway, or switch outages to seamlessly recover revenue in the background.

### Optimal Application
- Best Suited For:
  - `NETWORK_ERROR`: Connection resets, socket dropouts, temporary DNS/API failures.
  - `TIMEOUT`: Gateway processing timeouts where status inquiry confirms no prior debit.
- Pre-conditions:
  - Current `retry_count` must be strictly less than 2 for standard transactions.
  - For high-value transactions (>= ₹50,000), `retry_count` must be strictly 0 (maximum 1 automatic retry allowed).
  - ML recovery probability should ideally exceed 40%, with an absolute floor of 15%.

### Execution Guidelines
- Backoff Interval: Enforce an exponential or jittered backoff (3 to 10 seconds) to avoid hammering a struggling bank switch.
- Idempotency Guarantee: Ensure the idempotent request key remains consistent to prevent unintended duplicate creations.

### Strict Contraindications
- NEVER execute RETRY for `SUSPECTED_FRAUD` or `DUPLICATE`.
- NEVER execute RETRY for `EXPIRED_CARD` (an expired card will never validate automatically).
- NEVER execute RETRY for `INSUFFICIENT_FUNDS` without waiting for customer funding replenishment.

---

## STRATEGY: PAYMENT_LINK

### Purpose and Execution
The PAYMENT_LINK strategy generates a secure, dynamic Razorpay checkout link sent directly to the customer via WhatsApp, SMS, or email. The link enables the customer to complete payment using their choice of payment method.

### Optimal Application
- Best Suited For:
  - `AUTHENTICATION_FAILURE`: Customer failed OTP or PIN; link allows fresh authentication without restarting checkout.
  - `EXPIRED_CARD`: Allows customer to input an updated card or seamlessly switch to UPI / NetBanking.
  - `INSUFFICIENT_FUNDS`: Customer can select an alternate funding account, credit card, or pay-later option.
- Target Persona: High-intent customers with reasonable historical success rates.

### Operational Parameters
- Link Validity Window: Standard expiration set to 30 minutes for checkout orders, or up to 24 hours for subscription renewals.
- Omnichannel Fallback: Primary delivery via WhatsApp (higher open and conversion rate in India), fallback to SMS and email.
- Pre-filled Context: Amount, order summary, and merchant branding pre-loaded to minimize checkout friction.

### Advantages and Risk Profile
- High conversion rate (typically 45% to 70% recovery when delivered within 5 minutes of failure).
- Completely eliminates duplicate debit risk because the customer controls authorization initiation.

---

## STRATEGY: NOTIFICATION

### Purpose and Execution
The NOTIFICATION strategy delivers an informative, supportive alert to the customer detailing the failure reason and providing actionable guidance on how to proceed. It does not initiate an immediate charge.

### Optimal Application
- Best Suited For:
  - `INSUFFICIENT_FUNDS`: Explains that the issuing bank reported insufficient funds and suggests topping up or using another account.
  - Transient failures where automated retry limits have been reached (`retry_count >= 2`).
  - Scheduled subscription renewals experiencing billing hiccups.

### Communication Guidelines
- Tone: Empathetic, professional, and clear. Avoid harsh technical jargon or accusatory language.
- Call to Action: Clear guidance, such as: *"Your bank was unable to complete the payment due to insufficient balance. Please ensure funds are available, or update your preferred payment method in your account settings."*
- Frequency Limits: Maximum 2 messages per incident; respect customer do-not-disturb (DND) preferences.

---

## STRATEGY: ESCALATE

### Purpose and Execution
The ESCALATE strategy immediately terminates all automated recovery workflows and hands the transaction over to the merchant's Fraud & Risk Operations department.

### Optimal Application
- Best Suited For:
  - `SUSPECTED_FRAUD`: Transactions triggering AML rules, abnormal card velocity, proxy/VPN anomalies, or blacklisted VPAs.
  - Transactions associated with compromised credentials or stolen card alerts.

### Operational Protocol
- Action Freeze: Block all automated retries, payment links, and notifications.
- Security Logging: Record full transaction telemetry, IP address, device fingerprint, and risk scores into the audit database.
- Fraud Analyst Review: Dedicated risk analyst inspects the order before any fulfillment or merchant goods dispatch can occur.

---

## STRATEGY: MANUAL_REVIEW

### Purpose and Execution
The MANUAL_REVIEW strategy queues the failed transaction for prioritized human investigation by merchant billing or customer support specialists.

### Optimal Application
- Best Suited For:
  - High-Value transactions (>= ₹50,000) that failed an initial automatic retry.
  - Transactions where the payment gateway returns an ambiguous or indeterminate status.
  - VIP customers (`customer_success_rate > 90%`) who experienced an unexpected failure.
  - Edge cases where ML probability and failure type present conflicting recovery signals.

### SOP for Manual Reviewers
1. Verify Bank Clearance: Check payment gateway dashboard for any delayed capture or settlement webhook.
2. Direct Customer Contact: If appropriate, reach out via phone or dedicated relationship manager for enterprise/high-value clients.
3. Remediation Choice: Manually authorize a secondary retry, issue a customized payment link, or re-generate invoice.

---

## STRATEGY: NO_ACTION

### Purpose and Execution
The NO_ACTION strategy explicitly halts all recovery mechanisms, archiving the transaction record without active customer outreach or system retries.

### Optimal Application
- Best Suited For:
  - `DUPLICATE`: Redundant payment attempts that mirror an already active or processed order.
  - Low Recovery Probability: Transactions with ML recovery probabilities below 15% (`recovery_probability < 0.15`) where recovery cost exceeds expected value.
  - Customer explicitly opted out or abandoned session with zero engagement.

### Business and Economic Rationale
- Prevents Double Debiting: Crucial protection against charging customers twice on repeated browser submissions.
- Cost Efficiency: Avoids burning SMS/WhatsApp messaging fees and gateway API quotas on unviable recovery attempts.
- Customer Experience: Protects customers from spam notifications regarding ghost or redundant transactions.