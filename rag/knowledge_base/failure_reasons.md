# Payment Failure Reasons and Diagnostics

## OVERVIEW AND CLASSIFICATION CRITERIA

Payment failures occur across multiple transaction stages: customer authentication, merchant gateway integration, acquiring bank processing, card network/NPCI switching, and issuing bank account debit. Understanding the root cause is critical for selecting the correct recovery strategy without introducing duplicate charge risk, customer churn, or fraud exposure.

Failure categories are divided into:
1. Transient Infrastructure Failures (NETWORK_ERROR, TIMEOUT): High recovery potential via automated retry or short-interval follow-up.
2. Customer Authorization and Instrument Failures (AUTHENTICATION_FAILURE, EXPIRED_CARD, INSUFFICIENT_FUNDS): Moderate recovery potential requiring customer action, payment method updates, or alternative payment links.
3. System Integrity and Security Rejections (DUPLICATE, SUSPECTED_FRAUD): Zero automatic retry tolerance; requires immediate escalation, manual investigation, or deliberate inaction.

---

## NETWORK_ERROR

### Definition and Technical Root Cause
A NETWORK_ERROR indicates a transient disruption in communication between the merchant server, Razorpay payment gateway, payment aggregator switch, or the banking partner host. Common causes include:
- Socket dropouts or connection resets during TCP/TLS handshakes with acquiring bank APIs.
- Temporary packet loss on cellular or Wi-Fi networks during UPI intent or mobile checkout.
- Internal gateway switch routing latency or microservice connectivity glitches.

### Payment Method Nuances
- UPI: NPCI switch latency or transient dropouts between UPI apps (Google Pay, PhonePe, Paytm) and the remitter bank switch.
- CARD: Gateway-to-card network (Visa, Mastercard, RuPay) connection interruption prior to authorization payload delivery.
- NETBANKING: Core banking gateway server connection timeout during redirect.
- WALLET: Partner wallet API connectivity drop during balance check or debit request.

### Recommended Recovery Approach
- Primary Action: RETRY after an exponential backoff or short pause (typically 3 to 5 seconds).
- Secondary Action: If retry count reaches policy limits, transition to NOTIFICATION or PAYMENT_LINK.
- Recovery Potential: High (typically 65% to 85% probability on first retry).
- Risk Considerations: Verify that no previous attempt was debited before executing an automated retry.

---

## TIMEOUT

### Definition and Technical Root Cause
A TIMEOUT occurs when a payment authorization request exceeds the maximum allotted processing window (typically 30 to 120 seconds) without receiving a definitive success or failure confirmation from the issuing bank or payment network.

### Technical Drivers
- Issuing bank core banking systems experiencing high transaction loads or maintenance downtime.
- Customer session expiration during two-factor authentication (3D Secure OTP or biometric check).
- Payment Service Provider (PSP) webhook acknowledgment delays.

### Payment Method Nuances
- UPI: NPCI timeout where the transaction is pending in the UPI queue and customer approval was not delivered within the time limit.
- CARD: 3D Secure authentication session timeout where the customer failed to submit the OTP within 180 seconds.
- NETBANKING: User left the bank login portal unattended or network connection dropped before bank callback.
- WALLET: Third-party wallet provider response timeout.

### Recommended Recovery Approach
- Primary Action: RETRY after a status check query confirms the payment has not been captured or settled.
- Fallback Action: NOTIFICATION informing the customer of the pending state or offering a PAYMENT_LINK.
- Recovery Potential: High (typically 60% to 80% probability).
- Risk Considerations: Critical duplicate debit hazard. Always verify idempotency and ensure status inquiry does not show a captured payment before re-initiating.

---

## INSUFFICIENT_FUNDS

### Definition and Technical Root Cause
The customer's bank account, credit line, or prepaid wallet balance does not contain enough cleared funds to cover the transaction value. The issuing bank directly returns an insufficient funds decline code (e.g., ISO 8583 response code 51).

### Technical and Operational Dynamics
- Immediate retries against the same account almost always fail and may incur bank decline penalty charges or customer annoyance.
- Customer account balance may be replenished on salary cycles, bill payments, or fund transfers between accounts.

### Payment Method Nuances
- UPI: Remitter bank savings/current account balance is below transaction amount.
- CARD: Credit card credit limit exhausted or debit card available balance insufficient.
- WALLET: Stored balance insufficient; customer needs to top up wallet or bind a secondary card/bank.
- NETBANKING: Account balance insufficient at the time of debit confirmation.

### Recommended Recovery Approach
- Primary Action: NOTIFICATION sent via SMS, WhatsApp, or email alerting the customer in a supportive, discreet manner.
- Secondary Action: PAYMENT_LINK providing the customer with an immediate opportunity to pay using an alternate bank account, credit card, or payment method.
- Strict Policy: NEVER attempt immediate automatic RETRY. Retries without customer funding updates are counterproductive.
- Recovery Potential: Moderate (30% to 55% recovery rate when assisted by a seamless payment link; under 5% with blind retries).

---

## EXPIRED_CARD

### Definition and Technical Root Cause
The payment was initiated using a credit or debit card whose validity period (MM/YY) has passed, or a stored card-on-file token (COFT) that has expired or been revoked by the card brand.

### Operational Context
- Automated retries of the identical card credential will consistently produce a 100% decline rate.
- Customers often have newly issued replacement cards from their bank with updated expiration dates and CVVs.

### Payment Method Nuances
- CARD: Directly applicable. Card authorization declined by card scheme or issuer due to expired card record.
- Other Rails (UPI, NETBANKING, WALLET): Not directly applicable; if tagged on other rails, suspect metadata mismatch.

### Recommended Recovery Approach
- Primary Action: PAYMENT_LINK prompting the customer to enter updated card details or switch to UPI / NetBanking.
- Secondary Action: NOTIFICATION advising that the saved card has expired and providing a quick link to manage payment methods.
- Strict Policy: Automatic RETRY is completely prohibited.
- Recovery Potential: Moderate to High (45% to 65% when the payment link offers instant UPI or alternative card entry).

---

## AUTHENTICATION_FAILURE

### Definition and Technical Root Cause
The mandatory customer authentication step failed or was aborted. In the Indian payments ecosystem under RBI regulations, this is predominantly an Additional Factor of Authentication (AFA) / 3D Secure 2.0 or UPI PIN verification failure.

### Root Cause Factors
- Customer entered an incorrect SMS OTP or the 3DS OTP expired.
- UPI PIN entered incorrectly multiple times, triggering a temporary security freeze.
- Biometric authentication failure or app permission denial on mobile banking apps.
- Issuing bank ACS (Access Control Server) failure or certificate verification issue.

### Payment Method Nuances
- UPI: Incorrect UPI PIN or transaction limit per UPI PIN exceeded.
- CARD: 3DS OTP entered incorrectly, SMS delivery delay exceeding OTP validity, or bank ACS outage.
- NETBANKING: Failed internet banking login password, profile password, or transaction OTP.
- WALLET: Wallet login MPIN or mobile number verification failed.

### Recommended Recovery Approach
- Primary Action: PAYMENT_LINK sent directly to the customer so they can re-initiate the payment with a fresh authentication attempt.
- Secondary Action: NOTIFICATION guiding the customer to retry with correct credentials or choose another payment method.
- Strict Policy: Automatic RETRY cannot succeed without user input; do not fire automated silent retries.
- Recovery Potential: Moderate to High (50% to 75% when prompted immediately while customer purchase intent is fresh).

---

## DUPLICATE

### Definition and Technical Root Cause
The transaction payload matches an already submitted or in-flight payment request, sharing identical order ID, idempotency key, customer identifiers, and transaction amount.

### Root Cause Factors
- Customer clicked the "Pay" button multiple times due to slow UI feedback.
- Network re-transmission of identical HTTP requests.
- Asynchronous webhook retry arriving while order processing is already in progress.

### Recommended Recovery Approach
- Primary Action: NO_ACTION. Cease all recovery attempts immediately.
- Secondary Action: MANUAL_REVIEW if the primary transaction status is ambiguous or unreconciled in the ledger.
- Strict Policy: NEVER automatically retry duplicate payments. Automatic retry risks double-debiting the customer, creating merchant dispute liabilities and payment gateway compliance penalties.
- Recovery Potential: N/A (Intentional rejection to safeguard the customer).

---

## SUSPECTED_FRAUD

### Definition and Technical Root Cause
The payment has been flagged by automated fraud detection engines, risk rule matrices, or payment gateway machine learning models as having elevated fraud risk indicators.

### Risk Indicators
- Geolocation mismatch (IP address country differs significantly from issuing bank country).
- Velocity anomalies (rapid successive attempts across multiple card numbers from a single device or IP).
- Stolen card or credential stuffing indicators.
- High-risk BIN lists or blacklisted VPAs/customer entities.

### Recommended Recovery Approach
- Primary Action: ESCALATE immediately to the merchant's Fraud & Risk Operations team.
- Secondary Action: MANUAL_REVIEW with security hold on order fulfillment.
- Strict Policy: NEVER attempt automatic RETRY, and do NOT send automated customer notifications that reveal fraud detection triggers.
- Recovery Potential: Very low genuine recovery rate; financial and legal risk of processing fraudulent transactions heavily outweighs potential recovery value.