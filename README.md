# AI Revenue Recovery Agent

An AI-powered payment recovery system that intelligently identifies failed payment recovery opportunities, predicts the probability of successful recovery using Machine Learning, retrieves relevant payment-recovery policies using RAG, uses an LLM to recommend the best recovery action, validates that recommendation through deterministic guardrails, and finally executes the approved action in a safe simulated environment.

> **Razorpay AI Buildathon Project**
>
> **Important:** All payment actions in this project are simulated. No real payment transactions are performed.

---

## 1. Project Overview

Payment failures result in lost revenue and can negatively affect customer experience.

A failed payment does not always mean that the transaction should simply be abandoned. Different failure types require different recovery strategies.

For example:

- `NETWORK_ERROR` may be worth retrying.
- `INSUFFICIENT_FUNDS` may require another payment method.
- `EXPIRED_CARD` may require a payment-method update.
- `DUPLICATE` should not be retried.
- `SUSPECTED_FRAUD` should be escalated for manual review.

The goal of this project is to build an intelligent **Revenue Recovery Agent** that makes these decisions automatically while maintaining strict safety controls.

The system combines:

1. **Machine Learning** — predicts recovery probability and expected recovery value.
2. **RAG (Retrieval-Augmented Generation)** — retrieves relevant payment policies and recovery knowledge.
3. **LLM (Gemini)** — reasons over the payment context and retrieved knowledge to recommend an action.
4. **Guardrails** — independently validates the LLM recommendation.
5. **Action Executor** — safely simulates the approved recovery action.
6. **Backend API** — exposes the recovery pipeline through APIs.
7. **Streamlit Dashboard** — provides a visual control room for monitoring and executing recovery operations.
8. **Audit Logging** — records decisions and execution results.

---

# 2. System Architecture

```text
                    FAILED PAYMENTS
                          |
                          v
                 +-----------------+
                 |   ML PRIORITY   |
                 |     ENGINE      |
                 +--------+--------+
                          |
                          v
              Recovery Probability
              Expected Recovery Value
              Priority Level
                          |
                          v
                 +-----------------+
                 |       RAG       |
                 | Knowledge Base  |
                 +--------+--------+
                          |
                          v
                Relevant Policies
                Failure Information
                Recovery Strategies
                          |
                          v
                 +-----------------+
                 |       LLM       |
                 |     Gemini      |
                 +--------+--------+
                          |
                          v
                 Recommended Action
                          |
                          v
                 +-----------------+
                 |   GUARDRAILS    |
                 | Safety + Policy |
                 |     Checks      |
                 +--------+--------+
                          |
                    +-----+-----+
                    |           |
                 APPROVED     BLOCKED
                    |           |
                    v           v
             +------------+   MANUAL
             |  EXECUTOR  |   REVIEW
             +-----+------+
                   |
                   v
             Simulated Action
                   |
                   v
             Execution Result
                   |
                   v
              Audit Logging
```

The core design separates **prediction, reasoning, safety, and execution**.

```text
ML
 |
 | Which payments are worth attempting to recover?
 v
RAG
 |
 | What does the project knowledge say about this failure?
 v
LLM
 |
 | What action should be recommended?
 v
Guardrails
 |
 | Is that action safe and allowed?
 v
Executor
 |
 | Simulate the approved action
 v
Audit Log
```

---

# 3. Project Structure

```text
Razor_pay_agent/
|
+-- backend/
|   +-- actions/
|   |   +-- action_executor.py
|   |
|   +-- agent/
|   |   +-- recovery_pipeline.py
|   |
|   +-- api/
|   |
|   +-- database/
|   |
|   +-- policies/
|   |
|   +-- main.py
|
+-- data/
|   +-- customers.csv
|   +-- merchants.csv
|   +-- payments.csv
|   +-- ml_dataset.csv
|   +-- prioritized_payments.csv
|   +-- recovery_audit_log.csv
|   +-- recovery_outcomes.csv
|   +-- generate_data.py
|
+-- evaluation/
|   +-- model_comparison.csv
|
+-- frontend/
|   +-- dashboard.py
|
+-- ml/
|   +-- ML model/training files
|
+-- rag/
|   +-- knowledge_base/
|   +-- rag_engine.py
|   +-- llm_agent.py
|   +-- guardrails.py
|
+-- .env
+-- .gitignore
+-- README.md
+-- requirements.txt
```

> Do not commit `.env` to GitHub. It should contain secrets such as API keys and is excluded through `.gitignore`.

---

# 4. Machine Learning Layer

The ML component identifies valuable recovery opportunities.

For each failed payment, the model estimates:

- Recovery probability
- Expected recovery value
- Priority

## Recovery Probability

Recovery probability represents the model's estimate of how likely a failed payment is to be successfully recovered.

Example:

```text
Payment Amount: ₹60,063.19
Recovery Probability: 77.48%
```

This means the model estimates approximately a 77.48% probability of recovery based on the features used by the model.

It is a prediction, not a guarantee.

---

## Expected Recovery Value

The project calculates:

```text
Expected Recovery Value
= Payment Amount × Recovery Probability
```

Example:

```text
Payment Amount       = ₹60,063.19
Recovery Probability = 0.7748

Expected Recovery
≈ ₹60,063.19 × 0.7748
≈ ₹46,537.11
```

This is primarily a **prioritization estimate**.

It helps distinguish between opportunities such as:

```text
Payment A
₹10,000 × 90%
≈ ₹9,000 expected recovery

Payment B
₹100,000 × 50%
≈ ₹50,000 expected recovery
```

Although Payment A has a higher probability, Payment B has a larger potential recovery value.

Therefore, expected recovery helps prioritize opportunities according to potential business impact.

> Expected recovery is not money that has already been recovered.

---

# 5. Payment Prioritization

After predictions are generated, payments can be ranked according to their recovery opportunity.

Example:

```text
Payment ID     Amount       Probability    Expected Recovery    Priority
---------------------------------------------------------------------------
PAY_004769     ₹60,063      77.48%         ₹46,537              HIGH
PAY_017627     ₹37,491      78.79%         ₹29,539              MEDIUM
PAY_007811     ₹46,180      62.10%         ₹28,678              MEDIUM
```

The highest-value opportunities can then be passed to the recovery agent.

---

# 6. RAG Layer

The project uses **Retrieval-Augmented Generation (RAG)** to provide the LLM with relevant domain knowledge.

Instead of asking the LLM to make a decision entirely from general pretrained knowledge, the system retrieves relevant information from a project-specific knowledge base.

The knowledge base contains information such as:

- Payment failure reasons
- Recovery strategies
- Merchant policies
- Recovery rules
- Action restrictions

Example:

```text
NETWORK_ERROR
    -> RETRY may be appropriate.

INSUFFICIENT_FUNDS
    -> Request another payment method.

EXPIRED_CARD
    -> Ask the customer to update the payment method.

DUPLICATE
    -> Do not retry.

SUSPECTED_FRAUD
    -> Manual review.
```

## RAG Pipeline

```text
Payment Context
      |
      v
Query Creation
      |
      v
Knowledge Base
      |
      v
Similarity Retrieval
      |
      v
Top Relevant Knowledge Chunks
      |
      v
LLM Context
```

For example, if:

```text
Failure Type = NETWORK_ERROR
```

the RAG engine retrieves knowledge related to network failures and recovery strategies.

The retrieved information is then provided to Gemini along with the payment information.

---

# 7. Why RAG?

Without RAG:

```text
Payment -> LLM -> Action
```

The LLM would primarily rely on its general knowledge.

With RAG:

```text
Payment
   |
   v
Relevant project knowledge
   |
   v
LLM
   |
   v
Recommended Action
```

This grounds the recommendation in the application's specific recovery policies.

It also allows the knowledge base to be updated independently of the LLM model.

---

# 8. LLM Layer

The project uses **Google Gemini** as the reasoning component.

The LLM receives:

1. Payment information
2. ML prediction
3. Recovery probability
4. Expected recovery value
5. Failure type
6. Retry count
7. Retrieved RAG knowledge
8. Recovery policies

The LLM then produces a structured recommendation.

Typical output:

```text
FAILURE_EXPLANATION:
The payment failed due to a temporary network issue.

RECOMMENDED_ACTION:
RETRY

REASONING:
The retrieved knowledge indicates that retry is appropriate for
network errors. The payment has not been retried before and has
a high recovery probability.

RISK:
Low

AUTOMATIC_ACTION:
YES
```

The LLM is therefore responsible for **reasoning and recommendation**, not unrestricted execution.

---

# 9. Guardrail Layer

The guardrail layer is one of the most important safety components.

The LLM can recommend an action, but the recommendation is not automatically trusted.

The guardrail independently checks whether the action is allowed.

Conceptually:

```text
LLM Recommendation
       |
       v
Guardrail Checks
       |
       +---- Allowed ----> Executor
       |
       +---- Blocked ----> Manual Review
```

The guardrail considers payment and decision attributes such as:

- Failure type
- Recovery probability
- Retry count
- Payment amount
- LLM-recommended action
- High-risk conditions
- Duplicate-payment conditions
- Fraud-related conditions
- Recovery policy constraints

The exact thresholds and policies are implemented in `rag/guardrails.py`.

---

# 10. Example Guardrail Decisions

## Example 1 — Network Error

```text
Failure: NETWORK_ERROR
Recovery Probability: 77.48%
Retry Count: 0
LLM Action: RETRY
```

The action can pass the guardrail.

```text
GUARDRAIL STATUS: PASSED
FINAL ACTION: RETRY
```

---

## Example 2 — Suspected Fraud

```text
Failure: SUSPECTED_FRAUD
LLM Action: RETRY
```

The guardrail should prevent an unsafe automatic retry.

```text
GUARDRAIL STATUS: BLOCKED
FINAL ACTION: MANUAL_REVIEW
```

---

## Example 3 — Duplicate Payment

```text
Failure: DUPLICATE
LLM Action: RETRY
```

The guardrail should reject the retry because retrying a duplicate payment can create an unsafe situation.

```text
GUARDRAIL STATUS: BLOCKED
FINAL ACTION: MANUAL_REVIEW
```

The key principle is:

> The LLM recommends. The guardrail decides whether the recommendation is allowed to reach the executor.

---

# 11. Action Executor

Once an action passes the guardrails, it is sent to the action executor.

The executor currently operates in **simulation/test mode**.

Possible simulated actions include:

```text
RETRY
NOTIFICATION
PAYMENT_METHOD_UPDATE
MANUAL_REVIEW
NO_ACTION
```

Example:

```text
Payment ID           : PAY_TEST
Amount               : ₹50,000.00
Final Action         : RETRY
Recovery Probability : 75.00%

EXECUTION RESULT
----------------
payment_id: PAY_TEST
action: RETRY
status: RECOVERED
recovered_amount: 50000.0
message: Payment successfully recovered after retry.
```

No real payment gateway transaction is performed.

---

# 12. Important Difference: Expected vs Recovered Amount

The dashboard distinguishes between:

### Expected Recovery

This comes from the ML model:

```text
Expected Recovery
= Amount × Recovery Probability
```

It is a prediction used for prioritization.

### Recovered Amount

This comes from the simulated executor after an action is run.

For example:

```text
Amount = ₹50,000
Recovery Probability = 75%

Expected Recovery = ₹37,500
```

If the simulated retry succeeds:

```text
Recovered Amount = ₹50,000
```

If it fails:

```text
Recovered Amount = ₹0
```

Therefore:

```text
Expected Recovery != Guaranteed Recovery
```

---

# 13. Recovery Pipeline

The integrated orchestration is implemented in:

```text
backend/agent/recovery_pipeline.py
```

It connects the existing project components into a single workflow.

Conceptually:

```text
Prioritized Payment
       |
       v
Payment Context
       |
       v
RAG Retrieval
       |
       v
Gemini Recommendation
       |
       v
Guardrail Evaluation
       |
       v
Approved / Blocked Action
       |
       v
Simulated Execution
       |
       v
Audit Log
```

The pipeline also supports:

- Single-payment processing
- Batch processing
- Gemini failure fallback
- Deterministic guardrail enforcement
- Audit persistence
- Metrics generation

---

# 14. Gemini Failure Handling

LLM services can occasionally be unavailable.

The system therefore does not allow an LLM failure to bypass the safety layer.

If Gemini is unavailable or fails repeatedly, the pipeline can safely fall back to:

```text
MANUAL_REVIEW
```

The failure is also recorded in the audit information.

This follows the principle:

```text
LLM failure
    |
    v
Safe fallback
    |
    v
Manual Review
```

rather than:

```text
LLM failure
    |
    v
Unsafe automatic execution
```

---

# 15. Backend

The backend provides an API layer around the recovery system.

The main backend entry point is:

```text
backend/main.py
```

The API allows the frontend to communicate with the recovery pipeline.

Typical operations include:

```text
Health Check
Get Payments
Recover Payment
Run Recovery Batch
Get Metrics
Get Audit Information
```

The backend acts as the bridge between:

```text
Frontend
    |
    v
API
    |
    v
Recovery Pipeline
    |
    v
ML + RAG + Gemini + Guardrails + Executor
```

The backend is intentionally separated from the frontend so that the recovery logic can be used by other clients in the future.

---

# 16. API Endpoints

The backend exposes endpoints for the main system operations.

Typical endpoint categories include:

```text
GET  /health
GET  /payments
POST /recover
POST /batch
GET  /metrics
GET  /audit
```

The exact implementation is available in:

```text
backend/main.py
```

and the integrated recovery logic is available in:

```text
backend/agent/recovery_pipeline.py
```

---

# 17. Frontend Dashboard

The frontend is implemented using **Streamlit**.

Main file:

```text
frontend/dashboard.py
```

The dashboard acts as an **AI Revenue Recovery Control Room**.

It provides visibility into:

- Failed payment value
- Expected recovery
- Simulated recovered amount
- Recovery rate
- Number of processed payments
- Highest-priority opportunities
- ML predictions
- Recommended actions
- Final guardrail-approved actions
- Execution status
- RAG sources
- LLM reasoning
- Audit trail

---

# 18. Dashboard Workflow

The dashboard provides a visual representation of the complete AI pipeline.

A typical workflow is:

```text
1. Load prioritized payments
          |
          v
2. Select highest-priority opportunities
          |
          v
3. Click "Run Recovery Batch"
          |
          v
4. Payments go through:
   ML context
   RAG
   Gemini
   Guardrails
   Executor
          |
          v
5. Dashboard displays results
```

The dashboard also supports running an individual selected payment.

---

# 19. Dashboard Metrics

The dashboard displays metrics such as:

### Failed Payment Value

Total value of the selected/processed failed payments.

### Expected Recovery

Sum of ML-estimated expected recovery values.

### Recovered (Simulated)

Total amount recovered by the simulated executor.

### Recovery Rate

A summary of simulated recovery performance.

### Payments Processed

Number of payments that have gone through the recovery workflow.

> These metrics represent the project's simulated environment and should not be interpreted as real production payment recovery statistics.

---

# 20. Audit Trail

The system records important information about each recovery decision.

Audit information can include:

```text
timestamp
payment_id
amount
failure_type
recovery_probability
expected_recovery_value
priority
RAG sources
LLM recommendation
guardrail decision
final action
execution status
recovered amount
```

This is useful for:

- Debugging
- Explainability
- Monitoring
- Evaluating AI decisions
- Understanding why an action was taken
- Demonstrating safety controls

---

# 21. Example End-to-End Decision

Consider:

```text
Payment ID: PAY_004769
Amount: ₹60,063.19
Failure: NETWORK_ERROR
Retry Count: 0
```

### Step 1 — ML

The ML model predicts:

```text
Recovery Probability = 77.48%
```

Expected recovery:

```text
₹60,063.19 × 0.7748
≈ ₹46,537.11
```

The payment receives a high priority.

---

### Step 2 — RAG

The RAG engine retrieves relevant knowledge:

```text
failure_reasons.md
recovery_strategies.md
merchant_policies.md
```

The retrieved information indicates that network errors are generally suitable for retry.

---

### Step 3 — Gemini

Gemini receives the payment context and retrieved knowledge.

It recommends:

```text
RECOMMENDED_ACTION = RETRY
```

---

### Step 4 — Guardrail

The guardrail checks:

```text
Failure type
Retry count
Recovery probability
Payment conditions
LLM action
Safety policies
```

The action passes.

```text
GUARDRAIL STATUS = PASSED
FINAL ACTION = RETRY
```

---

### Step 5 — Executor

The simulated executor runs:

```text
RETRY
```

It returns either:

```text
RECOVERED
```

or:

```text
FAILED
```

---

### Step 6 — Audit

The complete decision and execution result are stored for later inspection.

---

# 22. Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Razor_pay_agent
```

Create a virtual environment:

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 23. Environment Variables

Create a `.env` file in the project root.

Example:

```env
GEMINI_API_KEY=your_api_key_here
```

Never commit this file to GitHub.

The `.gitignore` should contain:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

---

# 24. Running the Backend

From the project root:

```bash
uvicorn backend.main:app --reload
```

The API server will start locally.

A typical local address is:

```text
http://127.0.0.1:8000
```

FastAPI also provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

---

# 25. Running the Frontend

Open another terminal from the project root.

Run:

```bash
streamlit run frontend/dashboard.py
```

Streamlit will display a local URL, typically:

```text
http://localhost:8501
```

Open that URL in a browser.

---

# 26. Recommended Demo Flow

For a project demonstration:

### Step 1

Start the backend:

```bash
uvicorn backend.main:app --reload
```

### Step 2

Start the dashboard:

```bash
streamlit run frontend/dashboard.py
```

### Step 3

Open the Streamlit URL.

### Step 4

Show the top-level metrics:

```text
Failed Payment Value
Expected Recovery
Recovered (Simulated)
Recovery Rate
Payments Processed
```

### Step 5

Show the recovery opportunities table.

Point out:

```text
Payment ID
Amount
Failure Reason
Recovery Probability
Expected Recovery
Priority
Recommended Action
Final Action
Status
```

### Step 6

Run a recovery batch.

Explain:

```text
The selected payments are sent through the complete AI pipeline:
ML -> RAG -> Gemini -> Guardrails -> Executor
```

### Step 7

Open a payment decision.

Show:

```text
Payment & ML
Guardrail & Execution
RAG Context and LLM Reasoning
Audit Trail
```

This demonstrates that the system is explainable rather than a black box.

---

# 27. Example Demo Explanation

A concise explanation for a presentation:

> "Our system starts with failed payments and uses an ML model to estimate recovery probability and expected recovery value. This allows us to prioritize the payments with the highest recovery opportunity. The selected payment is then passed to a RAG system, which retrieves relevant payment failure and recovery policies from our knowledge base. Gemini uses this context together with the payment information to recommend an action such as retry, notification, payment-method update, or manual review. However, the LLM does not directly execute anything. A deterministic guardrail layer independently validates the recommendation against safety and payment policies. Only approved actions reach the simulated executor. Finally, the complete decision and execution result are stored in the audit trail and displayed through our Streamlit control room."

---

# 28. Technology Stack

## Machine Learning

- Python
- Pandas
- Scikit-learn
- Supervised learning
- Probability prediction
- Recovery prioritization

## Generative AI

- Google Gemini
- LLM-based reasoning
- Structured recommendations

## RAG

- Python
- Knowledge-base documents
- Text chunking
- Embeddings/similarity retrieval
- Context augmentation

## Backend

- Python
- FastAPI
- Uvicorn
- REST APIs

## Frontend

- Streamlit

## Data

- CSV datasets
- Simulated payment data
- Recovery outcomes
- Audit logs

---

# 29. Safety Design

A major design principle is that **LLM output is not trusted blindly**.

The architecture uses multiple stages:

```text
Prediction
    ↓
Retrieval
    ↓
LLM Recommendation
    ↓
Deterministic Guardrails
    ↓
Execution
```

This reduces the risk of an LLM making an unsafe payment-related decision.

Examples of actions that should receive additional scrutiny include:

- Suspected fraud
- Duplicate transactions
- Low-confidence recovery opportunities
- Excessive retries
- High-risk payment scenarios

The guardrail layer provides a deterministic control point before execution.

---

# 30. Explainability

The system provides multiple levels of explainability.

### ML

Shows:

```text
Recovery Probability
Expected Recovery
Priority
```

### RAG

Shows:

```text
Retrieved knowledge sources
Relevant knowledge chunks
```

### LLM

Shows:

```text
Failure explanation
Recommended action
Reasoning
Risk
Automatic action decision
```

### Guardrails

Shows:

```text
Guardrail status
Final action
Guardrail reason
```

### Executor

Shows:

```text
Execution status
Recovered amount
Execution message
```

This makes it possible to trace a decision from the original payment all the way to the final simulated result.

---

# 31. Failure Handling

The system is designed to fail safely.

Examples:

### Gemini unavailable

Fallback:

```text
MANUAL_REVIEW
```

### Guardrail rejects LLM action

Result:

```text
Action is not executed automatically.
```

### Simulated retry fails

Result:

```text
Execution status = FAILED
Recovered amount = ₹0
```

The system therefore separates:

```text
Recommendation
```

from:

```text
Approved action
```

and:

```text
Execution result
```

---

# 32. Current Project Scope

This project is a prototype / buildathon demonstration.

The action executor is simulated.

Therefore:

- No real payment is processed.
- No real customer is charged.
- No real refund is performed.
- Recovery outcomes are simulated.
- The ML model operates on project data.
- The backend runs locally unless deployed.
- Gemini requires a valid API key.

A production system would require additional components such as:

- Real payment gateway integration
- Authentication and authorization
- Production databases
- Rate limiting
- Secrets management
- Distributed monitoring
- More extensive fraud detection
- Human approval workflows
- Production-grade observability
- Security and compliance controls
- Extensive offline and online model evaluation

---

# 33. Future Improvements

Possible future improvements include:

### Better ML Models

- Gradient boosting
- XGBoost/LightGBM
- Calibrated probability models
- Better feature engineering
- Online learning

### Better RAG

- Vector databases
- Better embedding models
- Metadata filtering
- Hybrid retrieval
- Reranking

### Agent Improvements

- Multi-step recovery strategies
- Adaptive retry scheduling
- Customer communication generation
- Dynamic payment-method recommendations

### Safety

- More sophisticated policy engines
- Human-in-the-loop approval
- Transaction-level risk scoring
- Rate limits
- Stronger fraud controls

### Production

- PostgreSQL
- Redis
- Docker
- Cloud deployment
- Authentication
- Monitoring
- CI/CD

---

# 34. Key Takeaways

The main idea of the project is:

```text
ML decides WHAT IS WORTH RECOVERING
                ↓
RAG provides WHAT THE POLICIES SAY
                ↓
LLM recommends WHAT TO DO
                ↓
GUARDRAILS decide WHETHER IT IS SAFE
                ↓
EXECUTOR performs the SIMULATED ACTION
                ↓
AUDIT LOG records WHAT HAPPENED
```

This architecture combines predictive ML, retrieval-augmented generation, LLM reasoning, deterministic safety controls, and an actionable interface into one revenue-recovery workflow.

---

# 35. Project Status

Current implementation includes:

- [x] Payment dataset
- [x] ML recovery prediction
- [x] Recovery prioritization
- [x] Expected recovery calculation
- [x] RAG knowledge base
- [x] RAG retrieval
- [x] Gemini LLM reasoning
- [x] Guardrail validation
- [x] Simulated action executor
- [x] Integrated recovery pipeline
- [x] Gemini failure fallback
- [x] Audit logging
- [x] Recovery metrics
- [x] FastAPI backend
- [x] Streamlit dashboard
- [x] Batch recovery workflow
- [x] Single-payment execution workflow
- [x] Safety-oriented execution flow

---

## Disclaimer

This project is a simulated AI revenue recovery prototype developed for the Razorpay AI Buildathon.

It is intended for demonstration and experimentation only and does not perform real payment recovery or financial transactions.
