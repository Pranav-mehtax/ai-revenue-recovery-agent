# AI Revenue Recovery Agent

A Razorpay AI Buildathon prototype that prioritizes failed payments with the existing ML model, retrieves policy context with the existing TF-IDF RAG system, asks Gemini for a recommendation, applies deterministic guardrails, and executes only a simulated/Test Mode action.

## Run (Windows PowerShell)

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn backend.main:app --reload
.\venv\Scripts\python.exe -m streamlit run frontend\dashboard.py
```

The API is available at `http://127.0.0.1:8000/docs`; the dashboard opens in Streamlit. Use **Run Recovery Batch** to process the top ML-prioritized opportunities. Audit events are stored in `data/recovery_audit_log.csv`.

This prototype never initiates real payment processing. Expected recovery is an ML economic estimate, while recovered amount is only recorded after simulated execution.
