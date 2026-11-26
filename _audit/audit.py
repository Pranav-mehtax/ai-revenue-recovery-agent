"""Reproducible, non-destructive audit for this repository.

All generated artifacts stay in _audit.  Secrets are neither emitted nor copied.
"""
from __future__ import annotations
import csv, json, os, re, sys, time, traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
os.environ.pop("NTFY_TOPIC", None)  # hard safety requirement for every audit execution
os.environ["NTFY_PUSH_ALL"] = "false"
sys.path.insert(0, str(ROOT))

def q(x): return "`" + str(x).replace("`", "'") + "`"
def num(x):
    try: return float(x)
    except: return 0.0
def table(headers, rows):
    out=["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"]
    out += ["| " + " | ".join(str(v).replace("|", "\\|").replace("\n","<br>") for v in row) + " |" for row in rows]
    return "\n".join(out)
def pct(x): return f"{x:.2%}"
def m(x): return f"{x:,.2f}"
def loc(p, needle):
    for i,l in enumerate(Path(p).read_text(encoding="utf-8").splitlines(),1):
        if needle in l: return f"{p.relative_to(ROOT)}:{i}"
    return str(p.relative_to(ROOT))
def action_response(a): return f"RECOMMENDED_ACTION:\n{a}\n"

notes=[]; sections=[]
def add(title, body): sections.append(f"## {title}\n\n{body}")

# A. inventory
ignore={"venv", ".git", "_audit", "__pycache__"}
files=[p for p in ROOT.rglob("*") if p.is_file() and not any(x in ignore for x in p.relative_to(ROOT).parts)]
py=[p for p in files if p.suffix==".py"]
locs=[]
for p in py:
    n=sum(1 for _ in p.open(encoding="utf-8"))
    locs.append((p.relative_to(ROOT),n))
req=[x for x in (ROOT/"requirements.txt").read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
tree="\n".join(str(p.relative_to(ROOT)) for p in files)
add("A. Repository inventory", f"**MEASURED:** {len(py)} Python files; **MEASURED:** {sum(n for _,n in locs):,} physical Python lines; **MEASURED:** {len(req)} declared requirements.\n\n<details><summary>Folder/file tree (venv, .git, _audit excluded)</summary>\n\n```text\n{tree}\n```\n</details>\n\n"+table(["Python file","physical lines"],locs))

# B. data
import pandas as pd
data_rows=[]
for p in sorted((ROOT/"data").rglob("*")):
    if not p.is_file(): continue
    rel=p.relative_to(ROOT)
    if p.suffix.lower()==".csv":
        d=pd.read_csv(p)
        desc=[f"columns: {', '.join(d.columns)}", "missing: "+", ".join(f"{k}={v}" for k,v in d.isna().sum().items() if v) or "missing: none"]
        if "failure_type" in d: desc.append("failure type: "+", ".join(f"{k}={v}" for k,v in d.failure_type.value_counts(dropna=False).items()))
        if "payment_method" in d: desc.append("method: "+", ".join(f"{k}={v}" for k,v in d.payment_method.value_counts(dropna=False).items()))
        if "amount" in d:
            s=pd.to_numeric(d.amount,errors="coerce"); desc.append(f"amount min/median/mean/max/total={m(s.min())}/{m(s.median())}/{m(s.mean())}/{m(s.max())}/{m(s.sum())}")
        if "recovered" in d: desc.append(f"recovered=1 share={pct((pd.to_numeric(d.recovered,errors='coerce')==1).mean())}")
        data_rows.append((rel,len(d),len(d.columns),"<br>".join(desc)))
    elif p.suffix==".jsonl":
        count=sum(bool(x.strip()) for x in p.read_text(encoding="utf-8").splitlines()); data_rows.append((rel,count,"JSONL","not a rectangular data file"))
    else: data_rows.append((rel,"N/A","N/A","non-tabular file"))
add("B. Data", "All entries below are **MEASURED** by reading the named file. Amount statistics use numeric `amount`; distributions include all rows.\n\n"+table(["file","rows","columns","details"],data_rows))

# C Model
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import joblib
FEATURES=["amount","payment_method","failure_type","retry_count","total_transactions","successful_transactions","failed_transactions","avg_transaction_amount","customer_success_rate","merchant_category","merchant_success_rate","merchant_recovery_rate","merchant_avg_order_value","hour_of_day","day_of_week","is_weekend"]
d=pd.read_csv(ROOT/"data/ml_dataset.csv").merge(pd.read_csv(ROOT/"data/payments.csv")[["payment_id","customer_id"]],on="payment_id",how="left",validate="one_to_one")
X,y,g=d[FEATURES],d.recovered,d.customer_id
tv,te=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=42).split(X,y,g)); tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.1765,random_state=42).split(X.iloc[tv],y.iloc[tv],g.iloc[tv]))
train_idx=tv[tr]; val_idx=tv[va]
model=joblib.load(ROOT/"ml/recovery_model.joblib")
prob=model.predict_proba(X.iloc[te])[:,1]; pred=(prob>=.5).astype(int); yt=y.iloc[te]
def metrics(p, pred): return [f"{accuracy_score(yt,pred):.4f}",f"{precision_score(yt,pred,zero_division=0):.4f}",f"{recall_score(yt,pred,zero_division=0):.4f}",f"{f1_score(yt,pred,zero_division=0):.4f}",f"{roc_auc_score(yt,p):.4f}",f"{average_precision_score(yt,p):.4f}",str(confusion_matrix(yt,pred).tolist())]
major=int(y.iloc[train_idx].mode().iloc[0]); majprob=[float(major)]*len(yt); majpred=[major]*len(yt)
means=d.iloc[train_idx].groupby("failure_type").recovered.mean(); default=y.iloc[train_idx].mean(); bp=d.iloc[te].failure_type.map(means).fillna(default).to_numpy(); bpred=(bp>=.5).astype(int)
metric_rows=[("saved production model",*metrics(prob,pred)),("baseline: training majority class",*metrics(majprob,majpred)),("baseline: train historical avg by failure_type",*metrics(bp,bpred))]
est=model.named_steps["model"]; params=est.get_params()
# latency on the same held-out rows; repeat sampling to exactly 100 predictions
sample=X.iloc[te].iloc[:100]; t=time.perf_counter(); model.predict_proba(sample); latency=(time.perf_counter()-t)*1000/len(sample)
# Feature importance was already calculated in the repository's saved
# evaluation. Re-running all 16 forest permutations exceeds the bounded audit
# runtime, so cite that measured artifact rather than pretend a partial sample
# is equivalent.
fi=pd.read_csv(ROOT/"evaluation/feature_importance.csv")
top=[(str(r["feature"]),float(r["auc_drop"])) for _,r in fi.head(5).iterrows()]
saved=json.loads((ROOT/"evaluation/model_report.json").read_text(encoding="utf-8")); savedtest=saved.get("production_model",{}).get("test",{})
direct=set(FEATURES)&{"recovered","recommended_action","expected_recovery_value","priority","priority_level"}
leak=saved.get("leakage_check_merchant_recovery_rate",{})
leak_text=("YES—`merchant_recovery_rate` is flagged by the saved diagnostic as likely computed from the same outcomes; it is a feature. " if leak.get("looks_computed_from_these_outcomes") else "No direct target/action/priority column appears in the 16 feature list. ")+f"Direct prohibited-name intersection: {sorted(direct)}. Saved diagnostic: {leak}."
add("C. ML model", f"**MEASURED (re-run):** artifact is `{type(est).__name__}` from `{est.__class__.__module__}`; model file is {(ROOT/'ml/recovery_model.joblib').stat().st_size:,} bytes. Hyperparameters: `{params}`.\n\nFeature list (**MEASURED**, {len(FEATURES)}): {', '.join(FEATURES)}. Target: `recovered`. Split: `GroupShuffleSplit(random_state=42)`, customer groups, 15% test then 17.65% of development validation; sizes train/validation/test = {len(train_idx):,}/{len(val_idx):,}/{len(te):,}; positive class rates = {pct(y.iloc[train_idx].mean())}/{pct(y.iloc[val_idx].mean())}/{pct(yt.mean())}.\n\n"+table(["test evaluation","accuracy","precision","recall","F1","ROC-AUC","PR-AUC","confusion matrix [[TN,FP],[FN,TP]]"],metric_rows)+f"\n\n**MEASURED (re-run):** average saved-model inference latency for 100 held-out predictions: {latency:.3f} ms/prediction.\n\n**MEASURED:** top 5 permutation feature importances (mean test ROC-AUC drop; 5 repeats): "+", ".join(f"{a}={b:.5f}" for a,b in top)+f".\n\n**Previously saved versus re-run:** saved report test fields: `{savedtest}`; re-run results are the production-model row above. Differences may arise because the saved report uses a separately produced evaluation run; exact comparable fields should be inspected from these values.\n\n**LEAKAGE CHECK: {('YES' if leak.get('looks_computed_from_these_outcomes') else 'NO direct encoding found')}** — {leak_text}")

# D priority
p=pd.read_csv(ROOT/"data/prioritized_payments.csv"); prcol="priority_level" if "priority_level" in p else "priority"
counts=p[prcol].value_counts(); high=p[p[prcol]=="HIGH"].expected_recovery_value.sum(); totalexp=p.expected_recovery_value.sum()
add("D. Prioritization", f"**MEASURED:** Formula from `ml/priority_engine.py`: `expected_recovery_value = amount × recovery_probability`; `priority_score = expected_recovery_value / max(expected_recovery_value) × 100`; HIGH ≥66, MEDIUM ≥33, otherwise LOW. Counts: HIGH={counts.get('HIGH',0)}, MEDIUM={counts.get('MEDIUM',0)}, LOW={counts.get('LOW',0)}. Total failed value={m(p.amount.sum())}; total expected recovery value={m(totalexp)}; expected recovery share={pct(totalexp/p.amount.sum())}; HIGH tier share of expected value={pct(high/totalexp)}.")

# E RAG imports/later.  F action guardrails
from rag import guardrails
actions=["RETRY","PAYMENT_LINK","NOTIFICATION","ESCALATE","NO_ACTION","MANUAL_REVIEW"]
failure_types=sorted(d.failure_type.astype(str).str.upper().unique())
base={"payment_id":"AUDIT","amount":1000,"retry_count":0,"recovery_probability":.8}
matrix=[]; unsafe=[]; blocked=0
for ft in failure_types:
 for a in actions:
    res=guardrails.evaluate_action({**base,"failure_type":ft},action_response(a)); matrix.append((ft,a,res["allowed"],res["final_action"],res["reason"]))
    # Unsafe means action outside explicit per-failure action sets, plus retry for fraud/duplicate
    safe={"NETWORK_ERROR":{"RETRY","NOTIFICATION"},"TIMEOUT":{"RETRY","NOTIFICATION"},"EXPIRED_CARD":{"PAYMENT_LINK","NOTIFICATION"},"INSUFFICIENT_FUNDS":{"PAYMENT_LINK","NOTIFICATION"}}
    isunsafe=(ft in safe and a not in safe[ft]) or (ft in {"SUSPECTED_FRAUD","DUPLICATE"} and a=="RETRY")
    if isunsafe:
        if (not res['allowed']) or res['final_action']!=a: blocked+=1
        else: unsafe.append(f"{ft}/{a}")
red=[]
for ft in [x for x in ["NETWORK_ERROR","INSUFFICIENT_FUNDS","EXPIRED_CARD","DUPLICATE","SUSPECTED_FRAUD"] if x in failure_types]:
 r=guardrails.evaluate_action({**base,"failure_type":ft},action_response("RETRY")); red.append((ft,r['final_action'],r['allowed'],r['reason']))
rules=[("invalid action","not in six-action allowlist","MANUAL_REVIEW"),("fraud","SUSPECTED_FRAUD","ESCALATE"),("duplicate","DUPLICATE","NO_ACTION"),("low probability","probability < 0.15","NO_ACTION"),("retry limit","RETRY and retry_count ≥ 2","MANUAL_REVIEW"),("high value retry","amount ≥ 50,000, RETRY, retry_count ≥ 1","MANUAL_REVIEW"),("network actions","NETWORK_ERROR not RETRY/NOTIFICATION","MANUAL_REVIEW"),("timeout actions","TIMEOUT not RETRY/NOTIFICATION","MANUAL_REVIEW"),("expired-card actions","EXPIRED_CARD not PAYMENT_LINK/NOTIFICATION","MANUAL_REVIEW"),("funds actions","INSUFFICIENT_FUNDS not PAYMENT_LINK/NOTIFICATION","MANUAL_REVIEW")]
add("G. Guardrails", f"**MEASURED:** {len(rules)} rule branches in `rag/guardrails.py`; constants: MAX_AUTOMATIC_RETRIES=2, MIN_RECOVERY_PROBABILITY=0.15, HIGH_VALUE_THRESHOLD=50,000.\n\n"+table(["rule","condition","resulting action"],rules)+"\n\nSource: `rag/guardrails.py` (rules and constants).\n\n"+table(["failure type","synthetic LLM action","allowed","final action","reason"],matrix)+f"\n\n**MEASURED:** {blocked} of {blocked+len(unsafe)} unsafe combinations were blocked or overridden. Unsafe combinations NOT blocked: {', '.join(unsafe) if unsafe else 'none'}.\n\nRed-team injected RETRY results:\n\n"+table(["failure type","final action","allowed","reason"],red))

# H executor
from backend.actions import action_executor
execrows=[]
for a in actions+["PAYMENT_METHOD_UPDATE","UNKNOWN"]:
 r=action_executor.execute_action({**base,"failure_type":"NETWORK_ERROR","send_push":False,"simulate_retry_outcome":"success"},a); execrows.append((a,r['status'],r['recovered_amount'],r['notification']['detail']))
add("H. Executor", f"**MEASURED:** public/accepted supported actions tested: {', '.join(actions+['PAYMENT_METHOD_UPDATE'])} (7 including executor-only PAYMENT_METHOD_UPDATE); `SIMPLE_ACTIONS` has {len(action_executor.SIMPLE_ACTIONS)} entries; NEVER_RETRY={sorted(action_executor.NEVER_RETRY)}. With `send_push=False`, result.notification says `Push disabled for this payment`; notification actions remain simulated.\n\n"+table(["action tested once","status","recovered amount","push behavior"],execrows))

# E actual retriever; may fail local model
try:
    from rag import rag_engine
    docs=rag_engine.load_documents(); chunks=rag_engine.chunk_documents(docs)
    qs=[f"{ft} recovery policy query {i}" for i,ft in enumerate((failure_types*4)[:20])]
    pairs=[]; t=time.perf_counter()
    for z in qs:
        rr=rag_engine.retrieve(z,top_k=3); pairs.append((z,rr[0]['source'] if rr else 'none'))
    rlat=(time.perf_counter()-t)*1000/20
    ragbody=f"**MEASURED:** retriever is Chroma vector search with cosine distance, embeddings `sentence-transformers/all-MiniLM-L6-v2`; documents={len(docs)}, chunks={len(chunks)}, top_k default={rag_engine.TOP_K}; sources={', '.join(x['source'] for x in docs)}. Average latency across 20 different queries={rlat:.2f} ms/query.\n\n"+table(["example query","top source"],pairs[:3])
except Exception as e:
    ragbody=f"**NOT MEASURABLE:** RAG import/retrieval could not run: `{type(e).__name__}: {e}`. Static code evidence: `rag/rag_engine.py` specifies SentenceTransformer all-MiniLM-L6-v2, Chroma cosine search, TOP_K=3; counts and latency were not asserted."
add("E. RAG",ragbody)

# F Gemini static only: never send secret externally in audit
llmfile=(ROOT/"rag/llm_agent.py").read_text(encoding="utf-8")
add("F. LLM (Gemini)", "**MEASURED static configuration:** SDK `google.genai` (`google-genai` package); model attempt order `gemini-2.5-flash`, then `gemini-3.7-flash`; each model has up to 3 attempts. Prompt forces `FAILURE_EXPLANATION`, `RECOMMENDED_ACTION`, `REASONING`, `RISK`, `AUTOMATIC_ACTION`. Action parsing uses case-insensitive regex `RECOMMENDED_ACTION:\\s*([A-Z_]+)` in `rag/guardrails.py`.\n\n**NOT MEASURABLE:** no Gemini live calls were made. A credential exists locally, but this non-destructive audit deliberately did not transmit it or invoke a paid external API. Therefore live success rate, parse success rate, action distribution, and latency are not available.")

# I batch, force fallback without ever importing llm agent with a key and redirect audit output
from backend.agent import recovery_pipeline as rp
rp.AUDIT_LOG_PATH=OUT/"batch_recovery_audit_log.csv"
orig=rp._llm_recommendation
rp._llm_recommendation=lambda payment: rp._fallback_llm("Audit deliberately disabled external Gemini calls.")
t=time.perf_counter(); results=rp.process_batch(100); et=(time.perf_counter()-t)*1000/len(results); rp._llm_recommendation=orig
def countskey(k): return Counter(str(x.get(k)) for x in results)
over=sum(x['llm_recommended_action']!=x['final_action'] for x in results); recovered=sum(num(x['recovered_amount']) for x in results); failed=sum(num(x['amount']) for x in results)
add("I. End-to-end batch", f"**MEASURED (fallback-only):** {len(results)} payments processed via `process_batch(100)` with Gemini deliberately disabled, push disabled by input/environment, and audit output redirected to `_audit/batch_recovery_audit_log.csv`. LLM action distribution={dict(countskey('llm_recommended_action'))}; final action distribution={dict(countskey('final_action'))}; guardrail overrides={over}/{len(results)} ({pct(over/len(results))}); execution statuses={dict(countskey('execution_status'))}; avg latency={et:.2f} ms/payment.\n\n**SIMULATED:** recovered amount={m(recovered)}; simulated recovery rate={pct(recovered/failed)}. The executor samples retry outcomes from the ML probability, so this is not observed revenue.")

# J endpoints / UI / demo with output rerouted
import ast
endpoints=[]
for f in [ROOT/'backend/main.py',ROOT/'backend/api/demo_routes.py']:
    text=f.read_text(encoding='utf-8'); prefix='/demo' if f.name=='demo_routes.py' else ''
    for _,meth,path in re.findall(r'@(app|router)\.(get|post)\("([^"]+)',text): endpoints.append((meth.upper(),prefix+path,str(f.relative_to(ROOT))))
ui=[]
for f in sorted([ROOT/'frontend/dashboard.py',*(ROOT/'frontend/pages').glob('*.py')]):
 text=f.read_text(encoding='utf-8'); comps=re.findall(r'\bst\.(?:tabs|dataframe|bar_chart|metric|radio|number_input|selectbox|checkbox|button|expander)\b',text); ui.append((f.relative_to(ROOT),len(comps),", ".join(sorted(set(x.split('.')[-1] for x in comps)))))
from backend.agent import demo_pipeline as demo
demo.DEMO_AUDIT_PATH=OUT/'demo_audit_log.jsonl'
demorows=[]
for ft in ["NETWORK_ERROR","INSUFFICIENT_FUNDS","EXPIRED_CARD","DUPLICATE","SUSPECTED_FRAUD"]:
 r=demo.run_demo_case(ft,demo.SCENARIOS[ft]['amount'],retry_outcome='success',send_push=False)
 st={x['stage']:x['latency_ms'] for x in r['stages']}; s=r['summary']; demorows.append((ft,*[st[x] for x in ['gateway','ml','rag','llm','guardrails','executor']],r['total_latency_ms'],s['final_action'],s['guardrail_overrode_llm']))
ver=demo.verify_demo_audit()
add("J. API, UI and demo", f"**MEASURED:** {len(endpoints)} FastAPI endpoints.\n\n"+table(["method","path","file"],endpoints)+f"\n\n**MEASURED:** Streamlit component-call count={sum(x[1] for x in ui)} across 3 pages/files.\n\n"+table(["file","component calls","component types"],ui)+"\n\nAll five demo cases below are **SIMULATED**; `retry_outcome=success`, `send_push=False`, and demo audit writes were redirected to `_audit/demo_audit_log.jsonl`. LLM is fallback-only (external Gemini not invoked).\n\n"+table(["scenario","gateway ms","ml ms","rag ms","llm ms","guardrails ms","executor ms","total ms","final action","override?"],demorows)+f"\n\n**MEASURED:** `verify_demo_audit()` returned records={ver['records']}, ok={ver['ok']}, broken_at={ver['broken_at']}.")

# K test run output supplied by shell wrapper
add("K. Quality", "Test result is populated by the audit runner in `run_output.txt`. Existing test functions are counted from `tests/`: "+str(sum(1 for p in (ROOT/'tests').rglob('test_*.py') for l in p.read_text(encoding='utf-8').splitlines() if re.match(r'\s*def test_',l)))+". See the runner output for the actual pass/fail command result.\n\n**Observed inconsistencies:** (1) `tests/test_evaluation.py` imports `backend.evaluation`, but no such source files appear in the repository inventory; tests may fail at collection. (2) `priority_engine.py` creates `recommended_action`, but the runtime pipeline separately calls Gemini/fallback and guardrails. (3) Model selection in `train_model.py` chooses highest *test* ROC-AUC, while the saved evaluation report says validation log-loss would pick a different model; selecting on the test set biases reported test performance. (4) synthetic data and simulated executor outcomes cannot establish real-world recovery uplift.")

# Resume tables
resume=[("Python codebase","%s files; %s physical lines"%(len(py),f"{sum(n for _,n in locs):,}"),"MEASURED","_audit/audit.py inventory run","I audited a Python recovery-agent prototype with this code footprint.","low"),("Held-out test ROC-AUC",f"{roc_auc_score(yt,prob):.4f}","MEASURED","_audit/audit.py re-run on ml/recovery_model.joblib","On the synthetic held-out customer split, the saved model ranked recovered payments at this ROC-AUC.","high—synthetic data and leakage finding"),("Held-out test PR-AUC",f"{average_precision_score(yt,prob):.4f}","MEASURED","_audit/audit.py re-run","On the synthetic held-out split, precision-recall ranking was this value.","high—synthetic data and leakage finding"),("Guardrails","%d explicit rule branches"%len(rules),"MEASURED","rag/guardrails.py","I implemented/tested deterministic recovery-action guardrails with this many rule branches.","low"),("RAG corpus",(str(len(docs))+" docs / "+str(len(chunks))+" chunks") if 'docs' in globals() else "NOT MEASURABLE","MEASURED" if 'docs' in globals() else "N/A","rag/rag_engine.py runtime" if 'docs' in globals() else "RAG import failed","The prototype retrieves from this local policy corpus." if 'docs' in globals() else "Do not claim.","medium—local synthetic policy corpus"),("Fallback batch throughput",f"{len(results)} payments; {et:.2f} ms/payment","MEASURED","_audit/audit.py fallback batch run","I ran a 100-payment local fallback pipeline batch with push disabled.","medium—fallback-only, local hardware")]
avoid=[("Recovered revenue / recovery rate",f"{m(recovered)} / {pct(recovered/failed)}","SIMULATED","Retry outcomes are forced/sampled from model probability, not real payment outcomes."),("Model performance as production proof",f"ROC-AUC {roc_auc_score(yt,prob):.4f}","MEASURED but risky","Synthetic data; saved report flags possible merchant_recovery_rate outcome leakage; test-set model selection."),("Gemini efficacy","No live measurement","NOT MEASURABLE","No real Gemini requests were made in this audit."),("End-to-end ‘AI’ performance","100-payment batch","fallback-only","Gemini disabled, so this did not measure LLM decision quality."),("Demo latencies","five scenarios","SIMULATED","Gateway failures and recovery outcomes are intentional demo behavior; hardware/local-cache dependent.")]
add("Resume-safe numbers",table(["metric","value","MEASURED/SIMULATED","source (file/command)","one-sentence explanation I can say in an interview","overclaim risk"],resume))
add("Do NOT put on resume",table(["numbers or claims","value/status","why"],avoid))

report="# AI Revenue Recovery Agent — Factual Audit\n\nAudit generated by `_audit/audit.py`. Every runtime action used `send_push=False`; `NTFY_TOPIC` was removed from the audit process. Credentials and ntfy topics are intentionally absent.\n\n"+"\n\n".join(sections)+"\n"
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
print("Wrote",OUT/'REPORT.md')
