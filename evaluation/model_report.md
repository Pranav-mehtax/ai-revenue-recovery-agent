# Model evaluation report

Data: 14,935 failed payments from 4,736 customers, split **by customer** (10,432 train / 2,214 validation / 2,289 test) so the model is always scored on customers it never saw. Overall recovery rate: 61.3%.

## 1. Why this model

| Model | Val AUC | Test AUC | Test log loss | Test Brier | Calibration error (ECE) |
|---|---|---|---|---|---|
| Logistic Regression | 0.711 | 0.718 | 0.600 | 0.206 | 0.009 |
| Random Forest | 0.728 | 0.731 | 0.610 | 0.211 | 0.095 |
| XGBoost | 0.734 | 0.728 | 0.590 | 0.202 | 0.025 |
| Baseline: failure-type average | 0.670 | 0.695 | 0.601 | 0.207 | 0.005 |
| Baseline: overall average | 0.500 | 0.500 | 0.671 | 0.239 | 0.008 |

Production model: **Random Forest** (`ml/recovery_model.joblib`), test AUC 0.732, Brier skill 12.2% better than guessing the average rate.

Probabilities, not just rankings, feed the money estimate (expected recovery = amount x probability), so log loss, Brier score and calibration matter as much as AUC.

> Note: `train_model.py` picks the highest **test** AUC (Random Forest), but ranking by **validation** log loss picks XGBoost. Choosing on the test set makes the reported test score slightly optimistic; select on validation and report test once.

## 2. Which features matter most

Top features by permutation importance (drop in test AUC when shuffled):

1. `failure_type`: 0.1811
2. `retry_count`: 0.0195
3. `customer_success_rate`: 0.0081
4. `merchant_recovery_rate`: 0.0024
5. `payment_method`: 0.0020
6. `successful_transactions`: 0.0016

By feature group (shuffle test, and AUC change when retrained without the group):

| Group | Columns | Shuffle AUC drop |
|---|---|---|
| Payment details | amount, payment_method, failure_type, retry_count | 0.2028 |
| Customer history | total_transactions, successful_transactions, failed_transactions, avg_transaction_amount, customer_success_rate | 0.0322 |
| Merchant profile | merchant_category, merchant_success_rate, merchant_recovery_rate, merchant_avg_order_value | 0.0037 |
| Time of day and week | hour_of_day, day_of_week, is_weekend | 0.0004 |

| Retrained without | Test AUC | Change |
|---|---|---|
| All features (production) | 0.731 | +0.000 |
| Without: Payment details | 0.572 | -0.160 |
| Without: Customer history | 0.716 | -0.015 |
| Without: Merchant profile | 0.730 | -0.001 |
| Without: Time of day and week | 0.730 | -0.001 |
| Without merchant_recovery_rate only | 0.729 | -0.003 |
| Without both merchant rates | 0.730 | -0.001 |

## 3. Why priority = amount x probability

Share of all recoverable rupees captured by working only the top 20% of payments (test set):

| Ordering | Top 10% | Top 20% | Top 30% |
|---|---|---|---|
| Expected value: amount x probability (current) | 38.4% | 56.7% | 69.1% |
| Probability only | 13.0% | 25.6% | 37.8% |
| Amount only | 36.0% | 55.1% | 67.5% |
| Oracle (knows the outcome) | 47.2% | 68.4% | 82.0% |
| Random order | 10.0% | 20.0% | 30.0% |

Expected vs actual recovered rupees on the test set: Rs 4,396,509 expected, Rs 5,385,886 actual (ratio 0.82; 1.00 is perfect).

### How the HIGH / MEDIUM / LOW tiers behave

| Scheme | HIGH | MEDIUM | LOW | Share of recovered Rs in HIGH |
|---|---|---|---|---|
| Current: divide by the maximum (66/33) | 6 | 29 | 2,254 | 4.2% |
| Divide by 99th percentile (66/33) | 73 | 216 | 2,000 | 19.5% |
| Rank-based: top 10% / next 30% | 228 | 687 | 1,374 | 38.4% |

`priority_engine.py` divides every expected value by the single largest one. Because payment amounts are heavy-tailed, one large payment can push almost everyone else into LOW; the table shows whether that happens on your data and what the alternatives would give.

## 4. Leakage check: `merchant_recovery_rate`

Across 100 merchants, the column differs from the observed recovery rate of the same payments by 0.0809 on average (correlation 0.625); pure sampling noise alone would give about 0.0397.

The match is within what a genuine per-merchant statistic would show, so there is no sign it was computed from these outcomes.

## Limitations

- The data is synthetic; these numbers describe the system, not real Razorpay traffic.
- Customer history columns look like whole-period aggregates, which can include later transactions. A time-based split would be a stricter test.
- Recovery outcomes are single labelled events, so uplift from actually acting is not measured.
