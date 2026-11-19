# Technical Documentation: UPI Sentinel

## 1. Architecture
Synthetic data -> feature engineering -> (rules + ML models + graph analysis) -> blended risk score with reasons -> Streamlit dashboard -> analyst decisions in SQLite.

| Module | File | Purpose |
|---|---|---|
| Data generator | src/generate_data.py | 16,934 synthetic transactions, 7 fraud and 4 legitimate look-alike scenarios |
| Features | src/features.py | Per-user baselines and behaviour signals, computed from past data only |
| Rules | src/rules.py | Rule engine with human-readable reason codes |
| ML | src/model.py | Isolation Forest (unsupervised) + Random Forest (supervised, balanced classes) |
| Graph | src/graph_analysis.py | Mule-beneficiary detection via sender-to-beneficiary graph (networkx) |
| Scoring | src/scoring.py | Blends all signals into a 0-100 score with reasons |
| Evaluation | src/evaluate.py | Time-based hold-out metrics, per-scenario rates, threshold sweep |
| Cases | src/db.py | SQLite analyst case tracking |
| Dashboard | app/dashboard.py | Overview, queue, case view, cases, evaluation, privacy |

## 2. Features
- Baseline deviation: amount_ratio, amount_z (per-user median/MAD)
- Behaviour firsts: new_beneficiary, new_device, new_city (counted after 15 transactions of history)
- Timing and velocity: hour, odd_hour, txn_count_10m, amount_sum_1h, near_limit_count_1h
- Network: ben_fanin_3h (distinct senders paying one beneficiary within 3 hours)

## 3. Scoring
risk_score = 0.40 x rule_score + 45 x model_prob + 15 x graph_score, clipped to 0-100.
model_prob = 0.75 x Random Forest probability + 0.25 x Isolation Forest anomaly score.
Each flagged transaction lists its contributing reasons.

## 4. Evaluation (last 30% of transactions by time, threshold 35)
Hold-out: 5,081 transactions, 197 fraud.

| Metric | Value |
|---|---|
| Precision | 93.97% |
| Recall | 94.92% |
| F1 | 94.44% |
| False-positive rate | 0.25% |
| ROC-AUC / PR-AUC | 0.997 / 0.979 |
| Confusion matrix | TN 4872, FP 12, FN 10, TP 187 |

Per-scenario: ACCOUNT_TAKEOVER, HIGH_AMOUNT, MULE_FANIN, NEW_BENEFICIARY_HIGH_VALUE, ODD_HOUR and STRUCTURING are detected at 100%; VELOCITY_BURST at 86.7%. Legitimate look-alikes: LEGIT_NEW_BEN, LEGIT_NEW_DEVICE and LEGIT_TRAVEL are never flagged; LEGIT_FESTIVAL_SPIKE is flagged 76.9% of the time (10 of 13), the main known weakness.

Threshold trade-off: raising the threshold increases precision but lowers recall, and lowering it does the opposite. The full sweep is in data/threshold_sweep.csv, and the dashboard slider lets analysts choose their own operating point.

## 5. Limitations and future work
- Synthetic data is simpler than real traffic; scores are indicative, not production guarantees.
- Genuine festival-season spikes still resemble fraud; adding merchant category and seasonality features would help.
- Future: real-time streaming scoring, analyst feedback loop to retrain, deeper graph analysis (money-hop chains), device fingerprint features.