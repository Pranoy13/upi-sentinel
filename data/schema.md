# Transaction schema (synthetic data only)

| column | description |
|---|---|
| txn_id | unique id, e.g. T000123 |
| user_id | masked sender id, e.g. U0042 |
| beneficiary_id | masked receiver id, e.g. B0301 |
| amount | INR |
| timestamp | datetime (Oct 2026, 30 days) |
| device_id | device used by the sender |
| city | city of the transaction |
| channel | UPI_APP / QR_SCAN / WEB / COLLECT_REQUEST |
| txn_type | P2P / P2M / BILL_PAY |
| is_fraud | 1 = fraud, 0 = legitimate (ground truth, only for evaluation) |
| scenario | NORMAL, LEGIT_* controls, or a fraud scenario name |

## Contract between modules
`src/scoring.py` must expose:
`score_transactions(df) -> df` with new columns `risk_score` (0-100) and `reasons` (list of strings).