# Dataset Description

## Overview
UPI Sentinel is developed and evaluated entirely on a **synthetic** dataset produced by `src/generate_data.py` (fixed random seed, fully reproducible). No real users, accounts, devices or transactions are used, and the system never connects to or initiates real financial transactions.

| Property | Value |
|---|---|
| Transactions | 16,934 |
| Senders (users) | 300 |
| Beneficiaries | 600 regular + 15 mule-style accounts |
| Period | 30 days (1 to 30 October 2026) |
| Fraud transactions | 585 (3.45%) |
| Labels | `is_fraud` (0/1) and `scenario` (ground truth, used only for training and evaluation) |

## Schema
| Column | Description |
|---|---|
| txn_id | Unique transaction id |
| user_id | Masked sender id (U0001 to U0300) |
| beneficiary_id | Masked receiver id (B0001 to B0615) |
| amount | Amount in INR |
| timestamp | Date and time of the transaction |
| device_id | Device used by the sender |
| city | City of the transaction |
| channel | UPI_APP, QR_SCAN, WEB, COLLECT_REQUEST |
| txn_type | P2P, P2M, BILL_PAY |
| is_fraud | Ground-truth label |
| scenario | NORMAL, a LEGIT_* control case, or a fraud scenario |

## How normal behaviour is simulated
Each user has their own profile: home city, 1 to 2 devices, a typical amount (log-normal, around INR 500), a daily transaction rate, active hours, and a small set of known beneficiaries. About 90% of normal transactions go to known beneficiaries; about 10% go to a new beneficiary with a small amount (`LEGIT_NEW_BEN`).

## Fraud scenarios (585 transactions)
| Scenario | Txns | Pattern |
|---|---|---|
| VELOCITY_BURST | 210 | 6 to 11 rapid payments within seconds to new beneficiaries |
| ACCOUNT_TAKEOVER | 96 | New device and new city, then several high-value transfers to new beneficiaries |
| STRUCTURING | 91 | Repeated amounts just under INR 10,000 to the same beneficiary |
| MULE_FANIN | 66 | Many different victims pay one mule beneficiary within hours |
| NEW_BENEFICIARY_HIGH_VALUE | 50 | Tiny test payment, then a large transfer to the same new beneficiary |
| ODD_HOUR | 42 | High-value payments between 2 AM and 5 AM, outside the user's usual hours |
| HIGH_AMOUNT | 30 | Single amount 10 to 25 times the user's typical amount |

## Legitimate control cases (hard negatives, 1,803 transactions)
These look suspicious on the surface but are genuine. They test whether the system avoids false positives.
| Scenario | Txns | Pattern |
|---|---|---|
| LEGIT_NEW_BEN | 1,648 | Small payment to a new beneficiary |
| LEGIT_TRAVEL | 65 | User transacts from another city with normal amounts and known beneficiaries |
| LEGIT_NEW_DEVICE | 50 | User switches to a new phone, then behaves normally |
| LEGIT_FESTIVAL_SPIKE | 40 | One larger purchase (5 to 10 times typical) to a known merchant |

The remaining 14,546 transactions are `NORMAL`.

## Derived features (`src/features.py`)
All features use only past data for each user, so there is no label or future leakage.
- **Baseline deviation:** `amount_ratio`, `amount_z` (robust median/MAD per user)
- **Behaviour firsts:** `new_beneficiary`, `new_device`, `new_city` (counted only after 15 transactions of history)
- **Timing:** `hour`, `odd_hour`
- **Velocity and structuring:** `txn_count_10m`, `amount_sum_1h`, `near_limit_count_1h`
- **Network:** `ben_fanin_3h` (distinct senders paying one beneficiary in 3 hours)

## Evaluation protocol
Transactions are sorted by time. The first 70% are used for training the ML model and the last 30% are a hold-out test set, so the system is always tested on the "future". We report precision, recall, F1, false-positive rate, ROC-AUC, PR-AUC, a confusion matrix, per-scenario detection rates, and the false-flag rate on legitimate control cases, plus a threshold sweep (`data/threshold_sweep.csv`).

## Limitations
The data is simulated and cleaner than real UPI traffic, so reported scores are indicative, not production guarantees. Real deployment would need real labelled data under proper governance, drift monitoring, and analyst feedback loops.