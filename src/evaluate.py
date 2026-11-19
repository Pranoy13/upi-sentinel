"""Evaluation: detection performance and false positives on a time-based hold-out."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_FRACTION = 0.30      # last 30% of transactions by time (model trains on the first 70%)


def demo_scores(raw):
    """Simple placeholder scorer, used only to test this script before scoring.py exists."""
    from src.features import build_features
    f = build_features(raw)
    s = (30 * (f["txn_count_10m"] >= 4)
         + 30 * (f["amount_ratio"] >= 8)
         + 25 * ((f["new_device"] == 1) & (f["new_city"] == 1))
         + 25 * (f["ben_fanin_3h"] >= 4)
         + 20 * (f["near_limit_count_1h"] >= 3)
         + 15 * (f["odd_hour"] == 1)
         + 15 * (f["new_beneficiary"] == 1))
    f["risk_score"] = np.clip(s, 0, 100)
    return f


def load_scored(demo):
    raw = pd.read_csv("data/transactions.csv", parse_dates=["timestamp"])
    if demo:
        return demo_scores(raw)
    try:
        from src.scoring import score_transactions
        return score_transactions(raw)
    except ImportError:
        if os.path.exists("data/scored.csv"):
            print("src/scoring.py not found, using data/scored.csv")
            return pd.read_csv("data/scored.csv", parse_dates=["timestamp"])
        sys.exit("No scoring available yet. Use --demo, or wait for Person B's src/scoring.py")


def metrics_at(y, s, thr):
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"threshold": int(thr), "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "fpr": round(fpr, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use placeholder scorer")
    ap.add_argument("--threshold", type=int, default=35)
    args = ap.parse_args()

    df = load_scored(args.demo).sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * (1 - TEST_FRACTION))
    test = df.iloc[cut:].copy()
    y, s = test["is_fraud"].to_numpy(), test["risk_score"].to_numpy()

    m = metrics_at(y, s, args.threshold)
    m["accuracy"] = round((m["tp"] + m["tn"]) / len(test), 4)
    m["roc_auc"] = round(float(roc_auc_score(y, s)), 4)
    m["pr_auc"] = round(float(average_precision_score(y, s)), 4)

    # per-scenario: fraud scenarios -> detection rate, legit scenarios -> false-flag rate
    test["flagged"] = (test["risk_score"] >= args.threshold).astype(int)
    per = {}
    for sc, g in test.groupby("scenario"):
        per[sc] = {"type": "fraud" if g["is_fraud"].max() == 1 else "legit",
                   "count": int(len(g)), "flagged": int(g["flagged"].sum()),
                   "flag_rate": round(float(g["flagged"].mean()), 4),
                   "avg_score": round(float(g["risk_score"].mean()), 1)}

    sweep = [metrics_at(y, s, t) for t in range(10, 100, 5)]

    out = {"mode": "demo" if args.demo else "real", "threshold": args.threshold,
           "test_rows": int(len(test)), "test_fraud": int(y.sum()),
           "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "fpr": m["fpr"],
           "accuracy": m["accuracy"], "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
           "confusion_matrix": {"tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]},
           "per_scenario": per, "threshold_sweep": sweep}

    # demo results never overwrite the real metrics file that the dashboard reads
    path = "data/metrics_demo.json" if args.demo else "data/metrics.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    pd.DataFrame(sweep).to_csv("data/threshold_sweep.csv", index=False)

    print(f"\n=== Evaluation ({out['mode']}), threshold >= {args.threshold} ===")
    print(f"Hold-out: {out['test_rows']} txns, {out['test_fraud']} fraud")
    print(f"Precision {m['precision']:.2%} | Recall {m['recall']:.2%} | F1 {m['f1']:.2%} "
          f"| False-positive rate {m['fpr']:.2%}")
    print(f"ROC-AUC {m['roc_auc']} | PR-AUC {m['pr_auc']}")
    print(f"Confusion matrix: {out['confusion_matrix']}\n")
    print(pd.DataFrame(per).T.sort_values(["type", "flag_rate"], ascending=[True, False]))
    print(f"\nSaved {path} and data/threshold_sweep.csv")


if __name__ == "__main__":
    main()