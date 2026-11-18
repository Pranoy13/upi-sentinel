"""Feature engineering: per-user behaviour baselines and fraud signals."""
import numpy as np
import pandas as pd

MIN_HISTORY = 15          # txns a user needs before "new" things count as suspicious


def _window_count(t_query, t_events, window):
    """For each time in t_query, count events in (t - window, t]. t_events must be sorted."""
    hi = np.searchsorted(t_events, t_query, side="right")
    lo = np.searchsorted(t_events, t_query - window, side="right")
    return hi - lo


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    df["ts"] = (df["timestamp"] - pd.Timestamp("1970-01-01")).dt.total_seconds()
    df["hour"] = df["timestamp"].dt.hour

    # ---- per-user baselines (robust: median / MAD) ----
    g = df.groupby("user_id")["amount"]
    df["user_median"] = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median())
    df["amount_ratio"] = df["amount"] / df["user_median"]
    df["amount_z"] = (df["amount"] - df["user_median"]) / (1.4826 * mad + 1.0)

    night = (df["hour"] < 5).astype(int)
    df["user_night_share"] = night.groupby(df["user_id"]).transform("mean")
    df["odd_hour"] = ((night == 1) & (df["user_night_share"] < 0.02)).astype(int)

    # ---- history-aware "firsts" (past only) ----
    df["user_txn_idx"] = df.groupby("user_id").cumcount()
    has_history = df["user_txn_idx"] >= MIN_HISTORY
    df["new_beneficiary"] = ((df.groupby(["user_id", "beneficiary_id"]).cumcount() == 0)
                             & has_history).astype(int)
    df["new_device"] = ((df.groupby(["user_id", "device_id"]).cumcount() == 0)
                        & has_history).astype(int)
    df["new_city"] = ((df.groupby(["user_id", "city"]).cumcount() == 0)
                      & has_history).astype(int)

    # ---- velocity and structuring (per user) ----
    df["txn_count_10m"] = 0
    df["amount_sum_1h"] = 0.0
    df["near_limit_count_1h"] = 0
    near_limit = (df["amount"] >= 9000) & (df["amount"] < 10000)

    for _, idx in df.groupby("user_id").groups.items():
        idx = np.asarray(idx)
        t = df.loc[idx, "ts"].to_numpy()                 # already sorted
        df.loc[idx, "txn_count_10m"] = _window_count(t, t, 600)
        cs = np.concatenate([[0.0], np.cumsum(df.loc[idx, "amount"].to_numpy())])
        hi = np.searchsorted(t, t, side="right")
        lo = np.searchsorted(t, t - 3600, side="right")
        df.loc[idx, "amount_sum_1h"] = cs[hi] - cs[lo]
        t_nl = t[near_limit.loc[idx].to_numpy()]
        df.loc[idx, "near_limit_count_1h"] = _window_count(t, t_nl, 3600)

    # ---- beneficiary fan-in: distinct senders paying one beneficiary in 3h ----
    df["ben_fanin_3h"] = 1
    for _, idx in df.groupby("beneficiary_id").groups.items():
        idx = np.asarray(idx)
        t = df.loc[idx, "ts"].to_numpy()
        users = df.loc[idx, "user_id"].to_numpy()
        lo = np.searchsorted(t, t - 10800, side="right")
        df.loc[idx, "ben_fanin_3h"] = [len(set(users[l:i + 1])) for i, l in enumerate(lo)]

    return df


FEATURE_COLS = ["amount", "amount_ratio", "amount_z", "hour", "odd_hour",
                "new_beneficiary", "new_device", "new_city", "txn_count_10m",
                "amount_sum_1h", "near_limit_count_1h", "ben_fanin_3h"]


if __name__ == "__main__":
    raw = pd.read_csv("data/transactions.csv", parse_dates=["timestamp"])
    feats = build_features(raw)
    feats.to_csv("data/features.csv", index=False)
    print(f"Saved {len(feats)} rows to data/features.csv\n")
    pd.set_option("display.width", 200)
    print(feats.groupby("scenario")[FEATURE_COLS[1:]].mean().round(2))