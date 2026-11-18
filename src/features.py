
"""Feature engineering: per-user behaviour baselines and fraud signals."""

from collections import defaultdict, deque, Counter

import numpy as np
import pandas as pd


MIN_HISTORY = 15


def _window_count(t_query, t_events, window):
    """
    Count events in the interval (t - window, t].
    t_events must be sorted.
    """
    hi = np.searchsorted(t_events, t_query, side="right")
    lo = np.searchsorted(t_events, t_query - window, side="right")
    return hi - lo


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate chronological, history-aware transaction features.

    Behavioural baselines use previous transactions only.
    Original transaction row order is preserved in the output.
    """

    required = {
        "timestamp",
        "user_id",
        "beneficiary_id",
        "amount",
        "device_id",
        "city",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}"
        )

    if df.empty:
        return df.copy()

    df = df.copy()

    # Preserve original row positions.
    df["_original_position"] = np.arange(len(df))

    # Convert timestamps and validate them.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps found in transaction data."
        )

    # Ensure amount is numeric.
    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    if df["amount"].isna().any():
        raise ValueError(
            "Invalid or missing transaction amounts found."
        )

    if (df["amount"] < 0).any():
        raise ValueError(
            "Negative transaction amounts found."
        )

    # Process transactions chronologically.
    df = df.sort_values(
        ["timestamp", "_original_position"],
        kind="mergesort"
    ).reset_index(drop=True)

    # Timestamp features.
    df["ts"] = (
        df["timestamp"] - pd.Timestamp("1970-01-01")
    ).dt.total_seconds()

    df["hour"] = df["timestamp"].dt.hour

    # -------------------------------------------------
    # Per-user historical baselines (past transactions)
    # -------------------------------------------------

    df["user_median"] = np.nan
    df["amount_ratio"] = 1.0
    df["amount_z"] = 0.0
    df["user_night_share"] = 0.0

    for _, idx in df.groupby("user_id", sort=False).groups.items():

        idx = np.asarray(list(idx))

        amounts = df.loc[idx, "amount"]

        # Expanding median shifted by one transaction:
        # the current transaction is excluded.
        past_median = amounts.expanding(
            min_periods=1
        ).median().shift(1)

        # Expanding mean and standard deviation, also past-only.
        past_mean = amounts.expanding(
            min_periods=2
        ).mean().shift(1)

        past_std = amounts.expanding(
            min_periods=2
        ).std(ddof=0).shift(1)

        # Fill cold-start baseline values.
        past_median = past_median.fillna(0)
        past_mean = past_mean.fillna(amounts)
        past_std = past_std.fillna(0)

        df.loc[idx, "user_median"] = past_median.to_numpy()

        ratio = np.where(
            past_median.to_numpy() > 0,
            amounts.to_numpy() / np.maximum(
                past_median.to_numpy(), 1.0
            ),
            1.0
        )

        df.loc[idx, "amount_ratio"] = ratio

        z_score = (
            amounts.to_numpy() - past_mean.to_numpy()
        ) / (
            past_std.to_numpy() + 1.0
        )

        df.loc[idx, "amount_z"] = z_score

        # Historical night-time behaviour.
        night = (
            df.loc[idx, "timestamp"].dt.hour < 5
        ).astype(int)

        past_night_share = night.expanding(
            min_periods=1
        ).mean().shift(1).fillna(0)

        df.loc[idx, "user_night_share"] = (
            past_night_share.to_numpy()
        )

    # Odd-hour activity.
    night = (df["hour"] < 5).astype(int)

    df["odd_hour"] = (
        (night == 1) &
        (df["user_night_share"] < 0.02)
    ).astype(int)

    # -------------------------------------------------
    # History-aware first-time behaviour
    # -------------------------------------------------

    df["user_txn_idx"] = (
        df.groupby("user_id", sort=False).cumcount()
    )

    has_history = df["user_txn_idx"] >= MIN_HISTORY

    df["new_beneficiary"] = (
        (
            df.groupby(
                ["user_id", "beneficiary_id"],
                sort=False
            ).cumcount() == 0
        ) & has_history
    ).astype(int)

    df["new_device"] = (
        (
            df.groupby(
                ["user_id", "device_id"],
                sort=False
            ).cumcount() == 0
        ) & has_history
    ).astype(int)

    df["new_city"] = (
        (
            df.groupby(
                ["user_id", "city"],
                sort=False
            ).cumcount() == 0
        ) & has_history
    ).astype(int)

    # -------------------------------------------------
    # Velocity and structuring features
    # -------------------------------------------------

    df["txn_count_10m"] = 0
    df["amount_sum_1h"] = 0.0
    df["near_limit_count_1h"] = 0

    near_limit = (
        (df["amount"] >= 9000) &
        (df["amount"] < 10000)
    )

    for _, idx in df.groupby("user_id", sort=False).groups.items():

        idx = np.asarray(list(idx))

        t = df.loc[idx, "ts"].to_numpy()
        amounts = df.loc[idx, "amount"].to_numpy()

        # Transactions in the previous 10 minutes,
        # including the current transaction.
        df.loc[idx, "txn_count_10m"] = _window_count(
            t, t, 600
        )

        # Amount transferred in the previous hour.
        cumulative_amount = np.concatenate([
            [0.0],
            np.cumsum(amounts)
        ])

        hi = np.searchsorted(t, t, side="right")
        lo = np.searchsorted(t, t - 3600, side="right")

        df.loc[idx, "amount_sum_1h"] = (
            cumulative_amount[hi] -
            cumulative_amount[lo]
        )

        # Near-limit transactions in the previous hour.
        t_near_limit = t[
            near_limit.loc[idx].to_numpy()
        ]

        df.loc[idx, "near_limit_count_1h"] = (
            _window_count(t, t_near_limit, 3600)
        )

    # -------------------------------------------------
    # Beneficiary fan-in: distinct senders in three hours
    # -------------------------------------------------

    df["ben_fanin_3h"] = 0

    for _, idx in df.groupby(
        "beneficiary_id",
        sort=False
    ).groups.items():

        idx = np.asarray(list(idx))

        timestamps = df.loc[idx, "timestamp"].to_numpy()
        users = df.loc[idx, "user_id"].to_numpy()

        sender_counts = Counter()
        window = deque()

        fanin_scores = []

        for timestamp, user in zip(timestamps, users):

            if pd.isna(user):
                fanin_scores.append(0)
                continue

            cutoff = (
                timestamp - np.timedelta64(3, "h")
            )

            # Remove transactions outside the three-hour window.
            while window and window[0][0] < cutoff:
                old_time, old_user = window.popleft()

                sender_counts[old_user] -= 1

                if sender_counts[old_user] == 0:
                    del sender_counts[old_user]

            window.append((timestamp, user))
            sender_counts[user] += 1

            fanin_scores.append(len(sender_counts))

        df.loc[idx, "ben_fanin_3h"] = fanin_scores

    # Restore original transaction order.
    df = df.sort_values(
        "_original_position"
    ).drop(columns=["_original_position"])

    df = df.reset_index(drop=True)

    return df


FEATURE_COLS = [
    "amount",
    "amount_ratio",
    "amount_z",
    "hour",
    "odd_hour",
    "new_beneficiary",
    "new_device",
    "new_city",
    "txn_count_10m",
    "amount_sum_1h",
    "near_limit_count_1h",
    "ben_fanin_3h",
]


if __name__ == "__main__":

    raw = pd.read_csv(
        "data/transactions.csv",
        parse_dates=["timestamp"]
    )

    feats = build_features(raw)

    feats.to_csv(
        "data/features.csv",
        index=False
    )

    print(
        f"Saved {len(feats)} rows to data/features.csv\n"
    )

    pd.set_option("display.width", 200)

    print(
        feats.groupby("scenario")[FEATURE_COLS]
        .mean()
        .round(2)
    )
