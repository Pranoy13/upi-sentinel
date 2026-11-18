
import numpy as np
import pandas as pd


def _result(points, reasons, index):
    """Return aligned rule points and human-readable reasons."""
    points = pd.Series(points, index=index).fillna(0).astype(float)
    reasons = pd.Series(reasons, index=index).fillna("").astype(str)
    return points, reasons


def high_amount_vs_baseline(df):
    """Detect transactions unusually large for the user's normal behaviour."""
    ratio = df["amount_ratio"].fillna(0)
    z_score = df["amount_z"].fillna(0)

    points = np.select(
        [
            (ratio >= 10) | (z_score >= 5),
            (ratio >= 5) | (z_score >= 3),
            (ratio >= 3) | (z_score >= 2),
        ],
        [25, 18, 10],
        default=0,
    )

    reasons = np.select(
        [
            points == 25,
            points == 18,
            points == 10,
        ],
        [
            "Amount is {:.1f}x this user's typical amount".format(0) ,
            "Amount is unusually high compared with user baseline",
            "Amount is moderately above user baseline",
        ],
        default="",
    )

    # Include actual transaction ratio in the explanation.
    reasons = np.where(
        points > 0,
        [
            f"Amount is {r:.1f}x this user's typical amount"
            for r in ratio
        ],
        "",
    )

    return _result(points, reasons, df.index)


def velocity_burst(df):
    """Detect unusually frequent transactions within ten minutes."""
    count = df["txn_count_10m"].fillna(0)

    points = np.select(
        [count >= 8, count >= 5, count >= 3],
        [25, 18, 10],
        default=0,
    )

    reasons = np.where(
        points > 0,
        [f"{int(c)} transactions made by this user in the last 10 minutes"
         for c in count],
        "",
    )

    return _result(points, reasons, df.index)


def new_beneficiary_high_value(df):
    """Detect high-value payments to previously unseen beneficiaries."""
    new_ben = df["new_beneficiary"].fillna(0).astype(bool)
    ratio = df["amount_ratio"].fillna(0)
    amount = df["amount"].fillna(0)

    points = np.where(
        new_ben & ((ratio >= 5) | (amount >= 10000)),
        25,
        np.where(
            new_ben & ((ratio >= 3) | (amount >= 5000)),
            15,
            0,
        ),
    )

    reasons = np.where(
        points > 0,
        [
            f"New beneficiary received a high-value payment of Rs. {a:,.0f}"
            for a in amount
        ],
        "",
    )

    return _result(points, reasons, df.index)


def new_device_and_city(df):
    """Detect simultaneous new-device and new-city activity."""
    new_device = df["new_device"].fillna(0).astype(bool)
    new_city = df["new_city"].fillna(0).astype(bool)

    points = np.where(new_device & new_city, 25, 0)

    reasons = np.where(
        points > 0,
        "Transaction originated from both a new device and a new city",
        "",
    )

    return _result(points, reasons, df.index)


def odd_hour(df):
    """Give a small risk contribution for unusual transaction hours."""
    odd = df["odd_hour"].fillna(0).astype(bool)
    night_share = df["user_night_share"].fillna(0)

    # Reduce risk when the user regularly transacts at night.
    points = np.where(
        odd & (night_share < 0.10),
        8,
        0,
    )

    reasons = np.where(
        points > 0,
        [
            f"Transaction at {int(h):02d}:00 during an unusual hour"
            for h in df["hour"].fillna(0)
        ],
        "",
    )

    return _result(points, reasons, df.index)


def structuring(df):
    """Detect repeated transactions near the common Rs. 10,000 threshold."""
    count = df["near_limit_count_1h"].fillna(0)

    points = np.select(
        [count >= 5, count >= 3, count >= 2],
        [25, 18, 10],
        default=0,
    )

    reasons = np.where(
        points > 0,
        [
            f"{int(c)} near-limit transactions in the last hour"
            for c in count
        ],
        "",
    )

    return _result(points, reasons, df.index)


def mule_fanin(df):
    """Detect beneficiaries receiving payments from many distinct senders."""
    fanin = df["ben_fanin_3h"].fillna(0)

    points = np.select(
        [fanin >= 12, fanin >= 8, fanin >= 5],
        [30, 22, 12],
        default=0,
    )

    reasons = np.where(
        points > 0,
        [
            f"Beneficiary received payments from {int(c)} distinct senders "
            "within 3 hours"
            for c in fanin
        ],
        "",
    )

    return _result(points, reasons, df.index)


RULES = {
    "HIGH_AMOUNT_VS_BASELINE": high_amount_vs_baseline,
    "VELOCITY_BURST": velocity_burst,
    "NEW_BENEFICIARY_HIGH_VALUE": new_beneficiary_high_value,
    "NEW_DEVICE_AND_CITY": new_device_and_city,
    "ODD_HOUR": odd_hour,
    "STRUCTURING": structuring,
    "MULE_FANIN": mule_fanin,
}


def apply_rules(df):
    """
    Apply all rules.

    Returns:
        rule_score: Combined rule score, capped at 100.
        reasons: List of human-readable contributing factors.
    """
    rule_score = pd.Series(0.0, index=df.index)
    all_reasons = [[] for _ in range(len(df))]

    for rule_name, rule_function in RULES.items():
        points, reasons = rule_function(df)

        rule_score += points

        for position, reason in enumerate(reasons):
            if reason:
                all_reasons[position].append(reason)

    return (
        rule_score.clip(upper=100),
        pd.Series(all_reasons, index=df.index),
    )
