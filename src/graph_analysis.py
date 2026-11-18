
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import networkx as nx


def compute_graph_scores(df):
    """
    Build a chronological sender-to-beneficiary graph.

    Detect beneficiaries receiving payments from many distinct
    senders within a rolling three-hour window.

    Returns:
        Pandas Series containing graph scores between 0 and 1.
    """

    required = {
        "user_id",
        "beneficiary_id",
        "timestamp",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required graph columns: {missing}"
        )

    if df.empty:
        return pd.Series(0.0, index=df.index, name="graph_score")

    data = df.copy()

    data["original_position"] = np.arange(len(data))

    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )

    if data["timestamp"].isna().any():
        raise ValueError("Invalid timestamps found in graph data.")

    data = data.sort_values(
        ["timestamp", "_original_index"]
    )

    # Directed graph: sender -> beneficiary.
    graph = nx.DiGraph()

    # Rolling three-hour transaction history per beneficiary.
    windows = defaultdict(deque)

    # Store the number of distinct senders in each rolling window.
    scores = np.zeros(len(data), dtype=float)

    window_duration = pd.Timedelta(hours=3)

    for row in data.itertuples(index=False):

        user = row.user_id
        beneficiary = row.beneficiary_id
        timestamp = row.timestamp
        position = row.original_position

        if pd.isna(user) or pd.isna(beneficiary):
            continue

        window = windows[beneficiary]

        # Remove transactions older than three hours.
        cutoff = timestamp - window_duration

        while window and window[0][0] < cutoff:
            window.popleft()

        # Count distinct senders in the current rolling window.
        window.append((timestamp, user))

        distinct_senders = len({
            sender for _, sender in window
        })

        # Historical graph degree BEFORE adding this transaction.
        historical_degree = (
            graph.in_degree(beneficiary)
            if graph.has_node(beneficiary)
            else 0
        )

        # Typical historical incoming degree.
        existing_degrees = [
            degree
            for _, degree in graph.in_degree()
            if degree > 0
        ]

        typical_degree = (
            float(np.median(existing_degrees))
            if existing_degrees
            else 1.0
        )

        # Rolling fan-in component.
        fanin_component = min(
            distinct_senders / 12.0,
            1.0
        )

        # Historical degree compared with typical beneficiaries.
        degree_ratio = (
            (historical_degree + 1) /
            (typical_degree * 3 + 1)
        )

        degree_component = min(degree_ratio, 1.0)

        # Combined graph suspicion score.
        score = (
            0.75 * fanin_component +
            0.25 * degree_component
        )

        scores[position] = min(max(score, 0.0), 1.0)

        # Add current transaction to historical graph.
        graph.add_edge(user, beneficiary)

    return pd.Series(
        scores,
        index=df.index,
        name="graph_score"
    )
