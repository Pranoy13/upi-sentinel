
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
        return pd.Series(
            0.0,
            index=df.index,
            name="graph_score"
        )

    data = df.copy()

    # Preserve original row positions.
    data["original_position"] = np.arange(len(data))

    # Convert timestamps.
    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    if data["timestamp"].isna().any():
        raise ValueError("Invalid timestamps found in graph data.")

    # Process transactions chronologically.
    data = data.sort_values(
        ["timestamp", "original_position"]
    )

    # Directed graph: sender -> beneficiary.
    graph = nx.DiGraph()

    # Rolling three-hour transaction history per beneficiary.
    windows = defaultdict(deque)

    # Store scores in original transaction order.
    scores = np.zeros(len(data), dtype=float)

    # Maintain historical incoming degrees.
    incoming_degrees = []

    window_duration = pd.Timedelta(hours=3)

    for row in data.itertuples(index=False):

        user = row.user_id
        beneficiary = row.beneficiary_id
        timestamp = row.timestamp
        position = row.original_position

        if pd.isna(user) or pd.isna(beneficiary):
            continue

        # Get this beneficiary's rolling history.
        window = windows[beneficiary]

        cutoff = timestamp - window_duration

        # Remove transactions older than three hours.
        while window and window[0][0] < cutoff:
            window.popleft()

        # Add current sender to rolling history.
        window.append((timestamp, user))

        # Count distinct senders in the rolling window.
        distinct_senders = len({
            sender for _, sender in window
        })

        # Historical incoming degree before current transaction.
        historical_degree = (
            graph.in_degree(beneficiary)
            if graph.has_node(beneficiary)
            else 0
        )

        # Calculate typical historical incoming degree.
        typical_degree = (
            float(np.median(incoming_degrees))
            if incoming_degrees
            else 1.0
        )

        # Rolling fan-in component.
        fanin_component = min(
            distinct_senders / 12.0,
            1.0
        )

        # Compare beneficiary degree with typical degree.
        degree_ratio = (
            (historical_degree + 1) /
            (typical_degree * 3 + 1)
        )

        degree_component = min(
            degree_ratio,
            1.0
        )

        # Combined graph suspicion score.
        score = (
            0.75 * fanin_component +
            0.25 * degree_component
        )

        scores[position] = min(
            max(score, 0.0),
            1.0
        )

        # Add current transaction to historical graph.
        previous_degree = historical_degree

        graph.add_edge(user, beneficiary)

        updated_degree = graph.in_degree(beneficiary)

        # Update degree history only when a new sender adds an edge.
        if updated_degree > previous_degree:
            incoming_degrees.append(updated_degree)

    return pd.Series(
        scores,
        index=df.index,
        name="graph_score"
    )
