
import pandas as pd

from src.features import build_features
from src.rules import apply_rules
from src.model import predict_proba
from src.graph_analysis import compute_graph_scores


def score_transactions(df):
    """
    Combine rule-based, machine-learning and graph-based
    fraud detection into one final risk score.

    Returns:
        Original transaction columns plus:
        rule_score, model_prob, graph_score,
        risk_score and reasons.
    """

    if df.empty:
        result = df.copy()
        result["rule_score"] = 0.0
        result["model_prob"] = 0.0
        result["graph_score"] = 0.0
        result["risk_score"] = 0.0
        result["reasons"] = [[] for _ in range(len(result))]
        return result

    # Preserve original transaction data.
    result = df.copy()

    # Generate behavioural features.
    features = build_features(df.copy())

    # 1. Rule-based detection.
    rule_score, rule_reasons = apply_rules(features)

    # 2. Machine-learning prediction.
    model_predictions = predict_proba(features)
    model_prob = model_predictions["model_prob"]

    # 3. Graph-based detection.
    graph_score = compute_graph_scores(df)

    # Combine all three detection approaches.
    risk_score = (
        0.40 * rule_score +
        45.0 * model_prob +
        15.0 * graph_score
    )

    risk_score = risk_score.clip(0, 100)

    # Generate understandable explanations.
    all_reasons = []

    for position in range(len(result)):

        reasons = list(rule_reasons.iloc[position])

        if model_prob.iloc[position] >= 0.70:
            reasons.append(
                "Machine learning model detected high fraud probability"
            )
        elif model_prob.iloc[position] >= 0.40:
            reasons.append(
                "Machine learning model detected suspicious behaviour"
            )

        if graph_score.iloc[position] >= 0.50:
            reasons.append(
                "Beneficiary shows suspicious payment network activity"
            )

        if not reasons:
            reasons.append("No major fraud indicators detected")

        all_reasons.append(reasons)

    # Add final scores and explanations.
    result["rule_score"] = rule_score
    result["model_prob"] = model_prob
    result["graph_score"] = graph_score
    result["risk_score"] = risk_score
    result["reasons"] = all_reasons

    return result


if __name__ == "__main__":

    input_path = "data/transactions.csv"
    output_path = "data/scored.csv"

    print("Loading transactions...")

    transactions = pd.read_csv(input_path)

    print("Scoring transactions...")

    scored = score_transactions(transactions)

    # Store reasons as readable text in CSV.
    scored["reasons"] = scored["reasons"].apply(
        lambda items: " | ".join(items)
    )

    scored.to_csv(output_path, index=False)

    print("\nSCORING COMPLETE")
    print("Total transactions:", len(scored))
    print("Average risk score:", round(scored["risk_score"].mean(), 2))
    print("High-risk transactions:", int((scored["risk_score"] >= 35).sum()))
    print("Saved output:", output_path)
