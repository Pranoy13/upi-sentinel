"""UPI Sentinel - Streamlit risk dashboard."""
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import db  # noqa: E402

DATA = os.path.join(ROOT, "data")

st.set_page_config(page_title="UPI Sentinel", page_icon="🛡️", layout="wide")

# ---------- dark, clean styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetric"] {
        background: #161b26; border: 1px solid #262d3d;
        padding: 14px 18px; border-radius: 12px;
    }
    div[data-testid="stMetricLabel"] {color: #9aa4b8;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- data loading ----------
@st.cache_data
def load_data():
    scored_path = os.path.join(DATA, "scored.csv")
    demo = False
    if os.path.exists(scored_path):
        df = pd.read_csv(scored_path, parse_dates=["timestamp"])
    else:
        demo = True
        df = pd.read_csv(os.path.join(DATA, "features.csv"), parse_dates=["timestamp"])
        s = (
            np.clip(df["amount_ratio"] - 1, 0, 8) * 6
            + np.clip(df["txn_count_10m"] - 1, 0, 6) * 8
            + df["new_device"] * 15
            + df["new_beneficiary"] * 10
            + np.clip(df["ben_fanin_3h"] - 1, 0, 6) * 6
            + df["odd_hour"] * 8
        )
        df["risk_score"] = np.clip(s, 0, 100).round(1)

        def make_reasons(r):
            out = []
            if r["amount_ratio"] >= 4:
                out.append(f"Amount {r['amount_ratio']:.1f}x user's typical")
            if r["txn_count_10m"] >= 4:
                out.append(f"{int(r['txn_count_10m'])} transactions in 10 minutes")
            if r["new_device"] == 1:
                out.append("New device")
            if r["new_beneficiary"] == 1:
                out.append("New beneficiary")
            if r["ben_fanin_3h"] >= 4:
                out.append(f"{int(r['ben_fanin_3h'])} payers to same beneficiary in 3h")
            if r["odd_hour"] == 1:
                out.append("Odd-hour activity")
            return " | ".join(out) if out else "No strong risk signals"

        df["reasons"] = df.apply(make_reasons, axis=1)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, demo


df, DEMO = load_data()
db.init_db()

# ---------- sidebar ----------
st.sidebar.title("🛡️ UPI Sentinel")
st.sidebar.caption("Explainable UPI fraud analytics (synthetic data)")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "High-Risk Queue", "Case View", "Cases", "Evaluation", "About / Privacy"],
)
threshold = st.sidebar.slider("Risk threshold", 0, 100, 60)

if DEMO:
    st.warning("⚠️ Demo scoring: data/scored.csv not found, using a temporary placeholder risk score.")

df["flagged"] = df["risk_score"] >= threshold

# ---------- helpers ----------
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")


# ---------- pages ----------
if page == "Overview":
    st.title("Overview")
    st.caption("Risk summary of synthetic UPI transactions at the chosen threshold.")

    flagged = df[df["flagged"]]
    open_cases = sum(1 for c in db.get_all_cases() if c["status"] == "Open")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total transactions", f"{len(df):,}")
    k2.metric(f"Flagged (score ≥ {threshold})", f"{len(flagged):,}", f"{len(flagged) / len(df):.1%} of all")
    k3.metric("Flagged amount", f"₹{flagged['amount'].sum():,.0f}")
    k4.metric("Open cases", f"{open_cases}")

    st.markdown("")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk score distribution")
        fig = px.histogram(df, x="risk_score", nbins=40, color_discrete_sequence=["#4c8dff"])
        fig.add_vline(x=threshold, line_dash="dash", line_color="#ff4b4b",
                      annotation_text=f"threshold {threshold}", annotation_position="top")
        fig.update_layout(**PLOT, height=380, xaxis_title="Risk score", yaxis_title="Transactions",
                          yaxis_type="log", margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Log scale: most transactions are low risk, the right tail is where fraud lives.")

    with c2:
        st.subheader("Flagged transactions by scenario")
        by_sc = flagged.groupby("scenario").size().reset_index(name="flagged").sort_values("flagged")
        by_sc["type"] = np.where(by_sc["scenario"].str.startswith("LEGIT"), "Legitimate (false positive)", "Fraud / other")
        fig2 = px.bar(by_sc, x="flagged", y="scenario", orientation="h", color="type",
                      color_discrete_map={"Legitimate (false positive)": "#f5a524", "Fraud / other": "#ff4b4b"})
        fig2.update_layout(**PLOT, height=380, xaxis_title="Flagged transactions", yaxis_title="",
                           margin=dict(t=30, b=10), legend_title_text="")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Orange bars are legitimate look-alike cases that were wrongly flagged.")

else:
    st.title(page)
    st.info("This page is coming in the next step.")