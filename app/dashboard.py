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

# ---------- pages (filled in next steps) ----------
if page == "Overview":
    st.title("Overview")
    st.write(f"Loaded {len(df):,} transactions.")
else:
    st.title(page)
    st.info("This page is coming in the next step.")