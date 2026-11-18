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

elif page == "High-Risk Queue":
    st.title("High-Risk Queue")
    st.caption("Flagged transactions, highest risk first. Pick one to open its case.")

    q = df[df["flagged"]].copy()

    f1, f2, f3 = st.columns(3)
    scen_opts = sorted(q["scenario"].unique())
    city_opts = sorted(q["city"].unique())
    sel_scen = f1.multiselect("Scenario", scen_opts)
    sel_city = f2.multiselect("City", city_opts)
    min_score = f3.slider("Min risk score", 0, 100, int(threshold))

    if sel_scen:
        q = q[q["scenario"].isin(sel_scen)]
    if sel_city:
        q = q[q["city"].isin(sel_city)]
    q = q[q["risk_score"] >= min_score].sort_values("risk_score", ascending=False)

    case_map = {c["txn_id"]: c["status"] for c in db.get_all_cases()}
    q["case_status"] = q["txn_id"].map(case_map).fillna("No case")

    st.write(f"**{len(q):,}** transactions match.")

    show = q[["txn_id", "timestamp", "user_id", "beneficiary_id", "amount", "city",
              "scenario", "risk_score", "case_status", "reasons"]].reset_index(drop=True)

    event = st.dataframe(
        show,
        use_container_width=True,
        height=450,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "risk_score": st.column_config.ProgressColumn("risk_score", min_value=0, max_value=100, format="%.0f"),
            "amount": st.column_config.NumberColumn("amount", format="₹%.0f"),
            "reasons": st.column_config.TextColumn("reasons", width="large"),
        },
    )

    rows = event.selection.rows
    if rows:
        chosen = show.iloc[rows[0]]["txn_id"]
        st.session_state["selected_txn"] = chosen
        st.success(f"Selected **{chosen}**. Open **Case View** in the sidebar to review it.")
    elif "selected_txn" in st.session_state:
        st.info(f"Currently selected: **{st.session_state['selected_txn']}**")

elif page == "Case View":
    st.title("Case View")
    st.caption("Evidence, timeline and decision for one flagged transaction.")

    flagged_ids = (
        df[df["flagged"]].sort_values("risk_score", ascending=False)["txn_id"].tolist()
    )
    if not flagged_ids:
        st.info("No flagged transactions at this threshold. Lower the risk threshold in the sidebar.")
        st.stop()

    pre = st.session_state.get("selected_txn")
    start = flagged_ids.index(pre) if pre in flagged_ids else 0
    chosen = st.selectbox("Transaction", flagged_ids, index=start)
    st.session_state["selected_txn"] = chosen

    row = df[df["txn_id"] == chosen].iloc[0]
    existing = db.get_case(chosen)
    cur_status = existing["status"] if existing else "No case yet"

    # ---------- header ----------
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Amount", f"₹{row['amount']:,.0f}")
    h2.metric("User", row["user_id"])
    h3.metric("Beneficiary", row["beneficiary_id"])
    h4.metric("Case status", cur_status)

    # ---------- evidence: gauge + reasons + key features ----------
    e1, e2 = st.columns([1, 1])

    with e1:
        st.subheader("Risk score")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(row["risk_score"]),
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff4b4b"},
                "steps": [
                    {"range": [0, 40], "color": "#1f3a2b"},
                    {"range": [40, 70], "color": "#4a3b1a"},
                    {"range": [70, 100], "color": "#4a1f24"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": threshold},
            },
        ))
        gauge.update_layout(**PLOT, height=260, margin=dict(t=20, b=10, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True)

        st.subheader("Why it was flagged")
        for r in str(row["reasons"]).split(" | "):
            st.markdown(f"- {r}")

    with e2:
        st.subheader("Key features")
        yes_no = lambda v: "🔴 Yes" if int(v) == 1 else "🟢 No"
        feats = pd.DataFrame({
            "Feature": [
                "Amount", "User's typical amount", "Amount vs typical",
                "Txns in last 10 min", "New device", "New beneficiary",
                "New city", "Odd hour", "Payers to this beneficiary (3h)",
            ],
            "Value": [
                f"₹{row['amount']:,.0f}", f"₹{row['user_median']:,.0f}", f"{row['amount_ratio']:.1f}x",
                str(int(row["txn_count_10m"])), yes_no(row["new_device"]), yes_no(row["new_beneficiary"]),
                yes_no(row["new_city"]), yes_no(row["odd_hour"]), str(int(row["ben_fanin_3h"])),
            ],
        })
        st.dataframe(feats, hide_index=True, use_container_width=True, height=360)
        st.caption(f"City: {row['city']} | Channel: {row['channel']} | Device: {row['device_id']} | {row['timestamp']}")

    # ---------- transaction timeline ----------
    st.subheader("Transaction timeline (this user)")
    u = df[df["user_id"] == row["user_id"]].sort_values("timestamp")
    normal = u[~u["flagged"]]
    flg = u[u["flagged"] & (u["txn_id"] != chosen)]
    sel = u[u["txn_id"] == chosen]

    tl = go.Figure()
    tl.add_trace(go.Scatter(x=u["timestamp"], y=u["amount"], mode="lines",
                            line=dict(color="#3a4256", width=1), hoverinfo="skip", showlegend=False))
    tl.add_trace(go.Scatter(x=normal["timestamp"], y=normal["amount"], mode="markers", name="Normal",
                            marker=dict(color="#4c8dff", size=8), text=normal["txn_id"]))
    tl.add_trace(go.Scatter(x=flg["timestamp"], y=flg["amount"], mode="markers", name="Flagged",
                            marker=dict(color="#ff4b4b", size=10), text=flg["txn_id"]))
    tl.add_trace(go.Scatter(x=sel["timestamp"], y=sel["amount"], mode="markers", name="Selected",
                            marker=dict(color="#ffd23f", size=18, symbol="star",
                                        line=dict(color="white", width=1)), text=sel["txn_id"]))
    tl.update_layout(**PLOT, height=380, xaxis_title="Time", yaxis_title="Amount (₹)",
                     margin=dict(t=20, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(tl, use_container_width=True)
    st.caption(f"{len(u)} transactions by {row['user_id']}. Red = flagged, yellow star = selected.")

    # ---------- related transactions ----------
    st.subheader("Other transactions to the same beneficiary")
    rel = df[(df["beneficiary_id"] == row["beneficiary_id"]) & (df["txn_id"] != chosen)]
    rel = rel.sort_values("timestamp")[["txn_id", "timestamp", "user_id", "amount", "city", "risk_score"]]
    if rel.empty:
        st.write("None. This beneficiary only received this one transaction.")
    else:
        st.write(f"**{len(rel)}** other transactions, from **{rel['user_id'].nunique()}** different payers.")
        st.dataframe(rel.tail(15), hide_index=True, use_container_width=True)

    # ---------- analyst decision ----------
    st.subheader("Analyst decision")
    note = st.text_area("Analyst note", value=existing["analyst_note"] if existing else "",
                        key=f"note_{chosen}", placeholder="Why are you making this call?")

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("🚨 Confirm Fraud", use_container_width=True):
        db.save_case(chosen, "Confirmed Fraud", note)
        st.rerun()
    if b2.button("✅ False Positive", use_container_width=True):
        db.save_case(chosen, "False Positive", note)
        st.rerun()
    if b3.button("⬆️ Escalate", use_container_width=True):
        db.save_case(chosen, "Escalated", note)
        st.rerun()
    if b4.button("💾 Save note (Open)", use_container_width=True):
        db.save_case(chosen, "Open", note)
        st.rerun()
elif page == "Cases":
    st.title("Cases")
    st.caption("All analyst decisions saved in the case database.")

    cases = pd.DataFrame([dict(c) for c in db.get_all_cases()])
    if cases.empty:
        st.info("No cases yet. Open a transaction in Case View and save a decision.")
        st.stop()

    merged = cases.merge(
        df[["txn_id", "timestamp", "user_id", "beneficiary_id", "amount", "scenario", "risk_score", "reasons"]],
        on="txn_id", how="left",
    ).sort_values("updated_at", ascending=False)

    statuses = ["Open", "Confirmed Fraud", "False Positive", "Escalated"]
    for col, s in zip(st.columns(4), statuses):
        col.metric(s, int((merged["status"] == s).sum()))

    sel_status = st.multiselect("Filter by status", statuses, default=statuses)
    view = merged[merged["status"].isin(sel_status)]

    st.dataframe(
        view[["txn_id", "status", "analyst_note", "updated_at", "user_id", "beneficiary_id",
              "amount", "scenario", "risk_score", "reasons"]],
        hide_index=True,
        use_container_width=True,
        height=420,
        column_config={
            "amount": st.column_config.NumberColumn("amount", format="₹%.0f"),
            "risk_score": st.column_config.ProgressColumn("risk_score", min_value=0, max_value=100, format="%.0f"),
            "analyst_note": st.column_config.TextColumn("analyst_note", width="large"),
            "reasons": st.column_config.TextColumn("reasons", width="large"),
        },
    )

    st.download_button(
        "⬇️ Export cases as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="upi_sentinel_cases.csv",
        mime="text/csv",
    )

elif page == "Evaluation":
    import json

    st.title("Evaluation")
    st.caption("Detection performance against ground truth labels (from Person A's metrics.json).")

    mpath = os.path.join(DATA, "metrics.json")
    if not os.path.exists(mpath):
        st.info("Metrics not generated yet. Person A will add data/metrics.json.")
        st.stop()

    with open(mpath, "r", encoding="utf-8") as f:
        m = json.load(f)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision", f"{m.get('precision', 0):.1%}")
    k2.metric("Recall", f"{m.get('recall', 0):.1%}")
    k3.metric("F1 score", f"{m.get('f1', 0):.1%}")
    k4.metric("False positive rate", f"{m.get('fpr', 0):.2%}")
    if "threshold" in m:
        st.caption(f"Evaluated at risk threshold: {m['threshold']}")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Confusion matrix")
        cm = m.get("confusion_matrix")
        if cm is None:
            st.write("Not available.")
        else:
            if isinstance(cm, dict):
                cm = [[cm.get("tn", 0), cm.get("fp", 0)], [cm.get("fn", 0), cm.get("tp", 0)]]
            fig = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                x=["Predicted legit", "Predicted fraud"], y=["Actual legit", "Actual fraud"],
            )
            fig.update_layout(**PLOT, height=380, margin=dict(t=20, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Detection rate by scenario")
        ps = m.get("per_scenario_detection") or m.get("per_scenario") or m.get("scenario_detection")
        if not ps:
            st.write("Not available.")
        else:
            if isinstance(ps, dict):
                ps_df = pd.DataFrame({"scenario": list(ps.keys()), "rate": list(ps.values())})
            else:
                ps_df = pd.DataFrame(ps)
                ps_df.columns = ["scenario", "rate"] + list(ps_df.columns[2:])
            if ps_df["rate"].max() > 1:
                ps_df["rate"] = ps_df["rate"] / 100
            ps_df = ps_df.sort_values("rate")
            fig2 = px.bar(ps_df, x="rate", y="scenario", orientation="h",
                          color_discrete_sequence=["#4c8dff"])
            fig2.update_layout(**PLOT, height=380, xaxis_title="Detection rate", yaxis_title="",
                               xaxis_tickformat=".0%", xaxis_range=[0, 1], margin=dict(t=20, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Raw metrics.json"):
        st.json(m)

elif page == "About / Privacy":
    st.title("About / Privacy")
    st.markdown(
        """
**UPI Sentinel** is an explainable fraud analytics prototype built for Quick Heal Cyber Arena 2026 (PS-17).

### Privacy safeguards
- **All data is synthetic.** No real people, accounts or transactions are used.
- **IDs are masked.** User, beneficiary and device IDs are generated tokens (for example U0267, B0605).
- **No payment network connection.** The system never connects to UPI, banks or any real financial system.
- **Read-only analytics.** The dashboard only reads scored data. It cannot move money or block payments.
- **Local case storage.** Analyst decisions are stored in a local SQLite file that is not committed to the repository.

### How scoring works
Each transaction gets a risk score from 0 to 100, built from per-user baselines, rule-based reason codes,
an anomaly and classification model, and graph analysis of mule beneficiaries. Every flag comes with
human-readable reasons so an analyst can see why it was raised.

### Limitations
Results are based on simulated fraud patterns and will differ on real traffic. The system supports
analysts and does not replace them.
        """
    )