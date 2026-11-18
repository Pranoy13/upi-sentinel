"""UPI Sentinel - Streamlit risk dashboard (synthetic data only)."""
import json
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

PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
FEATURE_SHOW = ["amount_ratio", "user_median", "new_device", "new_beneficiary", "new_city",
                "txn_count_10m", "amount_sum_1h", "near_limit_count_1h", "ben_fanin_3h", "odd_hour"]


# ---------- data loading ----------
@st.cache_data
def load_data():
    feats = pd.read_csv(os.path.join(DATA, "features.csv"), parse_dates=["timestamp"])
    scored_path = os.path.join(DATA, "scored.csv")
    if os.path.exists(scored_path):
        df = pd.read_csv(scored_path, parse_dates=["timestamp"])
        extra = [c for c in feats.columns if c not in df.columns]
        if extra:
            df = df.merge(feats[["txn_id"] + extra], on="txn_id", how="left")
        demo = False
    else:
        demo = True
        df = feats.copy()
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
    df["reasons"] = df["reasons"].fillna("No strong risk signals")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, demo


@st.cache_data
def load_metrics():
    path = os.path.join(DATA, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


# =====================================================================
# OVERVIEW
# =====================================================================
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
        st.plotly_chart(fig, width="stretch")
        st.caption("Log scale: most transactions are low risk, the right tail is where fraud lives.")

    with c2:
        st.subheader("Flagged transactions by scenario")
        by_sc = flagged.groupby("scenario").size().reset_index(name="flagged").sort_values("flagged")
        by_sc["type"] = np.where(by_sc["scenario"].str.startswith("LEGIT"),
                                 "Legitimate (false positive)", "Fraud / other")
        fig2 = px.bar(by_sc, x="flagged", y="scenario", orientation="h", color="type",
                      color_discrete_map={"Legitimate (false positive)": "#f5a524", "Fraud / other": "#ff4b4b"})
        fig2.update_layout(**PLOT, height=380, xaxis_title="Flagged transactions", yaxis_title="",
                           margin=dict(t=30, b=10), legend_title_text="")
        st.plotly_chart(fig2, width="stretch")
        st.caption("Orange bars are legitimate look-alike cases that were wrongly flagged.")


# =====================================================================
# HIGH-RISK QUEUE
# =====================================================================
elif page == "High-Risk Queue":
    st.title("High-Risk Queue")
    st.caption("Flagged transactions, highest risk first. Pick one to open its case.")

    q = df[df["flagged"]].copy()

    f1, f2, f3 = st.columns(3)
    sel_scen = f1.multiselect("Scenario", sorted(q["scenario"].unique()))
    sel_city = f2.multiselect("City", sorted(q["city"].unique()))
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
        width="stretch",
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


# =====================================================================
# CASE VIEW
# =====================================================================
elif page == "Case View":
    st.title("Case View")

    ranked = df[df["flagged"]].sort_values("risk_score", ascending=False)["txn_id"].tolist()
    current = st.session_state.get("selected_txn")
    options = ranked[:]
    if current and current not in options:
        options.insert(0, current)

    if not options:
        st.info("No flagged transactions at this threshold. Lower the risk threshold in the sidebar.")
        st.stop()

    idx = options.index(current) if current in options else 0
    txn_id = st.selectbox("Transaction", options, index=idx)
    st.session_state["selected_txn"] = txn_id

    row = df[df["txn_id"] == txn_id].iloc[0]
    existing = db.get_case(txn_id)

    st.caption(
        f"{row['txn_id']} · {row['user_id']} → {row['beneficiary_id']} · "
        f"₹{row['amount']:,.0f} · {row['timestamp']:%d %b %Y, %H:%M} · {row['city']} · {row['channel']}"
    )

    left, right = st.columns([1, 1.4])

    # ----- evidence panel -----
    with left:
        st.subheader("Risk score")
        score = float(row["risk_score"])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff4b4b" if score >= threshold else "#4c8dff"},
                "steps": [
                    {"range": [0, 40], "color": "#1f3a2b"},
                    {"range": [40, 70], "color": "#4a3d1c"},
                    {"range": [70, 100], "color": "#4a1f24"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": threshold},
            },
        ))
        gauge.update_layout(**PLOT, height=250, margin=dict(t=30, b=0, l=20, r=20))
        st.plotly_chart(gauge, width="stretch")

        st.subheader("Why it was flagged")
        for r in str(row["reasons"]).split(" | "):
            st.markdown(f"- {r}")

        sub = {k: row[k] for k in ["rule_score", "model_prob", "graph_score"] if k in row.index}
        if sub:
            s1, s2, s3 = st.columns(3)
            cols = [s1, s2, s3]
            for c, (k, v) in zip(cols, sub.items()):
                c.metric(k.replace("_", " "), f"{float(v):.2f}")

    with right:
        st.subheader("Key evidence")
        e1, e2, e3 = st.columns(3)
        if "user_median" in row.index and pd.notna(row["user_median"]):
            e1.metric("Amount", f"₹{row['amount']:,.0f}", f"typical ₹{row['user_median']:,.0f}", delta_color="off")
        else:
            e1.metric("Amount", f"₹{row['amount']:,.0f}")
        if "amount_ratio" in row.index and pd.notna(row["amount_ratio"]):
            e2.metric("Amount vs typical", f"{row['amount_ratio']:.1f}x")
        if "txn_count_10m" in row.index and pd.notna(row["txn_count_10m"]):
            e3.metric("Txns in 10 min", f"{int(row['txn_count_10m'])}")

        def yn(col):
            if col in row.index and pd.notna(row[col]):
                return "🔴 Yes" if int(row[col]) == 1 else "🟢 No"
            return "n/a"

        e4, e5, e6 = st.columns(3)
        e4.metric("New device", yn("new_device"))
        e5.metric("New beneficiary", yn("new_beneficiary"))
        e6.metric("New city", yn("new_city"))
        e7, e8, e9 = st.columns(3)
        e7.metric("Odd hour", yn("odd_hour"))
        if "ben_fanin_3h" in row.index and pd.notna(row["ben_fanin_3h"]):
            e8.metric("Payers to beneficiary (3h)", f"{int(row['ben_fanin_3h'])}")
        if "near_limit_count_1h" in row.index and pd.notna(row["near_limit_count_1h"]):
            e9.metric("Near-limit txns (1h)", f"{int(row['near_limit_count_1h'])}")

    # ----- timeline -----
    st.subheader("Transaction timeline")
    ut = df[df["user_id"] == row["user_id"]].sort_values("timestamp")
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=ut["timestamp"], y=ut["amount"], mode="lines", name="User activity",
                               line=dict(color="#4c8dff", width=1), hoverinfo="skip"))
    normal = ut[~ut["flagged"]]
    fig_t.add_trace(go.Scatter(x=normal["timestamp"], y=normal["amount"], mode="markers", name="Normal",
                               marker=dict(color="#4c8dff", size=8),
                               customdata=normal[["txn_id", "risk_score"]],
                               hovertemplate="%{customdata[0]}<br>₹%{y:,.0f}<br>score %{customdata[1]}<extra></extra>"))
    fl = ut[ut["flagged"]]
    fig_t.add_trace(go.Scatter(x=fl["timestamp"], y=fl["amount"], mode="markers", name="Flagged",
                               marker=dict(color="#ff4b4b", size=11),
                               customdata=fl[["txn_id", "risk_score"]],
                               hovertemplate="%{customdata[0]}<br>₹%{y:,.0f}<br>score %{customdata[1]}<extra></extra>"))
    fig_t.add_trace(go.Scatter(x=[row["timestamp"]], y=[row["amount"]], mode="markers", name="Selected",
                               marker=dict(color="#ffd400", size=20, symbol="star",
                                           line=dict(color="white", width=1)),
                               hoverinfo="skip"))
    fig_t.update_layout(**PLOT, height=380, xaxis_title="Time", yaxis_title="Amount (₹)",
                        margin=dict(t=20, b=10))
    st.plotly_chart(fig_t, width="stretch")
    st.caption(f"{len(ut)} transactions for {row['user_id']}. Red = flagged, yellow star = selected.")

    # ----- related transactions -----
    st.subheader("Other transactions to the same beneficiary")
    rel = df[(df["beneficiary_id"] == row["beneficiary_id"]) & (df["txn_id"] != txn_id)]
    rel = rel.sort_values("timestamp")
    if rel.empty:
        st.write("No other transactions to this beneficiary.")
    else:
        st.write(f"{len(rel)} other transaction(s), from {rel['user_id'].nunique()} different payer(s).")
        st.dataframe(
            rel[["txn_id", "timestamp", "user_id", "amount", "city", "scenario", "risk_score"]].head(50),
            width="stretch", hide_index=True, height=250,
        )

    # ----- analyst actions -----
    st.subheader("Analyst decision")
    if existing:
        st.info(f"Current status: **{existing['status']}** (updated {existing['updated_at']})")
    note = st.text_area("Analyst note", value=(existing["analyst_note"] if existing else ""),
                        key=f"note_{txn_id}", placeholder="Why are you making this decision?")
    b1, b2, b3 = st.columns(3)
    if b1.button("🚨 Confirm Fraud", width="stretch"):
        db.save_case(txn_id, "Confirmed Fraud", note)
        st.success("Saved: Confirmed Fraud")
        st.rerun()
    if b2.button("✅ False Positive", width="stretch"):
        db.save_case(txn_id, "False Positive", note)
        st.success("Saved: False Positive")
        st.rerun()
    if b3.button("⬆️ Escalate", width="stretch"):
        db.save_case(txn_id, "Escalated", note)
        st.success("Saved: Escalated")
        st.rerun()
    if not existing:
        if st.button("Open case (no decision yet)"):
            db.save_case(txn_id, "Open", note)
            st.rerun()


# =====================================================================
# CASES
# =====================================================================
elif page == "Cases":
    st.title("Cases")
    st.caption("All analyst decisions saved in the local case database.")

    cases = db.get_all_cases()
    if not cases:
        st.info("No cases yet. Open a transaction in Case View and save a decision.")
    else:
        cdf = pd.DataFrame(cases)
        info = df[["txn_id", "user_id", "amount", "scenario", "risk_score"]]
        cdf = cdf.merge(info, on="txn_id", how="left")

        statuses = ["Open", "Confirmed Fraud", "False Positive", "Escalated"]
        sel = st.multiselect("Filter by status", statuses, default=statuses)
        cdf = cdf[cdf["status"].isin(sel)]

        m1, m2, m3, m4 = st.columns(4)
        for col, s in zip([m1, m2, m3, m4], statuses):
            col.metric(s, int((pd.DataFrame(cases)["status"] == s).sum()))

        st.dataframe(cdf, width="stretch", hide_index=True)
        st.download_button("⬇️ Export cases (CSV)", cdf.to_csv(index=False).encode("utf-8"),
                           file_name="upi_sentinel_cases.csv", mime="text/csv")


# =====================================================================
# EVALUATION
# =====================================================================
elif page == "Evaluation":
    st.title("Evaluation")
    m = load_metrics()
    if not m:
        st.info("Metrics not generated yet. Waiting for data/metrics.json.")
    else:
        st.caption(
            f"Time-based hold-out: {m.get('test_rows', '?'):,} test transactions, "
            f"{m.get('test_fraud', '?')} fraud · evaluated at threshold {m.get('threshold', '?')}."
        )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Precision", f"{m.get('precision', 0):.1%}")
        k2.metric("Recall", f"{m.get('recall', 0):.1%}")
        k3.metric("F1 score", f"{m.get('f1', 0):.3f}")
        k4.metric("False positive rate", f"{m.get('fpr', 0):.2%}")

        k5, k6, k7 = st.columns(3)
        k5.metric("Accuracy", f"{m.get('accuracy', 0):.2%}")
        k6.metric("ROC-AUC", f"{m.get('roc_auc', 0):.3f}")
        k7.metric("PR-AUC", f"{m.get('pr_auc', 0):.3f}")

        c1, c2 = st.columns([1, 1.5])

        with c1:
            st.subheader("Confusion matrix")
            cm = m.get("confusion_matrix")
            if isinstance(cm, dict):
                z = [[cm.get("tn", 0), cm.get("fp", 0)], [cm.get("fn", 0), cm.get("tp", 0)]]
            elif isinstance(cm, list):
                z = cm
            else:
                z = None
            if z:
                fig_cm = px.imshow(z, text_auto=True, color_continuous_scale="Blues",
                                   x=["Predicted legit", "Predicted fraud"],
                                   y=["Actual legit", "Actual fraud"])
                fig_cm.update_layout(**PLOT, height=380, coloraxis_showscale=False, margin=dict(t=20, b=10))
                st.plotly_chart(fig_cm, width="stretch")
            else:
                st.info("Not available.")

        with c2:
            st.subheader("Flag rate by scenario")
            per = m.get("per_scenario", {})
            if per:
                ps_df = pd.DataFrame([
                    {
                        "scenario": name,
                        "type": v.get("type", ""),
                        "count": v.get("count", 0),
                        "flagged": v.get("flagged", 0),
                        "rate": float(v.get("flag_rate", 0)) * 100,
                        "avg_score": v.get("avg_score", 0),
                    }
                    for name, v in per.items()
                ]).sort_values("rate")
                ps_df["kind"] = np.where(ps_df["type"] == "legit", "Legitimate (want LOW)", "Fraud (want HIGH)")
                fig_ps = px.bar(
                    ps_df, x="rate", y="scenario", orientation="h", color="kind",
                    color_discrete_map={"Legitimate (want LOW)": "#f5a524", "Fraud (want HIGH)": "#ff4b4b"},
                    hover_data=["count", "flagged", "avg_score"],
                )
                fig_ps.update_layout(**PLOT, height=380, xaxis_title="% of transactions flagged",
                                     yaxis_title="", xaxis_range=[0, 105], legend_title_text="",
                                     margin=dict(t=20, b=10))
                st.plotly_chart(fig_ps, width="stretch")
                st.caption("Fraud scenarios should be flagged (red, high). Legitimate look-alikes should not be (orange, low).")
            else:
                st.info("Per-scenario results not available.")


# =====================================================================
# ABOUT / PRIVACY
# =====================================================================
else:
    st.title("About / Privacy")
    st.markdown(
        """
**UPI Sentinel** is a fraud analytics prototype built for Quick Heal Cyber Arena 2026 (PS-17).

### Privacy safeguards
- **All data is synthetic.** No real customers, accounts or transactions are used.
- **IDs are masked.** User, beneficiary and device identifiers are generated placeholders.
- **No payment network connection.** The system never talks to UPI, NPCI or any bank.
- **Read-only analytics.** It scores and explains transactions; it never moves money or blocks accounts.
- **Local case storage.** Analyst notes stay in a local SQLite file (`data/cases.db`), which is not committed to git.

### How a transaction gets its risk score
Rules with reason codes, an Isolation Forest + Random Forest model, and graph analysis of mule
beneficiaries are combined into one 0-100 score, with the reasons shown to the analyst.

*This is a hackathon prototype and must not be used for decisions about real transactions.*
        """
    )