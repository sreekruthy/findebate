import streamlit as st
import plotly.graph_objects as go
from debate_engine import run_debate
import json
import os

st.set_page_config(layout="wide")



def is_valid_result(result):
    # check if debate actually worked
    timeline = result.get("debate_summary", {}).get("timeline", [])
    return any(turn.get("summary") for turn in timeline)
# ---------------- CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0b1220;
}

.card {
    padding: 20px;
    border-radius: 16px;
    background: linear-gradient(145deg, #0f172a, #111827);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 20px rgba(0,0,0,0.3);
    margin-bottom: 15px;
}

.title {
    font-size: 14px;
    color: #9ca3af;
}

.big {
    font-size: 36px;
    font-weight: bold;
}

.green { color: #22c55e; }
.orange { color: #f59e0b; }
.blue { color: #3b82f6; }

.section {
    margin-top: 25px;
    padding: 15px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
    background-color: #0b1220;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("## FinDebate — AI Financial Analysis")

col1, col2 = st.columns([3,1])

with col1:
    company = st.radio("Select company", ["Apple", "Tesla"], horizontal=True, label_visibility="collapsed")

with col2:
    run = st.button("Run Analysis")


# ---------------- RUN ----------------
if run:
    result = run_debate(company)

    CACHE_FILE = f"last_working_{company.lower()}.json"

    if is_valid_result(result):
        # Save good result
        with open(CACHE_FILE, "w") as f:
            json.dump(result, f)
    else:
        # Load previous good result if exists
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                result = json.load(f)
            st.warning("Using last successful result (API limit hit)")
        else:
            st.error("No valid data available yet. Try again later.")

    # ---------------- HERO ----------------
    col1, col2 = st.columns(2)

    with col1:
        decision = result["final_decision"]["decision"]

        color = "green" if decision == "BUY" else "orange"

        st.markdown(f"""
        <div class="card">
            <div class="title">FINAL DECISION</div>
            <div class="big {color}">{decision}</div>
            Conviction: {result["final_decision"]["conviction"]}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        risk = result["risk_summary"]["risk_score"]

        st.markdown(f"""
        <div class="card">
            <div class="title">RISK SCORE</div>
            <div class="big orange">{risk}/10</div>
            Level: {result["risk_summary"]["risk_level"]}
        </div>
        """, unsafe_allow_html=True)

    # ---------------- AGENTS ----------------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Agent Scores")

    cols = st.columns(4)

    for col, (name, data) in zip(cols, result["agent_outputs"].items()):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="title">{name.capitalize()}</div>
                <div class="big blue">{data['score']}/10</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- CHARTS ----------------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Charts")

    col1, col2 = st.columns(2)

    radar = result["chart_data"]["radar"]

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar["scores"],
            theta=radar["labels"],
            fill='toself',
            line=dict(color="#3b82f6")
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0,10])),
            paper_bgcolor="#0b1220",
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

    bar = result["chart_data"]["bar"]

    with col2:
        fig2 = go.Figure([
            go.Bar(
                x=bar["agents"],
                y=bar["scores"],
                text=bar["scores"],
                textposition='auto',
                marker_color=["#22c55e", "#3b82f6", "#f59e0b", "#a855f7"]
            )
        ])

        fig2.update_layout(
            paper_bgcolor="#0b1220",
            font=dict(color="white"),
            yaxis=dict(range=[0,10])
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- DEBATE ----------------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Debate Summary")

    for turn in result["debate_summary"]["timeline"]:
        st.markdown(f"""
        <div class="card">
            <b>Round {turn['round']} — {turn['agent']}</b><br>
            {turn['summary']}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TIME ----------------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Time Horizons")

    horizons = result["final_decision"].get("time_horizons", {})

    cols = st.columns(3)
    for col, (k, v) in zip(cols, horizons.items()):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="title">{k.replace("_"," ").title()}</div>
                <div class="big">{v}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- THESIS ----------------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Investment Thesis")

    thesis = result["final_decision"].get("investment_thesis", "")

    if isinstance(thesis, list):
        for t in thesis:
            st.write("•", t)
    else:
        st.write(thesis)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- RISK ----------------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Primary Risk")

    st.write(result["risk_summary"].get("primary_risk", ""))

    st.markdown('</div>', unsafe_allow_html=True)