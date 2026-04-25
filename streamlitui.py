import streamlit as st
from debate_engine import run_debate

st.title("FinDebate — Financial Analysis")
company = st.radio("Select Company", ["Apple", "Tesla"])

if st.button("Run Analysis"):
    with st.spinner("Running agents and debate..."):
        result = run_debate(company)
    
    # Hero
    st.metric("Decision", result["final_decision"]["decision"])
    st.write("Conviction:", result["final_decision"]["conviction"])
    
    # Risk
    st.metric("Risk Score", f"{result['risk_summary']['risk_score']}/10")
    
    # Agent scores
    for name, data in result["agent_outputs"].items():
        st.write(f"{name}: {data['score']}/10")
    
    # Charts — use result["chart_data"]["radar"] and ["bar"]
    # Debate — use result["debate_summary"]["timeline"]