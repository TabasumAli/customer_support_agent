import pandas as pd
import streamlit as st

def initialize_escalation_state():
    if "pending_escalations" not in st.session_state:
        st.session_state.pending_escalations = []

def add_escalation_ticket(summary: str):
    ticket = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "status": "Pending"
    }
    st.session_state.pending_escalations.append(ticket)

def render_escalation_ui():
    st.header("📋 Pending Escalations Queue")
    if not st.session_state.pending_escalations:
        st.success("No pending human escalations.")
    else:
        for idx, item in enumerate(st.session_state.pending_escalations, 1):
            with st.expander(f"Ticket #{idx} | {item['timestamp']} | Status: {item['status']}"):
                st.write("**Issue Summary:**")
                st.write(item["summary"])