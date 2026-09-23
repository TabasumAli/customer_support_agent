import threading
from datetime import datetime

import streamlit as st


# ---------------------------------------------------------------------------
# Plain-Python, thread-safe store.
# Do NOT use st.session_state here — CrewAI calls these from a background
# thread with no Streamlit session context.
# ---------------------------------------------------------------------------
_escalation_lock = threading.Lock()
_escalations: list[dict] = []


def _add_ticket(summary: str) -> dict:
    ticket = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "status": "Pending",
    }
    with _escalation_lock:
        _escalations.append(ticket)
    return ticket


# ---------------------------------------------------------------------------
# Public API (safe from any thread)
# ---------------------------------------------------------------------------
def add_escalation_ticket(summary: str) -> dict:
    return _add_ticket(summary)


def get_escalations() -> list[dict]:
    with _escalation_lock:
        return list(_escalations)


def clear_escalations() -> None:
    with _escalation_lock:
        _escalations.clear()


# ---------------------------------------------------------------------------
# Streamlit UI helpers — only called from app.py during a script run.
# ---------------------------------------------------------------------------
def initialize_escalation_state() -> None:
    """No-op kept for backward compatibility with app.py imports."""
    return None


def render_escalation_ui() -> None:
    st.header("📋 Pending Escalations Queue")

    tickets = get_escalations()

    if not tickets:
        st.success("No pending human escalations.")
        return

    for idx, item in enumerate(tickets, 1):
        with st.expander(
            f"Ticket #{idx} | {item['timestamp']} | Status: {item['status']}"
        ):
            st.write("**Issue Summary:**")
            st.write(item["summary"])

    if st.button("🗑️ Clear all tickets"):
        clear_escalations()
        st.rerun()