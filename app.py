import sys
import logging

import streamlit as st

from agent import run_support_agent
from escalation import initialize_escalation_state, render_escalation_ui


# ---------------------------------------------------------------------------
# Logging — makes CrewAI errors visible in the terminal
# ---------------------------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Support Desk 2026",
    page_icon="🤖",
    layout="wide",
)

initialize_escalation_state()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_error" not in st.session_state:
    st.session_state.last_error = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ AI Config")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎧 Customer Support AI Agent")

if not groq_api_key:
    st.info("👈 Enter your Groq API Key in the sidebar to start.")
    st.stop()


# ---------------------------------------------------------------------------
# Persisted error panel (survives reruns)
# ---------------------------------------------------------------------------
if st.session_state.last_error:
    with st.expander("🔴 Last error (full traceback)", expanded=True):
        st.code(st.session_state.last_error, language="python")
        if st.button("Clear error"):
            st.session_state.last_error = None
            st.rerun()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["💬 Chat Assistant", "📋 Pending Human Escalations"])

with tab2:
    render_escalation_ui()


with tab1:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input(
        "Ask about policies, check order status, or request a representative..."
    )

    if user_input:
        # 1) User message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2) Short history context
        history_context = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in st.session_state.messages[-6:]
        )

        # 3) Call the agent
        with st.chat_message("assistant"):
            with st.spinner("Processing request..."):
                try:
                    response = run_support_agent(
                        user_input, history_context, groq_api_key
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    st.session_state.last_error = None

                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    st.session_state.last_error = tb
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"⚠️ **Error:** {e}",
                        }
                    )
                    st.error(f"Error processing agent task: {e}")

        # 4) Only rerun on success (keeps the error visible otherwise)
        if st.session_state.last_error is None:
            st.rerun()