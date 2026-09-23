import streamlit as st
from agent import run_support_agent
from escalation import initialize_escalation_state, render_escalation_ui

st.set_page_config(page_title="AI Support Desk 2026", page_icon="🤖", layout="wide")

initialize_escalation_state()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Config
st.sidebar.title("⚙️ AI Config")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

st.title("🎧 Customer Support AI Agent")

if not groq_api_key:
    st.info("👈 Enter your Groq API Key in the sidebar to start.")
    st.stop()

tab1, tab2 = st.tabs(["💬 Chat Assistant", "📋 Pending Human Escalations"])

with tab2:
    render_escalation_ui()

with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about policies, check order status, or request a representative...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        history_context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-6:]])

        with st.chat_message("assistant"):
            with st.spinner("Processing request..."):
                try:
                    response = run_support_agent(user_input, history_context, groq_api_key)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error processing agent task: {e}")

        st.rerun()