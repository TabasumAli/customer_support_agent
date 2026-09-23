import io
import contextlib
import traceback

from crewai import Agent, Task, Crew, Process, LLM

from tools import (
    query_knowledge_base,
    order_database_lookup,
    escalate_to_human,
)


def run_support_agent(user_prompt: str, chat_history: str, api_key: str) -> str:
    """
    Run the customer support agent and return the assistant's response text.

    Uses Groq's OpenAI-compatible endpoint so we bypass LiteLLM entirely.
    This avoids the `cache_breakpoint` parameter that Groq's native API
    rejects (a known CrewAI <-> LiteLLM <-> Groq incompatibility).
    """

    # ------------------------------------------------------------------
    # LLM: Groq via OpenAI-compatible endpoint (native OpenAI provider).
    # ------------------------------------------------------------------
    llm = LLM(
        model="openai/openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        temperature=0.1,
    )

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------
    support_agent = Agent(
        role="Customer Support Agent",
        goal=(
            "Assist customers with queries using knowledge base search, "
            "order databases, or escalate when needed."
        ),
        backstory=(
            "You are a helpful and polite customer support assistant for "
            "an e-commerce platform."
        ),
        tools=[
            query_knowledge_base,
            order_database_lookup,
            escalate_to_human,
        ],
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------
    support_task = Task(
        description=(
            f"Conversation History:\n{chat_history}\n\n"
            f"Latest User Inquiry: {user_prompt}\n\n"
            "Instructions:\n"
            "1. Extract Order ID if present and use 'Order Database Lookup'.\n"
            "2. If query is policy-related, query knowledge base.\n"
            "3. If query cannot be resolved or customer requests a human, "
            "invoke 'Escalate to Human Agent'.\n"
            "4. Provide a direct, clear response."
        ),
        expected_output=(
            "Clear response to the customer with either the answer or an "
            "escalation confirmation."
        ),
        agent=support_agent,
    )

    # ------------------------------------------------------------------
    # Crew
    # ------------------------------------------------------------------
    crew = Crew(
        agents=[support_agent],
        tasks=[support_task],
        process=Process.sequential,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Kick off, capturing stdout/stderr so errors bubble up cleanly.
    # ------------------------------------------------------------------
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            result = crew.kickoff()
    except Exception:
        tb = traceback.format_exc()
        raise RuntimeError(
            f"{tb}\n\n--- CrewAI captured output ---\n{buf.getvalue()}"
        ) from None

    # ------------------------------------------------------------------
    # Normalize result
    # ------------------------------------------------------------------
    if hasattr(result, "raw") and result.raw:
        return result.raw
    if hasattr(result, "output") and result.output:
        return result.output
    return str(result)