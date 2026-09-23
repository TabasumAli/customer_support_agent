from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
from tools import query_knowledge_base, order_database_lookup, escalate_to_human

def run_support_agent(user_prompt: str, chat_history: str, api_key: str) -> str:
    llm = ChatGroq(
        temperature=0.1,
        model_name="groq/openai/gpt-oss-120b",
        groq_api_key=api_key
    )

    support_agent = Agent(
        role="Customer Support Agent",
        goal="Assist customers with queries using knowledge base search, order databases, or escalate when needed.",
        backstory="You are a helpful and polite customer support assistant for an e-commerce platform.",
        tools=[query_knowledge_base, order_database_lookup, escalate_to_human],
        verbose=False,
        llm=llm
    )

    support_task = Task(
        description=(
            f"Conversation History:\n{chat_history}\n\n"
            f"Latest User Inquiry: {user_prompt}\n\n"
            "Instructions:\n"
            "1. Extract Order ID if present and use 'Order Database Lookup'.\n"
            "2. If query is policy-related, query knowledge base.\n"
            "3. If query cannot be resolved or customer requests a human, invoke 'Escalate to Human Agent'.\n"
            "4. Provide a direct, clear response."
        ),
        expected_output="Clear response to customer with answer or escalation state.",
        agent=support_agent
    )

    crew = Crew(
        agents=[support_agent],
        tasks=[support_task],
        process=Process.sequential
    )

    result = crew.kickoff()
    return str(result)