import json
from functools import lru_cache

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from crewai.tools import tool

from escalation import add_escalation_ticket


# ---------------------------------------------------------------------------
# Cached loaders — plain Python caching, safe from CrewAI background threads.
# Do NOT use st.cache_resource / st.cache_data here.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def load_faiss_and_chunks():
    index = faiss.read_index("data/index.faiss")
    with open("data/chunks.json", "r") as f:
        chunks = json.load(f)
    return index, chunks


@lru_cache(maxsize=1)
def load_orders_db() -> pd.DataFrame:
    return pd.read_csv("data/customer_orders_database.csv")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool("Query Knowledge Base")
def query_knowledge_base(query: str) -> str:
    """Search the vector index for support policies, warranty, and return
    information. Use for any policy / FAQ / general question that is not
    about a specific order."""
    embedder = load_embedder()
    faiss_idx, kb_chunks = load_faiss_and_chunks()

    q_vec = embedder.encode([query])
    D, I = faiss_idx.search(np.array(q_vec).astype("float32"), k=2)

    results = []
    for idx in I[0]:
        if 0 <= idx < len(kb_chunks):
            chunk = kb_chunks[idx]
            results.append(
                f"Content: {chunk['text']}\n"
                f"Source: {chunk['metadata']['source_filename']} "
                f"(Page {chunk['metadata']['page_number']})"
            )

    return "\n\n".join(results) if results else "No policy information found."


@tool("Order Database Lookup")
def order_database_lookup(order_id: str) -> str:
    """Look up an order in the CSV database by its Order ID
    (e.g. 'ORD-2026-1001'). Input must be a single order ID string."""
    orders_df = load_orders_db()
    clean_id = order_id.strip().upper()

    col_map = {col.lower(): col for col in orders_df.columns}
    id_col = col_map.get("order id") or col_map.get("order_id")

    if id_col is None:
        return (
            "Order DB misconfigured: no 'Order ID' column found. "
            f"Columns present: {list(orders_df.columns)}"
        )

    matched = orders_df[
        orders_df[id_col].astype(str).str.upper().str.strip() == clean_id
    ]

    if matched.empty:
        return f"No record found for Order ID: {clean_id}"

    rec = matched.iloc[0]
    return "\n".join(f"{str(col).title()}: {val}" for col, val in rec.items())


@tool("Escalate to Human Agent")
def escalate_to_human(issue_summary: str) -> str:
    """Escalate the current conversation to a human support agent. Use when
    the customer explicitly asks for a human, or the issue cannot be
    resolved with the other tools. Provide a short issue summary."""
    add_escalation_ticket(issue_summary)
    return (
        "ESCALATED: Your request has been forwarded to a human "
        "representative under pending tickets."
    )