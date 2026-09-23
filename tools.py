import json
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.tools import tool
from escalation import add_escalation_ticket

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_faiss_and_chunks():
    index = faiss.read_index("data/index.faiss")
    with open("data/chunks.json", "r") as f:
        chunks = json.load(f)
    return index, chunks

# ---------------------------------------------------------
# UPDATED: Load CSV instead of XLSX
# ---------------------------------------------------------
@st.cache_data
def load_orders_db():
    return pd.read_csv("data/customer_orders_database.csv")

@tool("Query Knowledge Base")
def query_knowledge_base(query: str) -> str:
    """Searches vector index for support policies, warranty, and return info."""
    embedder = load_embedder()
    faiss_idx, kb_chunks = load_faiss_and_chunks()
    
    q_vec = embedder.encode([query])
    D, I = faiss_idx.search(np.array(q_vec).astype("float32"), k=2)
    
    results = []
    for idx in I[0]:
        if idx < len(kb_chunks):
            chunk = kb_chunks[idx]
            results.append(f"Content: {chunk['text']}\nSource: {chunk['metadata']['source_filename']} (Page {chunk['metadata']['page_number']})")
    
    return "\n\n".join(results) if results else "No policy information found."

@tool("Order Database Lookup")
def order_database_lookup(order_id: str) -> str:
    """Searches CSV order database by Order ID (e.g. ORD-2026-1001)."""
    orders_df = load_orders_db()
    clean_id = order_id.strip().upper()
    
    # Check for column name format ('order id' or 'Order ID')
    col_map = {col.lower(): col for col in orders_df.columns}
    id_col = col_map.get("order id", "Order ID")
    
    matched = orders_df[orders_df[id_col].astype(str).str.upper() == clean_id]
    
    if matched.empty:
        return f"No record found for Order ID: {clean_id}"
    
    rec = matched.iloc[0]
    
    # Flexible formatting matching your CSV headers
    return "\n".join([f"{col.title()}: {val}" for col, val in rec.items()])

@tool("Escalate to Human Agent")
def escalate_to_human(issue_summary: str) -> str:
    """Escalates complex queries or explicit user requests to a human support agent queue."""
    add_escalation_ticket(issue_summary)
    return "ESCALATED: Your request has been forwarded to a human representative under pending tickets."