import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# 1. Define Policy Knowledge Base
kb_chunks = [
    {"title": "Return & Refund Policy", "text": "Customers can request a return within 30 days of delivery. Items must be unused and in original packaging. Refunds are processed within 5-7 business days back to the original payment method."},
    {"title": "Shipping & Delivery Policy", "text": "Standard shipping across Pakistan takes 3 to 5 business days. Express shipping is available for major cities (Lahore, Karachi, Islamabad) within 24-48 hours. Tracking details are emailed once shipped."},
    {"title": "Order Cancellation Policy", "text": "Orders can be cancelled at any time before they enter the 'Shipped' status. Once shipped, the customer must follow standard return policies after delivery."},
    {"title": "Warranty & Repairs", "text": "All tech accessories come with a 1-year limited warranty covering manufacturing defects. Physical damage, liquid contact, or unauthorized repairs void warranty eligibility."},
    {"title": "Payment Options", "text": "We accept Cash on Delivery (COD), Credit/Debit Cards (Visa, MasterCard), and local digital wallets including EasyPaisa and JazzCash."},
    {"title": "Human Escalation Policy", "text": "If an issue cannot be resolved via knowledge base or order tracking, or if the customer explicitly requests human support, the interaction must be escalated immediately with a ticket summary."}
]

# 2. Generate Vector Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [f"Category: {c['title']}\nContent: {c['text']}" for c in kb_chunks]
embeddings = model.encode(texts)

# 3. Save FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))
faiss.write_index(index, "data/index.faiss")

# 4. Save chunks.json and metadata.json
chunks_payload = []
metadata_payload = []

for idx, (chunk, text) in enumerate(zip(kb_chunks, texts)):
    chunk_id = f"chunk_{idx+1}"
    meta = {
        "chunk_id": chunk_id,
        "source_filename": "customer_support_policy_2026.pdf",
        "source_path": "/docs/customer_support_policy_2026.pdf",
        "page_number": (idx // 2) + 1,
        "chunk_number": idx + 1,
        "document_type": "PDF Guidelines",
        "text_length": len(text),
        "total_number_of_chunks": len(kb_chunks)
    }
    chunks_payload.append({"chunk_id": chunk_id, "text": text, "metadata": meta})
    metadata_payload.append(meta)

with open("data/chunks.json", "w") as f:
    json.dump(chunks_payload, f, indent=2)

with open("data/metadata.json", "w") as f:
    json.dump(metadata_payload, f, indent=2)

print("✅ Saved data/index.faiss, data/chunks.json, and data/metadata.json")