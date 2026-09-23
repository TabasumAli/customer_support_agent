"""
Build the FAISS knowledge base index from PDFs in docs/.

Run this whenever you add, remove, or edit files in docs/:
    python build_index.py

Outputs:
    data/index.faiss
    data/chunks.json
"""

import json
import re
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DOCS_DIR = Path("docs")
DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "index.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.json"

CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 50      # characters of overlap between consecutive chunks
EMBED_MODEL = "all-MiniLM-L6-v2"


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, cleaned_text), ...] for a PDF."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # Collapse whitespace/newlines so chunking is stable
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Sliding-window character chunking."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    if not DOCS_DIR.exists():
        raise SystemExit(f"Missing {DOCS_DIR}/ — nothing to index.")

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDFs found in {DOCS_DIR}/.")

    all_chunks: list[dict] = []

    for pdf in pdf_files:
        print(f"📄 Indexing {pdf.name} ...")
        for page_num, page_text in extract_pages(pdf):
            for chunk in chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP):
                chunk = chunk.strip()
                if len(chunk) < 40:          # skip tiny/empty fragments
                    continue
                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source_filename": pdf.name,
                        "page_number": page_num,
                    },
                })

    if not all_chunks:
        raise SystemExit("No text extracted — check that the PDFs contain selectable text (not scans).")

    print(f"✅ Total chunks: {len(all_chunks)}")

    print(f"🔤 Loading embedder: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("🧠 Encoding chunks ...")
    vectors = embedder.encode(
        [c["text"] for c in all_chunks],
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"💾 Wrote {INDEX_PATH}")
    print(f"💾 Wrote {CHUNKS_PATH}")


if __name__ == "__main__":
    main()