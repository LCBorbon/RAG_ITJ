# RAG Document Q&A System

*A document-grounded question answering system built with Python, ChromaDB, sentence-transformers, and Google Gemini.*

---

**Language:** Python 3.11+  
**LLM:** Google Gemini 2.5 Flash (free tier)  
**Vector DB:** ChromaDB (local, persistent)  
**Embeddings:** sentence-transformers / all-MiniLM-L6-v2  
**Interface:** FastAPI + CLI  

---

## Architecture

The system is split into two phases:

**Phase 1 — Ingestion (run once, offline)**

```
PDFs  ──►  Text Extraction  ──►  Chunking (800 chars / 150 overlap)
                                          │
                                Embedding (local, CPU)
                                          │
                                ChromaDB  (persistent on disk)
```

**Phase 2 — Query (real-time)**

```
User Question  ──►  Embed question  ──►  ChromaDB cosine search
                                                   │
                                           Top-5 chunks
                                                   │
                                      Build context prompt
                                                   │
                                      Gemini 2.5 Flash API
                                                   │
                                      Grounded answer + citations
```

### Component breakdown

- **PDF Extraction** — pypdf reads each page and prefixes it with `[Page N]` for citation tracking
- **Chunking** — recursive character splitter tries paragraph → sentence → word boundaries
- **Embeddings** — all-MiniLM-L6-v2 runs locally (no API cost), produces 384-dim vectors
- **Vector Store** — ChromaDB with cosine similarity, persisted to `./chroma_db` on disk
- **Retrieval** — top-5 most similar chunks are fetched per query
- **Generation** — Gemini 2.5 Flash receives a strict system prompt to answer only from context

---

## Setup Instructions

### 1. Prerequisites

- Python 3.11 or higher
- A Google Gemini API key — free at [aistudio.google.com](https://aistudio.google.com)

### 2. Install dependencies

```bash
pip install pypdf sentence-transformers chromadb google-genai fastapi uvicorn python-dotenv
```

### 3. Set your API key

Create a file named `.env` in the same folder as the script:

```
GEMINI_API_KEY=your-key-here
```

### 4. Add your PDF documents

Place your PDF files inside the `arxiv_pdfs/` folder (or pass a custom path with `--docs_dir`).

### 5. Ingest documents (run once)

```bash
python RAG_ITJ.py --ingest
```

This extracts text, chunks it, embeds locally, and stores everything in `./chroma_db`. Only needs to run once — or re-run when you add new PDFs.

### 6. Ask questions

**Option A — Interactive CLI:**

```bash
python RAG_ITJ.py --cli
```

**Option B — FastAPI server + JSON endpoint:**

```bash
uvicorn RAG_ITJ:app --reload

# Then call the API:
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is self-attention?", "top_k": 5}'
```

### Project structure

```
RAG_ITJ.py        <- main script (ingestion + retrieval + API + CLI)
.env              <- your Gemini API key (never commit this)
arxiv_pdfs/       <- put your PDF files here
chroma_db/        <- auto-created by --ingest (vector database)
requirements.txt  <- pip dependencies
```

---

## Key Technical Decisions & Trade-offs

| Component | Choice | Reason / Trade-off |
|-----------|--------|--------------------|
| **PDF extraction** | `pypdf` | Lightweight, no external dependencies. Limitation: scanned PDFs (images of text) return nothing — would need OCR fallback. |
| **Chunking** | Recursive splitter 800 chars / 150 overlap | Tries paragraph -> sentence -> word splits. Overlap prevents answers that span boundaries from being missed. Larger chunks = more context but less precise retrieval. |
| **Embeddings** | `all-MiniLM-L6-v2` (local) | Runs 100% offline, zero cost, fast on CPU. Trade-off: 384-dim vectors are slightly less accurate than OpenAI ada-002 but require no API key or internet. |
| **Vector DB** | ChromaDB (file-based) | Zero config, persistent across restarts, cosine similarity built-in. Trade-off: not horizontally scalable like Pinecone or Weaviate. |
| **Top-k retrieval** | k = 5 chunks | Balances context richness vs prompt size. Fewer chunks = faster + cheaper; more chunks = higher recall but risks diluting the context. |
| **LLM** | Gemini 2.5 Flash (free tier) | Free, capable, large context window. Trade-off vs Claude/GPT-4: slightly less instruction-following precision but sufficient for grounded QA. |
| **Anti-hallucination** | Strict system prompt | The system prompt explicitly forbids outside knowledge and requires citations. Simple but effective — no re-ranking or confidence scoring added yet. |
| **No LangChain** | Pure Python | Keeps dependencies minimal and every component transparent and swappable. Trade-off: more code written manually vs using high-level abstractions. |

---

## Known Limitations & Future Improvements

- **Scanned PDFs:** images-only PDFs produce no text. Add `pytesseract` for OCR fallback.
- **No conversation memory:** each question is independent. Add a message buffer for follow-up questions.
- **No re-ranking:** a cross-encoder re-ranker after retrieval would improve precision on ambiguous queries.
- **Streaming:** the API waits for the full answer. Adding SSE streaming would improve UX for long responses.
- **Single file:** the entire pipeline is in one script for simplicity. A production version would split into modules.

---

*Built as a take-home challenge. All answers are strictly grounded in the provided documents — the system will say "I could not find an answer" rather than hallucinate.*
