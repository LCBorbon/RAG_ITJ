from dotenv import load_dotenv
load_dotenv()
 
import os
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from fastapi import FastAPI
from pydantic import BaseModel
import argparse
 
# 1. PDF TEXT EXTRACTION
def extract_text_from_pdf(pdf_path):
 
    reader = PdfReader(pdf_path)
 
    pages = []
 
    for i, page in enumerate(reader.pages):
 
        text = page.extract_text() or ""
 
        if text.strip():
 
            pages.append(
                f"[Page {i+1}]\n{text}"
            )
 
    return "\n\n".join(pages)
 
 

# 2. TEXT CHUNKING
def recursive_split(text, chunk_size=800, overlap=150):
 
    separators = ["\n\n", "\n", ". ", " ", ""]
 
    def split(text, sep):
 
        parts = text.split(sep) if sep else list(text)
 
        chunks = []
        current = ""
 
        for part in parts:
 
            candidate = current + (sep if current else "") + part
 
            if len(candidate) <= chunk_size:
 
                current = candidate
 
            else:
 
                if current:
                    chunks.append(current)
 
                current = part
 
        if current:
            chunks.append(current)
 
        return chunks
 
    raw = split(text, separators[0])
 
    if not raw:
        return []
 
    chunks = [raw[0]]
 
    for i in range(1, len(raw)):
 
        chunks.append(
            raw[i-1][-overlap:] + "\n" + raw[i]
        )
 
    return chunks
 
 

# 3. LOAD PDF FILES
def load_pdfs(docs_dir):
 
    pdf_paths = []
 
    for p in Path(docs_dir).glob("*.pdf"):
 
        if p.name.startswith("._"):
            continue
 
        pdf_paths.append(p)
 
    return pdf_paths
 
 

# 4. RAG PIPELINE
SYSTEM = """
You are a document QA assistant.
 
Rules:
- Answer ONLY using the provided context chunks.
- If the answer is not present say:
'I could not find an answer in the documents.'
- Cite the document and page.
- Do not use outside knowledge.
"""
 
class RAGPipeline:
 
    def __init__(self):
 
        print("Loading embedding model...")
 
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
 
        print("Connecting to ChromaDB...")
 
        self.client = chromadb.PersistentClient(path="./chroma_db")
 
        self.collection = self.client.get_or_create_collection(
            name="rag_docs",
            metadata={"hnsw:space": "cosine"},
        )
 
        print("Connecting to LLM (Gemini)...")
 
        self.llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
 
 

# INGEST DOCUMENTS
    def ingest(self, docs_dir):
 
        pdf_paths = load_pdfs(docs_dir)
 
        print(f"Found {len(pdf_paths)} PDFs")
 
        total_chunks = 0
 
        for pdf_path in pdf_paths:
 
            print("Processing:", pdf_path.name)
 
            text = extract_text_from_pdf(pdf_path)
 
            chunks = recursive_split(text)
 
            embeddings = self.embedder.encode(chunks).tolist()
 
            ids = [
                f"{pdf_path.name}::{i}"
                for i in range(len(chunks))
            ]
 
            metadata = [
                {
                    "source": pdf_path.name,
                    "chunk_index": i
                }
                for i in range(len(chunks))
            ]
 
            self.collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadata,
            )
 
            total_chunks += len(chunks)
 
            print("Stored", len(chunks), "chunks")
 
        print("Total chunks stored:", total_chunks)
 
 

# RETRIEVAL
    def retrieve(self, question, top_k=5):
 
        q_vector = self.embedder.encode([question]).tolist()
 
        results = self.collection.query(
            query_embeddings=q_vector,
            n_results=top_k,
            include=["documents", "metadatas"]
        )
 
        chunks = []
 
        for text, meta in zip(
            results["documents"][0],
            results["metadatas"][0]
        ):
 
            chunks.append({
                "text": text,
                "source": meta["source"]
            })
 
        return chunks
 
 

# GENERATION
    def generate(self, question, chunks):
 
        context = "\n\n".join([
            f"{c['source']}\n{c['text']}"
            for c in chunks
        ])
 
        prompt = f"""
Context:
 
{context}
 
Question:
{question}
 
Answer strictly from the context above.
"""
 
        response = self.llm.models.generate_content(
            model="gemini-2.5-flash",
            contents=SYSTEM + "\n\n" + prompt,
        )
 
        return response.text
 
 

# ASK QUESTION
    def ask(self, question, top_k=5):
 
        chunks = self.retrieve(question, top_k)
 
        answer = self.generate(question, chunks)
 
        return answer
 
 

# FASTAPI SERVER
pipeline = None
 
app = FastAPI()
 
class AskRequest(BaseModel):
    question: str
    top_k: int = 5
 
 
@app.on_event("startup")
def startup():
 
    global pipeline
 
    pipeline = RAGPipeline()
 
 
@app.post("/ask")
def ask(req: AskRequest):
 
    answer = pipeline.ask(req.question, req.top_k)
 
    return {
        "answer": answer
    }
 
 

# CLI INTERFACE
def run_cli():
 
    pipeline = RAGPipeline()
 
    while True:
 
        q = input("Your question: ").strip()
 
        if q.lower() in ("exit", "quit"):
            break
 
        result = pipeline.ask(q)
 
        print("\nAnswer:\n", result)
 
 

# MAIN
def main():
 
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--cli", action="store_true")
    parser.add_argument(
        "--docs_dir",
        default=r"C:\Users\lcbor\Downloads\rag_dataset\arxiv_pdfs"
    )
 
    args = parser.parse_args()
 
    pipeline = RAGPipeline()
 
    if args.ingest:
 
        pipeline.ingest(args.docs_dir)
 
    elif args.cli:
 
        run_cli()
 
 
if __name__ == "__main__":
 
    main()