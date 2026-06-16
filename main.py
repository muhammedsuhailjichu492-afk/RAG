"""
RAG Knowledge Base Search Engine — FastAPI Backend
Uses Ollama for embeddings + LLM, ChromaDB for vector storage
"""

import os
import uuid
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import chromadb
from chromadb.config import Settings
import ollama
import PyPDF2
import docx

# ── Config ──────────────────────────────────────────────────────────────────
EMBED_MODEL   = os.getenv("EMBED_MODEL",   "nomic-embed-text")
LLM_MODEL     = os.getenv("LLM_MODEL",     "llama3.2")
CHROMA_PATH   = os.getenv("CHROMA_PATH",   "./chroma_db")
DOCS_PATH     = os.getenv("DOCS_PATH",     "./docs_store")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K         = int(os.getenv("TOP_K",         "5"))

Path(DOCS_PATH).mkdir(parents=True, exist_ok=True)

# ── ChromaDB ─────────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Knowledge Base", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K
    use_llm: bool = True


class QueryResult(BaseModel):
    answer: str
    sources: List[dict]
    chunks_used: int


class DeleteRequest(BaseModel):
    doc_id: str


# ── Text utilities ────────────────────────────────────────────────────────────
def extract_text(file_path: str, ext: str) -> str:
    ext = ext.lower()
    if ext == ".pdf":
        text = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
        return "\n".join(text)
    elif ext in (".docx", ".doc"):
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext in (".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".html"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def get_embedding(text: str) -> List[float]:
    resp = ollama.embed(model=EMBED_MODEL, input=text)
    return resp["embeddings"][0]


def doc_id_from_filename(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()[:12]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        ollama.list()
        ollama_ok = True
    except Exception:
        ollama_ok = False
    return {
        "status": "ok",
        "ollama": ollama_ok,
        "embed_model": EMBED_MODEL,
        "llm_model": LLM_MODEL,
        "chunks_indexed": collection.count(),
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix
    allowed = {".pdf", ".txt", ".md", ".docx", ".py", ".js", ".ts", ".json", ".csv", ".html"}
    if ext.lower() not in allowed:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}")

    save_path = os.path.join(DOCS_PATH, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = extract_text(save_path, ext)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(422, f"Failed to extract text: {e}")

    if not text.strip():
        raise HTTPException(422, "Document appears to be empty or unreadable.")

    chunks = chunk_text(text)
    doc_id = doc_id_from_filename(file.filename)

    # Remove existing chunks for this doc (re-upload)
    existing = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    embeddings, ids, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        embeddings.append(emb)
        ids.append(f"{doc_id}_chunk_{i}")
        documents.append(chunk)
        metadatas.append({
            "doc_id": doc_id,
            "filename": file.filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
        })

    collection.add(embeddings=embeddings, ids=ids, documents=documents, metadatas=metadatas)

    return {
        "message": "Document indexed successfully",
        "filename": file.filename,
        "doc_id": doc_id,
        "chunks": len(chunks),
        "characters": len(text),
    }


@app.get("/documents")
def list_documents():
    if collection.count() == 0:
        return {"documents": []}

    all_items = collection.get(include=["metadatas"])
    seen, docs = set(), []
    for meta in all_items["metadatas"]:
        did = meta["doc_id"]
        if did not in seen:
            seen.add(did)
            docs.append({
                "doc_id": did,
                "filename": meta["filename"],
                "total_chunks": meta["total_chunks"],
            })
    return {"documents": docs}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    existing = collection.get(where={"doc_id": doc_id})
    if not existing["ids"]:
        raise HTTPException(404, "Document not found.")
    collection.delete(ids=existing["ids"])
    # Remove file from disk
    docs = list_documents()["documents"]
    for d in docs:
        if d["doc_id"] == doc_id:
            fp = os.path.join(DOCS_PATH, d["filename"])
            if os.path.exists(fp):
                os.remove(fp)
            break
    return {"message": f"Document {doc_id} deleted.", "chunks_removed": len(existing["ids"])}


@app.post("/query", response_model=QueryResult)
def query_knowledge_base(req: QueryRequest):
    if collection.count() == 0:
        raise HTTPException(400, "Knowledge base is empty. Upload documents first.")

    query_emb = get_embedding(req.query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(req.top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks   = results["documents"][0]
    metas    = results["metadatas"][0]
    distances = results["distances"][0]

    sources = [
        {
            "filename": m["filename"],
            "chunk_index": m["chunk_index"],
            "relevance": round(1 - d, 4),
            "snippet": c[:200] + ("…" if len(c) > 200 else ""),
        }
        for c, m, d in zip(chunks, metas, distances)
    ]

    if not req.use_llm:
        return QueryResult(
            answer="(LLM disabled — showing retrieved chunks only)",
            sources=sources,
            chunks_used=len(chunks),
        )

    context = "\n\n---\n\n".join(
        f"[Source: {m['filename']}, chunk {m['chunk_index']+1}/{m['total_chunks']}]\n{c}"
        for c, m in zip(chunks, metas)
    )

    prompt = f"""You are a knowledgeable assistant. Answer the user's question using the context below.
Provide a detailed, well-structured answer with explanations. Use bullet points, examples, and elaboration where helpful.
Be thorough — do not give one-line answers. If something is only partially covered in the context, expand on it using your own knowledge.
If the topic is completely absent from the context, say "I couldn't find relevant information in the knowledge base."

CONTEXT:
{context}

QUESTION: {req.query}

ANSWER:"""

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": -1},
    )
    answer = resp["message"]["content"].strip()

    return QueryResult(answer=answer, sources=sources, chunks_used=len(chunks))


# ── Summarize all docs ────────────────────────────────────────────────────────
@app.get("/summarize")
def summarize_knowledge_base():
    if collection.count() == 0:
        raise HTTPException(400, "Knowledge base is empty. Upload documents first.")

    all_items = collection.get(include=["documents", "metadatas"])
    full_text = "\n\n".join(all_items["documents"][:20])

    prompt = f"""You are a knowledgeable assistant. Read the following content from a knowledge base and provide a comprehensive summary.
Include all key topics, main points, important details, and any conclusions. Structure the summary with clear sections and bullet points.

CONTENT:
{full_text}

SUMMARY:"""

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": -1},
    )
    return {"summary": resp["message"]["content"].strip()}


# ── Direct text query (no file needed) ───────────────────────────────────────
class TextQueryRequest(BaseModel):
    text: str
    question: str


@app.post("/query/text")
def query_from_text(req: TextQueryRequest):
    prompt = f"""You are a knowledgeable assistant. Answer the user's question based on the text provided below.
Provide a detailed, well-structured answer with explanations, bullet points, and examples where helpful.
Be thorough and comprehensive in your response.

TEXT:
{req.text}

QUESTION: {req.question}

ANSWER:"""

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": -1},
    )
    return {"answer": resp["message"]["content"].strip()}


# ── Debug error endpoint ──────────────────────────────────────────────────────
class DebugRequest(BaseModel):
    code: Optional[str] = ""
    error: str


@app.post("/debug")
def debug_error(req: DebugRequest):
    code_section = f"\nCODE:\n{req.code}\n" if req.code.strip() else ""
    prompt = f"""You are an expert Python debugger and software engineer.
A user has encountered an error. Analyze it carefully and provide:

1. **Error Explanation** — What the error means in plain English
2. **Root Cause** — Why it happened
3. **Fix** — The exact corrected code with explanation
4. **Prevention** — How to avoid this error in the future

Be detailed, clear, and provide working code examples.
{code_section}
ERROR:
{req.error}

SOLUTION:"""

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": -1},
    )
    return {"solution": resp["message"]["content"].strip()}


# ── Serve frontend ────────────────────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    def serve_ui():
        return FileResponse(str(frontend_dir / "index.html"))