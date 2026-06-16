# RAG Knowledge Base Search Engine

A local Retrieval-Augmented Generation (RAG) system with a VS Code-inspired UI. Index your documents and query them with natural language using Ollama for embeddings and LLM inference — fully offline.

## How It Works

1. **Upload** — PDF, DOCX, TXT, MD, PY, JS, JSON, CSV, or HTML files are uploaded to the FastAPI backend.
2. **Chunk & Embed** — Each document is split into overlapping text chunks. Ollama's `nomic-embed-text` model generates vector embeddings for each chunk.
3. **Index** — Embeddings are stored in ChromaDB (persistent local vector database) with metadata like filename and chunk index.
4. **Query** — Your question is embedded the same way, and cosine similarity search finds the top-K most relevant chunks.
5. **Answer** — Retrieved chunks are passed as context to Ollama's LLM (`llama3.2` by default), which generates a grounded answer.

## Prerequisites

- **Ollama** installed and running: https://ollama.com
- **Python 3.10+**

Pull the required models:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Setup

```bash
cd backend
pip install -r requirements.txt
```

## Run

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in your browser, or visit `http://localhost:8000` (the backend also serves the frontend).

## VS Code Integration

Add this to `.vscode/tasks.json` to launch with a task:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Start RAG Server",
      "type": "shell",
      "command": "uvicorn main:app --reload --port 8000",
      "options": { "cwd": "${workspaceFolder}/backend" },
      "group": "build",
      "presentation": { "reveal": "always", "panel": "new" }
    }
  ]
}
```

Run it via `Terminal → Run Task → Start RAG Server`.

## Configuration

| Env Variable   | Default              | Description                     |
|----------------|----------------------|---------------------------------|
| `EMBED_MODEL`  | `nomic-embed-text`   | Ollama embedding model          |
| `LLM_MODEL`    | `llama3.2`           | Ollama chat model               |
| `CHUNK_SIZE`   | `500`                | Words per chunk                 |
| `CHUNK_OVERLAP`| `50`                 | Overlap between chunks          |
| `TOP_K`        | `5`                  | Number of chunks retrieved      |
| `CHROMA_PATH`  | `./chroma_db`        | ChromaDB persistence directory  |

## Supported File Types

`.pdf` · `.docx` · `.txt` · `.md` · `.py` · `.js` · `.ts` · `.json` · `.csv` · `.html`

## API Endpoints

| Method | Endpoint              | Description                     |
|--------|-----------------------|---------------------------------|
| GET    | `/health`             | Backend + Ollama status         |
| POST   | `/upload`             | Upload and index a file         |
| GET    | `/documents`          | List all indexed documents      |
| DELETE | `/documents/{doc_id}` | Remove a document from index    |
| POST   | `/query`              | Query the knowledge base        |
