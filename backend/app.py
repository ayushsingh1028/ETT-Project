import os # Reloading...
import shutil
import uuid
import tempfile
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent          # langchain-rag-tutorial-main/
CHROMA_PATH  = str(BASE_DIR / "chroma")
DATA_PATH    = str(BASE_DIR / "data" / "books")
FRONTEND_DIR = BASE_DIR / "frontend"
DB_PATH      = str(BASE_DIR / "chat_history.db")

# ── database setup ───────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Sessions table: stores individual chat threads
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id TEXT PRIMARY KEY,
                  title TEXT,
                  created_at DATETIME)''')
    
    # Messages table: linked to a session
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT,
                  role TEXT,
                  content TEXT,
                  sources TEXT,
                  timestamp DATETIME,
                  FOREIGN KEY(session_id) REFERENCES sessions(id))''')
    conn.commit()
    conn.close()

init_db()

# ── prompt ───────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# ── app ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Knowledge Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend static files
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# ── shared embedding function & DB (loaded once) ─────────────────────────────
print("⏳  Loading embedding model & database …")
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Open Chroma once
_vector_db: Optional[Chroma] = None

def get_db() -> Chroma:
    global _vector_db
    if _vector_db is None:
        _vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return _vector_db

def db_exists() -> bool:
    return os.path.exists(CHROMA_PATH) and any(
        f.endswith(".bin") or f.endswith(".parquet") or f.endswith(".sqlite3")
        for _, _, files in os.walk(CHROMA_PATH)
        for f in files
    )

# Pre-load DB if it exists
if db_exists():
    get_db()
    print("✅  Database & Embedding model ready (2026 Edition).")
else:
    print("⚠️   Database folder empty. Rebuild required.")



# ── pydantic models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    session_id: str  # Critical for session-based chat
    k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    context_chunks: List[str]
    relevance_scores: List[float]


class StatusResponse(BaseModel):
    db_ready: bool
    chunk_count: int
    data_files: list[str]


class HistoryItem(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    sources: Optional[List[str]] = None
    timestamp: str

class ChatSession(BaseModel):
    id: str
    title: str
    created_at: str


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    """Redirect root to the frontend app."""
    frontend_index = FRONTEND_DIR / "index.html"
    if frontend_index.exists():
        return FileResponse(str(frontend_index))
    return {"status": "ok", "message": "RAG Knowledge Assistant API is running 🚀"}


@app.get("/status", response_model=StatusResponse, tags=["database"])
def status():
    """Check whether the Chroma DB is initialized and how many chunks it has."""
    data_files: list[str] = []
    if os.path.exists(DATA_PATH):
        data_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".md")]

    if not db_exists():
        return StatusResponse(db_ready=False, chunk_count=0, data_files=data_files)

    try:
        db = get_db()
        count = db._collection.count()
        return StatusResponse(db_ready=True, chunk_count=count, data_files=data_files)
    except Exception as e:
        return StatusResponse(db_ready=False, chunk_count=0, data_files=data_files)


@app.post("/query", response_model=QueryResponse, tags=["chat"])
def query(req: QueryRequest):
    """Query the RAG system and store the interaction with deep logging."""
    print(f"\n[QUERY] Session: {req.session_id} | Question: {req.question}")
    
    try:
        # STEP 1: Search
        print("DEBUG: 1. Searching Chroma DB...")
        db = get_db()
        results = db.similarity_search_with_relevance_scores(req.question, k=req.k)
        print(f"DEBUG: 1. Found {len(results)} results.")
        
        if len(results) == 0:
            return QueryResponse(answer="No relevant context found.", sources=[], context_chunks=[], relevance_scores=[])

        chunks = [doc.page_content for doc, _ in results]
        scores = [float(score) for _, score in results]
        sources = [doc.metadata.get("source", "Unknown") for doc, _ in results]

        # STEP 2: Prompt
        print("DEBUG: 2. Preparing Prompt...")
        context_text = "\n\n---\n\n".join(chunks)
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        final_prompt = prompt.format(context=context_text, question=req.question)

        # STEP 3: Model call using Direct REST API (Targeting Gemini 2.5 Flash)
        print(f"DEBUG: 3. Calling AI Model (Direct REST v1 - Gemini 2.5 Flash)...")
        try:
            import requests
            api_key = os.getenv("GOOGLE_API_KEY")
            # Using the exact name found in models_list.json: gemini-2.5-flash
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": final_prompt}]
                }]
            }
            headers = {'Content-Type': 'application/json'}
            
            response = requests.post(url, json=payload, headers=headers)
            res_json = response.json()
            
            if response.status_code == 200:
                answer = res_json['candidates'][0]['content']['parts'][0]['text']
                print(f"      ✅ Direct REST Success with gemini-2.5-flash")
            else:
                raise Exception(f"API Error {response.status_code}: {res_json.get('error', {}).get('message', 'Unknown error')}")
                    
        except Exception as rest_err:
            print(f"      ❌ Direct REST Failed: {rest_err}")
            return QueryResponse(answer=f"AI Error (Direct REST failed): {rest_err}", sources=sources, context_chunks=chunks, relevance_scores=scores)





        # STEP 4: DB Write
        print("DEBUG: 4. Saving interaction to SQLite...")
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                      (req.session_id, "user", req.question, datetime.now().isoformat()))
            c.execute("INSERT INTO messages (session_id, role, content, sources, timestamp) VALUES (?, ?, ?, ?, ?)",
                      (req.session_id, "ai", answer, json.dumps(sources), datetime.now().isoformat()))
            
            # Title update
            c.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (req.session_id,))
            if c.fetchone()[0] <= 2:
                title = req.question[:30] + "..." if len(req.question) > 30 else req.question
                c.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, req.session_id))
            conn.commit()
            conn.close()
        except Exception as db_err:
            print(f"ERROR: Database write failed: {db_err}")
            # we still return the answer even if DB fails
            
        return QueryResponse(
            answer=answer,
            sources=sources,
            context_chunks=chunks,
            relevance_scores=scores,
        )

    except Exception as e:
        print(f"CRITICAL ERROR in /query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions", response_model=List[ChatSession], tags=["chat"])
def get_sessions():
    """List all chat sessions."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM sessions ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [ChatSession(id=r['id'], title=r['title'], created_at=r['created_at']) for r in rows]


@app.post("/sessions", response_model=ChatSession, tags=["chat"])
def create_session():
    """Create a new chat session."""
    s_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
              (s_id, "New Chat", now))
    conn.commit()
    conn.close()
    return ChatSession(id=s_id, title="New Chat", created_at=now)


@app.get("/sessions/{session_id}/messages", response_model=List[HistoryItem], tags=["chat"])
def get_session_messages(session_id: str):
    """Fetch all messages for a specific session."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    
    msgs = []
    for r in rows:
        msgs.append(HistoryItem(
            id=r['id'],
            session_id=r['session_id'],
            role=r['role'],
            content=r['content'],
            sources=json.loads(r['sources']) if r['sources'] else None,
            timestamp=r['timestamp']
        ))
    return msgs


@app.delete("/history", tags=["chat"])
def clear_history():
    """Wipe all sessions and messages."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM messages")
        c.execute("DELETE FROM sessions")
        conn.commit()
        conn.close()
        return {"message": "All history cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", tags=["database"])
async def upload_document(file: UploadFile = File(...)):
    """Upload a .md document to the data/books folder."""
    if not file.filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are supported.")

    os.makedirs(DATA_PATH, exist_ok=True)
    dest = os.path.join(DATA_PATH, file.filename)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    return {"message": f"File '{file.filename}' uploaded successfully.", "path": dest}


@app.post("/rebuild", tags=["database"])
def rebuild_database():
    """Re-index all .md files in data/books into Chroma."""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="data/books folder not found.")

    md_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".md")]
    if not md_files:
        raise HTTPException(status_code=404, detail="No .md files found in data/books.")

    documents: list[Document] = []
    for filename in md_files:
        path = os.path.join(DATA_PATH, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append(Document(page_content=text, metadata={"source": filename}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True
    )
    chunks = splitter.split_documents(documents)

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_PATH)
    db.persist()

    global _vector_db
    _vector_db = db

    return {
        "message": "Database rebuilt successfully.",
        "files_indexed": len(md_files),
        "chunks_created": len(chunks),
    }


@app.delete("/database", tags=["database"])
def delete_database():
    """Wipe the Chroma vector store."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        global _vector_db
        _vector_db = None
        return {"message": "Database deleted successfully."}
    return {"message": "No database found to delete."}
