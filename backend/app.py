import os
import shutil
import uuid
import tempfile
from pathlib import Path
from typing import Optional

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

# ── shared embedding function (loaded once) ──────────────────────────────────
print("⏳  Loading embedding model …")
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
print("✅  Embedding model ready.")


# ── helpers ──────────────────────────────────────────────────────────────────
def get_db() -> Chroma:
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


def db_exists() -> bool:
    return os.path.exists(CHROMA_PATH) and any(
        f.endswith(".bin") or f.endswith(".parquet") or f.endswith(".sqlite3")
        for _, _, files in os.walk(CHROMA_PATH)
        for f in files
    )


# ── pydantic models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    context_chunks: list[str]
    relevance_scores: list[float]


class StatusResponse(BaseModel):
    db_ready: bool
    chunk_count: int
    data_files: list[str]


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


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest):
    """Run a RAG query against the Chroma vector store."""
    if not db_exists():
        raise HTTPException(
            status_code=503,
            detail="Database not initialised. Upload documents and call /rebuild first."
        )

    db = get_db()
    results = db.similarity_search_with_relevance_scores(req.question, k=req.k)

    if not results:
        raise HTTPException(status_code=404, detail="No matching results found.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=req.question)

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    response = model.invoke(prompt)

    sources  = [doc.metadata.get("source", "unknown") for doc, _ in results]
    scores   = [round(float(score), 4) for _, score in results]
    chunks   = [doc.page_content for doc, _ in results]

    return QueryResponse(
        answer=response.content,
        sources=sources,
        context_chunks=chunks,
        relevance_scores=scores,
    )


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
        return {"message": "Database deleted successfully."}
    return {"message": "No database found to delete."}
