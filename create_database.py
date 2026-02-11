from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

def main():
    print(">>> MAIN STARTED")
    generate_data_store()
    print(">>> MAIN FINISHED")

def generate_data_store():
    print(">>> LOADING DOCUMENTS")
    documents = load_documents()
    print(f">>> LOADED {len(documents)} DOCUMENTS")

    if len(documents) == 0:
        print("❌ NO DOCUMENTS FOUND — EXITING")
        return

    print(">>> SPLITTING TEXT")
    chunks = split_text(documents)
    print(f">>> CREATED {len(chunks)} CHUNKS")

    print(">>> SAVING TO CHROMA")
    save_to_chroma(chunks)

def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".md"):
            file_path = os.path.join(DATA_PATH, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(
                    Document(page_content=text, metadata={"source": filename})
                )
    return documents

def split_text(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
chunk_overlap=50,

        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks,
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),

        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f">>> SAVED {len(chunks)} CHUNKS TO {CHROMA_PATH}")

if __name__ == "__main__":
    load_dotenv()
    main()
