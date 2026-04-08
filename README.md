# KnowledgeAI: Premium RAG Chat Assistant (2026 Edition)

KnowledgeAI is a state-of-the-art Retrieval-Augmented Generation (RAG) application that transforms your static documents into an interactive, multi-session chat experience. Built with a premium dark-mode aesthetic and powered by Google's latest **Gemini 2.5 Flash** model.

![UI Sneak Peek](https://img.shields.io/badge/UI-Modern_Glassmorphism-blueviolet)
![Engine](https://img.shields.io/badge/Engine-RAG_Pipeline-green)
![Version](https://img.shields.io/badge/Version-1.5.0_Beta-orange)

## 🚀 Key Features

- **Multi-Session Intelligence**: Persistent chat history stored in SQLite. Create, resume, and manage multiple discussion threads seamlessly.
- **RAG Engine**: Advanced document retrieval using **ChromaDB** and HuggingFace embeddings (`all-mpnet-base-v2`).
- **Premium UI/UX**: A stunning dark-mode interface with glassmorphism effects, smooth animations, and responsive sidebars.
- **Auto-Naming Threads**: Automatically generates session titles based on your first query.
- **Source Referencing**: Every answer includes clickable references to the original document chunks.
- **Monitoring**: Real-time database status and chunk count tracking.

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **AI Model**: Google Gemini 2.5 Flash (via Direct REST Integration)
- **Vector Store**: ChromaDB
- **Database**: SQLite (Chat History)
- **Frontend**: Vanilla JavaScript, Semantic HTML5, Custom CSS3
- **Styling**: Google Fonts (Inter), Material Icons

## 📂 Project Structure

```text
├── backend/
│   ├── app.py           # Main FastAPI server & RAG logic
│   └── data/            # Source documents (PDF, MD, etc.)
├── frontend/
│   └── index.html       # Single-page premium dashboard
├── chroma/              # Vector database storage
├── chat_history.db      # SQLite persistent storage
├── .env                 # API Keys & Secrets
└── requirements.txt     # Backend dependencies
```

## ⚙️ Installation & Setup

1. **Clone the repository**:
   cd langchain-rag-tutorial-main

2. **Set up Virtual Environment**:
   ```bash
   python -m venv env
   .\env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_key_here
   ```

5. **Initialize Database**:
   Place your documents in `data/books/` and run:
   ```bash
   python populate_database.py --reset
   ```

6. **Launch the Server**:
   ```bash
   .\env\Scripts\python -m uvicorn backend.app:app --reload
   ```
   Open your browser at `http://localhost:8000/app/`.

---
*Created by Antigravity AI - Designed for Knowledge Seekers.*
