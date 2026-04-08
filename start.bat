@echo off
echo.
echo  ╔═══════════════════════════════════════╗
echo  ║     KnowledgeAI RAG Assistant         ║
echo  ║     Starting backend server...        ║
echo  ╚═══════════════════════════════════════╝
echo.

cd /d "%~dp0"

IF NOT EXIST "env\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\env\
    echo Please create it with: python -m venv env
    echo Then install deps: env\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting FastAPI backend on http://localhost:8000 ...
echo Frontend available at: http://localhost:8000
echo API Docs at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop.
echo.

env\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
