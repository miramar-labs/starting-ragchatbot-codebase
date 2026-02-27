# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (from repo root)
./run.sh

# Run the server manually (must cd into backend first — imports are relative)
cd backend && uvicorn app:app --reload --port 8000
```

There are no tests. `main.py` at the root is a stub and is not used by the application.

## Environment

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. The `.env` file must be in the repo root; `config.py` calls `load_dotenv()` which resolves relative to the working directory at startup.

## Architecture

The app is a FastAPI backend (`backend/app.py`) that serves the frontend as static files and exposes two API endpoints: `POST /api/query` and `GET /api/courses`.

**All backend imports are relative** — the server must be started from inside the `backend/` directory (as `run.sh` does). Running `uvicorn` from the repo root will fail with import errors.

### RAG query flow

```
POST /api/query
  └── RAGSystem.query()
        ├── AIGenerator.generate_response()   ← first Claude API call (tool_choice: auto)
        │     └── if stop_reason == "tool_use"
        │           └── ToolManager.execute_tool("search_course_content")
        │                 └── VectorStore.search()   ← ChromaDB semantic search
        │           └── AIGenerator._handle_tool_execution()  ← second Claude API call (no tools)
        └── SessionManager.add_exchange()     ← stores last 2 turns in memory (not persisted)
```

Claude always receives `tools` but decides whether to call `search_course_content`. General knowledge questions are answered without a search; course-specific questions trigger one search maximum.

### VectorStore (ChromaDB)

Two collections, both using `all-MiniLM-L6-v2` embeddings:
- `course_catalog` — one document per course (title, instructor, link, lessons JSON). Used only to resolve a fuzzy course name to its exact title.
- `course_content` — one document per `CourseChunk`. Filtered by `course_title` and/or `lesson_number` metadata at query time.

ChromaDB is persisted to `backend/chroma_db/`. On startup, `app.py` loads all `.txt/.pdf/.docx` files from `../docs/` (relative to `backend/`), skipping courses whose title already exists in the catalog.

### Document processing

`DocumentProcessor.process_course_document()` expects this file format:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <title>
Lesson Link: <url>
...content...

Lesson 2: <title>
...
```

The lesson loop uses a **flush-on-next-marker** pattern: content is accumulated line by line and flushed into chunks only when the next `Lesson N:` header is encountered. The final lesson is flushed in a separate post-loop block. `chunk_text()` splits on sentence boundaries and applies a sliding-window overlap so adjacent chunks share boundary sentences.

### Extending tools

New tools implement the `Tool` ABC in `search_tools.py` (requires `get_tool_definition()` returning an Anthropic tool schema, and `execute(**kwargs)`), then are registered via `ToolManager.register_tool()` in `RAGSystem.__init__()`. The `AIGenerator` passes all registered tool definitions to every Claude call automatically.

### Key config values (`backend/config.py`)

| Setting | Value |
|---|---|
| Model | `claude-sonnet-4-20250514` |
| Embedding model | `all-MiniLM-L6-v2` |
| Chunk size | 800 chars |
| Chunk overlap | 100 chars |
| Max search results | 5 |
| Conversation history | 2 turns (in-memory only, lost on restart) |
