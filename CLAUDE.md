# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

Always use `uv` to run Python commands and manage dependencies — never use `pip` or `python` directly.

```bash
# Install dependencies
uv sync

# Start the server (from project root)
./run.sh

# Or manually from the backend directory
cd backend && uv run uvicorn app:app --reload --port 8000
```

The server runs at `http://localhost:8000`. The frontend is served as static files by FastAPI — there is no separate frontend build step.

Requires a `.env` file in the project root (copy from `.env.example`):
```
ANTHROPIC_API_KEY=your-key-here
```

## Architecture

This is a full-stack RAG chatbot. The backend is a single FastAPI process that serves both the API and the frontend static files.

### Request flow for a user query

1. **Frontend** (`frontend/script.js`) — POSTs `{ query, session_id }` to `/api/query`
2. **`app.py`** — receives the request, creates a session if needed, delegates to `RAGSystem`
3. **`rag_system.py`** — fetches conversation history, calls `AIGenerator`
4. **`ai_generator.py`** — makes a first Claude API call with the `search_course_content` tool available (`tool_choice: auto`)
5. **`search_tools.py` + `vector_store.py`** — if Claude decides to search, `CourseSearchTool` queries ChromaDB and returns formatted chunks
6. **`ai_generator.py`** — makes a second Claude API call with the search results in context to synthesize the final answer
7. Response flows back through `RAGSystem` (which saves the exchange to session history) → `app.py` → browser

### Key design decisions

- **Two ChromaDB collections**: `course_catalog` stores one entry per course (title, instructor, link, lessons as JSON); `course_content` stores individual text chunks with `course_title` and `lesson_number` metadata for filtering.
- **Semantic course resolution**: when a search specifies a `course_name`, it is fuzzy-matched against `course_catalog` via embedding search before filtering `course_content`. This tolerates partial or approximate course names.
- **Tool-use pattern**: Claude drives the retrieval decision. `AIGenerator` handles the two-turn agentic loop (initial call → tool execution → final synthesis call) internally.
- **Session history**: stored in-memory in `SessionManager` (not persisted across server restarts). History is passed as plain text appended to the system prompt, not as structured messages.
- **Idempotent document loading**: on startup, already-indexed course titles are checked so re-starting the server does not re-index documents.

### Course document format

Files in `docs/` must follow this structure for the parser in `document_processor.py` to extract metadata correctly:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>
Lesson 0: <title>
Lesson Link: <url>
<lesson content...>
Lesson 1: <title>
...
```

The course `title` is used as the primary key in ChromaDB — duplicate titles are skipped on load.

### Configuration

All tunable parameters live in `backend/config.py`:
- `ANTHROPIC_MODEL` — Claude model used for generation
- `EMBEDDING_MODEL` — sentence-transformers model for ChromaDB (`all-MiniLM-L6-v2`)
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — text chunking parameters (800 / 100 chars)
- `MAX_RESULTS` — number of chunks returned per search (5)
- `MAX_HISTORY` — conversation exchanges retained per session (2)
- `CHROMA_PATH` — ChromaDB persistence directory (`backend/chroma_db`)
