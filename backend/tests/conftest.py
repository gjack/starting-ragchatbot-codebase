"""
Shared pytest fixtures for the RAG chatbot test suite.

Provides:
  - mock_rag  : a fully-mocked RAGSystem (no ChromaDB / Anthropic API needed)
  - test_client: a Starlette TestClient backed by a minimal FastAPI app that
                 mirrors the API surface of app.py but never mounts static files
                 and never imports RAGSystem at module level.
"""
import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional


# ---------------------------------------------------------------------------
# Pydantic models — mirror of app.py so the test app has no dep on app.py
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    source_links: List[Optional[str]]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------

def make_test_app(rag) -> FastAPI:
    """
    Build a test FastAPI app wired to the given (mock) RAGSystem.

    Identical API surface to app.py:  POST /api/query,  GET /api/courses,
    DELETE /api/session/{session_id}.  Static file mounts and startup events
    are intentionally omitted so imports never touch the filesystem.
    """
    test_app = FastAPI(title="Test RAG App")

    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag.session_manager.create_session()
            answer, sources, source_links = rag.query(request.query, session_id)
            return QueryResponse(
                answer=answer,
                sources=sources,
                source_links=source_links,
                session_id=session_id,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        rag.session_manager.clear_session(session_id)
        return {"status": "ok"}

    return test_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag():
    """
    A MagicMock that stands in for RAGSystem.

    Default return values are set to realistic shapes so tests that don't care
    about specific values pass without additional setup.
    """
    rag = MagicMock()
    rag.query.return_value = (
        "This is the AI-generated answer.",
        ["Intro to Python - Lesson 2"],
        ["https://example.com/lesson2"],
    )
    rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Intro to Python", "Advanced RAG", "MCP Fundamentals"],
    }
    rag.session_manager.create_session.return_value = "generated-session-id"
    return rag


@pytest.fixture
def test_client(mock_rag):
    """
    A Starlette TestClient wrapping a test FastAPI app backed by mock_rag.

    Import-safe: never imports app.py, never touches ../frontend, never starts
    ChromaDB or the Anthropic client.
    """
    app = make_test_app(mock_rag)
    with TestClient(app) as client:
        yield client
