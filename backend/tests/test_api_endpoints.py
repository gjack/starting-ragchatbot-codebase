"""
Integration tests for the FastAPI API endpoints.

Uses test_client and mock_rag fixtures from conftest.py.  No real ChromaDB,
Anthropic API key, or static-file directory is required.

Covered endpoints:
  POST   /api/query
  GET    /api/courses
  DELETE /api/session/{session_id}
"""
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:

    def test_returns_200_on_success(self, test_client):
        resp = test_client.post("/api/query", json={"query": "What is RAG?"})
        assert resp.status_code == 200

    def test_response_body_has_answer(self, test_client):
        resp = test_client.post("/api/query", json={"query": "What is RAG?"})
        assert "answer" in resp.json()
        assert isinstance(resp.json()["answer"], str)

    def test_response_answer_matches_rag_output(self, test_client, mock_rag):
        mock_rag.query.return_value = ("Custom answer text.", [], [])
        resp = test_client.post("/api/query", json={"query": "question"})
        assert resp.json()["answer"] == "Custom answer text."

    def test_response_body_has_sources_list(self, test_client, mock_rag):
        mock_rag.query.return_value = ("answer", ["Course A - Lesson 1"], [])
        resp = test_client.post("/api/query", json={"query": "question"})
        assert resp.json()["sources"] == ["Course A - Lesson 1"]

    def test_response_body_has_source_links_list(self, test_client, mock_rag):
        mock_rag.query.return_value = ("answer", [], ["https://example.com"])
        resp = test_client.post("/api/query", json={"query": "question"})
        assert resp.json()["source_links"] == ["https://example.com"]

    def test_response_body_has_session_id(self, test_client):
        resp = test_client.post("/api/query", json={"query": "question"})
        assert "session_id" in resp.json()
        assert isinstance(resp.json()["session_id"], str)

    def test_creates_session_when_none_provided(self, test_client, mock_rag):
        """When no session_id is in the request body, create_session() is called."""
        test_client.post("/api/query", json={"query": "question"})
        mock_rag.session_manager.create_session.assert_called_once()

    def test_uses_provided_session_id_without_creating_new(self, test_client, mock_rag):
        """When session_id is supplied, create_session() must NOT be called."""
        test_client.post(
            "/api/query",
            json={"query": "question", "session_id": "existing-session"},
        )
        mock_rag.session_manager.create_session.assert_not_called()

    def test_passes_provided_session_id_to_rag(self, test_client, mock_rag):
        test_client.post(
            "/api/query",
            json={"query": "lesson question", "session_id": "my-session"},
        )
        mock_rag.query.assert_called_once_with("lesson question", "my-session")

    def test_generated_session_id_appears_in_response(self, test_client, mock_rag):
        mock_rag.session_manager.create_session.return_value = "new-abc-123"
        resp = test_client.post("/api/query", json={"query": "question"})
        assert resp.json()["session_id"] == "new-abc-123"

    def test_missing_query_field_returns_422(self, test_client):
        """FastAPI validation: omitting the required 'query' field → 422."""
        resp = test_client.post("/api/query", json={"session_id": "s1"})
        assert resp.status_code == 422

    def test_non_string_query_returns_422(self, test_client):
        """FastAPI validation: query must be a string → 422."""
        resp = test_client.post("/api/query", json={"query": 123})
        assert resp.status_code == 422

    def test_rag_exception_returns_500(self, test_client, mock_rag):
        mock_rag.query.side_effect = RuntimeError("AI backend failure")
        resp = test_client.post("/api/query", json={"query": "question"})
        assert resp.status_code == 500

    def test_500_detail_includes_error_message(self, test_client, mock_rag):
        mock_rag.query.side_effect = RuntimeError("AI backend failure")
        resp = test_client.post("/api/query", json={"query": "question"})
        assert "AI backend failure" in resp.json()["detail"]

    def test_empty_query_string_is_valid(self, test_client):
        """Empty string passes Pydantic validation — semantic checks are the AI's job."""
        resp = test_client.post("/api/query", json={"query": ""})
        assert resp.status_code == 200

    def test_sources_and_links_can_be_empty_lists(self, test_client, mock_rag):
        mock_rag.query.return_value = ("answer", [], [])
        resp = test_client.post("/api/query", json={"query": "question"})
        assert resp.json()["sources"] == []
        assert resp.json()["source_links"] == []


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:

    def test_returns_200_on_success(self, test_client):
        resp = test_client.get("/api/courses")
        assert resp.status_code == 200

    def test_response_has_total_courses(self, test_client):
        resp = test_client.get("/api/courses")
        assert "total_courses" in resp.json()

    def test_total_courses_matches_analytics(self, test_client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 5,
            "course_titles": ["A", "B", "C", "D", "E"],
        }
        resp = test_client.get("/api/courses")
        assert resp.json()["total_courses"] == 5

    def test_response_has_course_titles(self, test_client):
        resp = test_client.get("/api/courses")
        assert "course_titles" in resp.json()
        assert isinstance(resp.json()["course_titles"], list)

    def test_course_titles_matches_analytics(self, test_client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Intro to Python", "Advanced RAG"],
        }
        resp = test_client.get("/api/courses")
        assert resp.json()["course_titles"] == ["Intro to Python", "Advanced RAG"]

    def test_zero_courses_returns_empty_list(self, test_client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        resp = test_client.get("/api/courses")
        assert resp.json()["total_courses"] == 0
        assert resp.json()["course_titles"] == []

    def test_analytics_exception_returns_500(self, test_client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("DB unavailable")
        resp = test_client.get("/api/courses")
        assert resp.status_code == 500

    def test_500_detail_includes_error_message(self, test_client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("DB unavailable")
        resp = test_client.get("/api/courses")
        assert "DB unavailable" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

class TestDeleteSessionEndpoint:

    def test_returns_200_on_success(self, test_client):
        resp = test_client.delete("/api/session/abc123")
        assert resp.status_code == 200

    def test_response_body_is_ok_status(self, test_client):
        resp = test_client.delete("/api/session/abc123")
        assert resp.json() == {"status": "ok"}

    def test_calls_clear_session_with_correct_id(self, test_client, mock_rag):
        test_client.delete("/api/session/my-session-id")
        mock_rag.session_manager.clear_session.assert_called_once_with("my-session-id")

    def test_session_id_is_captured_from_url_path(self, test_client, mock_rag):
        """The session_id path parameter is passed verbatim to clear_session()."""
        test_client.delete("/api/session/unique-id-xyz-789")
        mock_rag.session_manager.clear_session.assert_called_once_with("unique-id-xyz-789")
