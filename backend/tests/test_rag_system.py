"""
Tests for RAGSystem.query() in rag_system.py.

Patches all heavy dependencies (ChromaDB, Anthropic) so tests run without
real infrastructure. Validates the orchestration logic: tools are passed to
the AI, history is saved, sources are collected and reset.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Helper: build a RAGSystem with every external dependency mocked
# ---------------------------------------------------------------------------


def make_rag():
    """
    Construct a RAGSystem with VectorStore, AIGenerator, DocumentProcessor,
    and SessionManager all replaced by MagicMocks.
    Returns (rag, mocks_dict).
    """
    with (
        patch("rag_system.VectorStore") as mock_vs_cls,
        patch("rag_system.AIGenerator") as mock_ai_cls,
        patch("rag_system.DocumentProcessor") as mock_dp_cls,
        patch("rag_system.SessionManager") as mock_sm_cls,
    ):

        from rag_system import RAGSystem

        config = MagicMock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "/tmp/test_chroma"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        config.ANTHROPIC_API_KEY = "test-key"
        config.ANTHROPIC_MODEL = "claude-test"

        rag = RAGSystem(config)

        mocks = {
            "vector_store": mock_vs_cls.return_value,
            "ai_generator": mock_ai_cls.return_value,
            "document_processor": mock_dp_cls.return_value,
            "session_manager": mock_sm_cls.return_value,
        }

    return rag, mocks


# ---------------------------------------------------------------------------
# Basic return-value shape
# ---------------------------------------------------------------------------


class TestRAGSystemQueryReturnShape:

    def test_query_returns_three_element_tuple(self):
        """query() must return (response, sources, source_links) — a 3-tuple."""
        rag, mocks = make_rag()
        mocks["ai_generator"].generate_response.return_value = "Some answer"
        mocks["session_manager"].get_conversation_history.return_value = None

        result = rag.query("What is RAG?")

        assert isinstance(result, tuple), "query() must return a tuple"
        assert len(result) == 3, (
            f"query() must return exactly 3 values (answer, sources, source_links), "
            f"got {len(result)}"
        )

    def test_query_first_element_is_ai_response(self):
        """The first element of the tuple is the string returned by AIGenerator."""
        rag, mocks = make_rag()
        mocks["ai_generator"].generate_response.return_value = (
            "Detailed answer about RAG"
        )
        mocks["session_manager"].get_conversation_history.return_value = None

        answer, _, _ = rag.query("Explain RAG")

        assert answer == "Detailed answer about RAG"

    def test_query_sources_and_links_are_lists(self):
        """sources and source_links in the returned tuple must be lists."""
        rag, mocks = make_rag()
        mocks["ai_generator"].generate_response.return_value = "Answer"
        mocks["session_manager"].get_conversation_history.return_value = None

        _, sources, source_links = rag.query("question")

        assert isinstance(sources, list)
        assert isinstance(source_links, list)


# ---------------------------------------------------------------------------
# Tool passing
# ---------------------------------------------------------------------------


class TestRAGSystemToolUsage:

    def test_query_passes_tool_definitions_to_ai_generator(self):
        """generate_response() is called with a non-empty 'tools' list."""
        rag, mocks = make_rag()
        mocks["ai_generator"].generate_response.return_value = "Answer"
        mocks["session_manager"].get_conversation_history.return_value = None

        rag.query("What does lesson 3 cover?")

        call_kwargs = mocks["ai_generator"].generate_response.call_args.kwargs
        tools = call_kwargs.get("tools")
        assert tools is not None, "generate_response() must receive a 'tools' argument"
        assert len(tools) > 0, "tools list must not be empty"

    def test_query_passes_tool_manager_to_ai_generator(self):
        """generate_response() receives the ToolManager instance."""
        rag, mocks = make_rag()
        mocks["ai_generator"].generate_response.return_value = "Answer"
        mocks["session_manager"].get_conversation_history.return_value = None

        rag.query("Content question")

        call_kwargs = mocks["ai_generator"].generate_response.call_args.kwargs
        tool_manager = call_kwargs.get("tool_manager")
        assert (
            tool_manager is not None
        ), "generate_response() must receive 'tool_manager' so it can execute tool calls"
        assert tool_manager is rag.tool_manager

    def test_tool_definitions_include_search_course_content(self):
        """The tools list passed to AI includes 'search_course_content'."""
        rag, mocks = make_rag()
        mocks["ai_generator"].generate_response.return_value = "Answer"
        mocks["session_manager"].get_conversation_history.return_value = None

        rag.query("What does the MCP course cover?")

        call_kwargs = mocks["ai_generator"].generate_response.call_args.kwargs
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert (
            "search_course_content" in tool_names
        ), f"'search_course_content' must be in the tools list; got {tool_names}"


# ---------------------------------------------------------------------------
# Session / history management
# ---------------------------------------------------------------------------


class TestRAGSystemSessionHandling:

    def test_query_fetches_history_when_session_id_provided(self):
        """get_conversation_history() is called when a session_id is given."""
        rag, mocks = make_rag()
        mocks["ai_generator"].generate_response.return_value = "Answer"
        mocks["session_manager"].get_conversation_history.return_value = (
            "User: hi\nAssistant: hello"
        )

        rag.query("follow up", session_id="session_1")

        mocks["session_manager"].get_conversation_history.assert_called_once_with(
            "session_1"
        )

    def test_query_skips_history_when_no_session(self):
        """get_conversation_history() is NOT called when session_id is None."""
        rag, mocks = make_rag()
        mocks["ai_generator"].generate_response.return_value = "Answer"

        rag.query("standalone question", session_id=None)

        mocks["session_manager"].get_conversation_history.assert_not_called()

    def test_query_saves_exchange_after_response(self):
        """add_exchange() is called with the query and response when a session exists."""
        rag, mocks = make_rag()
        mocks["ai_generator"].generate_response.return_value = "The answer"
        mocks["session_manager"].get_conversation_history.return_value = None

        rag.query("What is RAG?", session_id="session_42")

        mocks["session_manager"].add_exchange.assert_called_once()
        args = mocks["session_manager"].add_exchange.call_args.args
        assert args[0] == "session_42"
        assert "RAG" in args[1]  # original user query
        assert args[2] == "The answer"  # AI response

    def test_query_does_not_save_history_without_session(self):
        """add_exchange() is NOT called when no session_id is provided."""
        rag, mocks = make_rag()
        mocks["ai_generator"].generate_response.return_value = "Answer"

        rag.query("question", session_id=None)

        mocks["session_manager"].add_exchange.assert_not_called()


# ---------------------------------------------------------------------------
# Source collection and reset
# ---------------------------------------------------------------------------


class TestRAGSystemSourceHandling:

    def test_sources_are_collected_from_tool_manager(self):
        """Sources returned by the tool manager appear in query()'s second return value."""
        rag, mocks = make_rag()
        mocks["ai_generator"].generate_response.return_value = "Answer"
        mocks["session_manager"].get_conversation_history.return_value = None

        # Inject sources into the real tool_manager via patching
        rag.tool_manager.get_last_sources = MagicMock(
            return_value=["Course A - Lesson 1"]
        )
        rag.tool_manager.get_last_source_links = MagicMock(
            return_value=["https://example.com"]
        )
        rag.tool_manager.reset_sources = MagicMock()

        _, sources, links = rag.query("content question", session_id=None)

        assert sources == ["Course A - Lesson 1"]
        assert links == ["https://example.com"]

    def test_sources_reset_after_query(self):
        """reset_sources() is called on the tool manager after sources are retrieved."""
        rag, mocks = make_rag()
        mocks["ai_generator"].generate_response.return_value = "Answer"
        mocks["session_manager"].get_conversation_history.return_value = None

        rag.tool_manager.get_last_sources = MagicMock(return_value=[])
        rag.tool_manager.get_last_source_links = MagicMock(return_value=[])
        rag.tool_manager.reset_sources = MagicMock()

        rag.query("question", session_id=None)

        rag.tool_manager.reset_sources.assert_called_once()


# ---------------------------------------------------------------------------
# Error propagation (causes HTTP 500 in app.py)
# ---------------------------------------------------------------------------


class TestRAGSystemErrorPropagation:

    def test_exception_from_ai_generator_propagates(self):
        """
        If AIGenerator raises, the exception bubbles up through query().
        app.py catches it and returns HTTP 500, which the frontend shows as 'Query failed'.
        """
        rag, mocks = make_rag()
        mocks["ai_generator"].generate_response.side_effect = RuntimeError(
            "API failure"
        )
        mocks["session_manager"].get_conversation_history.return_value = None

        with pytest.raises(RuntimeError, match="API failure"):
            rag.query("What is covered in lesson 1?")

    def test_exception_not_swallowed_silently(self):
        """query() must not catch exceptions and return a default string."""
        rag, mocks = make_rag()
        mocks["ai_generator"].generate_response.side_effect = ValueError(
            "bad model name"
        )
        mocks["session_manager"].get_conversation_history.return_value = None

        with pytest.raises(ValueError):
            rag.query("Any content question")


# ---------------------------------------------------------------------------
# get_course_analytics — regression test for missing method
# ---------------------------------------------------------------------------


class TestRAGSystemCourseAnalytics:

    def test_get_course_analytics_exists(self):
        """RAGSystem must expose get_course_analytics() so /api/courses doesn't 500."""
        rag, mocks = make_rag()
        assert hasattr(rag, "get_course_analytics"), (
            "RAGSystem is missing get_course_analytics(). "
            "app.py calls this on the /api/courses endpoint, causing AttributeError → HTTP 500."
        )

    def test_get_course_analytics_returns_dict_with_expected_keys(self):
        """get_course_analytics() returns a dict with 'total_courses' and 'course_titles'."""
        rag, mocks = make_rag()
        if not hasattr(rag, "get_course_analytics"):
            pytest.skip("get_course_analytics() not implemented yet")

        mocks["vector_store"].get_course_count.return_value = 2
        mocks["vector_store"].get_existing_course_titles.return_value = [
            "Intro to Python",
            "Advanced RAG",
        ]

        result = rag.get_course_analytics()

        assert "total_courses" in result, "Missing 'total_courses' key"
        assert "course_titles" in result, "Missing 'course_titles' key"
        assert result["total_courses"] == 2
        assert "Intro to Python" in result["course_titles"]
