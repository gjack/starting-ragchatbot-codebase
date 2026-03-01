"""
Tests for CourseSearchTool.execute() in search_tools.py.

Validates that the tool correctly formats results, handles errors,
passes filters to the vector store, and tracks sources/links.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from unittest.mock import MagicMock, patch
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_store(search_return: SearchResults) -> MagicMock:
    store = MagicMock()
    store.search.return_value = search_return
    store.get_lesson_link.return_value = "https://example.com/lesson"
    store.get_course_link.return_value = "https://example.com/course"
    return store


def make_results(docs, metadatas) -> SearchResults:
    return SearchResults(
        documents=docs,
        metadata=metadatas,
        distances=[0.1] * len(docs)
    )


# ---------------------------------------------------------------------------
# CourseSearchTool.execute() — happy path
# ---------------------------------------------------------------------------

class TestCourseSearchToolExecute:

    def test_execute_returns_formatted_results(self):
        """execute() returns a non-empty formatted string when results exist."""
        results = make_results(
            docs=["Chunk about Python loops."],
            metadatas=[{"course_title": "Intro to Python", "lesson_number": 2}]
        )
        store = make_mock_store(results)
        tool = CourseSearchTool(store)

        output = tool.execute(query="python loops")

        assert "Intro to Python" in output
        assert "Lesson 2" in output
        assert "Chunk about Python loops." in output

    def test_execute_formats_header_without_lesson(self):
        """Result header shows course title only when lesson_number is absent."""
        results = make_results(
            docs=["General overview content."],
            metadatas=[{"course_title": "Data Science 101"}]   # no lesson_number key
        )
        store = make_mock_store(results)
        tool = CourseSearchTool(store)

        output = tool.execute(query="overview")

        assert "Data Science 101" in output
        assert "Lesson" not in output

    def test_execute_multiple_results_joined(self):
        """Multiple result chunks are joined by double newlines."""
        results = make_results(
            docs=["Chunk A.", "Chunk B."],
            metadatas=[
                {"course_title": "Course X", "lesson_number": 1},
                {"course_title": "Course X", "lesson_number": 2},
            ]
        )
        store = make_mock_store(results)
        tool = CourseSearchTool(store)

        output = tool.execute(query="topic")

        assert "Chunk A." in output
        assert "Chunk B." in output

    # ---------------------------------------------------------------------------
    # Empty / error results
    # ---------------------------------------------------------------------------

    def test_execute_empty_results_returns_no_content_message(self):
        """execute() reports no results found when search returns nothing."""
        store = make_mock_store(SearchResults(documents=[], metadata=[], distances=[]))
        tool = CourseSearchTool(store)

        output = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in output

    def test_execute_empty_results_with_course_filter_mentions_course(self):
        """Empty results message includes the course filter name."""
        store = make_mock_store(SearchResults(documents=[], metadata=[], distances=[]))
        tool = CourseSearchTool(store)

        output = tool.execute(query="topic", course_name="MCP Course")

        assert "MCP Course" in output

    def test_execute_empty_results_with_lesson_filter_mentions_lesson(self):
        """Empty results message includes the lesson filter number."""
        store = make_mock_store(SearchResults(documents=[], metadata=[], distances=[]))
        tool = CourseSearchTool(store)

        output = tool.execute(query="topic", lesson_number=3)

        assert "3" in output

    def test_execute_returns_error_string_on_search_error(self):
        """execute() surfaces the error message when SearchResults carries an error."""
        error_results = SearchResults(
            documents=[], metadata=[], distances=[],
            error="Search error: collection is empty"
        )
        store = make_mock_store(error_results)
        tool = CourseSearchTool(store)

        output = tool.execute(query="anything")

        assert "Search error" in output

    # ---------------------------------------------------------------------------
    # Filter passthrough to vector store
    # ---------------------------------------------------------------------------

    def test_execute_passes_query_to_store(self):
        """store.search() is called with the exact query string."""
        store = make_mock_store(SearchResults(documents=[], metadata=[], distances=[]))
        tool = CourseSearchTool(store)

        tool.execute(query="what is RAG?")

        store.search.assert_called_once()
        call_kwargs = store.search.call_args
        assert call_kwargs.kwargs.get("query") == "what is RAG?" or \
               call_kwargs.args[0] == "what is RAG?"

    def test_execute_passes_course_name_filter(self):
        """store.search() receives course_name when provided."""
        store = make_mock_store(SearchResults(documents=[], metadata=[], distances=[]))
        tool = CourseSearchTool(store)

        tool.execute(query="topic", course_name="MCP")

        _, kwargs = store.search.call_args
        assert kwargs.get("course_name") == "MCP"

    def test_execute_passes_lesson_number_filter(self):
        """store.search() receives lesson_number when provided."""
        store = make_mock_store(SearchResults(documents=[], metadata=[], distances=[]))
        tool = CourseSearchTool(store)

        tool.execute(query="topic", lesson_number=5)

        _, kwargs = store.search.call_args
        assert kwargs.get("lesson_number") == 5

    # ---------------------------------------------------------------------------
    # Source tracking
    # ---------------------------------------------------------------------------

    def test_execute_updates_last_sources(self):
        """last_sources is populated after a successful execute()."""
        results = make_results(
            docs=["Content"],
            metadatas=[{"course_title": "AI Fundamentals", "lesson_number": 1}]
        )
        store = make_mock_store(results)
        tool = CourseSearchTool(store)

        tool.execute(query="neural networks")

        assert len(tool.last_sources) == 1
        assert "AI Fundamentals" in tool.last_sources[0]

    def test_execute_updates_last_source_links(self):
        """last_source_links is populated after a successful execute()."""
        results = make_results(
            docs=["Content"],
            metadatas=[{"course_title": "AI Fundamentals", "lesson_number": 1}]
        )
        store = make_mock_store(results)
        store.get_lesson_link.return_value = "https://deeplearning.ai/lesson1"
        tool = CourseSearchTool(store)

        tool.execute(query="neural networks")

        assert len(tool.last_source_links) == 1
        assert tool.last_source_links[0] == "https://deeplearning.ai/lesson1"

    def test_execute_source_link_uses_course_link_when_no_lesson(self):
        """get_course_link() is used when metadata has no lesson_number."""
        results = make_results(
            docs=["Content"],
            metadatas=[{"course_title": "AI Fundamentals"}]
        )
        store = make_mock_store(results)
        store.get_course_link.return_value = "https://deeplearning.ai/course"
        tool = CourseSearchTool(store)

        tool.execute(query="overview")

        store.get_course_link.assert_called_once_with("AI Fundamentals")
        assert tool.last_source_links[0] == "https://deeplearning.ai/course"

    def test_execute_clears_previous_sources_on_empty_results(self):
        """Empty results leave last_sources empty (not stale from a prior call)."""
        results = make_results(
            docs=["Content"],
            metadatas=[{"course_title": "Course A", "lesson_number": 1}]
        )
        store = make_mock_store(results)
        tool = CourseSearchTool(store)
        tool.execute(query="first query")   # populates sources

        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        tool.execute(query="second query")  # should clear sources

        assert tool.last_sources == []
        assert tool.last_source_links == []


# ---------------------------------------------------------------------------
# ToolManager integration
# ---------------------------------------------------------------------------

class TestToolManagerIntegration:

    def test_tool_manager_registers_and_executes_search_tool(self):
        """ToolManager.execute_tool() routes to CourseSearchTool.execute()."""
        results = make_results(
            docs=["Some content."],
            metadatas=[{"course_title": "Test Course", "lesson_number": 1}]
        )
        store = make_mock_store(results)
        tool = CourseSearchTool(store)
        manager = ToolManager()
        manager.register_tool(tool)

        output = manager.execute_tool("search_course_content", query="test")

        assert "Test Course" in output

    def test_tool_manager_returns_error_for_unknown_tool(self):
        """ToolManager returns an error string for unregistered tool names."""
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="test")
        assert "not found" in result.lower() or "nonexistent_tool" in result

    def test_tool_manager_get_last_sources_after_search(self):
        """get_last_sources() returns sources set by the search tool."""
        results = make_results(
            docs=["Content"],
            metadatas=[{"course_title": "My Course", "lesson_number": 2}]
        )
        store = make_mock_store(results)
        tool = CourseSearchTool(store)
        manager = ToolManager()
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="topic")

        sources = manager.get_last_sources()

        assert len(sources) > 0
        assert "My Course" in sources[0]

    def test_tool_manager_reset_sources_clears_all(self):
        """reset_sources() empties last_sources on all registered tools."""
        results = make_results(
            docs=["Content"],
            metadatas=[{"course_title": "My Course", "lesson_number": 2}]
        )
        store = make_mock_store(results)
        tool = CourseSearchTool(store)
        manager = ToolManager()
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="topic")

        manager.reset_sources()

        assert manager.get_last_sources() == []
        assert manager.get_last_source_links() == []
