from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
        self.last_source_links = []  # Track source links from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, course_name=course_name, lesson_number=lesson_number
        )

        # Handle errors
        if results.error:
            self.last_sources = []
            self.last_source_links = []
            return results.error

        # Handle empty results
        if results.is_empty():
            self.last_sources = []
            self.last_source_links = []
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []
        source_links = []

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source label and link for the UI
            source = course_title
            if lesson_num is not None:
                source += f" - Lesson {lesson_num}"
                link = self.store.get_lesson_link(course_title, lesson_num)
            else:
                link = self.store.get_course_link(course_title)
            sources.append(source)
            source_links.append(link)

            formatted.append(f"{header}\n{doc}")

        self.last_sources = sources
        self.last_source_links = source_links

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving a course outline (title, link, and lesson list)"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []
        self.last_source_links = []

    def get_tool_definition(self) -> Dict[str, Any]:
        return {
            "name": "get_course_outline",
            "description": "Retrieve the full outline of a course: its title, link, and list of all lessons with their numbers and titles",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_title": {
                        "type": "string",
                        "description": "Course title to look up (partial matches work, e.g. 'MCP', 'Introduction')",
                    }
                },
                "required": ["course_title"],
            },
        }

    def execute(self, course_title: str) -> str:
        outline = self.store.get_course_outline(course_title)
        if not outline:
            return f"No course found matching '{course_title}'."

        self.last_sources = [outline["title"]]
        self.last_source_links = [outline["course_link"]]

        lines = [
            f"Course: {outline['title']}",
            f"Link: {outline['course_link']}",
            "",
            "Lessons:",
        ]
        for lesson in outline["lessons"]:
            lines.append(
                f"  Lesson {lesson['lesson_number']}: {lesson['lesson_title']}"
            )

        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def get_last_source_links(self) -> list:
        """Get source links from the last search operation"""
        for tool in self.tools.values():
            if hasattr(tool, "last_source_links") and tool.last_source_links:
                return tool.last_source_links
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []
            if hasattr(tool, "last_source_links"):
                tool.last_source_links = []
