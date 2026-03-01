"""
Tests for AIGenerator in ai_generator.py.

Validates the two-turn tool-use loop: initial call → tool execution → final synthesis.
All Anthropic API calls are mocked so no real API key is needed.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch, call

import ai_generator as ai_module
from ai_generator import AIGenerator

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_generator() -> AIGenerator:
    """Return an AIGenerator with a mocked Anthropic client."""
    with patch("ai_generator.anthropic.Anthropic"):
        gen = AIGenerator(api_key="test-key", model="claude-test")
    return gen


def text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def tool_use_block(name: str, inputs: dict, tool_id: str = "toolu_01") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = inputs
    return block


def mock_response(stop_reason: str, content: list) -> MagicMock:
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# Direct (no-tool) response path
# ---------------------------------------------------------------------------


class TestDirectResponse:

    def test_returns_text_when_stop_reason_is_end_turn(self):
        """generate_response() returns text directly when Claude doesn't use a tool."""
        gen = make_generator()
        gen.client.messages.create.return_value = mock_response(
            stop_reason="end_turn",
            content=[text_block("Paris is the capital of France.")],
        )

        result = gen.generate_response(query="What is the capital of France?")

        assert result == "Paris is the capital of France."

    def test_no_tool_manager_call_on_direct_response(self):
        """tool_manager.execute_tool() is NOT called when stop_reason is end_turn."""
        gen = make_generator()
        gen.client.messages.create.return_value = mock_response(
            stop_reason="end_turn", content=[text_block("Some answer")]
        )
        mock_tool_manager = MagicMock()

        gen.generate_response(query="General question", tool_manager=mock_tool_manager)

        mock_tool_manager.execute_tool.assert_not_called()

    def test_only_one_api_call_on_direct_response(self):
        """Only a single Claude API call is made when no tool is used."""
        gen = make_generator()
        gen.client.messages.create.return_value = mock_response(
            stop_reason="end_turn", content=[text_block("Answer")]
        )

        gen.generate_response(query="Simple question")

        assert gen.client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Tool-use path
# ---------------------------------------------------------------------------


class TestToolUsePath:

    def _setup_two_turn(
        self,
        gen: AIGenerator,
        tool_name: str,
        tool_inputs: dict,
        tool_result: str,
        final_answer: str,
    ):
        """Configure the mocked client for a two-turn tool-use exchange."""
        first_response = mock_response(
            stop_reason="tool_use", content=[tool_use_block(tool_name, tool_inputs)]
        )
        second_response = mock_response(
            stop_reason="end_turn", content=[text_block(final_answer)]
        )
        gen.client.messages.create.side_effect = [first_response, second_response]
        return first_response

    def test_tool_use_triggers_tool_execution(self):
        """When stop_reason is tool_use, tool_manager.execute_tool() is called."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Search result content"

        self._setup_two_turn(
            gen,
            tool_name="search_course_content",
            tool_inputs={"query": "python decorators"},
            tool_result="Search result content",
            final_answer="Decorators are...",
        )

        gen.generate_response(
            query="What are python decorators?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="python decorators"
        )

    def test_tool_use_makes_two_api_calls(self):
        """The tool-use path always results in exactly two Claude API calls."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "result"

        self._setup_two_turn(
            gen, "search_course_content", {"query": "topic"}, "result", "Answer"
        )

        gen.generate_response(
            query="Content question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert gen.client.messages.create.call_count == 2

    def test_final_answer_returned_after_tool_use(self):
        """generate_response() returns the text from the second Claude call."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "search results"

        self._setup_two_turn(
            gen,
            "search_course_content",
            {"query": "topic"},
            "search results",
            "Here is the synthesized answer.",
        )

        result = gen.generate_response(
            query="Content question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert result == "Here is the synthesized answer."

    def test_second_api_call_messages_contain_tool_result(self):
        """The messages sent to the second API call include a tool_result block."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Retrieved content here"

        self._setup_two_turn(
            gen,
            "search_course_content",
            {"query": "topic"},
            "Retrieved content here",
            "Final answer",
        )

        gen.generate_response(
            query="Content question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_call_kwargs = gen.client.messages.create.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]

        # Find a message whose content contains a tool_result block
        tool_result_found = False
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_result_found = True
                        assert (
                            block["content"] == "Retrieved content here"
                        ), f"tool_result content mismatch: {block['content']}"
        assert tool_result_found, (
            "No tool_result block found in the second API call's messages. "
            "Tool results must be passed back to Claude to synthesize a response."
        )

    def test_tool_result_includes_tool_use_id(self):
        """Each tool_result block carries the matching tool_use_id."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "result"

        # Use a specific tool_use_id so we can verify it
        first_response = mock_response(
            stop_reason="tool_use",
            content=[
                tool_use_block(
                    "search_course_content", {"query": "x"}, tool_id="toolu_abc123"
                )
            ],
        )
        second_response = mock_response(
            stop_reason="end_turn", content=[text_block("Answer")]
        )
        gen.client.messages.create.side_effect = [first_response, second_response]

        gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_call_kwargs = gen.client.messages.create.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]

        tool_use_id_found = False
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        assert block.get("tool_use_id") == "toolu_abc123"
                        tool_use_id_found = True
        assert (
            tool_use_id_found
        ), "tool_result block with tool_use_id not found in second call messages"

    def test_assistant_tool_use_block_included_in_second_call_messages(self):
        """Second call messages include the assistant's original tool_use response."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "result"

        first_response = mock_response(
            stop_reason="tool_use",
            content=[tool_use_block("search_course_content", {"query": "topic"})],
        )
        second_response = mock_response(
            stop_reason="end_turn", content=[text_block("Answer")]
        )
        gen.client.messages.create.side_effect = [first_response, second_response]

        gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_call_kwargs = gen.client.messages.create.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]

        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        assert (
            len(assistant_messages) >= 1
        ), "Second API call must include the assistant's tool_use turn in messages."

    def test_conversation_history_included_in_system_prompt(self):
        """Conversation history is appended to the system prompt when provided."""
        gen = make_generator()
        gen.client.messages.create.return_value = mock_response(
            stop_reason="end_turn", content=[text_block("Answer")]
        )

        gen.generate_response(
            query="Follow-up question",
            conversation_history="User: Hello\nAssistant: Hi there",
        )

        call_kwargs = gen.client.messages.create.call_args.kwargs
        system_prompt = call_kwargs["system"]
        assert "Hello" in system_prompt
        assert "Hi there" in system_prompt

    def test_tools_added_to_first_api_call(self):
        """Tool definitions are passed to the first Claude API call."""
        gen = make_generator()
        gen.client.messages.create.return_value = mock_response(
            stop_reason="end_turn", content=[text_block("Answer")]
        )
        tool_defs = [{"name": "search_course_content", "input_schema": {}}]

        gen.generate_response(query="question", tools=tool_defs)

        call_kwargs = gen.client.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tool_defs

    def test_tool_choice_auto_set_when_tools_provided(self):
        """tool_choice is set to auto when tools are provided."""
        gen = make_generator()
        gen.client.messages.create.return_value = mock_response(
            stop_reason="end_turn", content=[text_block("Answer")]
        )

        gen.generate_response(
            query="question", tools=[{"name": "search_course_content"}]
        )

        call_kwargs = gen.client.messages.create.call_args.kwargs
        assert call_kwargs.get("tool_choice") == {"type": "auto"}


# ---------------------------------------------------------------------------
# Two-round tool-use path
# ---------------------------------------------------------------------------


class TestTwoRoundToolUse:
    """Tests for the sequential tool loop (up to 2 rounds before synthesis)."""

    def _setup_three_turn(
        self,
        gen: AIGenerator,
        tool_a_name: str,
        tool_a_inputs: dict,
        tool_b_name: str,
        tool_b_inputs: dict,
        final_answer: str,
    ):
        """
        Configure the mocked client for a two-round tool-use exchange:
          call 1 → tool_use (tool A)
          call 2 → tool_use (tool B)   [between-round, tools still offered]
          call 3 → end_turn            [synthesis, no tools]
        """
        call1 = mock_response(
            "tool_use", [tool_use_block(tool_a_name, tool_a_inputs, "id_A")]
        )
        call2 = mock_response(
            "tool_use", [tool_use_block(tool_b_name, tool_b_inputs, "id_B")]
        )
        call3 = mock_response("end_turn", [text_block(final_answer)])
        gen.client.messages.create.side_effect = [call1, call2, call3]

    def test_two_rounds_makes_three_api_calls(self):
        """Two tool rounds produce exactly three Claude API calls."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "some result"

        self._setup_three_turn(
            gen,
            "get_course_outline",
            {"course_name": "Python 101"},
            "search_course_content",
            {"query": "lesson 3 title"},
            "Final synthesized answer",
        )

        gen.generate_response(
            query="What is lesson 3 of Python 101 about?",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert gen.client.messages.create.call_count == 3

    def test_two_rounds_executes_both_tools(self):
        """execute_tool is called once per round with the correct arguments."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "some result"

        self._setup_three_turn(
            gen,
            "get_course_outline",
            {"course_name": "Python 101"},
            "search_course_content",
            {"query": "lesson 3 title"},
            "Final synthesized answer",
        )

        gen.generate_response(
            query="question",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_name="Python 101"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="lesson 3 title"
        )

    def test_two_rounds_final_synthesis_has_no_tools(self):
        """The third (synthesis) API call must NOT include a 'tools' key."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "some result"

        self._setup_three_turn(
            gen,
            "get_course_outline",
            {"course_name": "Python 101"},
            "search_course_content",
            {"query": "lesson 3 title"},
            "Final synthesized answer",
        )

        gen.generate_response(
            query="question",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        third_call_kwargs = gen.client.messages.create.call_args_list[2].kwargs
        assert (
            "tools" not in third_call_kwargs
        ), "Synthesis call (index 2) must not include 'tools' so Claude produces text."

    def test_two_rounds_returns_synthesis_text(self):
        """generate_response() returns the text from the synthesis call."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "some result"

        self._setup_three_turn(
            gen,
            "get_course_outline",
            {"course_name": "Python 101"},
            "search_course_content",
            {"query": "lesson 3 title"},
            "Final synthesized answer",
        )

        result = gen.generate_response(
            query="question",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert result == "Final synthesized answer"

    def test_tool_exception_triggers_early_synthesis(self):
        """A tool execution error ends the loop early and triggers synthesis."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = RuntimeError("DB unavailable")

        call1 = mock_response(
            "tool_use",
            [tool_use_block("search_course_content", {"query": "topic"}, "id_err")],
        )
        call2 = mock_response("end_turn", [text_block("Synthesized despite error")])
        gen.client.messages.create.side_effect = [call1, call2]

        result = gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert gen.client.messages.create.call_count == 2
        assert result == "Synthesized despite error"
