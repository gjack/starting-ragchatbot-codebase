import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use **get_course_outline** for questions about course structure, lesson lists, or outlines (e.g. "What lessons are in X?", "Show me the outline of X", "How many lessons does X have?")
  - Return the course title, course link, and all lesson numbers and titles
- Use **search_course_content** for questions about specific topics or content within a course
- **Maximum 2 sequential tool calls per query** — use a second call only when the first result is needed to form the second query (e.g. look up a lesson title, then search for related content)
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_loop(response, api_params, tool_manager, tools)

        # Return direct response
        return response.content[0].text

    def _handle_tool_loop(
        self,
        initial_response,
        base_params: Dict[str, Any],
        tool_manager,
        tools,
        max_rounds: int = 2,
    ) -> str:
        """
        Execute up to max_rounds of tool-use before synthesizing a final answer.

        After each tool round the between-round call still offers tools (tool_choice:
        auto) so Claude can request a second lookup.  The final synthesis call uses
        self.base_params (no tools) to guarantee Claude produces text.
        """
        messages = list(base_params["messages"])
        current_response = initial_response

        for round_num in range(max_rounds):
            # Append assistant's tool-use turn
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute every tool block in this round
            tool_results = []
            terminated_early = False
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(
                            content_block.name, **content_block.input
                        )
                    except Exception as e:
                        result = f"Tool execution error: {e}"
                        terminated_early = True
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": result,
                        }
                    )
                    if terminated_early:
                        break

            messages.append({"role": "user", "content": tool_results})

            if terminated_early:
                break

            # Between rounds: offer tools again so Claude can chain a second lookup
            if round_num < max_rounds - 1:
                between_params = {
                    k: v for k, v in base_params.items() if k != "messages"
                }
                next_response = self.client.messages.create(
                    **between_params, messages=messages
                )
                if next_response.stop_reason != "tool_use":
                    # Claude synthesized directly — return without an extra call
                    return next_response.content[0].text
                current_response = next_response

        return self._synthesize(messages, base_params["system"])

    def _synthesize(self, messages: list, system: str) -> str:
        """Make a final synthesis call with no tools so Claude must produce text."""
        final = self.client.messages.create(
            **self.base_params,
            messages=messages,
            system=system,
        )
        return final.content[0].text
