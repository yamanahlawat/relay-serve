from typing import Any, Sequence

from loguru import logger
from mcp.types import EmbeddedResource, ImageContent, TextContent

from app.model_context_protocol.schemas.tools import MCPTool, ToolCall, ToolResult
from app.model_context_protocol.services.tool_execution import MCPToolExecutionService


class AnthropicToolHandler:
    """
    Handles tool-related operations for Anthropic provider.
    Manages tool formatting, execution, and result handling.
    """

    def __init__(self) -> None:
        self.mcp_service = MCPToolExecutionService()

    def format_tools(self, tools: Sequence[MCPTool]) -> list[dict]:
        """
        Format MCP tools for Anthropic API format.
        Args:
            tools: Sequence of MCPTool objects
        Returns:
            List of formatted tools for Anthropic API
        """
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]

    async def execute_tool(self, name: str, arguments: dict[str, Any], call_id: str | None = None) -> ToolResult:
        """
        Execute a tool via MCP service.
        Args:
            name: Tool name
            arguments: Tool arguments
            call_id: Optional call identifier
        Returns:
            Tool execution result
        """
        tool_call = ToolCall(name=name, arguments=arguments, call_id=call_id)
        return await self.mcp_service.execute_tool(tool_call=tool_call)

    def format_tool_messages(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: str,
        tool_id: str,
    ) -> list[dict[str, Any]]:
        """
        Format tool call and result as messages for Anthropic API.
        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            tool_result: Tool execution result
            tool_id: Tool call identifier
        Returns:
            List of formatted messages
        """
        return [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": tool_args,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": tool_result,
                    }
                ],
            },
        ]

    def format_tool_result(self, content: list[TextContent | ImageContent | EmbeddedResource]) -> str:
        """
        Format tool result content for message history.
        Args:
            content: List of content objects from tool execution
        Returns:
            Formatted string result
        """
        text_parts = []
        for item in content:
            if isinstance(item, TextContent):
                text_parts.append(item.text)
            elif isinstance(item, ImageContent):
                # TODO: Handle image content
                # Future: Handle image content if needed
                logger.warning("Image content not displayed")
                raise NotImplementedError("Image content not supported")
            elif isinstance(item, EmbeddedResource):
                # TODO: Handle embedded resources
                # Future: Handle embedded resources if needed
                logger.warning("Embedded resource not displayed")
                raise NotImplementedError("Embedded resource not supported")
        return " ".join(text_parts)
