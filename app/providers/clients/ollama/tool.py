from typing import Any, Sequence

from loguru import logger
from mcp.types import EmbeddedResource, ImageContent, TextContent

from app.model_context_protocol.schemas.tools import MCPTool, ToolCall, ToolResult
from app.model_context_protocol.services.tool_execution import mcp_tool_service


class OllamaToolHandler:
    """
    Handles tool-related operations for Ollama provider.
    """

    def __init__(self) -> None:
        self.mcp_service = mcp_tool_service

    def format_tools(self, tools: Sequence[MCPTool]) -> list[dict]:
        """
        Format MCP tools for Ollama API.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": tool.input_schema.get("properties", {}),
                        "required": tool.input_schema.get("required", []),
                    },
                },
            }
            for tool in tools
        ]

    async def execute_tool(self, name: str, arguments: dict[str, Any], call_id: str | None = None) -> ToolResult:
        """
        Execute a tool via MCP service.
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
        """Format tool call and result as messages."""
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_args,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_id,
            },
        ]

    def format_tool_result(self, content: list[TextContent | ImageContent | EmbeddedResource]) -> str:
        """
        Format tool result content for message history.
        """
        text_parts = []
        for item in content:
            if isinstance(item, TextContent):
                text_parts.append(item.text)
            elif isinstance(item, ImageContent):
                logger.warning("Image content not displayed")
                raise NotImplementedError("Image content not supported")
            elif isinstance(item, EmbeddedResource):
                logger.warning("Embedded resource not displayed")
                raise NotImplementedError("Embedded resource not supported")
        return " ".join(text_parts)
