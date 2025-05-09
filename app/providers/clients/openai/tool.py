import json
from typing import Any, Sequence

from loguru import logger
from mcp.types import EmbeddedResource, ImageContent, TextContent

from app.chat.schemas.stream import ToolExecution
from app.model_context_protocol.schemas.tools import MCPTool, ToolCall, ToolResult
from app.model_context_protocol.services.tool_execution import MCPToolExecutionService
from app.providers.clients.openai.schemas import OpenAIFunction, OpenAIFunctionParameters, OpenAITool


class OpenAIToolHandler:
    """
    Handles tool-related operations for OpenAI provider.
    """

    def __init__(self) -> None:
        self.mcp_service = MCPToolExecutionService()

    def format_tools(self, tools: Sequence[MCPTool]) -> list[dict]:
        """
        Format MCP tools for OpenAI API as function definitions.
        """
        return [
            OpenAITool(
                function=OpenAIFunction(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=OpenAIFunctionParameters(
                        properties=tool.input_schema.get("properties", {}),
                        required=tool.input_schema.get("required", []),
                        additionalProperties=False,
                    ),
                )
            ).model_dump()
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
        tool_call: ToolExecution,
        result: str,
    ) -> list[dict[str, Any]]:
        """Format tool call and result as messages."""
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id,
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
