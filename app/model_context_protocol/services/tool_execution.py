from typing import Sequence

from loguru import logger

from app.database.session import AsyncSessionLocal
from app.model_context_protocol.exceptions import MCPToolError
from app.model_context_protocol.schemas.tools import MCPTool, ToolCall, ToolResult
from app.model_context_protocol.services.lifecycle import mcp_lifecycle_manager


class MCPToolExecutionService:
    """
    Service for executing MCP tools and managing tool metadata.

    This service is responsible for discovering available tools from running servers,
    maintaining a tool registry, and executing tool calls on the appropriate servers.
    """

    def __init__(self) -> None:
        self.lifecycle_manager = mcp_lifecycle_manager
        self._tool_cache: dict[str, MCPTool] = {}

    async def get_available_tools(self, refresh: bool = False) -> Sequence[MCPTool]:
        """
        Get all available tools from running MCP servers.
        Args:
            refresh: Force refresh the tool cache
        Returns:
            List of available tools with their metadata
        """
        if self._tool_cache and not refresh:
            return list(self._tool_cache.values())

        tools = []
        running_servers = await self.lifecycle_manager.get_running_servers()

        for server_name, session in running_servers.items():
            response = await session.list_tools()
            for tool in response.tools:
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description,
                    server_name=server_name,
                    input_schema=tool.inputSchema,
                )
                tools.append(mcp_tool)
                self._tool_cache[tool.name] = mcp_tool
        return tools

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call on the appropriate MCP server.
        Args:
            tool_call: Tool call details
        Returns:
            Result of the tool execution
        Raises:
            MCPToolError: If tool execution fails
        """
        async with AsyncSessionLocal() as db:
            # Get tool metadata
            tool = self._tool_cache.get(tool_call.name)
            if not tool:
                # Refresh cache and try again
                await self.get_available_tools(refresh=True)
                tool = self._tool_cache.get(tool_call.name)
                if not tool:
                    raise MCPToolError(f"Tool {tool_call.name} not found")

            logger.info(f"Executing tool: {tool_call.name} with arguments: {tool_call.arguments}")
            # Get server session
            session = await self.lifecycle_manager.get_server_session(db, tool.server_name)

            # Execute tool
            result = await session.call_tool(name=tool_call.name, arguments=tool_call.arguments)

            logger.info(f"Got tool results for tool: {tool_call.name}")

            return ToolResult(content=result.content, call_id=tool_call.call_id)


# Create a singleton instance of the service
mcp_tool_service = MCPToolExecutionService()
