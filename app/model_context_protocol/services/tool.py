from typing import Sequence

from loguru import logger

from app.model_context_protocol.exceptions import MCPToolError
from app.model_context_protocol.initialize import mcp_registry
from app.model_context_protocol.schemas.tools import MCPTool, ToolCall, ToolResult


class MCPToolService:
    """
    Service for managing MCP tool operations.
    """

    def __init__(self) -> None:
        self.registry = mcp_registry
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
        running_servers = await self.registry.get_running_servers()

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
        session = await self.registry.get_server_session(tool.server_name)

        # Execute tool
        result = await session.call_tool(name=tool_call.name, arguments=tool_call.arguments)

        return ToolResult(content=result.content, call_id=tool_call.call_id)

    async def refresh_tools(self) -> None:
        """
        Force refresh the tool cache.
        """
        self._tool_cache.clear()
        await self.get_available_tools(refresh=True)
