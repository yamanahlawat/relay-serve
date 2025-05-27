import asyncio
from typing import Sequence

from loguru import logger

from app.database.session import AsyncSessionLocal
from app.model_context_protocol.constants import MCPEventType
from app.model_context_protocol.exceptions import MCPToolError
from app.model_context_protocol.schemas.tools import MCPTool, ToolCall, ToolResult
from app.model_context_protocol.services.event_bus import mcp_event_bus
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
        self._initialized: bool = False
        # Tracking active tool executions by server
        self._active_executions: dict[str, set[str]] = {}
        # Lock for thread-safe access to active executions
        self._active_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """
        Initialize the tool service and register event handlers.
        This should be called during application startup.
        """
        if self._initialized:
            return

        # Register for server events
        await mcp_event_bus.subscribe(MCPEventType.SERVER_SHUTDOWN, self._handle_server_shutdown)
        logger.info("Initialized MCP tool execution service with event handlers")
        self._initialized = True

    async def _handle_server_shutdown(self, data: dict[str, str]) -> None:
        """
        Handle server shutdown events from the event bus.
        Args:
            data: Event data containing server_name
        """
        server_name = data.get("server_name")
        if server_name:
            await self.clear_server_tools(server_name)

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
            # Use the safer method to get tools from this server
            server_tools = await self._safely_get_server_tools(server_name, session)
            tools.extend(server_tools)

            # Update the tool cache with these tools
            for tool in server_tools:
                self._tool_cache[tool.name] = tool
        return tools

    async def _safely_get_server_tools(self, server_name: str, session) -> list[MCPTool]:
        """
        Safely retrieve tools from a server session, handling closed resources gracefully.
        This method contains error handling to prevent issues when accessing potentially
        closed or unavailable server sessions.
        Args:
            server_name: Name of the server to get tools from
            session: Server client session

        Returns:
            List of MCPTool objects from this server
        """
        tools = []
        try:
            response = await session.list_tools()
            for tool in response.tools:
                tools.append(
                    MCPTool(
                        name=tool.name,
                        description=tool.description,
                        server_name=server_name,
                        input_schema=tool.inputSchema,
                    )
                )
        except Exception as e:
            # Handle all exceptions including ClosedResourceError
            logger.warning(f"Error retrieving tools from server {server_name}: {e}")

        return tools

    async def has_active_executions(self, server_name: str) -> bool:
        """
        Check if a server has any active tool executions running.
        Args:
            server_name: Name of the server to check
        Returns:
            True if there are active executions, False otherwise
        """
        async with self._active_lock:
            return server_name in self._active_executions and bool(self._active_executions[server_name])

    async def wait_for_active_executions(self, server_name: str, timeout: float = 5.0) -> bool:
        """
        Wait for any active executions to complete with a timeout.
        Args:
            server_name: Name of the server to wait for
            timeout: Maximum time to wait in seconds
        Returns:
            True if all executions completed, False if timeout was reached
        """
        start_time = asyncio.get_event_loop().time()
        while await self.has_active_executions(server_name):
            # Check if we've exceeded the timeout
            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > timeout:
                async with self._active_lock:
                    count = len(self._active_executions.get(server_name, set()))
                logger.warning(f"Timeout waiting for {count} active executions to complete for server {server_name}")
                return False

            # Wait a small amount of time before checking again
            await asyncio.sleep(0.1)

        return True

    async def clear_server_tools(self, server_name: str) -> None:
        """
        Clear all tools for a specific server from the cache.
        This should be called when a server is shut down or disabled.
        Args:
            server_name: Name of the server whose tools should be cleared
        """
        # Wait for active executions to complete before clearing tools
        active_count = 0
        async with self._active_lock:
            if server_name in self._active_executions:
                active_count = len(self._active_executions[server_name])

        if active_count > 0:
            logger.info(
                f"Waiting for {active_count} active executions to complete before clearing tools for {server_name}"
            )
            await self.wait_for_active_executions(server_name)

        # Find all tools associated with this server
        tools_to_remove = []
        for tool_name, tool in self._tool_cache.items():
            if tool.server_name == server_name:
                tools_to_remove.append(tool_name)

        # Remove those tools from the cache
        for tool_name in tools_to_remove:
            self._tool_cache.pop(tool_name, None)

        logger.info(f"Cleared {len(tools_to_remove)} tools for server {server_name} from tool cache")

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

            server_name = tool.server_name
            call_id = tool_call.call_id

            # Register this execution as active for the server
            async with self._active_lock:
                if server_name not in self._active_executions:
                    self._active_executions[server_name] = set()
                self._active_executions[server_name].add(call_id)

            try:
                logger.info(f"Executing tool: {tool_call.name} with arguments: {tool_call.arguments}")
                # Get server session
                session = await self.lifecycle_manager.get_server_session(db, server_name)

                # Execute tool
                result = await session.call_tool(name=tool_call.name, arguments=tool_call.arguments)

                logger.info(f"Got tool results for tool: {tool_call.name}")

                # Publish tool execution event
                await mcp_event_bus.publish(
                    MCPEventType.TOOL_EXECUTED,
                    {
                        "tool_name": tool_call.name,
                        "server_name": server_name,
                        "call_id": call_id,
                        "success": True,
                    },
                )

                return ToolResult(content=result.content, call_id=call_id)
            finally:
                # Remove this execution from active list when completed
                async with self._active_lock:
                    if server_name in self._active_executions:
                        self._active_executions[server_name].discard(call_id)
                        # Clean up empty sets
                        if not self._active_executions[server_name]:
                            self._active_executions.pop(server_name, None)


# Create a singleton instance of the service
mcp_tool_service = MCPToolExecutionService()
