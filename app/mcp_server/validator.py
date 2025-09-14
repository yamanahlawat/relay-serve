import asyncio

from loguru import logger

from app.mcp_server.constants import ServerType
from app.mcp_server.schema import MCPServerBase
from app.mcp_server.utils import create_server_instance_from_config


class MCPServerValidator:
    """
    Service for validating MCP servers.
    """

    async def validate_server(self, server_name: str, config: MCPServerBase) -> tuple[bool, str | None]:
        """
        Validate server connectivity and configuration.
        Args:
            server_name: Name of the server for identification
            config: Server configuration to validate
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if config.server_type == ServerType.STDIO:
                return await self._validate_stdio_server(server_name, config)
            elif config.server_type == ServerType.STREAMABLE_HTTP:
                return await self._validate_streamable_http_server(server_name, config)
            else:
                return False, f"Unsupported server type: {config.server_type}"

        except asyncio.TimeoutError:
            logger.warning(f"Server validation timed out for {server_name}")
            return False, "Server validation timed out"
        except ConnectionError as e:
            logger.warning(f"Connection error validating server {server_name}: {e}")
            return False, f"Connection error: {e}"
        except ValueError as e:
            logger.warning(f"Configuration error for server {server_name}: {e}")
            return False, f"Configuration error: {e}"
        except Exception as e:
            logger.exception(f"Unexpected error validating server {server_name}")
            return False, f"Unexpected error: {e}"

    async def _validate_stdio_server(self, server_name: str, config: MCPServerBase) -> tuple[bool, str | None]:
        """Validate stdio server connectivity."""
        try:
            # Create pydantic-ai MCP server instance using shared utility
            mcp_server = create_server_instance_from_config(config)

            # Use the server as an async context manager to do a validation
            async with mcp_server as session:
                # Try to list tools as a health check
                await asyncio.wait_for(session.list_tools(), timeout=3.0)
                logger.info(f"Successfully validated stdio server {server_name}")
                return True, None

        except asyncio.TimeoutError:
            return False, "Server validation timed out"
        except ConnectionError as e:
            logger.warning(f"Connection error for stdio server {server_name}: {e}")
            return False, f"Connection error: {e}"
        except FileNotFoundError as e:
            logger.warning(f"Command not found for stdio server {server_name}: {e}")
            return False, f"Command not found: {e}"
        except PermissionError as e:
            logger.warning(f"Permission denied for stdio server {server_name}: {e}")
            return False, f"Permission denied: {e}"
        except Exception as e:
            logger.exception(f"Error validating stdio server {server_name}")
            return False, str(e)

    async def _validate_streamable_http_server(
        self, server_name: str, config: MCPServerBase
    ) -> tuple[bool, str | None]:
        """
        Validate streamable HTTP server connectivity.
        """
        try:
            # Create pydantic-ai MCP server instance using shared utility
            mcp_server = create_server_instance_from_config(config)

            # Use the server as an async context manager to do a validation
            async with mcp_server as session:
                # Try to list tools as a health check
                await asyncio.wait_for(session.list_tools(), timeout=3.0)
                logger.info(f"Successfully validated streamable HTTP server {server_name}")
                return True, None

        except asyncio.TimeoutError:
            return False, "Server validation timed out"
        except ConnectionError as e:
            logger.warning(f"Connection error for HTTP server {server_name}: {e}")
            return False, f"Connection error: {e}"
        except ValueError as e:
            logger.warning(f"Invalid URL for HTTP server {server_name}: {e}")
            return False, f"Invalid URL: {e}"
        except Exception as e:
            logger.exception(f"Error validating streamable HTTP server {server_name}")
            return False, str(e)
