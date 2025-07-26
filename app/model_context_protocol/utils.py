"""Shared utilities for MCP server management."""

from loguru import logger
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP

from app.model_context_protocol.constants import ServerType


def create_server_instance(server_type, command, config=None, env=None):
    """Create MCP server instance from configuration parameters."""
    config = config or {}

    try:
        if server_type == ServerType.STDIO:
            return MCPServerStdio(
                command=command,
                args=config.get("args", []),
                env=env or {},
                tool_prefix=config.get("tool_prefix"),
                timeout=config.get("timeout", 5.0),
                cwd=config.get("cwd"),
            )
        elif server_type == ServerType.STREAMABLE_HTTP:
            return MCPServerStreamableHTTP(
                url=command,
                tool_prefix=config.get("tool_prefix"),
                timeout=config.get("timeout", 5.0),
                headers=config.get("headers"),
                read_timeout=config.get("sse_read_timeout", 300.0),
            )
        else:
            logger.warning(f"Unsupported server type: {server_type}")
            return None
    except Exception as e:
        logger.error(f"Failed to create server instance: {e}")
        return None


def create_server_instance_from_db(db_server):
    """Create MCP server instance from database model."""
    return create_server_instance(
        server_type=db_server.server_type, command=db_server.command, config=db_server.config, env=db_server.env
    )


def create_server_instance_from_config(config):
    """Create MCP server instance from config schema."""
    # Get environment variables
    env_vars = {}
    if hasattr(config, "env") and config.env:
        env_vars = {key: value.get_secret_value() for key, value in config.env.items()}

    # Get command args from config
    server_config = getattr(config, "config", {}) or {}
    command_args = server_config.get("args", [])

    # Create modified config for stdio servers
    if config.server_type == ServerType.STDIO:
        modified_config = server_config.copy()
        modified_config["args"] = command_args
        return create_server_instance(
            server_type=config.server_type, command=config.command, config=modified_config, env=env_vars
        )
    else:
        return create_server_instance(
            server_type=config.server_type, command=config.command, config=server_config, env=env_vars
        )
