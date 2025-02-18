from app.core.config import settings
from app.model_context_protocol.schemas.servers import MCPConfig, MCPServerConfig
from app.model_context_protocol.services.registry import MCPServerRegistry

MCP_SERVERS = {
    "tavily-search": {
        "command": "python",
        "args": [
            "-m",
            "mcp_server_tavily",
            "--api-key",
            settings.TAVILY_SEARCH_API_KEY.get_secret_value() if settings.TAVILY_SEARCH_API_KEY else "",
        ],
        "enabled": True,
    },
    "sentry": {
        "command": "uvx",
        "args": [
            "mcp-server-sentry",
            "--auth-token",
            settings.SENTRY_AUTH_TOKEN.get_secret_value(),
        ],
        "enabled": False,
    },
}

mcp_config = MCPConfig(servers={key: MCPServerConfig(**value) for key, value in MCP_SERVERS.items()})

# Instantiate the MCP Server Registry
mcp_registry = MCPServerRegistry(config=mcp_config)
