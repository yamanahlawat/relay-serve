from app.core.config import settings
from app.model_context_protocol.schemas.servers import MCPConfig, MCPServerConfig
from app.model_context_protocol.services.registry import MCPServerRegistry

MCP_SERVERS = {
    # Direct subprocess execution (original implementation)
    "tavily-search": {
        "command": "python",
        "args": [
            "-m",
            "mcp_server_tavily",
        ],
        "env": {
            "TAVILY_API_KEY": settings.TAVILY_SEARCH_API_KEY.get_secret_value(),
        },
        "enabled": False,
    },
    # Docker-based MCP servers using docker_image helper
    # Docker MCP Toolkit Gateway
    "docker-mcp-gateway": {
        "command": "docker",
        "args": ["run", "-i", "--rm", "alpine/socat", "STDIO", "TCP:host.docker.internal:8811"],
        "enabled": True,
    },
}

# Create the MCP configuration
mcp_config = MCPConfig(servers={key: MCPServerConfig(**value) for key, value in MCP_SERVERS.items()})

# Instantiate the MCP Server Registry
mcp_registry = MCPServerRegistry(config=mcp_config)
