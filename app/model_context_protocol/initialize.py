from app.model_context_protocol.schemas.servers import MCPConfig, MCPServerConfig
from app.model_context_protocol.registry import MCPServerRegistry

MCP_SERVERS = {
    "tavily-search": {
        "command": "python",
        "args": [
            "-m",
            "mcp_server_tavily",
            "--api-key",
            "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",
        ],
        "enabled": True,
    },
}

mcp_config = MCPConfig(servers={key: MCPServerConfig(**value) for key, value in MCP_SERVERS.items()})

# Instantiate the MCP Server Registry
mcp_registry = MCPServerRegistry(config=mcp_config)
