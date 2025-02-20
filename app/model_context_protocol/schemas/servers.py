from pydantic import BaseModel, Field

from app.model_context_protocol.schemas.tools import BaseTool


class MCPServerConfig(BaseModel):
    """
    Individual MCP server configuration.
    """

    command: str
    args: list[str]
    enabled: bool = True
    env: dict[str, str] | None = None


class MCPConfig(BaseModel):
    """
    Global MCP configuration.
    """

    servers: dict[str, MCPServerConfig] = Field(
        default_factory=dict, description="Map of server names to their configurations"
    )

    @property
    def enabled_servers(self) -> dict[str, MCPServerConfig]:
        """
        Get only enabled servers.
        """
        return {name: config for name, config in self.servers.items() if config.enabled}


class MCPServerTools(BaseModel):
    name: str
    tools: list[BaseTool]
