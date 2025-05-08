from pydantic import BaseModel, Field

from app.model_context_protocol.schemas.tools import BaseTool


class MCPServerConfig(BaseModel):
    """
    Individual MCP server configuration.

    This class supports both direct command execution and Docker-based MCP servers.
    For Docker-based servers, the command should be 'docker' and args should include
    the 'run' command and necessary parameters.

    Examples:
    1. Direct execution:
       command: "python"
       args: ["-m", "mcp_server_tavily"]

    2. Docker Gateway:
       command: "docker"
       args: ["run", "--rm", "-i", "alpine/socat", "STDIO", "TCP:host.docker.internal:8811"]

    3. Docker Container:
       command: "docker"
       args: ["run", "-i", "--rm", "-e", "API_KEY", "mcp/server-name"]
       env: {"API_KEY": "your-api-key"}
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
        default_factory=dict,
        description="Map of server names to their configurations",
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
