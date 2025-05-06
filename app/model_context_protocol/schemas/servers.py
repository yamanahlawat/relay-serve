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

    4. Docker Container with advanced options:
       docker_image: "modelcontextprotocol/server-tavily"
       docker_tag: "latest"
       docker_network: "relay_network"
       docker_container_name: "mcp-tavily"
       docker_labels: {"mcp.server": "tavily", "mcp.version": "1.0"}
       docker_resources: {"memory": "512m", "cpu-shares": "512"}
       env: {"API_KEY": "your-api-key"}
    """

    command: str
    args: list[str]
    enabled: bool = True
    env: dict[str, str] | None = None

    @property
    def is_docker(self) -> bool:
        """Check if this is a Docker-based command"""
        return self.command == "docker" and len(self.args) > 0 and self.args[0] == "run"


class MCPConfig(BaseModel):
    """
    Global MCP configuration.
    """

    servers: dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="Map of server names to their configurations",
    )
    # Docker configuration
    docker_network: str = Field(default="relay_network", description="Default network for Docker containers")
    gateway_host: str = Field(default="host.docker.internal", description="Host for Docker MCP Toolkit Gateway")
    gateway_port: int = Field(default=8811, description="Port for Docker MCP Toolkit Gateway")

    @property
    def enabled_servers(self) -> dict[str, MCPServerConfig]:
        """
        Get only enabled servers.
        """
        return {name: config for name, config in self.servers.items() if config.enabled}


class MCPServerTools(BaseModel):
    name: str
    tools: list[BaseTool]
