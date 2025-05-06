from typing import Any

from pydantic import BaseModel, Field, model_validator

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

    # Docker-specific helper fields (not stored, used for convenience)
    docker_image: str | None = None
    docker_tag: str = "latest"
    docker_network: str | None = None
    docker_volumes: list[str] | None = None
    docker_ports: list[str] | None = None
    docker_container_name: str | None = None
    docker_labels: dict[str, str] | None = None
    docker_resources: dict[str, str] | None = None
    docker_auto_remove: bool = True  # Corresponds to --rm flag

    @property
    def is_docker(self) -> bool:
        """Check if this is a Docker-based command"""
        return self.command == "docker" and len(self.args) > 0 and self.args[0] == "run"

    @model_validator(mode="before")
    @classmethod
    def build_docker_command(cls, data: Any) -> Any:
        """If docker_image is provided, build the docker command and args"""
        if not isinstance(data, dict):
            return data

        # Make a copy to avoid modifying the input data
        values = dict(data)

        if values.get("docker_image") and not (
            values.get("command") == "docker"
            and values.get("args")
            and len(values.get("args", [])) > 0
            and values.get("args")[0] == "run"
        ):
            # Start with basic docker run command
            docker_args = ["run"]

            # Add auto-remove flag if specified
            if values.get("docker_auto_remove", True):
                docker_args.append("--rm")

            # Always run in interactive mode for MCP servers
            docker_args.append("-i")

            # Add container name if specified
            if values.get("docker_container_name"):
                docker_args.extend(["--name", values["docker_container_name"]])

            # Add network if specified
            if values.get("docker_network"):
                docker_args.extend(["--network", values["docker_network"]])

            # Add volumes if specified
            if values.get("docker_volumes"):
                for volume in values["docker_volumes"]:
                    docker_args.extend(["-v", volume])

            # Add ports if specified
            if values.get("docker_ports"):
                for port in values["docker_ports"]:
                    docker_args.extend(["-p", port])

            # Add labels if specified
            if values.get("docker_labels"):
                for key, value in values["docker_labels"].items():
                    # Escape any quotes in the value
                    safe_value = value.replace('"', '\\"') if isinstance(value, str) else value
                    docker_args.extend(["--label", f"{key}={safe_value}"])

            # Add resource limits if specified
            if values.get("docker_resources"):
                for key, value in values["docker_resources"].items():
                    docker_args.extend([f"--{key}", str(value)])

            # Add environment variables from env dict
            # Use -e KEY for secrets to avoid exposing values in process listings
            if values.get("env"):
                for key, value in values["env"].items():
                    # Check if this might be a sensitive value (API key, token, etc.)
                    is_sensitive = any(s in key.lower() for s in ["key", "token", "secret", "password", "auth"])

                    if is_sensitive:
                        # For sensitive values, just pass the key and let Docker read from host env
                        docker_args.extend(["-e", key])
                    else:
                        # For non-sensitive values, include the value directly
                        # Escape any quotes in the value
                        safe_value = value.replace('"', '\\"') if isinstance(value, str) else value
                        docker_args.extend(["-e", f"{key}={safe_value}"])

            # Add image name with tag
            tag = values.get("docker_tag", "latest")
            docker_args.append(f"{values['docker_image']}:{tag}")

            # Set the command and args
            values["command"] = "docker"
            values["args"] = docker_args

        return values


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
