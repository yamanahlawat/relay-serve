"""MCP Server Initialization and Configuration

This module configures the MCP servers used by the application. All MCP servers use the same
standardized configuration approach based on command and arguments.

MCP Server Configuration:
- command: The command to execute (REQUIRED)
- args: List of arguments passed to the command (REQUIRED)
- env: Optional environment variables (optional)
- enabled: Whether to automatically start this server (default: True)

The configuration supports various execution methods:

1. Direct execution with Python:
   ```python
   "command": "python",
   "args": ["-m", "mcp_server_module"]
   ```

2. NPX-based execution:
   ```python
   "command": "npx",
   "args": ["-y", "@modelcontextprotocol/server-name"]
   ```

3. Docker-based execution:
   ```python
   "command": "docker",
   "args": ["run", "-i", "--rm", "--name", "mcp-server", "mcp/server-image"]
   ```

4. Docker with environment variables:
   ```python
   "command": "docker",
   "args": ["run", "-i", "--rm", "-e", "API_KEY", "mcp/server-image"],
   "env": {"API_KEY": "your-api-key"}
   ```

The MCP_SERVERS dictionary contains default configurations that will be used to seed
the database if no server configurations exist.
"""

from app.core.config import settings
from app.model_context_protocol.services.registry import MCPServerRegistry

MCP_SERVERS = {
    # Docker-based MCP Toolkit Gateway
    "docker-mcp-gateway": {
        "command": "docker",
        "args": [
            "run",
            "-i",
            "--rm",
            "--name",
            "mcp-gateway",
            "alpine/socat",
            "STDIO",
            "TCP:host.docker.internal:8811",
        ],
        "enabled": True,
    },
    "puppeteer": {
        "command": "docker",
        "args": ["run", "-i", "--rm", "--init", "-e", "DOCKER_CONTAINER=true", "mcp/puppeteer"],
        "enabled": True,
    },
    # Direct subprocess execution with Python
    "tavily-search": {
        "command": "python",
        "args": [
            "-m",
            "mcp_server_tavily",
        ],
        "env": {
            "TAVILY_API_KEY": settings.TAVILY_SEARCH_API_KEY,
        },
        "enabled": True,
    },
}

# Instantiate the MCP Server Registry
mcp_registry = MCPServerRegistry()
