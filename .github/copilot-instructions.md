# Relay Serve - AI Coding Agent Instructions

## Core Development Principles

**Simplicity & Code Quality**:

- **No duplicate code / dead code** - Eliminate redundancy and unused code immediately
- **Single responsibility principle** - Each function/component should have one clear purpose
- **No premature optimization** - Code readability matters more than micro-optimizations
- **Embrace simplicity** - Simple solutions are preferred over complex ones

**Code Philosophy**:

- Writing less code is better than writing more code
- Writing simple code is better than writing complex code
- Writing no code is better than writing any code
- Every line of code written today becomes technical debt tomorrow
- **Follow KISS principle** - Keep It Simple, Stupid

## Architecture Overview

Relay is a FastAPI-based LLM Studio with a modular domain-driven architecture. The system manages chat sessions, LLM providers, and Model Context Protocol (MCP) servers for extensible AI workflows.

### Core Components

- **`app/ai/`** - Pydantic-AI integration with streaming chat completion using SSE
- **`app/chat/`** - Chat session and message management with PostgreSQL persistence
- **`app/llms/`** - Multi-provider LLM management (OpenAI, Anthropic, Ollama, etc.)
- **`app/model_context_protocol/`** - MCP server lifecycle management and tool execution
- **`app/database/`** - SQLAlchemy with async PostgreSQL, generic CRUD pattern
- **`app/core/`** - Configuration management with pydantic-settings

### Key Architectural Patterns

**Domain Service Pattern**: Each module has a domain service (e.g., `ChatSessionService`, `MCPServerDomainService`) that handles business logic and coordinates between CRUD operations and external services.

**Generic CRUD Base**: All data access uses `CRUDBase[ModelType, CreateSchemaType, UpdateSchemaType]` from `app/database/crud.py` - extend this for new models.

**Async Session Management**: Database operations use dependency injection of `AsyncSession` from `app/database/session.py`.

**Pydantic Validation**: All API schemas inherit from `BaseModel` with clear separation between Create/Update/InDB/Response schemas.

## MCP Server Integration

The MCP system is the most complex component:

- **Lifecycle Management**: `MCPServerLifecycleManager` handles server startup/shutdown during app lifespan
- **Process Management**: `MCPProcessManager` manages individual server processes (stdio/HTTP)
- **Tool Execution**: `MCPToolService` handles tool calls with event-driven architecture
- **Server Types**: Support for `stdio` (subprocess) and `streamable_http` (HTTP transport)

MCP servers are configured via `config: dict` field - validation is handled by pydantic-ai classes, not custom schemas.

## Development Workflows

**Database Migrations**: Use Alembic with `alembic_postgresql_enum` for enum changes:
```bash
alembic revision --autogenerate -m "description"
alembic upgrade head
```

**Code Quality**: Pre-commit hooks with Ruff for formatting/linting:
```bash
pre-commit run --all-files
```

**Dependencies**: Use `uv` for dependency management:
```bash
uv add package_name
uv sync
```

## Critical Integration Points

**Streaming Responses**: Use `StreamingResponse` with SSE for real-time chat via `app/ai/services/sse.py`

**Agent Integration**: Pydantic-AI agents use MCP servers as toolsets - see `get_mcp_servers_for_agent()` in domain service

**Error Handling**: Domain-specific exceptions (e.g., `MCPServerError`, `SessionNotFoundException`) with consistent error response schemas

**Configuration**: Environment-based settings in `app/core/config.py` with type validation and database DSN assembly

## Project-Specific Conventions

- Models use `TimeStampedBase` for automatic created_at/updated_at timestamps
- All async database operations use `AsyncSession` with proper transaction handling
- MCP server configurations are stored as JSONB in PostgreSQL for flexibility
- Tools are dynamically registered and executed through the MCP protocol
- Chat attachments support multiple media types (images, documents, audio, video)

## Common Patterns

**Adding New Domain**: Create models → CRUD → schemas → service → router → tests
**MCP Server Types**: Extend `ServerType` enum and add handling in `MCPServerDomainService`
**New LLM Provider**: Add to `llm_registry` and create provider-specific configuration
**Chat Features**: Extend `ChatMessage` model and update streaming logic in `app/ai/services/chat.py`
