<div align="center">

# Relay Serve

**Open source chat API server**

A REST API server for managing chat conversations across multiple language model providers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com)

</div>

---

## What is Relay Serve?

Relay Serve is a **REST API server** that provides a consistent interface for chat applications to work with different language model providers. Instead of integrating with each provider individually, your application can use a single API to access OpenAI, Anthropic, Google, and others.

## Key Features

**Multiple Provider Support**

- OpenAI, Anthropic, Google Gemini, Groq, Mistral, Cohere, AWS Bedrock
- Single API endpoint works with any configured provider
- Provider factory pattern with Pydantic AI integration
- Easy to add new providers

**Session & Message Management**

- Persistent conversation storage with PostgreSQL
- Message history and context preservation
- File attachment support with image processing
- Structured session lifecycle management

**Model Context Protocol (MCP) Integration**

- Full MCP server support for tool integration
- Server lifecycle management and validation
- External tool and context integration

**Real-time Communication**

- Server-Sent Events (SSE) for streaming responses
- Async request handling with FastAPI
- Tool tracking and execution monitoring

**Developer Experience**

- Complete REST API with automatic documentation
- Database migrations with Alembic
- Type-safe Python code with Pydantic
- Modern dependency management with UV
- Docker development environment
- Application monitoring with Logfire

## 🛠️ Tech Stack

| Component          | Technology                   | Purpose                    |
| ------------------ | ---------------------------- | -------------------------- |
| **Backend**        | FastAPI (Python 3.13+)       | Async REST API server      |
| **Database**       | PostgreSQL                   | Primary data storage       |
| **Cache**          | Dragonfly (Redis-compatible) | Session and data caching   |
| **ORM**            | SQLAlchemy 2.0               | Database operations        |
| **AI Framework**   | Pydantic AI 1.0+             | LLM provider abstraction   |
| **Infrastructure** | Docker & Docker Compose      | Development environment    |
| **Migrations**     | Alembic                      | Database schema management |
| **Package Manager**| UV                           | Fast Python dependency management |
| **MCP**            | Model Context Protocol       | Tool and context integration |
| **Monitoring**     | Logfire                      | Application observability  |

## 🖥️ Frontend: Relay Connect

**[Relay Connect](https://github.com/yamanahlawat/relay-connect)** is our companion Next.js frontend that provides a beautiful chat interface for Relay Serve. Check out the repository for setup instructions and screenshots.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.13+ (for local development)
- Git

### Docker Setup (Recommended)

1. **Clone and configure**

```bash
git clone https://github.com/yamanahlawat/relay-serve.git
cd relay-serve
cp .env.template .env
# Edit .env with your configuration
```

2. **Start everything**

```bash
docker-compose up -d
```

3. **Access the API**

- API Documentation: http://localhost:8000/docs
- Interactive API: http://localhost:8000/redoc

### Local Development Setup

1. **Install uv and dependencies**

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: pip install uv

# Install project dependencies
uv sync --all-packages
```

## 📖 Configuration

### Environment Variables

Key configuration options in your `.env` file:

```env
# API Configuration
BASE_URL=http://localhost:8000
DEBUG=true
ENVIRONMENT=local

# Database
DATABASE__HOST=localhost
DATABASE__USER=relay_user
DATABASE__PASSWORD=your_password
DATABASE__DB=relay_db

# Redis Cache
REDIS__HOST=localhost
REDIS__PORT=6379

# CORS (for frontend)
ALLOWED_CORS_ORIGINS=["http://localhost:3000"]

# Optional: Monitoring
LOGFIRE_TOKEN=your_logfire_token
```

### LLM Provider Setup

Add your API keys through the Relay API or directly in the database:

- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Google**: `GOOGLE_API_KEY`
- **Groq**: `GROQ_API_KEY`

## 📚 API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Development

### Code Quality

```bash
# Linting and formatting
ruff check .
ruff format .

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Project Structure

```
app/
├── llm/             # LLM providers and chat services
│   ├── providers/   # Provider implementations (OpenAI, Anthropic, etc.)
│   ├── services/    # Chat, SSE streaming, and tool tracking
│   ├── schemas/     # LLM-related data models
│   └── dependencies/ # LLM dependency injection
├── session/         # Chat session management
├── message/         # Message handling and storage
├── attachment/      # File attachment processing
├── mcp_server/      # Model Context Protocol server
├── model/           # AI model definitions and management
├── provider/        # Provider configuration and management
├── core/            # Core infrastructure
│   ├── database/    # Database setup and CRUD operations
│   ├── storage/     # File storage interfaces
│   ├── image/       # Image processing utilities
│   └── config.py    # Application configuration
└── api/             # API versioning and schemas
```

## 🚀 Deployment

### Docker Production

```bash
# Build production image
docker build -f docker/dockerfile -t relay-serve:latest .

# Run with production config
docker-compose -f docker-compose.yml up -d
```

## 🤝 Contributing

We love your input! We want to make contributing as easy and transparent as possible. Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contribution Setup

1. **Fork the repository**
2. **Create your feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run the test suite**: `pytest`
5. **Run code quality checks**: `ruff check . && ruff format .`
6. **Commit your changes**: `git commit -m 'feat: add amazing feature'`
7. **Push to the branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Development Guidelines

- Follow [Conventional Commits](https://www.conventionalcommits.org/)
- Add tests for new features
- Update documentation as needed
- Ensure all CI checks pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - The web framework that powers our API
- [Pydantic AI](https://ai.pydantic.dev/) - For LLM provider abstractions
- [SQLAlchemy](https://www.sqlalchemy.org/) - Our database ORM
- [Alembic](https://alembic.sqlalchemy.org/) - Database migrations
- All the amazing LLM providers making AI accessible

## 📞 Support

- 💬 [Discord Community](https://discord.gg/V5AHYx72Yv)
- 💬 [GitHub Discussions](https://github.com/yamanahlawat/relay-serve/discussions)
- 🐛 [Issue Tracker](https://github.com/yamanahlawat/relay-serve/issues)
- 📧 [Contact Maintainer](https://github.com/yamanahlawat)

---

<div align="center">

**[⭐ Star us on GitHub](https://github.com/yamanahlawat/relay-serve)** if you find Relay Serve helpful!

</div>
