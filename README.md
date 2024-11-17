![Relay Github Header](./relay.svg)


# Relay Serve - Simple yet effective open source LLM Studio.

Relay is an open-source platform for interacting with Large Language Models (LLMs), designed to provide a unified interface for both local and cloud-based models. It supports project-specific knowledge bases and configurable search integrations.

> ⚠️ **Note:** This project is currently in active development and not yet ready for production use.

## 🌟 Key Features

- 🤖 Unified interface for multiple LLM providers (OpenAI, Anthropic, Ollama)
- 📚 Project-based knowledge management
- 🔄 Real-time chat with streaming responses
- 🎯 Configurable model parameters and settings
- 🔌 Extensible plugin system (coming soon)
- 🔒 Built-in security and rate limiting

## 🛠️ Tech Stack

- **Frontend**: React/Next.js with TailwindCSS
- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL with vector extensions
- **Cache**: Dragonfly (Redis compatible)
- **Infrastructure**: Docker & Docker Compose

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.12 or higher
- Git

### Development Setup

1. Clone the repository
```bash
git clone git@github.com:yamanahlawat/relay-serve.git
cd relay
