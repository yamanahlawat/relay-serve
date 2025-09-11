# Contributing to Relay Serve

We love your input! We want to make contributing to Relay Serve as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. **Fork the repo** and create your branch from `main`.
2. **Install dependencies**: `uv sync --all-packages`
3. **Make your changes** and add tests if applicable.
4. **Run the test suite**: `pytest`
5. **Run code quality checks**: `ruff check . && ruff format .`
6. **Run pre-commit hooks**: `pre-commit run --all-files`
7. **Ensure the description is clear** and the PR template is filled out.
8. **Submit the pull request**!

## Development Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose (for local development)
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yamanahlawat/relay-serve.git
   cd relay-serve
   ```

2. **Install dependencies**
   ```bash
   uv sync --all-packages
   ```

3. **Set up environment**
   ```bash
   cp .env.template .env
   # Edit .env with your configuration
   ```

4. **Start services with Docker**
   ```bash
   docker-compose up -d
   ```

5. **Run the development server**
   ```bash
   uv run fastapi dev app/main.py
   ```

### Code Style

We use several tools to maintain code quality:

- **Ruff**: For linting and formatting
- **Pre-commit**: For automated checks
- **Type hints**: Required for all new code

Run these commands before submitting:

```bash
# Format code
ruff format .

# Check for issues
ruff check .

# Run pre-commit hooks
pre-commit run --all-files
```

### Testing

- Write tests for new features and bug fixes
- Run the test suite: `pytest`
- Ensure all tests pass before submitting

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](LICENSE) that covers the project.

## Report bugs using GitHub Issues

We use GitHub Issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yamanahlawat/relay-serve/issues/new/choose).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
- Be specific!
- Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening)

## Feature Requests

Feature requests are welcome! Please use our [feature request template](https://github.com/yamanahlawat/relay-serve/issues/new/choose) and provide:

- Clear description of the feature
- Use cases and examples
- Any implementation ideas you might have

## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `style:` code style changes (formatting, etc.)
- `refactor:` code changes that neither fix bugs nor add features
- `test:` adding or updating tests
- `chore:` maintenance tasks

Example: `feat: add support for new LLM provider`

## Community

- Join our [Discord community](https://discord.gg/V5AHYx72Yv)
- Participate in [GitHub Discussions](https://github.com/yamanahlawat/relay-serve/discussions)
- Follow our [Code of Conduct](https://github.com/yamanahlawat/relay-serve/blob/main/CODE_OF_CONDUCT.md)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Don't hesitate to ask! You can:
- Open an issue
- Join our Discord
- Start a discussion

Thank you for contributing to Relay Serve! 🎉
