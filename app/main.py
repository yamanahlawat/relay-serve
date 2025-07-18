from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ai.services.sse import get_sse_manager
from app.api.v1.router import api_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Context manager to handle the lifespan of the application.
    """
    yield
    # Get the SSE manager instance
    manager = await get_sse_manager()
    # Clean up Redis connections
    await manager.cleanup()


relay = FastAPI(
    title="Relay",
    description="Simple yet effective open source LLM Studio.",
    lifespan=lifespan,
)

# Add CORS middleware if allowed origins are set
if settings.ALLOWED_CORS_ORIGINS:
    relay.add_middleware(
        middleware_class=CORSMiddleware,
        allow_origins=[str(url).rstrip("/") for url in settings.ALLOWED_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Include API router
relay.include_router(router=api_router, prefix=settings.API_URL)


# Initialize Sentry if DSN is provided
if settings.SENTRY_DSN:
    from app.core.sentry import init_sentry

    # Initialize Sentry
    init_sentry()


# Configure Logfire if token is provided
if settings.LOGFIRE_TOKEN:
    from app.core.logfire import configure_logfire

    # Configure Logfire
    configure_logfire()
