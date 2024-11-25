from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.sentry import init_sentry

relay = FastAPI(
    title="Relay",
    description="Simple yet effective open source LLM Studio.",
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


# Initialize Sentry
init_sentry()
