from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import settings

relay = FastAPI(
    title="Relay",
    description="Simple yet effective open source LLM Studio.",
)

# Include API router
relay.include_router(router=api_router, prefix=settings.API_URL)
