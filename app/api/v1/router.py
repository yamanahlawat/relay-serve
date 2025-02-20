from fastapi import APIRouter

from app.chat.router.attachment import router as attachment_router
from app.chat.router.chat import router as chat_router
from app.chat.router.message import router as message_router
from app.chat.router.session import router as session_router
from app.model_context_protocol.router import router as mcp_router
from app.providers.router import model_router, provider_router

api_router = APIRouter(prefix="/v1")


# Include all routers
# Provider Routers
api_router.include_router(router=provider_router)
api_router.include_router(router=model_router)
# Chat Routers
api_router.include_router(router=chat_router)
api_router.include_router(router=session_router)
api_router.include_router(router=message_router)
api_router.include_router(router=attachment_router)
# MCP Routers
api_router.include_router(router=mcp_router)
