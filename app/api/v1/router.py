from fastapi import APIRouter

from app.ai.router import router as ai_router
from app.chat.router.attachment import router as attachment_router
from app.chat.router.message import router as message_router
from app.chat.router.session import router as session_router
from app.llms.router.model import router as model_router
from app.llms.router.provider import router as provider_router
from app.model_context_protocol.router import router as mcp_router

api_router = APIRouter(prefix="/v1")


# Include all routers
api_router.include_router(router=ai_router)
# Chat Routers
api_router.include_router(router=session_router)
api_router.include_router(router=message_router)
api_router.include_router(router=attachment_router)
# LLM Management Routers
api_router.include_router(router=provider_router)
api_router.include_router(router=model_router)
# MCP Routers
api_router.include_router(router=mcp_router)
