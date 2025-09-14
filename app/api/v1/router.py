from fastapi import APIRouter

from app.llm.router import router as llm_router
from app.attachment.router import router as attachment_router
from app.mcp_server.router import router as mcp_router
from app.message.router import router as message_router
from app.model.router import router as model_router
from app.provider.router import router as provider_router
from app.session.router import router as session_router

api_router = APIRouter(prefix="/v1")


# Include all routers
api_router.include_router(router=llm_router)
# Chat Routers
api_router.include_router(router=session_router)
api_router.include_router(router=message_router)
api_router.include_router(router=attachment_router)
# LLM Management Routers
api_router.include_router(router=provider_router)
api_router.include_router(router=model_router)
# MCP Routers
api_router.include_router(router=mcp_router)
