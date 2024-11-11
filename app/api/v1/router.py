from fastapi import APIRouter

from app.providers.router import model_router, provider_router

api_router = APIRouter(prefix="/v1")


api_router.include_router(router=provider_router)
api_router.include_router(router=model_router)
