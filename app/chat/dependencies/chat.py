from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.services.completion import ChatCompletionService
from app.database.dependencies import get_db_session
from app.providers.dependencies.provider import get_provider_factory
from app.providers.factory import ProviderFactory


async def get_chat_service(
    db: AsyncSession = Depends(get_db_session),
    provider_factory: ProviderFactory = Depends(get_provider_factory),
) -> ChatCompletionService:
    return ChatCompletionService(
        db=db,
        provider_factory=provider_factory,
    )
