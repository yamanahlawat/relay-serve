from fastapi import Depends

from app.chat.services.completion import ChatCompletionService
from app.providers.dependencies.provider import get_provider_factory
from app.providers.factory import ProviderFactory


async def get_chat_service(
    provider_factory: ProviderFactory = Depends(get_provider_factory),
) -> ChatCompletionService:
    return ChatCompletionService(
        provider_factory=provider_factory,
    )
