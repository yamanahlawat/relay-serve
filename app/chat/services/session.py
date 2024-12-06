from typing import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.crud import crud_session
from app.chat.exceptions.session import ActiveSessionNotFoundException, SessionNotFoundException
from app.chat.models.session import ChatSession
from app.chat.schemas.session import SessionCreate, SessionUpdate
from app.providers.exceptions.model import InvalidModelProviderException
from app.providers.services import LLMModelService, LLMProviderService


class ChatSessionService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_session(self, session_in: SessionCreate) -> ChatSession:
        provider_service = LLMProviderService(db=self.db)
        model_service = LLMModelService(db=self.db)
        provider = await provider_service.get_provider(provider_id=session_in.provider_id)
        model = await model_service.get_model(llm_model_id=session_in.llm_model_id)
        if model.provider_id != provider.id:
            raise InvalidModelProviderException()
        return await crud_session.create(db=self.db, obj_in=session_in)

    async def list_sessions(self, offset: int = 0, limit: int = 10) -> Sequence[ChatSession]:
        return await crud_session.filter(db=self.db, offset=offset, limit=limit)

    async def get_session(self, session_id: UUID) -> ChatSession:
        session = await crud_session.get(self.db, id=session_id)
        if not session:
            raise SessionNotFoundException(session_id=session_id)
        return session

    async def get_active_session(self, session_id: UUID) -> ChatSession:
        session = await crud_session.get_active(db=self.db, id=session_id)
        if not session:
            raise ActiveSessionNotFoundException(session_id=session_id)
        return session

    async def update_session(self, session_id: UUID, session_in: SessionUpdate) -> ChatSession | None:
        session = await self.get_session(session_id)
        return await crud_session.update(db=self.db, id=session.id, obj_in=session_in)

    async def delete_session(self, session_id: UUID) -> None:
        session = await self.get_session(session_id)
        await crud_session.delete(db=self.db, id=session.id)
