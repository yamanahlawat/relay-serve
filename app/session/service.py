from typing import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.model.exceptions import InvalidModelProviderException
from app.model.service import LLMModelService
from app.provider.service import LLMProviderService
from app.session.crud import crud_session
from app.session.exceptions import ActiveSessionNotFoundException, SessionNotFoundException
from app.session.model import ChatSession
from app.session.schema import SessionCreate, SessionUpdate


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

    async def list_sessions(self, title: str | None = None, offset: int = 0, limit: int = 10) -> Sequence[ChatSession]:
        filters = [ChatSession.title.ilike(f"%{title}%")] if title else []
        return await crud_session.filter(db=self.db, filters=filters, offset=offset, limit=limit)

    async def get_session(self, session_id: UUID) -> ChatSession:
        session = await crud_session.get_with_relations(self.db, id=session_id)
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
        if session_in.provider_id:
            provider_service = LLMProviderService(db=self.db)
            await provider_service.get_provider(provider_id=session_in.provider_id)
        if session_in.llm_model_id:
            model_service = LLMModelService(db=self.db)
            await model_service.get_model(llm_model_id=session_in.llm_model_id)
        return await crud_session.update(db=self.db, id=session.id, obj_in=session_in)

    async def delete_session(self, session_id: UUID) -> None:
        session = await self.get_session(session_id)
        await crud_session.delete(db=self.db, id=session.id)
