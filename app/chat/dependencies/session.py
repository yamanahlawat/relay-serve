from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Path, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.crud import crud_session
from app.chat.models import ChatSession
from app.database.dependencies import get_db_session


async def validate_session(
    session_id: Annotated[UUID, Path],
    db: AsyncSession = Depends(get_db_session),
) -> ChatSession:
    """
    Validate that a session exists.
    """
    session = await crud_session.get(db, id=session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found",
        )
    return session
