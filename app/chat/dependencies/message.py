from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Path, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.crud import crud_message
from app.chat.models import ChatMessage
from app.database.dependencies import get_db_session


async def validate_message(
    message_id: Annotated[UUID, Path],
    session_id: Annotated[UUID, Path],
    db: AsyncSession = Depends(get_db_session),
) -> ChatMessage:
    """
    Validate that a message exists and belongs to the session.
    """
    message = await crud_message.get(db, id=message_id)
    if not message:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Message {message_id} not found")

    if message.session_id != session_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message belongs to a different session")

    return message
