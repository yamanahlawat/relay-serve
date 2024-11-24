from typing import Sequence

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.crud import crud_message
from app.chat.dependencies import validate_message, validate_session
from app.chat.models import ChatMessage, ChatSession
from app.chat.schemas.message import MessageCreate, MessageRead, MessageUpdate
from app.database.dependencies import get_db_session

router = APIRouter(prefix="/messages", tags=["Chat Messages"])


@router.post("/{session_id}/", response_model=MessageRead, status_code=status.HTTP_201_CREATED)
async def create_message(
    message_in: MessageCreate,
    session: ChatSession = Depends(dependency=validate_session),
    db: AsyncSession = Depends(get_db_session),
) -> ChatMessage:
    """
    ## Create a New Chat Message
    Creates a new message in the specified chat session.

    ### Parameters
    - **session_id**: UUID of the chat session
    - **content**: Message content
    - **role**: Message role (user/assistant/system)
    - **parent_id**: Optional parent message ID for threading
    - **extra_data**: Optional additional data

    ### Returns
    The created message

    ### Raises
    - **404**: Session not found
    - **404**: Parent message not found (if parent_id provided)
    - **400**: Invalid role for message
    """
    # Verify parent message if provided
    if message_in.parent_id:
        parent = await crud_message.get(db, id=message_in.parent_id)
        if not parent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Parent message {message_in.parent_id} not found"
            )
        if parent.session_id != session.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Parent message belongs to a different session"
            )

    message = await crud_message.create_with_session(db=db, session_id=session.id, obj_in=message_in)
    return message


@router.get("/{session_id}/", response_model=list[MessageRead])
async def list_session_messages(
    session: ChatSession = Depends(dependency=validate_session),
    offset: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db_session),
) -> Sequence[ChatMessage]:
    """
    ## List Session Messages
    Retrieves messages from a specific chat session.

    ### Parameters
    - **session_id**: UUID of the chat session
    - **offset**: Number of messages to skip (default: 0)
    - **limit**: Maximum number of messages to return (default: 50)

    ### Returns
    List of messages in chronological order

    ### Raises
    - **404**: Session not found
    """
    messages = await crud_message.list_by_session(db=db, session_id=session.id, offset=offset, limit=limit)
    for message in messages:
        message.usage = message.get_usage()
    return messages


@router.get("/{session_id}/{message_id}/", response_model=MessageRead)
async def get_message(
    message: ChatMessage = Depends(dependency=validate_message),
) -> ChatMessage:
    """
    ## Get Message Details
    Retrieves details of a specific message.

    ### Parameters
    - **session_id**: UUID of the chat session
    - **message_id**: UUID of the message

    ### Returns
    The message details

    ### Raises
    - **404**: Session or message not found
    - **400**: Message doesn't belong to session
    """
    # Get usage metrics
    message.usage = message.get_usage()
    return message


@router.patch("/{session_id}/{message_id}/", response_model=MessageRead)
async def update_message(
    message_in: MessageUpdate,
    message: ChatMessage = Depends(dependency=validate_message),
    db: AsyncSession = Depends(get_db_session),
) -> ChatMessage | None:
    """
    ## Update Message
    Updates the details of a specific message.

    ### Parameters
    - **session_id**: UUID of the chat session
    - **message_id**: UUID of the message
    - **content**: Optional new content
    - **status**: Optional new status
    - **extra_data**: Optional data updates

    ### Returns
    The updated message

    ### Raises
    - **404**: Session or message not found
    - **400**: Message doesn't belong to session
    """
    return await crud_message.update(db=db, id=message.id, obj_in=message_in)


@router.delete("/{session_id}/{message_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(
    message: ChatMessage = Depends(dependency=validate_message),
    db: AsyncSession = Depends(get_db_session),
) -> None:
    """
    ## Delete Message
    Permanently deletes a specific message.

    ### Parameters
    - **session_id**: UUID of the chat session
    - **message_id**: UUID of the message

    ### Raises
    - **404**: Session or message not found
    - **400**: Message doesn't belong to session
    """
    await crud_message.delete(db, id=message.id)
