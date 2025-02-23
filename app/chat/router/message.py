from typing import Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.chat.dependencies import get_chat_message_service
from app.chat.exceptions import (
    InvalidMessageSessionException,
    InvalidParentMessageSessionException,
    MessageNotFoundException,
    ParentMessageNotFoundException,
)
from app.chat.models import ChatMessage
from app.chat.schemas import MessageRead, MessageUpdate
from app.chat.schemas.message import MessageCreate
from app.chat.services import ChatMessageService

router = APIRouter(prefix="/messages", tags=["Chat Messages"])


@router.post("/{session_id}/", response_model=MessageRead, status_code=status.HTTP_201_CREATED)
async def create_message(
    session_id: UUID,
    message_in: MessageCreate,
    service: ChatMessageService = Depends(get_chat_message_service),
) -> ChatMessage:
    """
    Create a new message in the specified chat session with support for file attachments.

    ### Parameters
    - **session_id**: UUID of the chat session
    - **Form Data**:
        - **content**: Message content (required)
        - **role**: Message role (user/assistant/system)
        - **status**: Message status
        - **parent_id**: Optional parent message ID for threading
        - **usage**: JSON string of usage statistics
        - **attachments**: List of file attachments
        - **extra_data**: JSON string of additional metadata

    ### Returns
    The created message with complete details

    ### Raises
    - **404**: Session not found
    - **404**: Parent message not found (if parent_id provided)
    - **400**: Invalid role for message
    - **400**: Invalid form data format
    - **413**: File too large
    """
    try:
        return await service.create_message(
            message_in=message_in,
            session_id=session_id,
        )
    except ParentMessageNotFoundException as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error.message,
        )
    except InvalidParentMessageSessionException as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error.message,
        )


@router.get("/{session_id}/", response_model=list[MessageRead])
async def list_session_messages(
    session_id: UUID,
    offset: int = 0,
    limit: int = 10,
    service: ChatMessageService = Depends(get_chat_message_service),
) -> Sequence[ChatMessage]:
    """
    ## List Session Messages
    Retrieves messages from a specific chat session.

    ### Parameters
    - **session_id**: UUID of the chat session
    - **offset**: Number of messages to skip (default: 0)
    - **limit**: Maximum number of messages to return (default: 10)

    ### Returns
    List of messages in chronological order

    ### Raises
    - **404**: Session not found
    """
    return await service.list_messages(session_id=session_id, offset=offset, limit=limit)


@router.get("/{session_id}/{message_id}/", response_model=MessageRead)
async def get_message(
    session_id: UUID,
    message_id: UUID,
    service: ChatMessageService = Depends(get_chat_message_service),
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
    try:
        return await service.get_message(message_id=message_id, session_id=session_id)
    except MessageNotFoundException as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error.message,
        )
    except InvalidMessageSessionException as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error.message,
        )


@router.patch("/{session_id}/{message_id}/", response_model=MessageRead)
async def update_message(
    message_in: MessageUpdate,
    message_id: UUID,
    session_id: UUID,
    service: ChatMessageService = Depends(get_chat_message_service),
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
    try:
        return await service.update_message(session_id=session_id, message_id=message_id, message_in=message_in)
    except MessageNotFoundException as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error.message,
        )
    except InvalidMessageSessionException as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error.message,
        )


@router.delete("/{session_id}/{message_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(
    message_id: UUID,
    session_id: UUID,
    service: ChatMessageService = Depends(get_chat_message_service),
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
    try:
        await service.delete_message(session_id=session_id, message_id=message_id)
    except MessageNotFoundException as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error.message,
        )
    except InvalidMessageSessionException as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error.message,
        )


@router.delete("/bulk/", status_code=status.HTTP_204_NO_CONTENT)
async def bulk_delete_messages(
    message_ids: list[UUID],
    service: ChatMessageService = Depends(get_chat_message_service),
) -> None:
    """
    ## Bulk Delete Messages
    Permanently deletes multiple messages by their IDs.
    ### Parameters
    - **message_ids**: List of UUIDs for the messages to delete.
    """
    await service.bulk_delete_messages(message_ids=message_ids)
