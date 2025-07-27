from typing import Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.chat.dependencies import get_chat_session_service
from app.chat.exceptions.session import SessionNotFoundException
from app.chat.models import ChatSession
from app.chat.schemas import SessionCreate, SessionRead, SessionUpdate
from app.chat.services import ChatSessionService

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post("/", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
async def create_chat_session(
    session_in: SessionCreate,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSession:
    """
    ## Create Chat Session
    Creates a new chat session with the specified provider and model.

    ### Parameters
    - **title**: Session title (1-255 chars)
    - **system_context**: Optional system instructions
    - **provider_id**: UUID of the LLM provider
    - **llm_model_id**: UUID of the model to use
    - **extra_data**: Optional additional data

    ### Returns
    The created chat session

    ### Raises
    - **404**: Session not found
    """
    return await service.create_session(session_in=session_in)


@router.get("/", response_model=list[SessionRead])
async def list_chat_sessions(
    offset: int = 0,
    limit: int = 10,
    title: str | None = None,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> Sequence[ChatSession]:
    """
    ## List Chat Sessions
    Retrieves a list of all chat sessions.

    ### Parameters
    - **offset**: Number of sessions to skip (default: 0)
    - **limit**: Maximum number of sessions to return (default: 50)

    ### Returns
    List of chat sessions

    ### Raises
    - **400**: Invalid request parameters
    """
    return await service.list_sessions(offset=offset, limit=limit, title=title)


@router.get("/{session_id}/", response_model=SessionRead)
async def get_chat_session(
    session_id: UUID,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSession:
    """
    ## Get Chat Session
    Retrieves details of a specific chat session.

    ### Parameters
    - **session_id**: UUID of the chat session

    ### Returns
    The chat session details

    ### Raises
    - **404**: Session not found
    """
    try:
        return await service.get_session(session_id=session_id)
    except SessionNotFoundException as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error.message)


@router.patch("/{session_id}/", response_model=SessionRead)
async def update_chat_session(
    session_in: SessionUpdate,
    session_id: UUID,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSession | None:
    """
    ## Update Chat Session
    Updates the details of a specific chat session.

    ### Parameters
    - **session_id**: UUID of the chat session
    - **title**: Optional new title
    - **system_context**: Optional new system instructions
    - **extra_data**: Optional additional data

    ### Returns
    The updated chat session

    ### Raises
    - **404**: Session not found
    - **400**: Invalid request parameters
    """
    try:
        return await service.update_session(session_id=session_id, session_in=session_in)
    except SessionNotFoundException as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error.message)


@router.delete("/{session_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    session_id: UUID,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> None:
    """
    ## Delete Chat Session
    Permanently deletes a specific chat session.

    ### Parameters
    - **session_id**: UUID of the chat session

    ### Raises
    - **404**: Session not found
    """
    try:
        await service.delete_session(session_id=session_id)
    except SessionNotFoundException as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error.message)
