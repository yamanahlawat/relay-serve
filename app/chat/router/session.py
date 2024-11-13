from typing import Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.crud import crud_session
from app.chat.schemas.session import SessionCreate, SessionRead, SessionUpdate
from app.database.dependencies import get_db_session
from app.providers.crud import crud_model, crud_provider

router = APIRouter(prefix="/sessions", tags=["Chat Sessions"])


@router.post("/", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
async def create_chat_session(
    session_in: SessionCreate,
    db: AsyncSession = Depends(get_db_session),
) -> SessionRead:
    """
    Creates a new chat session with the specified provider and model.
    Args:
        - **title**: Session title (1-255 chars)
        - **system_context**: Optional system instructions
        - **provider_id**: UUID of the LLM provider
        - **llm_model_id**: UUID of the model to use
        - **extra_data**: Optional additional data
    Returns:
        The created chat session
    Raises:
        - 404: Provider or model not found
        - 400: Invalid provider/model combination
    """
    # Verify provider exists
    provider = await crud_provider.get(db=db, id=session_in.provider_id)
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Provider {session_in.provider_id} not found"
        )

    # Verify model exists and belongs to provider
    model = await crud_model.get(db=db, id=session_in.llm_model_id)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {session_in.llm_model_id} not found")
    if model.provider_id != provider.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Model does not belong to the specified provider"
        )

    session = await crud_session.create(db=db, obj_in=session_in)
    return session


@router.get("/", response_model=list[SessionRead])
async def list_chat_sessions(
    offset: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db_session),
) -> Sequence[SessionRead]:
    """
    List chat sessions.
    Retrieves a paginated list of chat sessions, ordered by last message time.
    Args:
        - **offset**: Number of sessions to skip (default: 0)
        - **limit**: Maximum number of sessions to return (default: 10)
    Returns:
        List of chat sessions
    """
    return await crud_session.filter(db=db, offset=offset, limit=limit)


@router.get("/{session_id}/", response_model=SessionRead)
async def get_chat_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> SessionRead:
    """
    Get chat session details.
    Retrieves details of a specific chat session.
    Args:
        - **session_id**: UUID of the session
    Returns:
        The chat session details
    Raises:
        - 404: Session not found
    """
    session = await crud_session.get(db=db, id=session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat session {session_id} not found")
    return session


@router.patch("/{session_id}/", response_model=SessionRead)
async def update_chat_session(
    session_id: UUID,
    session_in: SessionUpdate,
    db: AsyncSession = Depends(get_db_session),
) -> SessionRead:
    """
    Update chat session.
    Updates the details of a specific chat session.
    Args:
        - **session_id**: UUID of the session
        - **title**: Optional new title
        - **status**: Optional new status
        - **system_context**: Optional new system context
        - **extra_data**: Optional data updates
    Returns:
        The updated chat session
    Raises:
        - 404: Session not found
    """
    session = await crud_session.get(db=db, id=session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat session {session_id} not found")

    session = await crud_session.update(db=db, id=session_id, obj_in=session_in)
    return session


@router.delete("/{session_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> None:
    """
    Delete chat session.
    Permanently deletes a chat session and all its messages.
    Args:
        - **session_id**: UUID of the session to delete
    Raises:
        - 404: Session not found
    """
    session = await crud_session.get(db=db, id=session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat session {session_id} not found")

    await crud_session.delete(db=db, id=session_id)
