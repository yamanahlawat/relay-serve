"""Modern chat router using pydantic_ai and mem0."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.ai.chat import chat_service
from app.api.schemas.error import ErrorResponseModel
from app.chat.dependencies.session import get_chat_session_service
from app.chat.schemas.chat import CompletionParams
from app.chat.services.session import ChatSessionService

router = APIRouter(prefix="/ai", tags=["AI Chat"])


@router.get(
    "/chat/{session_id}/stream",
    response_class=StreamingResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Session not found",
            "model": ErrorResponseModel,
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "AI service unavailable",
            "model": ErrorResponseModel,
        },
    },
)
async def stream_chat(
    session_id: UUID,
    message: str,
    user_id: str,
    params: CompletionParams = Depends(),
    session_service: ChatSessionService = Depends(get_chat_session_service),
) -> StreamingResponse:
    """
    ## Stream Chat Completion with AI

    Streams the completion using pydantic_ai and mem0 for memory management.

    ### Parameters
    - **session_id**: UUID of the chat session
    - **message**: The message to send to the AI
    - **user_id**: User identifier for memory management
    - **params**: Generation parameters

    ### Returns
    Server-sent events stream of the generated completion

    ### Raises
    - **404**: Session not found
    - **503**: AI service unavailable
    """
    try:
        # Get session with provider and model information
        session = await session_service.get_active_session(session_id=session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )

        # Stream the response
        async def generate_stream():
            """Generate the streaming response."""
            try:
                async for chunk in chat_service.stream_response(
                    provider=session.provider,
                    model=session.llm_model,
                    user_id=user_id,
                    session_id=str(session_id),
                    message=message,
                    system_prompt=session.system_context,
                    temperature=params.temperature,
                    max_tokens=params.max_tokens,
                ):
                    yield f"data: {chunk}\\n\\n"
            except Exception as e:
                yield f"data: Error: {str(e)}\\n\\n"
            finally:
                yield "data: [DONE]\\n\\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to generate response: {str(e)}",
        ) from e


@router.post(
    "/chat/{session_id}/complete",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Session not found",
            "model": ErrorResponseModel,
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "AI service unavailable",
            "model": ErrorResponseModel,
        },
    },
)
async def complete_chat(
    session_id: UUID,
    message: str,
    user_id: str,
    params: CompletionParams = Depends(),
    session_service: ChatSessionService = Depends(get_chat_session_service),
) -> dict[str, str]:
    """
    ## Complete Chat (Non-streaming)

    Generates a complete response using pydantic_ai and mem0.

    ### Parameters
    - **session_id**: UUID of the chat session
    - **message**: The message to send to the AI
    - **user_id**: User identifier for memory management
    - **params**: Generation parameters

    ### Returns
    Complete response from the AI

    ### Raises
    - **404**: Session not found
    - **503**: AI service unavailable
    """
    try:
        # Get session with provider and model information
        session = await session_service.get_active_session(session_id=session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )

        # Generate complete response
        response = await chat_service.generate_response(
            provider=session.provider,
            model=session.llm_model,
            user_id=user_id,
            session_id=str(session_id),
            message=message,
            system_prompt=session.system_context,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
        )

        return {"response": response}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to generate response: {str(e)}",
        ) from e


@router.get(
    "/chat/{session_id}/history",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Session not found",
            "model": ErrorResponseModel,
        },
    },
)
async def get_chat_history(
    session_id: UUID,
    user_id: str,
    limit: int = 50,
) -> dict[str, list[dict[str, str]]]:
    """
    ## Get Chat History

    Retrieves the conversation history for a session using mem0.

    ### Parameters
    - **session_id**: UUID of the chat session
    - **user_id**: User identifier for memory management
    - **limit**: Maximum number of messages to return

    ### Returns
    Conversation history
    """
    try:
        history = await chat_service.get_conversation_history(
            user_id=user_id,
            session_id=str(session_id),
            limit=limit,
        )

        return {"history": history}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to get chat history: {str(e)}",
        ) from e


@router.delete(
    "/chat/{session_id}/clear",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Session not found",
            "model": ErrorResponseModel,
        },
    },
)
async def clear_chat_session(
    session_id: UUID,
    user_id: str,
) -> None:
    """
    ## Clear Chat Session

    Clears all memories for a specific chat session.

    ### Parameters
    - **session_id**: UUID of the chat session to clear
    - **user_id**: User identifier for memory management

    ### Returns
    No content on successful clear
    """
    try:
        success = await chat_service.clear_session(
            user_id=user_id,
            session_id=str(session_id),
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Failed to clear session",
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to clear session: {str(e)}",
        ) from e
