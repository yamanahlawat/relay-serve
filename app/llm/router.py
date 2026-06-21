"""Chat router using pydantic_ai with message-based streaming."""

from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, status
from fastapi.responses import StreamingResponse

from app.api.schemas.error import ErrorResponseModel
from app.llm.dependencies.chat import ChatServiceDep
from app.llm.schemas.chat import CompletionParams
from app.llm.services import SSEManagerDep
from app.session.dependencies import ChatSessionServiceDep

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.get(
    "/complete/{session_id}/{message_id}/stream",
    response_class=StreamingResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Session, message or model not found",
            "model": ErrorResponseModel,
        },
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "description": "Rate limit exceeded",
            "model": ErrorResponseModel,
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "AI service unavailable",
            "model": ErrorResponseModel,
        },
    },
)
async def stream_completion(
    session_id: UUID,
    message_id: UUID,
    background_tasks: BackgroundTasks,
    session_service: ChatSessionServiceDep,
    chat_service: ChatServiceDep,
    sse_manager: SSEManagerDep,
    params: Annotated[CompletionParams, Depends()],
) -> StreamingResponse:
    """
    ## Stream Chat Completion

    Streams the completion for a previously created message using Server-Sent Events (SSE).

    ### Parameters
    - **session_id**: UUID of the chat session
    - **message_id**: UUID of the message to generate completion for
    - **params**: Generation parameters (temperature, max_tokens)

    ### Returns
    Server-sent events stream of the generated completion

    ### Raises
    - **404**: Session, message or model not found
    - **429**: Rate limit exceeded
    - **503**: AI service unavailable
    """
    # Get session with provider and model information
    session = await session_service.get_active_session(session_id=session_id)

    # Create the AI response generator
    async def streaming_response() -> AsyncGenerator[str, None]:
        """Generate AI responses."""
        async for chunk in chat_service.stream_response(
            provider=session.provider,
            model=session.llm_model,
            session_id=session_id,
            message_id=message_id,
            system_prompt=session.system_context,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            top_p=params.top_p,
        ):
            yield chunk

    # Use SSE manager to handle the streaming with Redis Pub/Sub
    return StreamingResponse(
        sse_manager.stream_generator(
            session_id=session_id,
            generator=streaming_response(),
            background_tasks=background_tasks,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


@router.post(
    "/complete/{session_id}/stop",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Session not found",
            "model": ErrorResponseModel,
        },
    },
)
async def stop_completion(
    session_id: UUID,
    sse_manager: SSEManagerDep,
) -> None:
    """
    ## Stop Chat Completion Stream

    Stops an ongoing streaming completion for the specified session.
    Uses Redis-based cancellation to stop the stream across distributed instances.

    ### Parameters
    - **session_id**: UUID of the chat session to stop streaming

    ### Returns
    No content on successful stop

    ### Raises
    - **404**: Session not found
    """
    await sse_manager.stop_stream(session_id=session_id)
