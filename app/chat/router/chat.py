from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, status
from fastapi.responses import StreamingResponse

from app.api.schemas.error import ErrorResponseModel
from app.chat.dependencies.chat import get_chat_service
from app.chat.schemas import CompletionParams
from app.chat.services.completion import ChatCompletionService
from app.chat.services.sse import SSEConnectionManager, get_sse_manager

router = APIRouter(prefix="/chat", tags=["Chat"])


# For the streaming completion
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
            "content": {
                "application/json": {
                    "examples": {
                        "Rate limit": {
                            "value": {
                                "detail": {
                                    "code": "rate_limit_error",
                                    "message": "Rate limit exceeded",
                                    "provider": "anthropic",
                                    "details": "Too many requests",
                                }
                            }
                        }
                    }
                }
            },
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Provider service unavailable",
            "model": ErrorResponseModel,
        },
    },
)
async def stream_completion(
    session_id: UUID,
    message_id: UUID,
    background_tasks: BackgroundTasks,
    params: CompletionParams = Depends(),
    sse_manager: SSEConnectionManager = Depends(get_sse_manager),
    chat_service: ChatCompletionService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    ## Stream Chat Completion

    Streams the completion for a previously created message using Server-Sent Events (SSE).

    ### Parameters
    - **session_id**: UUID of the chat session
    - **message_id**: UUID of the message to generate completion for
    - **params**: Generation parameters:
    - **max_tokens**: Maximum tokens to generate (default: 1024)
    - **temperature**: Temperature for generation (default: 0.7)

    ### Returns
    Server-sent events stream of the generated completion

    ### Raises
    - **404**: Session, message or model not found
    - **429**: Rate limit exceeded
    - **503**: Provider service unavailable
    """
    # Validate session, message, and model
    chat_session, provider, model = await chat_service.validate_message(session_id=session_id, message_id=message_id)

    provider_client = chat_service.get_provider_client(provider=provider)

    return StreamingResponse(
        sse_manager.stream_generator(
            session_id=session_id,
            generator=chat_service.generate_chat_stream(
                chat_session=chat_session,
                model=model,
                provider_client=provider_client,
                params=params,
                message_id=message_id,
            ),
            background_tasks=background_tasks,
        ),
        media_type="text/event-stream",
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
    sse_manager: SSEConnectionManager = Depends(get_sse_manager),
) -> None:
    """
    ## Stop Chat Completion Stream

    Stops an ongoing streaming completion for the specified session.
    Also cancels any ongoing API calls to the provider.

    ### Parameters
    - **session_id**: UUID of the chat session to stop streaming

    ### Returns
    No content on successful stop

    ### Raises
    - **404**: Session not found
    """
    await sse_manager.stop_stream(session_id=session_id)
