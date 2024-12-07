from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.error import ErrorResponseModel
from app.chat.schemas import CompletionParams, CompletionRequest, CompletionResponse
from app.chat.services.completion import ChatCompletionService
from app.chat.services.sse import SSEConnectionManager, get_sse_manager
from app.database.dependencies import get_db_session

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
    db: AsyncSession = Depends(get_db_session),
    sse_manager: SSEConnectionManager = Depends(get_sse_manager),
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
    service = ChatCompletionService(db=db)
    chat_session, provider, model = await service.validate_message(session_id=session_id, message_id=message_id)

    provider_client = service.get_provider_client(provider=provider)

    return StreamingResponse(
        sse_manager.stream_generator(
            session_id=session_id,
            generator=service.generate_stream(
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


# For the non streaming completion
@router.post(
    "/complete/{session_id}",
    response_model=CompletionResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Session, provider or model not found",
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
async def complete(
    session_id: UUID,
    request: CompletionRequest,
    db: AsyncSession = Depends(get_db_session),
) -> CompletionResponse:
    """
    ## Generate Chat Completion

    Generates a completion for the given prompt in a single request.

    ### Parameters
    - **session_id**: UUID of the chat session
    - **request**: Completion request:
    - **provider_id**: UUID of the LLM provider to use
    - **llm_model_id**: UUID of the model to use
    - **prompt**: Text prompt to generate completion for
    - **parent_id**: Optional parent message ID for threading
    - **max_tokens**: Maximum tokens to generate (default: 1024)
    - **temperature**: Temperature for generation (default: 0.7)

    ### Returns
    The generated completion with usage statistics

    ### Raises
    - **404**: Session, provider or model not found
    - **429**: Rate limit exceeded
    - **503**: Provider service unavailable
    """
    service = ChatCompletionService(db=db)

    # Validate request and get required models
    chat_session, provider, model = await service.validate_request(
        session_id=session_id,
    )

    # Create user message
    user_message = await service.create_user_message(
        session_id=session_id,
        content=request.prompt,
        provider=provider,
        model=model,
        parent_id=request.parent_id,
    )

    # Get provider client
    provider_client = service.get_provider_client(provider=provider)

    # Generate completion
    return await service.generate_complete(
        chat_session=chat_session,
        model=model,
        provider_client=provider_client,
        request=request,
        user_message=user_message,
    )
