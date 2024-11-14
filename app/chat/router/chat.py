from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.error import ErrorResponseModel
from app.chat.schemas import ChatRequest, ChatResponse
from app.chat.services.completion import ChatCompletionService
from app.database.dependencies import get_db_session

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "/complete/{session_id}/",
    response_model=ChatResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Session, provider or model not found",
            "model": ErrorResponseModel,
            "content": {
                "application/json": {
                    "examples": {
                        "Session not found": {"value": {"detail": "Chat session not found"}},
                        "Provider not found": {"value": {"detail": "Provider not found"}},
                    }
                }
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid request parameters",
            "model": ErrorResponseModel,
            "content": {
                "application/json": {
                    "examples": {"Invalid parameters": {"value": {"detail": "Invalid request parameters"}}}
                }
            },
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
            "content": {
                "application/json": {
                    "examples": {
                        "Connection error": {
                            "value": {
                                "detail": {
                                    "code": "connection_error",
                                    "message": "Failed to connect to Anthropic API",
                                    "provider": "anthropic",
                                    "details": "Connection error",
                                }
                            }
                        }
                    }
                }
            },
        },
    },
)
async def chat_complete(
    session_id: UUID,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db_session),
) -> ChatResponse | StreamingResponse:
    """
    ## Generate a Chat Completion for a Specific Session

    Creates a new message in the chat session and generates a response using the specified LLM provider. Supports both streaming and non-streaming responses.

    ### Parameters

    - **session_id**: UUID of the chat session
    - **request**: Chat completion request containing:
    - **provider_id**: UUID of the LLM provider
    - **llm_model_id**: UUID of the model to use
    - **prompt**: Input text prompt
    - **stream**: Whether to stream the response (default: False)
    - **max_tokens**: Maximum tokens to generate (default: 1024)
    - **temperature**: Temperature for generation (default: 0.7)
    - **parent_id**: Optional ID of the parent message for threading

    ### Returns

    - If `stream=False`: Generated text response with usage metrics
    - If `stream=True`: Server-sent events stream of generated text chunks

    ### Raises

    - **404**: Session, provider or model not found
    - **400**: Invalid request parameters or provider
    - **429**: Rate limit exceeded
    - **503**: Provider service unavailable
    - **500**: Unexpected server error
    """
    service = ChatCompletionService(db=db)
    # Validate request and get required models
    chat_session, provider, model = await service.validate_request(session_id=session_id, request=request)
    # Create user message
    user_message = await service.create_user_message(
        session_id=session_id,
        content=request.prompt,
        provider=provider,
        model=model,
        parent_id=request.parent_id,
    )

    # Get provider client
    provider_client = service.get_provider_client(provider=provider, user_message_id=user_message.id)

    if request.stream:
        return StreamingResponse(
            service.generate_stream(
                chat_session=chat_session,
                model=model,
                provider_client=provider_client,
                request=request,
                user_message=user_message,
            ),
            media_type="text/event-stream",
        )
    else:
        return await service.generate_complete(
            chat_session=chat_session,
            model=model,
            provider_client=provider_client,
            request=request,
            user_message=user_message,
        )
