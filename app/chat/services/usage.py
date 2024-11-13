from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.models import ChatMessage
from app.providers.models import LLMModel


class UsageTracker:
    """
    Service for tracking and analyzing usage metrics
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def update_message_usage(
        self,
        message_id: UUID,
        input_tokens: int,
        output_tokens: int,
        model: LLMModel,
    ) -> ChatMessage:
        """
        Update usage statistics for a message.
        Uses model's token costs for calculating total cost.

        Args:
            message_id: ID of the message to update
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            model: LLM model used for the message

        Returns:
            Updated ChatMessage instance

        Raises:
            HTTPException: If message not found
        """
        message = await self.db.get(ChatMessage, message_id)
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Message {message_id} not found",
            )

        # Update token counts
        message.input_tokens = input_tokens
        message.output_tokens = output_tokens

        # Calculate costs using model's rates
        message.input_cost = input_tokens * model.input_cost_per_token
        message.output_cost = output_tokens * model.output_cost_per_token

        await self.db.commit()
        await self.db.refresh(message)
        return message
