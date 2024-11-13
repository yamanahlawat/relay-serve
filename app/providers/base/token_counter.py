from abc import ABC, abstractmethod
from typing import Sequence

from app.chat.models import ChatMessage


class TokenCounterBase(ABC):
    """
    Base class for token counting implementations
    """

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in a single text string
        """
        pass

    @abstractmethod
    async def count_message_tokens(self, messages: Sequence[ChatMessage], prompt: str) -> tuple[int, int]:
        """
        Count tokens in a message sequence plus new prompt
        Returns tuple of (prompt_tokens, context_tokens)
        """
        pass
