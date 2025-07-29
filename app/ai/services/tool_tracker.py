"""Tool call tracking utilities for managing streaming transparency."""

from typing import Any

from loguru import logger


class ToolCallTracker:
    """
    Tool call tracker for managing tool call lifecycle.
    Tracks basic tool call information without complex argument accumulation.
    """

    def __init__(self) -> None:
        """Initialize the tool call tracker."""
        self._active_tool_calls: dict[str, dict[str, Any]] = {}
        # Map part index to tool call ID for tracking tool call deltas
        self._part_index_to_tool_call_id: dict[int, str] = {}

    def start_tool_call(self, tool_call_id: str, tool_name: str, part_index: int | None = None) -> None:
        """
        Start tracking a new tool call.

        Args:
            tool_call_id: Unique identifier for the tool call
            tool_name: Name of the tool being called
            part_index: Optional part index for mapping tool call deltas
        """
        self._active_tool_calls[tool_call_id] = {
            "tool_name": tool_name,
            "started": True,
            "completed": False,
        }

        # Map part index to tool call ID if provided
        if part_index is not None:
            self._part_index_to_tool_call_id[part_index] = tool_call_id

        logger.debug(f"Started tracking tool call: {tool_name} (ID: {tool_call_id}, part: {part_index})")

    def complete_tool_call(self, tool_call_id: str) -> None:
        """
        Mark a tool call as completed.

        Args:
            tool_call_id: Tool call identifier
        """
        if tool_call_id in self._active_tool_calls:
            self._active_tool_calls[tool_call_id]["completed"] = True
            logger.debug(f"Completed tool call: {tool_call_id}")

    def get_tool_info(self, tool_call_id: str) -> dict[str, Any] | None:
        """
        Get information about a tracked tool call.

        Args:
            tool_call_id: Tool call identifier

        Returns:
            Tool call information dict or None if not found
        """
        return self._active_tool_calls.get(tool_call_id)

    def cleanup_tool_call(self, tool_call_id: str) -> None:
        """
        Clean up tracking data for a completed tool call.

        Args:
            tool_call_id: Tool call identifier
        """
        self._active_tool_calls.pop(tool_call_id, None)
        logger.debug(f"Cleaned up tool call tracking: {tool_call_id}")

    def reset(self) -> None:
        """Reset all tracking state."""
        self._active_tool_calls.clear()
        self._part_index_to_tool_call_id.clear()
        logger.debug("Reset tool call tracker state")

    def get_tool_call_id_by_part_index(self, part_index: int) -> str | None:
        """
        Get the tool call ID associated with a part index.

        Args:
            part_index: The part index from the streaming event

        Returns:
            Tool call ID if found, None otherwise
        """
        return self._part_index_to_tool_call_id.get(part_index)
