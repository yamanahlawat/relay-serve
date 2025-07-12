"""Tool call tracking utilities for managing streaming transparency."""

import json
from typing import Any, Dict

from loguru import logger


class ToolCallTracker:
    """
    Tracks tool call state during streaming to manage transparency display.
    Centralizes logic for accumulating tool arguments and managing tool call lifecycle.
    """

    def __init__(self) -> None:
        """Initialize the tool call tracker."""
        self._tool_args_accumulator: Dict[str, str] = {}
        self._active_tool_calls: Dict[str, Dict[str, Any]] = {}
        # Map part index to tool call ID for tracking tool call deltas
        self._part_index_to_tool_call_id: Dict[int, str] = {}

    def start_tool_call(self, tool_call_id: str, tool_name: str, part_index: int | None = None) -> None:
        """
        Start tracking a new tool call.

        Args:
            tool_call_id: Unique identifier for the tool call
            tool_name: Name of the tool being called
            part_index: Optional part index for mapping tool call deltas
        """
        self._tool_args_accumulator[tool_call_id] = ""
        self._active_tool_calls[tool_call_id] = {
            "tool_name": tool_name,
            "started": True,
            "completed": False,
        }

        # Map part index to tool call ID if provided
        if part_index is not None:
            self._part_index_to_tool_call_id[part_index] = tool_call_id

        logger.debug(f"Started tracking tool call: {tool_name} (ID: {tool_call_id}, part: {part_index})")

    def accumulate_args(self, tool_call_id: str, args_delta: str) -> Dict[str, Any] | None:
        """
        Accumulate tool arguments and attempt to parse as JSON.

        Args:
            tool_call_id: Tool call identifier
            args_delta: Incremental arguments string

        Returns:
            Parsed arguments dict if valid JSON, None otherwise
        """
        if tool_call_id not in self._tool_args_accumulator:
            # Initialize if not already tracking
            self._tool_args_accumulator[tool_call_id] = ""

        # Accumulate the delta
        self._tool_args_accumulator[tool_call_id] += args_delta

        # Try to parse current accumulated args
        try:
            current_args = json.loads(self._tool_args_accumulator[tool_call_id])
            logger.debug(f"Successfully parsed args for {tool_call_id}: {current_args}")
            return current_args
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON yet, that's normal during streaming
            return None

    def complete_tool_call(self, tool_call_id: str) -> None:
        """
        Mark a tool call as completed.

        Args:
            tool_call_id: Tool call identifier
        """
        if tool_call_id in self._active_tool_calls:
            self._active_tool_calls[tool_call_id]["completed"] = True
            logger.debug(f"Completed tool call: {tool_call_id}")

    def get_tool_info(self, tool_call_id: str) -> Dict[str, Any] | None:
        """
        Get information about a tracked tool call.

        Args:
            tool_call_id: Tool call identifier

        Returns:
            Tool call information dict or None if not found
        """
        return self._active_tool_calls.get(tool_call_id)

    def get_accumulated_args(self, tool_call_id: str) -> str:
        """
        Get the accumulated arguments string for a tool call.

        Args:
            tool_call_id: Tool call identifier

        Returns:
            Accumulated arguments string
        """
        return self._tool_args_accumulator.get(tool_call_id, "")

    def cleanup_tool_call(self, tool_call_id: str) -> None:
        """
        Clean up tracking data for a completed tool call.

        Args:
            tool_call_id: Tool call identifier
        """
        self._tool_args_accumulator.pop(tool_call_id, None)
        self._active_tool_calls.pop(tool_call_id, None)
        logger.debug(f"Cleaned up tool call tracking: {tool_call_id}")

    def reset(self) -> None:
        """Reset all tracking state."""
        self._tool_args_accumulator.clear()
        self._active_tool_calls.clear()
        self._part_index_to_tool_call_id.clear()
        logger.debug("Reset tool call tracker state")

    @property
    def active_tool_calls(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active tool calls."""
        return self._active_tool_calls.copy()

    def get_tool_call_id_by_part_index(self, part_index: int) -> str | None:
        """
        Get the tool call ID associated with a part index.

        Args:
            part_index: The part index from the streaming event

        Returns:
            Tool call ID if found, None otherwise
        """
        return self._part_index_to_tool_call_id.get(part_index)
