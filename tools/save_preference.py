"""Tool: save_preference — Allow the agent to proactively save user preferences."""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from agent.utils import get_logger

logger = get_logger(__name__)


class SavePreferenceInput(BaseModel):
    """Arguments for the save_preference tool."""
    content: str = Field(
        ...,
        description="The preference or fact to save, e.g. 'User prefers concise answers' or 'User is a software engineer'.",
    )
    preference_type: str = Field(
        "personal",
        description="Category: personal, style, topic, or constraint.",
    )
    confidence: float = Field(
        0.9,
        description="Confidence level from 0.0 to 1.0. Use 0.9+ for explicit statements, 0.6-0.8 for inferred ones.",
    )


def make_save_preference(user_id: str):
    """Return a save_preference tool bound to the given user_id."""

    @tool("save_preference", args_schema=SavePreferenceInput, return_direct=False)
    def save_preference(
        content: str,
        preference_type: str = "personal",
        confidence: float = 0.9,
    ) -> str:
        """
        Save a user preference or fact to long-term memory.

        Call this when the user shares something worth remembering
        (name, preferences, constraints, interests, communication style, etc.).
        Do NOT call for one-off task requests like "Translate hello".

        Examples:
          - "I'm a nurse" → save_preference("User is a nurse", "personal")
          - "I like hiking" → save_preference("User likes hiking", "personal")
          - "Keep it short" → save_preference("User prefers concise responses", "style")
          - "Don't use markdown" → save_preference("User dislikes markdown formatting", "constraint")
        """
        try:
            from agent.memory.memory_manager import get_memory_manager
            from agent.interfaces import StorageType
            import uuid

            manager = get_memory_manager()
            key = f"pref_tool_{uuid.uuid4().hex[:8]}"

            # Use _run_async helper to bridge sync tool -> async storage
            from agent.memory.memory_manager import _run_async
            _run_async(
                manager.store_user_memory(
                    user_id=user_id,
                    key=key,
                    content=content,
                    metadata={
                        "preference_type": preference_type,
                        "confidence": confidence,
                        "source": "tool_call",
                    },
                    document_type=StorageType.USER_PREFERENCE,
                )
            )
            logger.info(f"Tool saved preference for {user_id}: {content[:60]}")
            return f"Saved preference: [{preference_type}] {content}"
        except Exception as e:
            logger.error(f"save_preference failed: {e}")
            return f"Failed to save preference: {e}"

    return save_preference
