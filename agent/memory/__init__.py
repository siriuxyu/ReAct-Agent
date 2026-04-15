"""
Memory Module

Provides memory management capabilities:
- ContextMemoryManager: Legacy memory manager (see manager.py)
- MemoryManager: Long-term memory with user-specific namespaces (see memory_manager.py)
- Uses VectorStorageBackend (ChromaDB) for persistent storage
"""

from .manager import ContextMemoryManager
from .memory_manager import (
    MemoryManager,
    get_memory_manager,
    get_memory_tools,
    is_memory_enabled,
    is_storage_available,
)
from .profile_store import ProfileMemoryRecord, build_profile_memory_block, search_profile_memories
from .recall import build_session_recall_block
from .runtime_recall import RuntimeMemoryContext, build_runtime_memory_context, build_runtime_recall_context
from .session_store import SessionStore, get_session_store
from .task_scratchpad import TaskScratchpad, build_task_scratchpad

__all__ = [
    "ContextMemoryManager",
    "MemoryManager",
    "ProfileMemoryRecord",
    "RuntimeMemoryContext",
    "SessionStore",
    "TaskScratchpad",
    "build_session_recall_block",
    "build_profile_memory_block",
    "build_runtime_memory_context",
    "build_runtime_recall_context",
    "build_task_scratchpad",
    "get_memory_manager",
    "get_memory_tools",
    "get_session_store",
    "is_memory_enabled",
    "is_storage_available",
    "search_profile_memories",
]
