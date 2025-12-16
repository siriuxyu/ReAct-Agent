"""
Memory Module

Provides memory management capabilities:
- ContextMemoryManager: Legacy memory manager (see manager.py)
- MemoryManager: Long-term memory with user-specific namespaces (see memory_manager.py)
- Uses VectorStorageBackend (ChromaDB) for persistent storage
"""

from .manager import ContextMemoryManager
from .memory_manager import (
    LangMemManager,
    get_langmem_manager,
    get_memory_store,
    get_memory_tools,
    is_langmem_enabled,
    is_storage_available,
)

__all__ = [
    "ContextMemoryManager",
    "LangMemManager",
    "get_langmem_manager",
    "get_memory_store",
    "get_memory_tools",
    "is_langmem_enabled",
    "is_storage_available",
]
