"""
Memory Synapse - Shared KV Cache Management
Provides unified memory/caching for agent-to-agent communication.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime, timedelta
import hashlib


@dataclass
class CacheEntry:
    """Single cache entry with TTL support."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: Optional[int] = None  # None = no expiry
    
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def touch(self):
        """Update access timestamp."""
        self.accessed_at = datetime.now()
        self.access_count += 1


@dataclass
class MemorySynapse:
    """
    Shared KV cache for agent communication.
    Supports TTL, LRU eviction, and namespace isolation.
    """
    
    _cache: Dict[str, CacheEntry] = field(default_factory=dict)
    _namespaces: Dict[str, Set[str]] = field(default_factory=dict)
    _max_size: int = 1000
    _default_ttl: int = 3600  # 1 hour default
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self._max_size = max_size
        self._default_ttl = default_ttl
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate fully-qualified cache key."""
        return f"{namespace}:{key}"
    
    def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl_seconds: Optional[int] = None
    ):
        """
        Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
            namespace: Namespace for isolation
            ttl_seconds: Time-to-live in seconds (None = use default)
        """
        fq_key = self._generate_key(namespace, key)
        
        # Evict if at capacity
        if fq_key not in self._cache and len(self._cache) >= self._max_size:
            self._evict_lru()
        
        entry = CacheEntry(
            key=fq_key,
            value=value,
            ttl_seconds=ttl_seconds or self._default_ttl
        )
        
        self._cache[fq_key] = entry
        
        # Track namespace
        if namespace not in self._namespaces:
            self._namespaces[namespace] = set()
        self._namespaces[namespace].add(fq_key)
    
    def get(
        self,
        key: str,
        namespace: str = "default",
        default: Any = None
    ) -> Any:
        """
        Retrieve a value from the cache.
        
        Args:
            key: Cache key
            namespace: Namespace
            default: Default if not found
            
        Returns:
            Cached value or default
        """
        fq_key = self._generate_key(namespace, key)
        
        if fq_key not in self._cache:
            return default
            
        entry = self._cache[fq_key]
        
        # Check expiry
        if entry.is_expired():
            self.delete(key, namespace)
            return default
        
        # Update access
        entry.touch()
        
        return entry.value
    
    def delete(self, key: str, namespace: str = "default"):
        """Delete a key from cache."""
        fq_key = self._generate_key(namespace, key)
        
        if fq_key in self._cache:
            del self._cache[fq_key]
        
        if namespace in self._namespaces:
            self._namespaces[namespace].discard(fq_key)
    
    def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if key exists and is not expired."""
        fq_key = self._generate_key(namespace, key)
        
        if fq_key not in self._cache:
            return False
            
        entry = self._cache[fq_key]
        
        if entry.is_expired():
            self.delete(key, namespace)
            return False
        
        return True
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Find LRU entry (oldest accessed_at)
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].accessed_at
        )
        
        entry = self._cache[lru_key]
        
        # Remove from namespace tracking
        namespace = lru_key.split(":")[0] if ":" in lru_key else "default"
        if namespace in self._namespaces:
            self._namespaces[namespace].discard(lru_key)
        
        del self._cache[lru_key]
    
    def clear_namespace(self, namespace: str):
        """Clear all entries in a namespace."""
        if namespace not in self._namespaces:
            return
            
        for fq_key in list(self._namespaces[namespace]):
            if fq_key in self._cache:
                del self._cache[fq_key]
        
        self._namespaces[namespace].clear()
    
    def clear_all(self):
        """Clear the entire cache."""
        self._cache.clear()
        self._namespaces.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        # Clean expired entries first
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for k in expired_keys:
            self.delete(k.split(":")[1], k.split(":")[0])
        
        return {
            "total_entries": len(self._cache),
            "max_size": self._max_size,
            "namespaces": list(self._namespaces.keys()),
            "total_accesses": sum(e.access_count for e in self._cache.values()),
            "by_namespace": {
                ns: len(keys) 
                for ns, keys in self._namespaces.items()
            }
        }
    
    # === Agent Council Integration ===
    
    def inject_council_context(
        self,
        council_id: str,
        agent_id: str,
        context: str
    ):
        """Inject context from a council agent into shared memory."""
        key = f"council:{council_id}:agent:{agent_id}:context"
        self.set(key, context, namespace="council", ttl_seconds=1800)
    
    def get_council_context(self, council_id: str) -> List[str]:
        """Get all context entries for a council."""
        ns = "council"
        prefix = f"council:{council_id}:agent:"
        
        context = []
        for key in self._namespaces.get(ns, []):
            if key.startswith(prefix) and key.endswith(":context"):
                entry = self._cache.get(key)
                if entry and not entry.is_expired():
                    context.append(entry.value)
        
        return context
    
    def inject_consensus(
        self,
        council_id: str,
        consensus: str
    ):
        """Inject final consensus result."""
        key = f"council:{council_id}:consensus"
        self.set(key, consensus, namespace="council", ttl_seconds=3600)
    
    def get_consensus(self, council_id: str) -> Optional[str]:
        """Get consensus result for a council."""
        return self.get(
            f"council:{council_id}:consensus",
            namespace="council"
        )
    
    # === Tool Result Caching ===
    
    def cache_tool_result(
        self,
        tool_name: str,
        tool_input: str,
        result: Any,
        ttl_seconds: int = 300
    ):
        """Cache tool execution results."""
        input_hash = hashlib.md5(tool_input.encode()).hexdigest()[:16]
        key = f"tool:{tool_name}:{input_hash}"
        self.set(key, result, namespace="tools", ttl_seconds=ttl_seconds)
    
    def get_cached_tool_result(
        self,
        tool_name: str,
        tool_input: str
    ) -> Optional[Any]:
        """Get cached tool result if available."""
        input_hash = hashlib.md5(tool_input.encode()).hexdigest()[:16]
        key = f"tool:{tool_name}:{input_hash}"
        return self.get(key, namespace="tools")


# Global instance
_synapse: Optional[MemorySynapse] = None


def get_synapse() -> MemorySynapse:
    """Get or create the global synapse instance."""
    global _synapse
    if _synapse is None:
        _synapse = MemorySynapse()
    return _synapse