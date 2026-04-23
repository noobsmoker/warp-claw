"""
Warp Cortex - Memory-efficient multi-agent AI architecture.

This module implements the core components of the Warp-Cortex architecture:
- Singleton weight sharing for O(1) model memory
- Topological synapse for context compression
- KV-cache sparsification for reduced memory usage
- Referential injection for asynchronous agent updates
"""

__version__ = "0.1.0"

from .singleton import WarpCortexSingleton
from .synapse import TopologicalSynapse
from .sparsification import SparseAttention
from .injection import ReferentialInjection
from .manager import WarpCortexManager

__all__ = [
    "WarpCortexSingleton",
    "TopologicalSynapse",
    "SparseAttention",
    "ReferentialInjection",
    "WarpCortexManager",
]