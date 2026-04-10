"""
Base Tool Interface
All Warp-Claw tools must implement this interface.
"""

import asyncio
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: int = 0
    tokens_estimate: int = 0


@dataclass
class BaseTool(ABC):
    """
    Abstract base class for all tools.
    All tools must inherit from this and implement execute().
    """
    
    name: str = ""
    description: str = ""
    version: str = "0.1.0"
    
    # Tool metadata
    category: str = "general"  # general, code, search, file, memory
    tags: List[str] = []
    requires_permissions: List[str] = []
    
    # Execution defaults
    default_timeout: int = 30  # seconds
    max_retries: int = 0
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given arguments.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            ToolResult with success/status
        """
        pass
    
    async def execute_with_timeout(
        self,
        timeout: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        """Execute with timeout protection."""
        timeout = timeout or self.default_timeout
        start_time = datetime.now()
        
        try:
            result = await asyncio.wait_for(
                self.execute(**kwargs),
                timeout=timeout
            )
            
            # Calculate execution time
            execution_time = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            result.execution_time_ms = execution_time
            
            return result
            
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                result=None,
                error=f"Execution timeout after {timeout}s"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Return the tool's JSON schema for OpenAI compatibility.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    
    def estimate_tokens(self, input_text: str) -> int:
        """Estimate token cost for input."""
        # Rough estimate: ~4 chars per token
        return len(input_text) // 4
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "tags": self.tags,
            "default_timeout": self.default_timeout
        }


class ToolRegistry:
    """Registry for available tools."""
    
    _tools: Dict[str, BaseTool] = {}
    
    @classmethod
    def register(cls, tool: BaseTool):
        """Register a tool."""
        cls._tools[tool.name] = tool
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return cls._tools.get(name)
    
    @classmethod
    def list_tools(cls) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [t.get_metadata() for t in cls._tools.values()]
    
    @classmethod
    def get_schemas(cls) -> List[Dict[str, Any]]:
        """Get all tool schemas."""
        return [t.get_schema() for t in cls._tools.values()]


# Decorator for easy tool registration
def tool(name: str, description: str, category: str = "general"):
    """Decorator to register a tool class."""
    def decorator(cls):
        cls.name = name
        cls.description = description
        cls.category = category
        ToolRegistry.register(cls())
        return cls
    return decorator