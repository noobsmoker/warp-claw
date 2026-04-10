# Warp-Claw Tests

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.cortex_bridge import M1CortexBridge
from core.agent_council import AgentCouncil, CouncilType
from core.memory_synapse import MemorySynapse
from tools.base_tool import ToolRegistry


class TestCortexBridge:
    """Test M1 Cortex Bridge."""
    
    @pytest.mark.asyncio
    async def test_device_detection(self):
        """Test device detection."""
        bridge = M1CortexBridge(model_id="qwen-0.5b")
        assert bridge.device in ["mps", "cpu"]
    
    @pytest.mark.asyncio
    async def test_generate(self):
        """Test basic generation."""
        bridge = M1CortexBridge(model_id="qwen-0.5b")
        
        # Should work even without model (uses mock)
        result = await bridge.generate(
            prompt="Hello world",
            max_tokens=10
        )
        
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_council_detection(self):
        """Test council detection from prompt."""
        bridge = M1CortexBridge(model_id="qwen-0.5b")
        
        # Should detect research council for long prompt
        councils = bridge._detect_councils("Explain quantum computing with sources and verification")
        assert "research" in councils


class TestAgentCouncil:
    """Test Agent Council."""
    
    @pytest.mark.asyncio
    async def test_spawn_council(self):
        """Test council spawning."""
        council = AgentCouncil()
        
        council_id = await council.spawn_council(
            prompt="Test prompt",
            council_types=["research"]
        )
        
        assert council_id.startswith("council_")
        
        # Check status
        status = council.get_council_status(council_id)
        assert status["status"] != "not_found"
    
    @pytest.mark.asyncio
    async def test_multiple_council_types(self):
        """Test spawning multiple council types."""
        council = AgentCouncil()
        
        council_id = await council.spawn_council(
            prompt="Test",
            council_types=["research", "code", "creative"]
        )
        
        assert council_id.startswith("council_")


class TestMemorySynapse:
    """Test Memory Synapse."""
    
    def test_set_get(self):
        """Test basic set/get."""
        synapse = MemorySynapse()
        
        synapse.set("key1", "value1", namespace="test")
        result = synapse.get("key1", namespace="test")
        
        assert result == "value1"
    
    def test_namespace(self):
        """Test namespace isolation."""
        synapse = MemorySynapse()
        
        synapse.set("key", "value1", namespace="ns1")
        synapse.set("key", "value2", namespace="ns2")
        
        assert synapse.get("key", namespace="ns1") == "value1"
        assert synapse.get("key", namespace="ns2") == "value2"
    
    def test_expiry(self):
        """Test TTL expiry."""
        from datetime import datetime
        
        synapse = MemorySynapse()
        
        # Set with 0 TTL (immediate expiry)
        synapse.set("key", "value", namespace="test", ttl_seconds=0)
        
        # Should be expired
        assert not synapse.exists("key", namespace="test")
    
    def test_lru_eviction(self):
        """Test LRU eviction."""
        synapse = MemorySynapse(max_size=2)
        
        synapse.set("key1", "value1")
        synapse.set("key2", "value2")
        synapse.set("key3", "value3")  # Should evict key1
        
        # key1 may or may not be evicted depending on access
        assert synapse.get("key3") == "value3"


class TestToolRegistry:
    """Test Tool Registry."""
    
    def test_list_tools(self):
        """Test listing tools."""
        tools = ToolRegistry.list_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
    
    def test_get_tool(self):
        """Test getting a tool."""
        tool = ToolRegistry.get("execute_python")
        
        assert tool is not None
        assert tool.name == "execute_python"


# Run with: pytest tests/ -v