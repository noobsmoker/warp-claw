"""
M1-Optimized Warp Cortex Bridge
Wraps and patches Warp Cortex for Apple Silicon MPS support.
"""

import os
import sys
import torch
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

# Add vendor path for Warp Cortex imports
VENDOR_PATH = Path(__file__).parent.parent.parent / "vendor" / "warp-cortex"
sys.path.insert(0, str(VENDOR_PATH))


@dataclass
class AgentCouncils:
    """Active agent councils for a generation session."""
    research: List[str] = field(default_factory=list)
    code: List[str] = field(default_factory=list)
    creative: List[str] = field(default_factory=list)
    meta: List[str] = field(default_factory=list)


@dataclass
class M1CortexBridge:
    """M1-optimized bridge to Warp Cortex engine."""
    
    model_id: str = "qwen-0.5b"
    device: str = "cpu"
    max_agents: int = 100
    
    # Internal state
    _engine: Optional[Any] = field(default=None, init=False)
    _tokenizer: Optional[Any] = field(default=None, init=False)
    _model: Optional[Any] = field(default=None, init=False)
    _active_councils: AgentCouncils = field(default_factory=AgentCouncils, init=False)
    _memory_cache: Dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        self.device = self._detect_device()
        
    def _detect_device(self) -> str:
        """Detect and configure the best available device."""
        if torch.backends.mps.is_available():
            # M1/M2/M3 optimization: configure for Metal Performance Shaders
            torch.set_default_dtype(torch.float32)
            # Disable MPS high watermark for memory efficiency
            if hasattr(torch.mps, 'set_low_memory_mode'):
                torch.mps.set_low_memory_mode(True)
            return "mps"
        return "cpu"
    
    def _load_model(self):
        """Load the model and tokenizer (lazy loading)."""
        if self._model is not None:
            return
            
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            # Fallback: create mock model for development/testing
            self._model = MockModel()
            self._tokenizer = MockTokenizer()
            return
            
        # Load from HuggingFace with M1 optimization
        model_config = self._load_model_config()
        repo = model_config.get("repo", "Qwen/Qwen2.5-0.5B-Instruct")
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            repo, 
            trust_remote_code=True
        )
        
        # Load model with MPS/CPU
        self._model = AutoModelForCausalLM.from_pretrained(
            repo,
            device_map=self.device,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        self.max_agents = model_config.get("max_agents", 100)
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from models.yaml."""
        import yaml
        config_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"
        
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                return config.get("models", {}).get(self.model_id, {})
        return {}
    
    async def generate(
        self,
        prompt: str,
        councils: Optional[List[str]] = None,
        tools: Optional[List[Dict]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """
        Main entry point: generate with optional council spawning.
        
        Args:
            prompt: Input prompt
            councils: Optional list of council names to activate
            tools: Optional list of tool definitions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Enable streaming output
            
        Returns:
            Generated text response
        """
        self._load_model()
        
        # Step 1: Detect councils from prompt if not specified
        if councils is None:
            councils = self._detect_councils(prompt)
        
        # Step 2: Execute tool calls if tools provided and needed
        tool_results = []
        if tools:
            tool_results = await self._execute_tools(prompt, tools)
            if tool_results:
                prompt = self._inject_tool_results(prompt, tool_results)
        
        # Step 3: Spawn side agents for active councils
        side_responses = []
        if councils:
            side_responses = await self._spawn_councils(prompt, councils)
        
        # Step 4: Run main generation
        if stream:
            return self._generate_stream(prompt, max_tokens, temperature)
        
        main_response = await self._generate_main(prompt, max_tokens, temperature)
        
        # Step 5: Merge with council responses
        final_response = self._merge_responses(main_response, side_responses)
        
        return final_response
    
    def _detect_councils(self, prompt: str) -> List[str]:
        """Detect which councils to activate based on prompt triggers."""
        import yaml
        config_path = Path(__file__).parent.parent.parent / "config" / "agents.yaml"
        
        active = []
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                councils = config.get("councils", {})
                
                prompt_upper = prompt.upper()
                for council_name, council_config in councils.items():
                    triggers = council_config.get("triggers", [])
                    if any(t in prompt_upper for t in triggers):
                        active.append(council_name)
        
        # Default: always run research for longer prompts
        if len(prompt) > 200 and "research" not in active:
            active.append("research")
            
        return active[:4]  # Max 4 councils
    
    async def _spawn_councils(self, prompt: str, council_names: List[str]) -> List[Dict[str, str]]:
        """Spawn side agents for each council."""
        import yaml
        results = []
        
        config_path = Path(__file__).parent.parent.parent / "config" / "agents.yaml"
        if not config_path.exists():
            return results
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
            councils = config.get("councils", {})
        
        for council_name in council_names:
            council_config = councils.get(council_name, {})
            agent_count = council_config.get("agent_count", 1)
            system_prompt = council_config.get("system_prompt", "")
            
            # Spawn agents for this council
            for i in range(agent_count):
                agent_id = f"{council_name}_{i}"
                
                # Create council-specific prompt
                council_prompt = f"{system_prompt}\n\nUser request: {prompt}"
                
                # Generate side agent response
                response = await self._generate_main(council_prompt, max_tokens=256, temperature=0.7)
                
                results.append({
                    "council": council_name,
                    "agent_id": agent_id,
                    "response": response
                })
                
                # Store in memory cache
                self._memory_cache[agent_id] = response
        
        return results
    
    async def _execute_tools(self, prompt: str, tools: List[Dict]) -> List[Dict]:
        """Execute tool calls if needed."""
        # Simple tool detection based on keywords
        results = []
        
        tool_names = [t.get("function", {}).get("name", "") for t in tools]
        
        if "execute_python" in tool_names and ("calculate" in prompt.lower() or "compute" in prompt.lower() or "fibonacci" in prompt.lower()):
            # Defer to code executor tool
            pass
            
        return results
    
    def _inject_tool_results(self, prompt: str, tool_results: List[Dict]) -> str:
        """Inject tool results into prompt context."""
        if not tool_results:
            return prompt
            
        results_text = "\n\nTool Results:\n"
        for result in tool_results:
            results_text += f"- {result.get('tool', 'unknown')}: {result.get('result', '')}\n"
            
        return prompt + results_text
    
    async def _generate_main(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Run main model generation."""
        if self._model is None or self._tokenizer is None:
            # Return mock response for testing
            return f"[Mock response to: {prompt[:50]}...]"
        
        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
            )
        
        # Decode
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):]
            
        return response.strip()
    
    def _generate_stream(self, prompt: str, max_tokens: int, temperature: float):
        """Generate with streaming (SSE)."""
        # For streaming, yield tokens as they come
        async def token_stream():
            if self._model is None or self._tokenizer is None:
                # Mock streaming
                for word in ["[Mock", " response", " streaming", "..."]:
                    yield f"data: {word}\n\n"
                    await asyncio.sleep(0.1)
                yield "data: [DONE]\n\n"
                return
                
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            generated = []
            for _ in range(max_tokens):
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=1,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
                    )
                
                new_token = outputs[0, -1].item()
                word = self._tokenizer.decode([new_token])
                
                if word == self._tokenizer.eos_token_id:
                    break
                    
                generated.append(word)
                yield f"data: {word}\n\n"
                
                inputs = {"input_ids": outputs}
            
            yield "data: [DONE]\n\n"
            
        return token_stream()
    
    def _merge_responses(self, main: str, side_responses: List[Dict]) -> str:
        """Merge main response with council responses via referential injection."""
        if not side_responses:
            return main
            
        # Group by council
        by_council = {}
        for sr in side_responses:
            council = sr["council"]
            if council not in by_council:
                by_council[council] = []
            by_council[council].append(sr["response"])
        
        # Add council summaries to main response
        merged = main + "\n\n"
        for council, responses in by_council.items():
            merged += f"\n[{council.upper()} COUNCIL]\n"
            for r in responses:
                merged += f"- {r[:200]}...\n"
                
        return merged
    
    def get_council_status(self, council_id: str) -> Dict[str, Any]:
        """Get status of a specific council/agent."""
        return {
            "agent_id": council_id,
            "status": "active" if council_id in self._memory_cache else "inactive",
            "cached_response": self._memory_cache.get(council_id, "")[:100]
        }
    
    def get_active_councils(self) -> Dict[str, int]:
        """Get count of active agents per council."""
        return {
            "research": len(self._active_councils.research),
            "code": len(self._active_councils.code),
            "creative": len(self._active_councils.creative),
            "meta": len(self._active_councils.meta)
        }


# Mock classes for development/testing without model downloads
class MockModel:
    """Mock model for testing."""
    
    def generate(self, **kwargs):
        import torch
        return torch.tensor([[1, 2, 3]])
    
    def to(self, device):
        return self


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    pad_token_id = 0
    eos_token_id = 1
    
    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[1, 2, 3]])}
    
    def decode(self, tokens, skip_special_tokens=False):
        return f"Generated: {tokens}"
    
    def from_pretrained(self, repo, trust_remote_code=True):
        return self()


# Global bridge instance
_bridge: Optional[M1CortexBridge] = None


def get_bridge(model_id: str = "qwen-0.5b") -> M1CortexBridge:
    """Get or create the global bridge instance."""
    global _bridge
    if _bridge is None or _bridge.model_id != model_id:
        _bridge = M1CortexBridge(model_id=model_id)
    return _bridge