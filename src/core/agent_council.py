"""
Multi-Agent Council Orchestrator
Manages spawning, coordination, and consensus merging of agent councils.
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import yaml
from pathlib import Path


class CouncilType(Enum):
    RESEARCH = "research"
    CODE = "code"
    CREATIVE = "creative"
    META = "meta"


@dataclass
class CouncilAgent:
    """Single agent within a council."""
    agent_id: str
    council_type: CouncilType
    system_prompt: str
    status: str = "pending"  # pending, running, completed, failed
    response: str = ""
    tokens_used: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Council:
    """A council containing multiple agents."""
    council_id: str
    council_type: CouncilType
    agents: List[CouncilAgent] = field(default_factory=list)
    status: str = "spawning"  # spawning, running, consensus, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_complete(self) -> bool:
        return all(a.status == "completed" for a in self.agents)
    
    def get_responses(self) -> List[str]:
        return [a.response for a in self.agents if a.response]


@dataclass
class AgentCouncil:
    """Main orchestrator for multi-agent councils."""
    
    _councils: Dict[str, Council] = field(default_factory=dict)
    _config: Dict[str, Any] = field(default_factory=dict)
    _model_generator: Optional[Callable] = field(default=None, init=False)
    
    def __post_init__(self):
        self._load_config()
    
    def _load_config(self):
        """Load council configuration."""
        config_path = Path(__file__).parent.parent.parent / "config" / "agents.yaml"
        
        if config_path.exists():
            with open(config_path) as f:
                self._config = yaml.safe_load(f)
    
    def set_generator(self, generator: Callable):
        """Set the model generation function."""
        self._model_generator = generator
    
    async def spawn_council(
        self,
        prompt: str,
        council_types: List[str],
        agent_counts: Optional[Dict[str, int]] = None
    ) -> str:
        """
        Spawn a new council with specified agents.
        
        Args:
            prompt: The user prompt to route to agents
            council_types: List of council type names (research, code, creative, meta)
            agent_counts: Optional override for agent counts per council
            
        Returns:
            council_id for tracking this council
        """
        council_id = f"council_{uuid.uuid4().hex[:8]}"
        agent_counts = agent_counts or {}
        
        councils_config = self._config.get("councils", {})
        
        created_councils = []
        
        for type_name in council_types:
            council_type = CouncilType(type_name)
            council_config = councils_config.get(type_name, {})
            
            # Determine agent count
            count = agent_counts.get(type_name, council_config.get("agent_count", 1))
            
            # Create agents for this council
            agents = []
            for i in range(count):
                agent_id = f"{council_id}_{type_name}_{i}"
                system_prompt = council_config.get("system_prompt", "")
                
                agent = CouncilAgent(
                    agent_id=agent_id,
                    council_type=council_type,
                    system_prompt=system_prompt
                )
                agents.append(agent)
            
            # Create council
            council = Council(
                council_id=council_id,
                council_type=council_type,
                agents=agents,
                status="running"
            )
            
            self._councils[council_id] = council
            created_councils.append(council)
        
        # Run all councils concurrently
        await self._run_councils(prompt, created_councils)
        
        # Perform consensus
        await self._run_consensus(council_id)
        
        return council_id
    
    async def _run_councils(self, prompt: str, councils: List[Council]):
        """Run all councils concurrently."""
        tasks = []
        
        for council in councils:
            for agent in council.agents:
                agent.status = "running"
                agent.started_at = datetime.now()
                
                # Create generation task
                task = self._run_agent(agent, prompt)
                tasks.append(task)
        
        # Wait for all agents
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_agent(self, agent: CouncilAgent, base_prompt: str):
        """Run a single agent with its system prompt."""
        full_prompt = f"{agent.system_prompt}\n\nUser request: {base_prompt}"
        
        try:
            if self._model_generator:
                response = await self._model_generator(
                    full_prompt,
                    max_tokens=256,
                    temperature=0.7
                )
            else:
                # Default: use mock response
                response = f"[{agent.council_type.value} agent response to: {base_prompt[:50]}...]"
            
            agent.response = response
            agent.status = "completed"
            agent.tokens_used = len(response.split())
            
        except Exception as e:
            agent.response = f"[Error: {str(e)}]"
            agent.status = "failed"
        
        finally:
            agent.completed_at = datetime.now()
    
    async def _run_consensus(self, council_id: str):
        """Merge council responses into consensus."""
        # Get all councils with this ID
        matching = [c for c in self._councils.values() if c.council_id == council_id]
        
        if not matching:
            return
        
        # Collect all responses
        all_responses = []
        for council in matching:
            all_responses.extend(council.get_responses())
            council.status = "consensus"
        
        # If we have a meta council, use it for final synthesis
        meta_councils = [c for c in matching if c.council_type == CouncilType.META]
        
        if meta_councils and all_responses:
            # Meta agent synthesizes
            meta = meta_councils[0]
            synthesis_prompt = (
                "Synthesize the following council responses into a coherent response:\n\n"
                + "\n".join(all_responses)
            )
            
            # Run meta agent
            for agent in meta.agents:
                await self._run_agent(agent, synthesis_prompt)
        
        # Mark complete
        for council in matching:
            council.status = "completed"
    
    def get_council_status(self, council_id: str) -> Dict[str, Any]:
        """Get status of a council."""
        matching = [c for c in self._councils.values() if c.council_id == council_id]
        
        if not matching:
            return {"status": "not_found", "council_id": council_id}
        
        council = matching[0]
        
        return {
            "council_id": council_id,
            "type": council.council_type.value,
            "status": council.status,
            "agent_count": len(council.agents),
            "completed_agents": sum(1 for a in council.agents if a.status == "completed"),
            "created_at": council.created_at.isoformat(),
            "responses": [
                {
                    "agent_id": a.agent_id,
                    "status": a.status,
                    "response": a.response[:200] if a.response else ""
                }
                for a in council.agents
            ]
        }
    
    def get_all_councils(self) -> List[Dict[str, Any]]:
        """Get all active councils."""
        return [
            {
                "council_id": c.council_id,
                "type": c.council_type.value,
                "status": c.status,
                "agent_count": len(c.agents)
            }
            for c in self._councils.values()
        ]
    
    def get_council_responses(self, council_id: str) -> List[str]:
        """Get all responses from a council."""
        matching = [c for c in self._councils.values() if c.council_id == council_id]
        
        if not matching:
            return []
        
        responses = []
        for council in matching:
            responses.extend(council.get_responses())
        
        return responses
    
    def clear_council(self, council_id: str):
        """Clear a council from memory."""
        self._councils = {k: v for k, v in self._councils.items() if k != council_id}


# Global instance
_council: Optional[AgentCouncil] = None


def get_council() -> AgentCouncil:
    """Get or create the global council instance."""
    global _council
    if _council is None:
        _council = AgentCouncil()
    return _council