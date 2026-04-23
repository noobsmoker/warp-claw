"""
WarpCortexManager - Orchestrates multi-agent spawning with shared resources.
"""

import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Callable
from .singleton import WarpCortexSingleton
from .synapse import TopologicalSynapse
from .injection import ReferentialInjection


class WarpCortexManager:
    """Manager for coordinating multiple Warp-Cortex agents."""

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.singleton = WarpCortexSingleton()
        self.synapse = TopologicalSynapse()
        self.injection = ReferentialInjection()
        self._executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context('spawn'))

        # Start injection processor
        asyncio.create_task(self.injection.start_injection_processor())

    def create_agent(self, task_func: Callable, args: tuple = ()) -> 'WarpCortexAgent':
        """Create a new agent instance."""
        return WarpCortexAgent(self.singleton, self.synapse, self.injection, task_func, args)

    async def spawn_subagents(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """Spawn multiple subagents concurrently."""
        loop = asyncio.get_event_loop()

        # Submit tasks to process pool
        futures = [
            loop.run_in_executor(self._executor, self._execute_task, task)
            for task in tasks
        ]

        # Wait for all to complete
        results = await asyncio.gather(*futures)
        return results

    def _execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a single task in a separate process."""
        task_func = task['func']
        args = task.get('args', ())
        kwargs = task.get('kwargs', {})

        try:
            return task_func(*args, **kwargs)
        except Exception as e:
            print(f"Task execution error: {e}")
            return None

    def shutdown(self) -> None:
        """Shutdown the manager and cleanup resources."""
        self._executor.shutdown(wait=True)
        self.injection.stop()


class WarpCortexAgent:
    """Individual agent instance with shared Warp-Cortex resources."""

    def __init__(self, singleton: WarpCortexSingleton, synapse: TopologicalSynapse,
                 injection: ReferentialInjection, task_func: Callable, args: tuple):
        self.singleton = singleton
        self.synapse = synapse
        self.injection = injection
        self.task_func = task_func
        self.args = args

    def run(self) -> Any:
        """Execute the agent's task."""
        try:
            result = self.task_func(*self.args)
            return result
        except Exception as e:
            print(f"Agent execution error: {e}")
            return None

    async def run_async(self) -> Any:
        """Execute the agent's task asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run)