#!/usr/bin/env python3
"""Test script for Warp Cortex multi-agent scalability."""

import psutil
import multiprocessing
import time
from warp_cortex import WarpCortexManager


def simple_task(agent_id):
    """Simple task function that each agent performs."""
    # Simulate some computational work
    result = 0
    for i in range(1000):
        result += i * agent_id
    return f"Agent {agent_id} completed task with result: {result}"


def test_scalability():
    """Test multi-agent scalability with Warp Cortex."""
    
    # Measure memory before spawning agents
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
    print(f"Memory usage before spawning agents: {memory_before:.2f} MB")
    
    # Record start time
    start_time = time.time()
    
    # Create WarpCortexManager and spawn 12 concurrent agents
    manager = WarpCortexManager()
    
    # Spawn 12 agents with the simple task
    agents = []
    for i in range(12):
        agent = manager.create_agent(simple_task, args=(i,))
        agents.append(agent)
    
    # Wait for all agents to complete
    results = []
    for agent in agents:
        result = agent.run()
        results.append(result)
    
    # Measure memory after spawning agents
    memory_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
    print(f"Memory usage after spawning agents: {memory_after:.2f} MB")
    
    # Record end time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Calculate memory efficiency
    memory_used = memory_after - memory_before
    memory_efficiency = len(results) / memory_used if memory_used > 0 else 0
    
    # Print results
    print(f"\n=== Test Results ===")
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"Memory efficiency: {memory_efficiency:.2f} agents/MB")
    print(f"Total agents completed: {len(results)}")
    print(f"\nSample results (first 3):")
    for result in results[:3]:
        print(f"  - {result}")
    
    return execution_time, memory_used, memory_efficiency


if __name__ == "__main__":
    test_scalability()