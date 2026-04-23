"""
Referential Injection for asynchronous KV-cache updates.
"""

import asyncio
import threading
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor


class ReferentialInjection:
    """Asynchronous injection of updates into primary agent context."""

    def __init__(self):
        self._update_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.Lock()

    async def start_injection_processor(self) -> None:
        """Start the background injection processor."""
        if self._running:
            return

        self._running = True
        asyncio.create_task(self._process_injections())

    async def _process_injections(self) -> None:
        """Process injection updates in the background."""
        while self._running:
            try:
                update = await asyncio.wait_for(self._update_queue.get(), timeout=1.0)
                await self._inject_update(update)
                self._update_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Injection processing error: {e}")

    async def _inject_update(self, update: Dict[str, Any]) -> None:
        """Inject an update into the primary context."""
        update_type = update.get('type', 'landmarks')

        if update_type == 'landmarks':
            # Update landmark buffer
            new_landmarks = update.get('data', [])
            print(f"Injected {len(new_landmarks)} new landmarks")
        elif update_type == 'context':
            # Update context
            new_context = update.get('data', '')
            print(f"Injected context update: {len(new_context)} chars")

    async def inject_update(self, update: Dict[str, Any]) -> None:
        """Public method to inject updates asynchronously."""
        await self._update_queue.put(update)

    def stop(self) -> None:
        """Stop the injection processor."""
        self._running = False
        self._executor.shutdown(wait=True)