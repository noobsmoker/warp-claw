"""
WebSocket Stream for Real-Time Agent Activity
Provides WebSocket endpoints for streaming agent reasoning.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from websockets import WebSocketServerProtocol, serve
import uuid


class StreamEventType(Enum):
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    AGENT_SPAWN = "agent_spawn"
    AGENT_THINK = "agent_think"
    AGENT_TOOL_CALL = "agent_tool_call"
    AGENT_TOOL_RESULT = "agent_tool_result"
    AGENT_COMPLETE = "agent_complete"
    COUNCIL_UPDATE = "council_update"
    COUNCIL_CONSENSUS = "council_consensus"
    ERROR = "error"


@dataclass
class StreamEvent:
    """A streaming event."""
    event_type: StreamEventType
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    
    def to_json(self) -> str:
        return json.dumps({
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "event_id": self.event_id
        })


@dataclass
class StreamClient:
    """Connected WebSocket client."""
    client_id: str
    websocket: WebSocketServerProtocol
    subscribed_events: Set[StreamEventType] = field(default_factory=set)
    subscribed_councils: Set[str] = field(default_factory=set)
    connected_at: datetime = field(default_factory=datetime.now)


@dataclass
class WebSocketStream:
    """
    Real-time streaming of agent activity via WebSocket.
    """
    
    _clients: Dict[str, StreamClient] = field(default_factory=dict)
    _broadcast_callback: Optional[Callable] = field(default=None, init=False)
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a new WebSocket connection."""
        client_id = str(uuid.uuid4())[:8]
        client = StreamClient(client_id=client_id, websocket=websocket)
        
        self._clients[client_id] = client
        
        # Send welcome event
        await self.send_event(client, StreamEvent(
            StreamEventType.CONNECT,
            {"client_id": client_id, "message": "Connected to Warp-Claw stream"}
        ))
        
        try:
            # Handle messages from client
            async for message in websocket:
                await self.handle_message(client, message)
                
        except Exception as e:
            await self.send_event(client, StreamEvent(
                StreamEventType.ERROR,
                {"error": str(e)}
            ))
            
        finally:
            # Cleanup
            if client_id in self._clients:
                del self._clients[client_id]
            
            await self.send_event(client, StreamEvent(
                StreamEventType.DISCONNECT,
                {"client_id": client_id}
            ))
    
    async def handle_message(self, client: StreamClient, message: str):
        """Handle incoming WebSocket message."""
        try:
            msg = json.loads(message)
            msg_type = msg.get("type")
            
            if msg_type == "subscribe":
                # Subscribe to event types
                events = msg.get("events", [])
                for e in events:
                    try:
                        client.subscribed_events.add(StreamEventType(e))
                    except ValueError:
                        pass
                
                # Subscribe to specific councils
                councils = msg.get("councils", [])
                client.subscribed_councils.update(councils)
                
                await self.send_event(client, StreamEvent(
                    StreamEventType.CONNECT,
                    {"subscribed": True, "events": [e.value for e in client.subscribed_events]}
                ))
                
            elif msg_type == "unsubscribe":
                events = msg.get("events", [])
                for e in events:
                    try:
                        client.subscribed_events.discard(StreamEventType(e))
                    except ValueError:
                        pass
                
                councils = msg.get("councils", [])
                client.subscribed_councils.difference_update(councils)
                
            elif msg_type == "ping":
                await self.send_event(client, StreamEvent(
                    StreamEventType.CONNECT,
                    {"type": "pong"}
                ))
                
        except json.JSONDecodeError:
            pass
    
    async def send_event(self, client: StreamClient, event: StreamEvent):
        """Send event to a specific client."""
        try:
            await client.websocket.send(event.to_json())
        except Exception:
            pass
    
    async def broadcast(
        self,
        event_type: StreamEventType,
        data: Dict[str, Any],
        council_filter: Optional[str] = None
    ):
        """Broadcast event to all subscribed clients."""
        event = StreamEvent(event_type, data)
        
        for client in self._clients.values():
            # Check subscriptions
            if event_type not in client.subscribed_events:
                continue
            
            # Check council filter
            if council_filter and council_filter not in client.subscribed_councils:
                continue
            
            await self.send_event(client, event)
    
    # === Convenience Methods ===
    
    async def broadcast_agent_spawn(
        self,
        agent_id: str,
        council_type: str
    ):
        """Broadcast agent spawn event."""
        await self.broadcast(
            StreamEventType.AGENT_SPAWN,
            {
                "agent_id": agent_id,
                "council_type": council_type
            }
        )
    
    async def broadcast_agent_think(
        self,
        agent_id: str,
        thought: str
    ):
        """Broadcast agent thinking."""
        await self.broadcast(
            StreamEventType.AGENT_THINK,
            {
                "agent_id": agent_id,
                "thought": thought[:500]  # Truncate for streaming
            }
        )
    
    async def broadcast_agent_complete(
        self,
        agent_id: str,
        response: str,
        tokens_used: int
    ):
        """Broadcast agent completion."""
        await self.broadcast(
            StreamEventType.AGENT_COMPLETE,
            {
                "agent_id": agent_id,
                "response": response[:500],
                "tokens_used": tokens_used
            }
        )
    
    async def broadcast_council_update(
        self,
        council_id: str,
        status: str,
        completed_agents: int,
        total_agents: int
    ):
        """Broadcast council status update."""
        await self.broadcast(
            StreamEventType.COUNCIL_UPDATE,
            {
                "council_id": council_id,
                "status": status,
                "completed_agents": completed_agents,
                "total_agents": total_agents
            },
            council_filter=council_id
        )
    
    async def broadcast_council_consensus(
        self,
        council_id: str,
        consensus: str
    ):
        """Broadcast council consensus result."""
        await self.broadcast(
            StreamEventType.COUNCIL_CONSENSUS,
            {
                "council_id": council_id,
                "consensus": consensus[:500]
            },
            council_filter=council_id
        )
    
    # === Server ===
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8001):
        """Start WebSocket server."""
        async with serve(self.handle_connection, host, port):
            print(f"WebSocket stream server started on ws://{host}:{port}")
            await asyncio.Future()  # Run forever


# Global instance
_stream: Optional[WebSocketStream] = None


def get_stream() -> WebSocketStream:
    """Get or create the global stream instance."""
    global _stream
    if _stream is None:
        _stream = WebSocketStream()
    return _stream


async def run_stream_server(host: str = "0.0.0.0", port: int = 8001):
    """Run the WebSocket stream server."""
    stream = get_stream()
    await stream.start_server(host, port)