"""
WebSocket client with robust reconnection logic and exponential backoff.
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import websockets
from websockets.exceptions import (
    WebSocketException,
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK
)

logger = logging.getLogger(__name__)

class WebSocketClient:
    def __init__(
        self,
        url: str,
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        backoff_factor: float = 2.0,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0
    ):
        """Initialize WebSocket client with reconnection settings."""
        self.url = url
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_factor = backoff_factor
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._retry_count = 0
        self._current_backoff = initial_backoff
        self._message_handlers: List[Callable] = []
        self._error_handlers: List[Callable] = []
        self._reconnect_handlers: List[Callable] = []
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    def add_message_handler(self, handler: Callable) -> None:
        """Add a message handler callback."""
        self._message_handlers.append(handler)

    def add_error_handler(self, handler: Callable) -> None:
        """Add an error handler callback."""
        self._error_handlers.append(handler)

    def add_reconnect_handler(self, handler: Callable) -> None:
        """Add a reconnect handler callback."""
        self._reconnect_handlers.append(handler)

    async def connect(self) -> None:
        """Connect to WebSocket server with retry logic."""
        while not self._stop_event.is_set():
            try:
                async with self._lock:
                    if self._connected:
                        return

                    self._ws = await websockets.connect(
                        self.url,
                        ping_interval=self.ping_interval,
                        ping_timeout=self.ping_timeout
                    )
                    self._connected = True
                    self._retry_count = 0
                    self._current_backoff = self.initial_backoff

                    logger.info(f"Connected to WebSocket server: {self.url}")
                    
                    # Notify reconnect handlers
                    for handler in self._reconnect_handlers:
                        try:
                            await handler()
                        except Exception as e:
                            logger.error(f"Reconnect handler failed: {str(e)}")

                    # Start message handling
                    asyncio.create_task(self._handle_messages())
                    return

            except Exception as e:
                logger.error(f"WebSocket connection failed: {str(e)}")
                await self._handle_connection_error()

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        self._stop_event.set()
        async with self._lock:
            if self._ws:
                await self._ws.close()
                self._ws = None
            self._connected = False

    async def send(self, message: Dict[str, Any]) -> None:
        """Send message to WebSocket server."""
        if not self._connected or not self._ws:
            raise RuntimeError("WebSocket not connected")

        try:
            await self._ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            await self._handle_connection_error()
            raise

    async def _handle_messages(self) -> None:
        """Handle incoming WebSocket messages."""
        while self._connected and not self._stop_event.is_set():
            try:
                if not self._ws:
                    break

                message = await self._ws.recv()
                data = json.loads(message)

                # Notify message handlers
                for handler in self._message_handlers:
                    try:
                        await handler(data)
                    except Exception as e:
                        logger.error(f"Message handler failed: {str(e)}")

            except ConnectionClosedOK:
                logger.info("WebSocket connection closed normally")
                break
            except ConnectionClosedError as e:
                logger.error(f"WebSocket connection closed with error: {str(e)}")
                await self._handle_connection_error()
                break
            except Exception as e:
                logger.error(f"Error handling message: {str(e)}")
                await self._handle_connection_error()
                break

    async def _handle_connection_error(self) -> None:
        """Handle connection errors with exponential backoff."""
        self._connected = False
        self._retry_count += 1

        if self._retry_count > self.max_retries:
            logger.error("Max retry attempts reached")
            self._stop_event.set()
            return

        # Calculate backoff time with jitter
        backoff = min(
            self._current_backoff * (1 + 0.1 * (asyncio.get_event_loop().time() % 1)),
            self.max_backoff
        )
        self._current_backoff = min(
            self._current_backoff * self.backoff_factor,
            self.max_backoff
        )

        # Notify error handlers
        for handler in self._error_handlers:
            try:
                await handler(self._retry_count, backoff)
            except Exception as e:
                logger.error(f"Error handler failed: {str(e)}")

        logger.info(f"Reconnecting in {backoff:.2f} seconds (attempt {self._retry_count})")
        await asyncio.sleep(backoff)
        await self.connect()

class WebSocketManager:
    """Manager for multiple WebSocket connections."""
    
    def __init__(self):
        """Initialize WebSocket manager."""
        self._clients: Dict[str, WebSocketClient] = {}
        self._lock = asyncio.Lock()

    async def add_client(
        self,
        name: str,
        url: str,
        **kwargs
    ) -> WebSocketClient:
        """Add a new WebSocket client."""
        async with self._lock:
            if name in self._clients:
                raise ValueError(f"WebSocket client {name} already exists")

            client = WebSocketClient(url, **kwargs)
            self._clients[name] = client
            await client.connect()
            return client

    async def remove_client(self, name: str) -> None:
        """Remove a WebSocket client."""
        async with self._lock:
            if name not in self._clients:
                return

            client = self._clients[name]
            await client.disconnect()
            del self._clients[name]

    def get_client(self, name: str) -> Optional[WebSocketClient]:
        """Get a WebSocket client by name."""
        return self._clients.get(name)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        for client in self._clients.values():
            if client._connected:
                try:
                    await client.send(message)
                except Exception as e:
                    logger.error(f"Failed to broadcast message: {str(e)}")

    async def close_all(self) -> None:
        """Close all WebSocket connections."""
        async with self._lock:
            for client in self._clients.values():
                await client.disconnect()
            self._clients.clear()

# Initialize global instance
ws_manager = WebSocketManager() 