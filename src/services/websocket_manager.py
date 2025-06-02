"""
WebSocket Manager - Resilient Real-time Data Streaming

This module implements a practical WebSocket manager with multiplexing,
automatic reconnection, and HTTP polling fallback for local deployment.
"""

import asyncio
import json
import time
import logging
import websockets
import aiohttp
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import random
import secrets

logger = logging.getLogger(__name__)

class StreamStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FALLBACK = "fallback"
    ERROR = "error"

class DataSourceType(Enum):
    MARKET_DATA = "market_data"
    ORDER_BOOK = "order_book"
    WHALE_ACTIVITY = "whale_activity"
    NETWORK_STATUS = "network_status"

@dataclass
class StreamConfig:
    url: str
    fallback_url: Optional[str] = None
    reconnect_attempts: int = 5
    reconnect_delay: float = 5.0
    ping_interval: float = 30.0
    timeout: float = 10.0
    enable_fallback: bool = True

@dataclass
class StreamMetrics:
    connection_attempts: int = 0
    successful_connections: int = 0
    messages_received: int = 0
    messages_lost: int = 0
    reconnections: int = 0
    fallback_activations: int = 0
    last_message_time: float = 0.0
    average_latency_ms: float = 0.0

class WebSocketManager:
    """
    Resilient WebSocket manager with multiplexing and fallback
    
    Features:
    - Single WebSocket connection with message multiplexing
    - Automatic reconnection with exponential backoff
    - HTTP polling fallback when WebSocket fails
    - Message queuing during disconnections
    - Real-time latency monitoring
    """
    
    def __init__(self):
        # Connection management
        self.active_streams: Dict[DataSourceType, StreamConfig] = {}
        self.stream_status: Dict[DataSourceType, StreamStatus] = {}
        self.websocket_connections: Dict[DataSourceType, Optional[websockets.WebSocketServerProtocol]] = {}
        
        # Fallback HTTP sessions
        self.http_sessions: Dict[DataSourceType, aiohttp.ClientSession] = {}
        self.fallback_timers: Dict[DataSourceType, asyncio.Task] = {}
        
        # Message handling
        self.message_handlers: Dict[DataSourceType, List[Callable]] = defaultdict(list)
        self.message_queue: Dict[DataSourceType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.missed_messages: Dict[DataSourceType, int] = defaultdict(int)
        
        # Performance tracking
        self.stream_metrics: Dict[DataSourceType, StreamMetrics] = {}
        self.connection_health = {}
        
        # Configuration
        self.max_queue_size = 1000
        self.health_check_interval = 30.0
        self.fallback_poll_interval = 5.0
        
        logger.info("📡 WebSocket Manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize WebSocket manager"""
        try:
            # Setup default stream configurations
            self._setup_default_streams()
            
            # Initialize HTTP sessions for fallback
            await self._initialize_http_sessions()
            
            # Start health monitoring
            asyncio.create_task(self._health_monitoring_loop())
            
            logger.info("✅ WebSocket Manager initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WebSocket Manager: {str(e)}")
            return False
    
    def _setup_default_streams(self):
        """Setup default stream configurations"""
        self.active_streams = {
            DataSourceType.MARKET_DATA: StreamConfig(
                url="wss://api.birdeye.so/socket.io/",
                fallback_url="https://api.birdeye.so/defi/history",
                reconnect_attempts=5,
                reconnect_delay=2.0
            ),
            DataSourceType.ORDER_BOOK: StreamConfig(
                url="wss://api.helius.xyz/v0/websocket",
                fallback_url="https://api.helius.xyz/v0/token-metadata",
                reconnect_attempts=3,
                reconnect_delay=5.0
            )
        }
        
        # Initialize metrics for each stream
        for source_type in self.active_streams:
            self.stream_metrics[source_type] = StreamMetrics()
            self.stream_status[source_type] = StreamStatus.DISCONNECTED
    
    async def _initialize_http_sessions(self):
        """Initialize HTTP sessions for fallback polling"""
        try:
            for source_type in self.active_streams:
                session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10.0),
                    connector=aiohttp.TCPConnector(limit=10)
                )
                self.http_sessions[source_type] = session
            
            logger.info("🌐 HTTP sessions initialized for fallback")
            
        except Exception as e:
            logger.error(f"Error initializing HTTP sessions: {str(e)}")
    
    async def connect_stream(self, source_type: DataSourceType, 
                           subscription_params: Dict[str, Any] = None) -> bool:
        """Connect to a specific data stream"""
        try:
            if source_type not in self.active_streams:
                logger.error(f"Unknown source type: {source_type}")
                return False
            
            config = self.active_streams[source_type]
            self.stream_status[source_type] = StreamStatus.CONNECTING
            
            # Attempt WebSocket connection
            success = await self._connect_websocket(source_type, config, subscription_params)
            
            if success:
                self.stream_status[source_type] = StreamStatus.CONNECTED
                self.stream_metrics[source_type].successful_connections += 1
                logger.info(f"✅ WebSocket connected: {source_type.value}")
                return True
            else:
                # Fall back to HTTP polling if enabled
                if config.enable_fallback:
                    await self._activate_fallback(source_type)
                    return True
                else:
                    self.stream_status[source_type] = StreamStatus.ERROR
                    return False
            
        except Exception as e:
            logger.error(f"Error connecting stream {source_type.value}: {str(e)}")
            self.stream_status[source_type] = StreamStatus.ERROR
            return False
    
    async def _connect_websocket(self, source_type: DataSourceType, 
                               config: StreamConfig, 
                               subscription_params: Dict[str, Any] = None) -> bool:
        """Attempt WebSocket connection with retries"""
        metrics = self.stream_metrics[source_type]
        
        for attempt in range(config.reconnect_attempts):
            try:
                metrics.connection_attempts += 1
                
                logger.info(f"🔌 Connecting to {source_type.value} WebSocket (attempt {attempt + 1})")
                
                websocket = await asyncio.wait_for(
                    websockets.connect(
                        config.url,
                        ping_interval=config.ping_interval,
                        ping_timeout=config.timeout
                    ),
                    timeout=config.timeout
                )
                
                self.websocket_connections[source_type] = websocket
                
                # Send subscription message if provided
                if subscription_params:
                    subscription_msg = json.dumps(subscription_params)
                    await websocket.send(subscription_msg)
                
                # Start message listening task
                asyncio.create_task(self._websocket_message_loop(source_type, websocket))
                
                return True
                
            except Exception as e:
                logger.warning(f"WebSocket connection attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < config.reconnect_attempts - 1:
                    delay = config.reconnect_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
        
        return False
    
    async def _websocket_message_loop(self, source_type: DataSourceType, 
                                    websocket: websockets.WebSocketServerProtocol):
        """Main WebSocket message receiving loop"""
        try:
            async for message in websocket:
                try:
                    start_time = time.time()
                    
                    # Parse message
                    if isinstance(message, str):
                        data = json.loads(message)
                    else:
                        data = message  # Binary data
                    
                    # Update metrics
                    metrics = self.stream_metrics[source_type]
                    metrics.messages_received += 1
                    metrics.last_message_time = time.time()
                    
                    # Calculate latency if timestamp available
                    if 'timestamp' in data:
                        latency = (time.time() - data['timestamp']) * 1000
                        current_avg = metrics.average_latency_ms
                        count = metrics.messages_received
                        metrics.average_latency_ms = (current_avg * (count - 1) + latency) / count
                    
                    # Process message
                    await self._process_incoming_message(source_type, data)
                    
                    processing_time = (time.time() - start_time) * 1000
                    logger.debug(f"Processed {source_type.value} message in {processing_time:.1f}ms")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from {source_type.value}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing message from {source_type.value}: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"WebSocket connection closed: {source_type.value}")
            await self._handle_disconnection(source_type)
        except Exception as e:
            logger.error(f"WebSocket message loop error for {source_type.value}: {str(e)}")
            await self._handle_disconnection(source_type)
    
    async def _process_incoming_message(self, source_type: DataSourceType, data: Dict[str, Any]):
        """Process incoming message and route to handlers"""
        try:
            # Add to message queue for processing
            queue = self.message_queue[source_type]
            
            if len(queue) >= self.max_queue_size:
                # Remove oldest message if queue is full
                queue.popleft()
                self.missed_messages[source_type] += 1
            
            queue.append({
                'timestamp': time.time(),
                'source': source_type.value,
                'data': data
            })
            
            # Call registered handlers
            handlers = self.message_handlers[source_type]
            for handler in handlers:
                try:
                    asyncio.create_task(handler(source_type, data))
                except Exception as e:
                    logger.error(f"Handler error for {source_type.value}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error processing incoming message: {str(e)}")
    
    async def _handle_disconnection(self, source_type: DataSourceType):
        """Handle WebSocket disconnection"""
        try:
            self.stream_status[source_type] = StreamStatus.RECONNECTING
            self.websocket_connections[source_type] = None
            
            metrics = self.stream_metrics[source_type]
            metrics.reconnections += 1
            
            # Attempt reconnection
            config = self.active_streams[source_type]
            reconnected = await self._connect_websocket(source_type, config)
            
            if reconnected:
                self.stream_status[source_type] = StreamStatus.CONNECTED
                logger.info(f"🔄 Reconnected to {source_type.value}")
            else:
                # Fall back to HTTP polling
                if config.enable_fallback:
                    await self._activate_fallback(source_type)
                else:
                    self.stream_status[source_type] = StreamStatus.ERROR
                    logger.error(f"❌ Failed to reconnect to {source_type.value}")
            
        except Exception as e:
            logger.error(f"Error handling disconnection for {source_type.value}: {str(e)}")
    
    async def _activate_fallback(self, source_type: DataSourceType):
        """Activate HTTP polling fallback"""
        try:
            config = self.active_streams[source_type]
            
            if not config.fallback_url:
                logger.warning(f"No fallback URL configured for {source_type.value}")
                return
            
            self.stream_status[source_type] = StreamStatus.FALLBACK
            self.stream_metrics[source_type].fallback_activations += 1
            
            # Start HTTP polling task
            fallback_task = asyncio.create_task(self._http_polling_loop(source_type))
            self.fallback_timers[source_type] = fallback_task
            
            logger.warning(f"📡 Activated HTTP fallback for {source_type.value}")
            
        except Exception as e:
            logger.error(f"Error activating fallback for {source_type.value}: {str(e)}")
    
    async def _http_polling_loop(self, source_type: DataSourceType):
        """HTTP polling fallback loop"""
        try:
            config = self.active_streams[source_type]
            session = self.http_sessions[source_type]
            
            while self.stream_status[source_type] == StreamStatus.FALLBACK:
                try:
                    # Make HTTP request
                    async with session.get(config.fallback_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            await self._process_incoming_message(source_type, data)
                        else:
                            logger.warning(f"HTTP fallback error {response.status} for {source_type.value}")
                    
                    # Wait before next poll
                    await asyncio.sleep(self.fallback_poll_interval)
                    
                    # Periodically try to reconnect WebSocket
                    if secrets.randbelow(10000) / 10000.0 < 0.1:  # 10% chance each poll
                        if await self._connect_websocket(source_type, config):
                            logger.info(f"🔄 Restored WebSocket connection for {source_type.value}")
                            self.stream_status[source_type] = StreamStatus.CONNECTED
                            break
                    
                except Exception as e:
                    logger.error(f"HTTP polling error for {source_type.value}: {str(e)}")
                    await asyncio.sleep(self.fallback_poll_interval)
            
        except Exception as e:
            logger.error(f"HTTP polling loop error for {source_type.value}: {str(e)}")
    
    def register_handler(self, source_type: DataSourceType, handler: Callable):
        """Register message handler for data source"""
        try:
            self.message_handlers[source_type].append(handler)
            logger.info(f"📝 Registered handler for {source_type.value}")
        except Exception as e:
            logger.error(f"Error registering handler: {str(e)}")
    
    def get_latest_messages(self, source_type: DataSourceType, count: int = 10) -> List[Dict[str, Any]]:
        """Get latest messages from source"""
        try:
            queue = self.message_queue[source_type]
            return list(queue)[-count:] if queue else []
        except Exception as e:
            logger.error(f"Error getting latest messages: {str(e)}")
            return []
    
    async def _health_monitoring_loop(self):
        """Monitor connection health"""
        while True:
            try:
                current_time = time.time()
                
                for source_type, metrics in self.stream_metrics.items():
                    # Check if we're receiving recent messages
                    time_since_last_message = current_time - metrics.last_message_time
                    
                    if time_since_last_message > 60:  # No messages for 1 minute
                        if self.stream_status[source_type] == StreamStatus.CONNECTED:
                            logger.warning(f"⚠️ No recent messages from {source_type.value}")
                            # Try reconnection
                            await self._handle_disconnection(source_type)
                    
                    # Update connection health
                    if self.stream_status[source_type] == StreamStatus.CONNECTED:
                        self.connection_health[source_type.value] = "healthy"
                    elif self.stream_status[source_type] == StreamStatus.FALLBACK:
                        self.connection_health[source_type.value] = "fallback"
                    else:
                        self.connection_health[source_type.value] = "unhealthy"
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {str(e)}")
                await asyncio.sleep(60)
    
    def get_stream_metrics(self) -> Dict[str, Any]:
        """Get comprehensive stream metrics"""
        return {
            "stream_status": {
                source.value: status.value 
                for source, status in self.stream_status.items()
            },
            "connection_health": self.connection_health,
            "metrics": {
                source.value: {
                    "messages_received": metrics.messages_received,
                    "messages_lost": self.missed_messages[source],
                    "reconnections": metrics.reconnections,
                    "fallback_activations": metrics.fallback_activations,
                    "average_latency_ms": metrics.average_latency_ms,
                    "last_message_age_s": time.time() - metrics.last_message_time if metrics.last_message_time > 0 else 0
                }
                for source, metrics in self.stream_metrics.items()
            },
            "queue_sizes": {
                source.value: len(queue) 
                for source, queue in self.message_queue.items()
            }
        }
    
    async def disconnect_all(self):
        """Disconnect all streams gracefully"""
        try:
            # Close WebSocket connections
            for source_type, websocket in self.websocket_connections.items():
                if websocket:
                    await websocket.close()
            
            # Cancel fallback tasks
            for source_type, task in self.fallback_timers.items():
                if task and not task.done():
                    task.cancel()
            
            # Close HTTP sessions
            for session in self.http_sessions.values():
                await session.close()
            
            logger.info("🔌 All streams disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting streams: {str(e)}")

# Example usage for threat detection integration
class ThreatDetectionHandler:
    """Example handler for processing market data for threat detection"""
    
    def __init__(self, threat_detector):
        self.threat_detector = threat_detector
    
    async def handle_market_data(self, source_type: DataSourceType, data: Dict[str, Any]):
        """Handle market data for threat analysis"""
        try:
            if source_type == DataSourceType.MARKET_DATA:
                # Extract relevant data for threat detection
                token_address = data.get('token_address')
                price_change = data.get('price_change_1m', 0.0)
                volume_spike = data.get('volume_spike', 0.0)
                
                # Trigger threat analysis
                if abs(price_change) > 0.1 or volume_spike > 5.0:
                    await self.threat_detector.analyze_market_event(data)
            
        except Exception as e:
            logger.error(f"Error in threat detection handler: {str(e)}") 