"""
System Communication API - Inter-Service Communication Hub

This module implements a lightweight FastAPI-based communication system
for coordinating between AI, Defense, and Security components in local deployment.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger(__name__)

class MessageType(Enum):
    AI_CONFIDENCE_UPDATE = "ai_confidence_update"
    THREAT_ALERT = "threat_alert"
    SECURITY_EVENT = "security_event"
    DEFENSE_ACTION = "defense_action"
    MODEL_UPDATE = "model_update"
    SYSTEM_STATUS = "system_status"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

# Pydantic Models for API
class AIConfidenceMessage(BaseModel):
    token_address: str
    prediction_confidence: float = Field(ge=0.0, le=1.0)
    model_id: str
    novelty_score: float = Field(ge=0.0, le=1.0)
    prediction_timestamp: float
    market_context: Dict[str, Any] = {}

class ThreatAlertMessage(BaseModel):
    threat_id: str
    threat_type: str
    threat_level: int = Field(ge=1, le=5)
    token_address: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Dict[str, Any] = {}
    recommended_action: str
    time_to_impact_seconds: Optional[float] = None

class SecurityEventMessage(BaseModel):
    event_id: str
    event_type: str
    severity: str
    key_id: Optional[str] = None
    description: str
    requires_action: bool = False

class DefenseActionMessage(BaseModel):
    action_id: str
    action_type: str
    token_address: str
    positions_affected: List[str] = []
    execution_time_ms: float
    success: bool
    capital_preserved: float = 0.0

class ModelUpdateMessage(BaseModel):
    model_id: str
    update_type: str  # "deployment", "rollback", "performance_update"
    performance_metrics: Dict[str, float] = {}
    deployment_stage: str
    traffic_allocation: float = 0.0

class SystemStatusMessage(BaseModel):
    component: str
    status: str  # "healthy", "degraded", "error", "critical"
    metrics: Dict[str, Any] = {}
    last_updated: float

@dataclass
class InternalMessage:
    """Internal message format for system communication"""
    message_id: str
    message_type: MessageType
    priority: Priority
    source_component: str
    target_component: Optional[str]
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None

class SystemCommunicationAPI:
    """
    Lightweight communication API for local component coordination
    
    Features:
    - FastAPI-based HTTP endpoints for component communication
    - Message queuing and routing
    - Priority-based message handling
    - Component health monitoring
    - Event correlation and tracking
    """
    
    def __init__(self, port: int = 8001):
        self.app = FastAPI(title="AntBot System Communication API", version="1.0.0")
        self.port = port
        
        # Message management
        self.message_queue = asyncio.PriorityQueue()
        self.message_history = deque(maxlen=1000)
        self.component_subscribers: Dict[str, List[str]] = {}
        
        # Component tracking
        self.registered_components = {}
        self.component_health = {}
        
        # Performance metrics
        self.api_metrics = {
            "total_messages": 0,
            "messages_by_type": {},
            "average_processing_time_ms": 0.0,
            "failed_deliveries": 0
        }
        
        # Setup API routes
        self._setup_routes()
        
        logger.info(f"🔗 System Communication API initialized on port {port}")
    
    async def start_server(self):
        """Start the communication API server"""
        try:
            # Start message processing loop
            asyncio.create_task(self._message_processing_loop())
            asyncio.create_task(self._health_monitoring_loop())
            
            # Start FastAPI server
            config = uvicorn.Config(
                app=self.app,
                host="127.0.0.1",
                port=self.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start communication API: {str(e)}")
    
    def _setup_routes(self):
        """Setup FastAPI routes for component communication"""
        
        @self.app.post("/api/v1/ai/confidence")
        async def ai_confidence_update(message: AIConfidenceMessage, background_tasks: BackgroundTasks):
            """Receive AI confidence updates"""
            try:
                internal_msg = InternalMessage(
                    message_id=f"ai_conf_{int(time.time() * 1000)}",
                    message_type=MessageType.AI_CONFIDENCE_UPDATE,
                    priority=Priority.MEDIUM,
                    source_component="adaptive_learning_system",
                    target_component="realtime_defense_system",
                    payload=message.dict()
                )
                
                background_tasks.add_task(self._queue_message, internal_msg)
                return {"status": "accepted", "message_id": internal_msg.message_id}
                
            except Exception as e:
                logger.error(f"Error processing AI confidence update: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/threats/alert")
        async def threat_alert(message: ThreatAlertMessage, background_tasks: BackgroundTasks):
            """Receive threat alerts from defense system"""
            try:
                priority = Priority.CRITICAL if message.threat_level >= 4 else Priority.HIGH
                
                internal_msg = InternalMessage(
                    message_id=f"threat_{message.threat_id}",
                    message_type=MessageType.THREAT_ALERT,
                    priority=priority,
                    source_component="realtime_defense_system",
                    target_component=None,  # Broadcast to all
                    payload=message.dict()
                )
                
                background_tasks.add_task(self._queue_message, internal_msg)
                return {"status": "accepted", "message_id": internal_msg.message_id}
                
            except Exception as e:
                logger.error(f"Error processing threat alert: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/security/event")
        async def security_event(message: SecurityEventMessage, background_tasks: BackgroundTasks):
            """Receive security events"""
            try:
                priority = Priority.CRITICAL if message.severity == "critical" else Priority.HIGH
                
                internal_msg = InternalMessage(
                    message_id=message.event_id,
                    message_type=MessageType.SECURITY_EVENT,
                    priority=priority,
                    source_component="secure_wallet_manager",
                    target_component="realtime_defense_system" if message.requires_action else None,
                    payload=message.dict()
                )
                
                background_tasks.add_task(self._queue_message, internal_msg)
                return {"status": "accepted", "message_id": internal_msg.message_id}
                
            except Exception as e:
                logger.error(f"Error processing security event: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/defense/action")
        async def defense_action_completed(message: DefenseActionMessage, background_tasks: BackgroundTasks):
            """Receive defense action completion notifications"""
            try:
                internal_msg = InternalMessage(
                    message_id=message.action_id,
                    message_type=MessageType.DEFENSE_ACTION,
                    priority=Priority.MEDIUM,
                    source_component="realtime_defense_system",
                    target_component="adaptive_learning_system",  # For learning feedback
                    payload=message.dict()
                )
                
                background_tasks.add_task(self._queue_message, internal_msg)
                return {"status": "accepted", "message_id": internal_msg.message_id}
                
            except Exception as e:
                logger.error(f"Error processing defense action: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/models/update")
        async def model_update(message: ModelUpdateMessage, background_tasks: BackgroundTasks):
            """Receive model update notifications"""
            try:
                internal_msg = InternalMessage(
                    message_id=f"model_{message.model_id}_{int(time.time())}",
                    message_type=MessageType.MODEL_UPDATE,
                    priority=Priority.MEDIUM,
                    source_component="model_deployment_manager",
                    target_component=None,  # Broadcast
                    payload=message.dict()
                )
                
                background_tasks.add_task(self._queue_message, internal_msg)
                return {"status": "accepted", "message_id": internal_msg.message_id}
                
            except Exception as e:
                logger.error(f"Error processing model update: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/components/register")
        async def register_component(component_name: str, callback_url: str):
            """Register component for message delivery"""
            try:
                self.registered_components[component_name] = {
                    "callback_url": callback_url,
                    "registered_at": time.time(),
                    "last_seen": time.time()
                }
                
                logger.info(f"📋 Component registered: {component_name} -> {callback_url}")
                return {"status": "registered", "component": component_name}
                
            except Exception as e:
                logger.error(f"Error registering component: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/system/status")
        async def get_system_status():
            """Get overall system status"""
            try:
                return {
                    "api_status": "healthy",
                    "registered_components": len(self.registered_components),
                    "message_queue_size": self.message_queue.qsize(),
                    "total_messages_processed": self.api_metrics["total_messages"],
                    "component_health": self.component_health,
                    "uptime_seconds": time.time() - self._start_time if hasattr(self, '_start_time') else 0
                }
                
            except Exception as e:
                logger.error(f"Error getting system status: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _queue_message(self, message: InternalMessage):
        """Queue message for processing"""
        try:
            # Priority queue uses tuple (priority, timestamp, message)
            # Lower priority number = higher priority
            priority_value = 6 - message.priority.value  # Invert for correct ordering
            await self.message_queue.put((priority_value, message.timestamp, message))
            
            # Update metrics
            self.api_metrics["total_messages"] += 1
            msg_type = message.message_type.value
            self.api_metrics["messages_by_type"][msg_type] = self.api_metrics["messages_by_type"].get(msg_type, 0) + 1
            
            logger.debug(f"Message queued: {message.message_type.value} (Priority: {message.priority.value})")
            
        except Exception as e:
            logger.error(f"Error queueing message: {str(e)}")
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        while True:
            try:
                # Get next message from priority queue
                priority, timestamp, message = await self.message_queue.get()
                
                start_time = time.time()
                
                # Process message based on type and target
                await self._process_message(message)
                
                # Update performance metrics
                processing_time = (time.time() - start_time) * 1000
                current_avg = self.api_metrics["average_processing_time_ms"]
                total_messages = self.api_metrics["total_messages"]
                self.api_metrics["average_processing_time_ms"] = (
                    (current_avg * (total_messages - 1) + processing_time) / total_messages
                )
                
                # Store in history
                self.message_history.append(message)
                
            except Exception as e:
                logger.error(f"Message processing error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message: InternalMessage):
        """Process individual message"""
        try:
            # Handle different message types
            if message.message_type == MessageType.AI_CONFIDENCE_UPDATE:
                await self._handle_ai_confidence_update(message)
            elif message.message_type == MessageType.THREAT_ALERT:
                await self._handle_threat_alert(message)
            elif message.message_type == MessageType.SECURITY_EVENT:
                await self._handle_security_event(message)
            elif message.message_type == MessageType.DEFENSE_ACTION:
                await self._handle_defense_action(message)
            elif message.message_type == MessageType.MODEL_UPDATE:
                await self._handle_model_update(message)
            
            logger.debug(f"Processed message: {message.message_id}")
            
        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {str(e)}")
            self.api_metrics["failed_deliveries"] += 1
    
    async def _handle_ai_confidence_update(self, message: InternalMessage):
        """Handle AI confidence update message"""
        # Forward to defense system for threat assessment adjustment
        if "realtime_defense_system" in self.registered_components:
            await self._deliver_to_component("realtime_defense_system", message)
    
    async def _handle_threat_alert(self, message: InternalMessage):
        """Handle threat alert message - broadcast to all components"""
        # Critical threats should reach all components
        for component_name in self.registered_components:
            await self._deliver_to_component(component_name, message)
    
    async def _handle_security_event(self, message: InternalMessage):
        """Handle security event message"""
        # Security events may need specific routing
        if message.target_component and message.target_component in self.registered_components:
            await self._deliver_to_component(message.target_component, message)
    
    async def _handle_defense_action(self, message: InternalMessage):
        """Handle defense action completion - send feedback to AI system"""
        if "adaptive_learning_system" in self.registered_components:
            await self._deliver_to_component("adaptive_learning_system", message)
    
    async def _handle_model_update(self, message: InternalMessage):
        """Handle model update notification - broadcast to interested components"""
        interested_components = ["realtime_defense_system", "secure_wallet_manager"]
        for component in interested_components:
            if component in self.registered_components:
                await self._deliver_to_component(component, message)
    
    async def _deliver_to_component(self, component_name: str, message: InternalMessage):
        """Deliver message to specific component"""
        try:
            # In a real implementation, this would make HTTP calls to component callbacks
            # For now, we'll log the delivery
            logger.info(f"📤 Delivering {message.message_type.value} to {component_name}")
            
            # Update component last_seen
            if component_name in self.registered_components:
                self.registered_components[component_name]["last_seen"] = time.time()
            
        except Exception as e:
            logger.error(f"Failed to deliver message to {component_name}: {str(e)}")
            self.api_metrics["failed_deliveries"] += 1
    
    async def _health_monitoring_loop(self):
        """Monitor component health"""
        while True:
            try:
                current_time = time.time()
                
                for component_name, info in self.registered_components.items():
                    last_seen = info["last_seen"]
                    age_seconds = current_time - last_seen
                    
                    if age_seconds < 60:  # Healthy
                        self.component_health[component_name] = "healthy"
                    elif age_seconds < 300:  # Warning
                        self.component_health[component_name] = "warning"
                    else:  # Unhealthy
                        self.component_health[component_name] = "unhealthy"
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring error: {str(e)}")
                await asyncio.sleep(60)
    
    def get_api_metrics(self) -> Dict[str, Any]:
        """Get API performance metrics"""
        return {
            "total_messages": self.api_metrics["total_messages"],
            "messages_by_type": self.api_metrics["messages_by_type"],
            "average_processing_time_ms": self.api_metrics["average_processing_time_ms"],
            "failed_deliveries": self.api_metrics["failed_deliveries"],
            "queue_size": self.message_queue.qsize(),
            "registered_components": len(self.registered_components),
            "healthy_components": sum(1 for status in self.component_health.values() if status == "healthy")
        }

# Utility functions for components to use
class ComponentClient:
    """Client for components to communicate with the API"""
    
    def __init__(self, component_name: str, api_base_url: str = "http://127.0.0.1:8001"):
        self.component_name = component_name
        self.api_base_url = api_base_url
        self.session = None  # Would use aiohttp.ClientSession in real implementation
    
    async def send_ai_confidence(self, token_address: str, confidence: float, 
                               model_id: str, novelty_score: float) -> bool:
        """Send AI confidence update"""
        # Implementation would use aiohttp to POST to /api/v1/ai/confidence
        logger.info(f"📊 Sending AI confidence: {confidence:.2f} for {token_address[:8]}...")
        return True
    
    async def send_threat_alert(self, threat_id: str, threat_type: str, 
                              threat_level: int, token_address: str, confidence: float) -> bool:
        """Send threat alert"""
        logger.warning(f"🚨 Sending threat alert: {threat_type} (Level {threat_level})")
        return True
    
    async def send_security_event(self, event_id: str, event_type: str, 
                                severity: str, description: str) -> bool:
        """Send security event"""
        logger.info(f"🔐 Sending security event: {event_type} ({severity})")
        return True 