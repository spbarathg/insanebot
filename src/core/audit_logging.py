"""
Production-Grade Audit Logging System

Comprehensive audit trail for trading bot operations:
- All trading decisions and executions
- System configuration changes
- Security events and access logs
- Performance metrics and errors
- Compliance reporting
- Real-time monitoring and alerting
"""

import os
import json
import time
import asyncio
import logging
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from contextlib import asynccontextmanager
import aiofiles
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from functools import wraps
import queue

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    TRADING_DECISION = "trading_decision"
    TRADE_EXECUTION = "trade_execution"
    SYSTEM_CONFIG_CHANGE = "system_config_change"
    SECURITY_EVENT = "security_event"
    ERROR_EVENT = "error_event"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    API_ACCESS = "api_access"
    PERFORMANCE_METRIC = "performance_metric"
    RISK_EVENT = "risk_event"
    AI_DECISION = "ai_decision"
    WALLET_ACCESS = "wallet_access"
    COMPLIANCE_CHECK = "compliance_check"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONFIG_CHANGE = "config_change"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    DATA_ACCESS = "data_access"
    API_REQUEST = "api_request"
    BOT_STATE_CHANGE = "bot_state_change"
    WALLET_OPERATION = "wallet_operation"
    BACKUP_OPERATION = "backup_operation"
    RECOVERY_OPERATION = "recovery_operation"
    SECRET_ACCESS = "secret_access"

class AuditLevel(Enum):
    """Audit event severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AuditSeverity(Enum):
    """Severity levels for audit events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Comprehensive audit event record"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: AuditEventType = AuditEventType.SYSTEM_CONFIG_CHANGE
    level: AuditLevel = AuditLevel.INFO
    component: str = "unknown"
    action: str = ""
    description: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    
    # Trading specific fields
    token_address: Optional[str] = None
    trade_amount: Optional[float] = None
    trade_price: Optional[float] = None
    trade_side: Optional[str] = None  # 'buy', 'sell'
    profit_loss: Optional[float] = None
    
    # AI/Decision specific fields
    ai_confidence: Optional[float] = None
    decision_factors: Optional[Dict[str, Any]] = None
    risk_score: Optional[float] = None
    
    # System specific fields
    system_metrics: Optional[Dict[str, Any]] = None
    configuration_changes: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Computed fields
    hash: Optional[str] = None
    
    def __post_init__(self):
        """Compute hash for integrity verification"""
        if self.hash is None:
            self.hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of event data for integrity"""
        data = {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'event_type': self.event_type.value,
            'level': self.level.value,
            'component': self.component,
            'action': self.action,
            'description': self.description,
            'context': self.context
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

class AuditLogWriter:
    """Asynchronous audit log writer with rotation and compression"""
    
    def __init__(self, log_dir: Path, max_file_size: int = 100 * 1024 * 1024):  # 100MB
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size
        self.current_file = None
        self.current_file_size = 0
        self.write_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.writer_thread = None
        self.running = False
        
        # Initialize current log file
        self._initialize_current_file()
    
    def _initialize_current_file(self):
        """Initialize current log file"""
        today = datetime.now().strftime("%Y%m%d")
        self.current_file = self.log_dir / f"audit_{today}.jsonl"
        
        if self.current_file.exists():
            self.current_file_size = self.current_file.stat().st_size
        else:
            self.current_file_size = 0
    
    def start(self):
        """Start the audit log writer"""
        if not self.running:
            self.running = True
            self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self.writer_thread.start()
            logger.info("Audit log writer started")
    
    def stop(self):
        """Stop the audit log writer"""
        self.running = False
        if self.writer_thread:
            self.writer_thread.join(timeout=5)
        logger.info("Audit log writer stopped")
    
    def _writer_loop(self):
        """Main writer loop"""
        while self.running:
            try:
                # Get event from queue with timeout
                try:
                    event = self.write_queue.get(timeout=1)
                except Empty:
                    continue
                
                # Write event to file
                self._write_event_sync(event)
                self.write_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in audit writer loop: {str(e)}")
    
    def _write_event_sync(self, event: AuditEvent):
        """Synchronously write event to file"""
        try:
            # Check if file rotation is needed
            if self._needs_rotation():
                self._rotate_file()
            
            # Write event
            event_json = json.dumps(asdict(event), default=str) + '\n'
            
            with open(self.current_file, 'a', encoding='utf-8') as f:
                f.write(event_json)
                f.flush()
            
            self.current_file_size += len(event_json.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Failed to write audit event: {str(e)}")
    
    def _needs_rotation(self) -> bool:
        """Check if log file needs rotation"""
        if self.current_file_size >= self.max_file_size:
            return True
        
        # Check if date has changed
        today = datetime.now().strftime("%Y%m%d")
        current_date = self.current_file.stem.split('_')[1]
        return today != current_date
    
    def _rotate_file(self):
        """Rotate current log file"""
        try:
            # Archive current file if it exists and has content
            if self.current_file.exists() and self.current_file_size > 0:
                timestamp = datetime.now().strftime("%H%M%S")
                archived_name = f"{self.current_file.stem}_{timestamp}.jsonl"
                archived_file = self.log_dir / archived_name
                self.current_file.rename(archived_file)
                
                # Compress archived file in background
                self.executor.submit(self._compress_file, archived_file)
            
            # Create new current file
            self._initialize_current_file()
            
        except Exception as e:
            logger.error(f"Failed to rotate audit log: {str(e)}")
    
    def _compress_file(self, file_path: Path):
        """Compress log file"""
        try:
            import gzip
            compressed_path = file_path.with_suffix('.jsonl.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original file
            file_path.unlink()
            logger.info(f"Compressed audit log: {compressed_path}")
            
        except Exception as e:
            logger.error(f"Failed to compress audit log: {str(e)}")
    
    async def write_event(self, event: AuditEvent):
        """Queue event for writing"""
        try:
            self.write_queue.put(event, timeout=1)
        except Exception as e:
            logger.error(f"Failed to queue audit event: {str(e)}")

class AuditLogReader:
    """Read and query audit logs"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
    
    def get_log_files(self) -> List[Path]:
        """Get all audit log files"""
        files = []
        for pattern in ['*.jsonl', '*.jsonl.gz']:
            files.extend(self.log_dir.glob(pattern))
        return sorted(files)
    
    async def read_events(self, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         event_types: Optional[List[AuditEventType]] = None,
                         component: Optional[str] = None,
                         limit: Optional[int] = None) -> List[AuditEvent]:
        """Read events with filtering"""
        events = []
        count = 0
        
        for log_file in self.get_log_files():
            if limit and count >= limit:
                break
                
            file_events = await self._read_file(log_file)
            for event in file_events:
                if limit and count >= limit:
                    break
                
                # Apply filters
                if start_time and event.timestamp < start_time.timestamp():
                    continue
                if end_time and event.timestamp > end_time.timestamp():
                    continue
                if event_types and event.event_type not in event_types:
                    continue
                if component and event.component != component:
                    continue
                
                events.append(event)
                count += 1
        
        return events
    
    async def _read_file(self, file_path: Path) -> List[AuditEvent]:
        """Read events from a single file"""
        events = []
        
        try:
            if file_path.suffix == '.gz':
                import gzip
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    content = f.read()
            else:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
            
            for line in content.strip().split('\n'):
                if line:
                    try:
                        event_data = json.loads(line)
                        # Convert enum strings back to enums
                        event_data['event_type'] = AuditEventType(event_data['event_type'])
                        event_data['level'] = AuditLevel(event_data['level'])
                        events.append(AuditEvent(**event_data))
                    except Exception as e:
                        logger.error(f"Failed to parse audit event: {str(e)}")
        
        except Exception as e:
            logger.error(f"Failed to read audit file {file_path}: {str(e)}")
        
        return events

class AuditLogger:
    """Main audit logging interface"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.log_dir = Path(self.config.get('log_dir', 'logs/audit'))
        self.max_file_size = self.config.get('max_file_size', 100 * 1024 * 1024)
        self.retention_days = self.config.get('retention_days', 90)
        
        # Initialize components
        self.writer = AuditLogWriter(self.log_dir, self.max_file_size)
        self.reader = AuditLogReader(self.log_dir)
        
        # Start writer
        self.writer.start()
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start periodic cleanup of old log files"""
        def cleanup_old_files():
            while True:
                try:
                    cutoff_time = time.time() - (self.retention_days * 24 * 3600)
                    for log_file in self.reader.get_log_files():
                        if log_file.stat().st_mtime < cutoff_time:
                            log_file.unlink()
                            logger.info(f"Deleted old audit log: {log_file}")
                except Exception as e:
                    logger.error(f"Error cleaning up audit logs: {str(e)}")
                
                # Sleep for 24 hours
                time.sleep(24 * 3600)
        
        cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
        cleanup_thread.start()
    
    async def log_event(self, event: AuditEvent):
        """Log an audit event"""
        await self.writer.write_event(event)
    
    async def log_trading_decision(self, 
                                  component: str,
                                  action: str,
                                  token_address: str,
                                  decision_factors: Dict[str, Any],
                                  ai_confidence: float,
                                  risk_score: float,
                                  context: Dict[str, Any] = None):
        """Log a trading decision"""
        event = AuditEvent(
            event_type=AuditEventType.TRADING_DECISION,
            level=AuditLevel.INFO,
            component=component,
            action=action,
            description=f"Trading decision: {action} for {token_address}",
            token_address=token_address,
            ai_confidence=ai_confidence,
            decision_factors=decision_factors,
            risk_score=risk_score,
            context=context or {},
            tags=['trading', 'decision']
        )
        await self.log_event(event)
    
    async def log_trade_execution(self,
                                 component: str,
                                 token_address: str,
                                 trade_side: str,
                                 trade_amount: float,
                                 trade_price: float,
                                 profit_loss: Optional[float] = None,
                                 context: Dict[str, Any] = None):
        """Log a trade execution"""
        event = AuditEvent(
            event_type=AuditEventType.TRADE_EXECUTION,
            level=AuditLevel.INFO,
            component=component,
            action=f"execute_{trade_side}",
            description=f"Executed {trade_side} of {trade_amount} {token_address} at {trade_price}",
            token_address=token_address,
            trade_amount=trade_amount,
            trade_price=trade_price,
            trade_side=trade_side,
            profit_loss=profit_loss,
            context=context or {},
            tags=['trading', 'execution']
        )
        await self.log_event(event)
    
    async def log_config_change(self,
                               component: str,
                               config_changes: Dict[str, Any],
                               user_id: Optional[str] = None,
                               context: Dict[str, Any] = None):
        """Log configuration changes"""
        event = AuditEvent(
            event_type=AuditEventType.SYSTEM_CONFIG_CHANGE,
            level=AuditLevel.WARNING,
            component=component,
            action="config_update",
            description=f"Configuration updated for {component}",
            user_id=user_id,
            configuration_changes=config_changes,
            context=context or {},
            tags=['config', 'change']
        )
        await self.log_event(event)
    
    async def log_security_event(self,
                                component: str,
                                action: str,
                                description: str,
                                level: AuditLevel = AuditLevel.WARNING,
                                user_id: Optional[str] = None,
                                ip_address: Optional[str] = None,
                                context: Dict[str, Any] = None):
        """Log security events"""
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_EVENT,
            level=level,
            component=component,
            action=action,
            description=description,
            user_id=user_id,
            ip_address=ip_address,
            context=context or {},
            tags=['security', action]
        )
        await self.log_event(event)
    
    async def log_error(self,
                       component: str,
                       error_type: str,
                       description: str,
                       error_details: Dict[str, Any],
                       level: AuditLevel = AuditLevel.ERROR,
                       context: Dict[str, Any] = None):
        """Log error events"""
        event = AuditEvent(
            event_type=AuditEventType.ERROR_EVENT,
            level=level,
            component=component,
            action=error_type,
            description=description,
            error_details=error_details,
            context=context or {},
            tags=['error', error_type]
        )
        await self.log_event(event)
    
    async def log_performance_metric(self,
                                   component: str,
                                   metrics: Dict[str, Any],
                                   context: Dict[str, Any] = None):
        """Log performance metrics"""
        event = AuditEvent(
            event_type=AuditEventType.PERFORMANCE_METRIC,
            level=AuditLevel.INFO,
            component=component,
            action="metrics_update",
            description=f"Performance metrics for {component}",
            system_metrics=metrics,
            context=context or {},
            tags=['performance', 'metrics']
        )
        await self.log_event(event)
    
    async def generate_compliance_report(self,
                                       start_date: datetime,
                                       end_date: datetime,
                                       output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Generate compliance report"""
        events = await self.reader.read_events(start_date, end_date)
        
        report = {
            'report_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_events': len(events),
                'trading_decisions': len([e for e in events if e.event_type == AuditEventType.TRADING_DECISION]),
                'trade_executions': len([e for e in events if e.event_type == AuditEventType.TRADE_EXECUTION]),
                'security_events': len([e for e in events if e.event_type == AuditEventType.SECURITY_EVENT]),
                'config_changes': len([e for e in events if e.event_type == AuditEventType.SYSTEM_CONFIG_CHANGE]),
                'error_events': len([e for e in events if e.event_type == AuditEventType.ERROR_EVENT])
            },
            'trading_summary': {
                'total_trades': len([e for e in events if e.event_type == AuditEventType.TRADE_EXECUTION]),
                'total_volume': sum(e.trade_amount or 0 for e in events if e.event_type == AuditEventType.TRADE_EXECUTION),
                'profit_loss': sum(e.profit_loss or 0 for e in events if e.event_type == AuditEventType.TRADE_EXECUTION and e.profit_loss)
            },
            'risk_events': [
                asdict(e) for e in events 
                if e.event_type in [AuditEventType.SECURITY_EVENT, AuditEventType.ERROR_EVENT] 
                and e.level in [AuditLevel.ERROR, AuditLevel.CRITICAL]
            ],
            'generated_at': datetime.now().isoformat(),
            'report_hash': hashlib.sha256(json.dumps(events, default=str).encode()).hexdigest()[:16]
        }
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def shutdown(self):
        """Shutdown audit logger"""
        self.writer.stop()

# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None

def initialize_audit_logging(config: Optional[Dict[str, Any]] = None) -> AuditLogger:
    """Initialize global audit logger"""
    global _audit_logger
    _audit_logger = AuditLogger(config)
    return _audit_logger

def get_audit_logger() -> Optional[AuditLogger]:
    """Get global audit logger instance"""
    return _audit_logger

@asynccontextmanager
async def audit_context(component: str, action: str, **kwargs):
    """Context manager for audit logging"""
    start_time = time.time()
    audit_logger = get_audit_logger()
    
    try:
        yield
        
        # Log successful completion
        if audit_logger:
            duration = time.time() - start_time
            await audit_logger.log_event(AuditEvent(
                event_type=AuditEventType.SYSTEM_CONFIG_CHANGE,
                level=AuditLevel.INFO,
                component=component,
                action=f"{action}_completed",
                description=f"Successfully completed {action}",
                context={**kwargs, 'duration_seconds': duration},
                tags=[component, action, 'success']
            ))
    
    except Exception as e:
        # Log failure
        if audit_logger:
            duration = time.time() - start_time
            await audit_logger.log_error(
                component=component,
                error_type=f"{action}_failed",
                description=f"Failed to complete {action}: {str(e)}",
                error_details={
                    'exception_type': type(e).__name__,
                    'exception_message': str(e),
                    'duration_seconds': duration
                },
                context=kwargs
            )
        raise

# Decorator for automatic audit logging
def audit_log(component: str, action: str, event_type: AuditEventType = AuditEventType.SYSTEM_CONFIG_CHANGE):
    """Decorator to automatically audit log function calls"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                if audit_logger:
                    duration = time.time() - start_time
                    await audit_logger.log_event(AuditEvent(
                        event_type=event_type,
                        level=AuditLevel.INFO,
                        component=component,
                        action=action,
                        description=f"Successfully executed {func.__name__}",
                        context={'duration_seconds': duration, 'function': func.__name__},
                        tags=[component, action, 'function_call']
                    ))
                
                return result
                
            except Exception as e:
                if audit_logger:
                    duration = time.time() - start_time
                    await audit_logger.log_error(
                        component=component,
                        error_type=f"{action}_error",
                        description=f"Error in {func.__name__}: {str(e)}",
                        error_details={
                            'function': func.__name__,
                            'exception_type': type(e).__name__,
                            'exception_message': str(e),
                            'duration_seconds': duration
                        }
                    )
                raise
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.create_task(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator 