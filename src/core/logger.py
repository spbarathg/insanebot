"""
SystemLogger - Advanced logging system for Ant Bot

Provides structured logging, log aggregation, filtering, rotation,
and real-time log analysis with performance monitoring.
"""

import logging
import json
import os
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import gzip
import shutil

@dataclass
class LogEvent:
    """Represents a structured log event"""
    timestamp: float
    level: str
    logger_name: str
    message: str
    component: str
    event_type: str
    context: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

@dataclass
class LogMetrics:
    """Metrics for log analysis"""
    total_logs: int = 0
    logs_by_level: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    logs_by_component: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_rate: float = 0.0
    avg_logs_per_minute: float = 0.0
    critical_errors: List[str] = field(default_factory=list)

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def __init__(self, include_extra=True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record):
        # Base log data
        log_data = {
            "timestamp": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if available
        if self.include_extra and hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Add common fields
        if hasattr(record, 'component'):
            log_data['component'] = record.component
        if hasattr(record, 'event_type'):
            log_data['event_type'] = record.event_type
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)

class LogFilter:
    """Advanced log filtering"""
    
    def __init__(self):
        self.filters = []
        self.component_filters = {}
        self.level_filters = {}
        
    def add_component_filter(self, component: str, min_level: str):
        """Add component-specific level filter"""
        self.component_filters[component] = getattr(logging, min_level.upper())
    
    def add_level_filter(self, min_level: str, max_level: str = None):
        """Add global level filter"""
        min_level_num = getattr(logging, min_level.upper())
        max_level_num = getattr(logging, max_level.upper()) if max_level else 100
        self.level_filters['global'] = (min_level_num, max_level_num)
    
    def should_log(self, record) -> bool:
        """Determine if record should be logged"""
        # Check component-specific filters
        component = getattr(record, 'component', None)
        if component and component in self.component_filters:
            return record.levelno >= self.component_filters[component]
        
        # Check global level filters
        if 'global' in self.level_filters:
            min_level, max_level = self.level_filters['global']
            return min_level <= record.levelno <= max_level
        
        return True

class LogAggregator:
    """Aggregates and analyzes log data"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.log_buffer: deque = deque(maxlen=window_size)
        self.metrics = LogMetrics()
        self.error_patterns = defaultdict(int)
        self.performance_logs = deque(maxlen=100)
        
    def add_log(self, log_event: LogEvent):
        """Add log event to aggregator"""
        self.log_buffer.append(log_event)
        self._update_metrics(log_event)
        self._analyze_patterns(log_event)
    
    def _update_metrics(self, log_event: LogEvent):
        """Update log metrics"""
        self.metrics.total_logs += 1
        self.metrics.logs_by_level[log_event.level] += 1
        self.metrics.logs_by_component[log_event.component] += 1
        
        # Calculate error rate
        total_errors = sum(
            count for level, count in self.metrics.logs_by_level.items()
            if level in ['ERROR', 'CRITICAL']
        )
        self.metrics.error_rate = total_errors / max(1, self.metrics.total_logs)
        
        # Track critical errors
        if log_event.level == 'CRITICAL':
            self.metrics.critical_errors.append(log_event.message)
            if len(self.metrics.critical_errors) > 10:
                self.metrics.critical_errors = self.metrics.critical_errors[-10:]
    
    def _analyze_patterns(self, log_event: LogEvent):
        """Analyze log patterns for issues"""
        if log_event.level in ['ERROR', 'CRITICAL']:
            # Extract error pattern
            error_key = f"{log_event.component}:{log_event.event_type}"
            self.error_patterns[error_key] += 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            "total_logs": self.metrics.total_logs,
            "logs_by_level": dict(self.metrics.logs_by_level),
            "logs_by_component": dict(self.metrics.logs_by_component),
            "error_rate": self.metrics.error_rate,
            "critical_errors": self.metrics.critical_errors.copy(),
            "top_error_patterns": dict(list(self.error_patterns.most_common(5))),
            "buffer_size": len(self.log_buffer)
        }

class SystemLogger:
    """Advanced system logging manager"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Core logging components
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self.formatters: Dict[str, logging.Formatter] = {}
        
        # Advanced features
        self.log_filter = LogFilter()
        self.log_aggregator = LogAggregator(self.config.get('buffer_size', 1000))
        self.log_buffer: deque = deque(maxlen=10000)
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.correlation_tracking = {}
        self.session_tracking = {}
        
        # Log routing and alerting
        self.log_routes: Dict[str, List[Callable]] = defaultdict(list)
        self.alert_callbacks: List[Callable] = []
        
        # Async processing
        self.log_queue = asyncio.Queue()
        self.processing_task = None
        self.is_running = False
        
        self._initialized = False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration"""
        return {
            "level": "INFO",
            "format": "structured",
            "buffer_size": 1000,
            "log_dir": "logs",
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "backup_count": 5,
            "compression": True,
            "console_output": True,
            "file_output": True,
            "structured_output": True,
            "performance_tracking": True,
            "correlation_tracking": True
        }
    
    async def initialize(self) -> bool:
        """Initialize the logging system"""
        try:
            # Create log directory
            log_dir = Path(self.config["log_dir"])
            log_dir.mkdir(exist_ok=True)
            
            # Setup formatters
            await self._setup_formatters()
            
            # Setup handlers
            await self._setup_handlers()
            
            # Setup loggers
            await self._setup_loggers()
            
            # Start async processing
            await self._start_async_processing()
            
            self._initialized = True
            logger = self.get_logger("SystemLogger")
            logger.info("SystemLogger initialized successfully")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize SystemLogger: {e}")
            return False
    
    async def _setup_formatters(self):
        """Setup log formatters"""
        # Structured JSON formatter
        self.formatters["structured"] = StructuredFormatter()
        
        # Standard text formatter
        self.formatters["standard"] = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Detailed formatter for development
        self.formatters["detailed"] = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )
    
    async def _setup_handlers(self):
        """Setup log handlers"""
        log_dir = Path(self.config["log_dir"])
        
        # Console handler
        if self.config.get("console_output", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.formatters["standard"])
            console_handler.setLevel(getattr(logging, self.config["level"]))
            self.handlers["console"] = console_handler
        
        # File handler with rotation
        if self.config.get("file_output", True):
            file_handler = RotatingFileHandler(
                log_dir / "ant_bot.log",
                maxBytes=self.config["max_file_size"],
                backupCount=self.config["backup_count"]
            )
            file_handler.setFormatter(self.formatters["detailed"])
            file_handler.setLevel(logging.DEBUG)
            self.handlers["file"] = file_handler
        
        # Structured JSON handler
        if self.config.get("structured_output", True):
            json_handler = RotatingFileHandler(
                log_dir / "ant_bot_structured.jsonl",
                maxBytes=self.config["max_file_size"],
                backupCount=self.config["backup_count"]
            )
            json_handler.setFormatter(self.formatters["structured"])
            json_handler.setLevel(logging.DEBUG)
            self.handlers["structured"] = json_handler
        
        # Error-specific handler
        error_handler = RotatingFileHandler(
            log_dir / "ant_bot_errors.log",
            maxBytes=self.config["max_file_size"],
            backupCount=self.config["backup_count"]
        )
        error_handler.setFormatter(self.formatters["detailed"])
        error_handler.setLevel(logging.ERROR)
        self.handlers["error"] = error_handler
        
        # Performance handler
        if self.config.get("performance_tracking", True):
            perf_handler = RotatingFileHandler(
                log_dir / "ant_bot_performance.log",
                maxBytes=self.config["max_file_size"],
                backupCount=self.config["backup_count"]
            )
            perf_handler.setFormatter(self.formatters["structured"])
            perf_handler.setLevel(logging.INFO)
            self.handlers["performance"] = perf_handler
    
    async def _setup_loggers(self):
        """Setup component loggers"""
        components = [
            "SystemLogger", "ConfigManager", "SecurityManager", "SystemMetrics",
            "MonetaryLayer", "WorkerLayer", "CarwashLayer", "IntelligenceLayer", "DataLayer",
            "ArchitectureIteration", "PerformanceAmplification", "PluginSystem",
            "FoundingQueen", "AntQueen", "WorkerAnts", "AntDrone", "AccountingAnt", "AntPrincess"
        ]
        
        for component in components:
            logger = logging.getLogger(component)
            logger.setLevel(logging.DEBUG)
            
            # Add handlers
            for handler in self.handlers.values():
                logger.addHandler(handler)
            
            # Store reference
            self.loggers[component] = logger
            
            # Add custom attributes
            self._enhance_logger(logger, component)
    
    def _enhance_logger(self, logger: logging.Logger, component: str):
        """Enhance logger with custom functionality"""
        original_log = logger._log
        
        def enhanced_log(level, msg, args, exc_info=None, extra=None, stack_info=False, **kwargs):
            # Add component information
            if extra is None:
                extra = {}
            extra['component'] = component
            
            # Add correlation tracking
            if self.config.get("correlation_tracking", True):
                correlation_id = self._get_correlation_id()
                if correlation_id:
                    extra['correlation_id'] = correlation_id
            
            # Create log event for aggregation
            log_event = LogEvent(
                timestamp=time.time(),
                level=logging.getLevelName(level),
                logger_name=logger.name,
                message=msg % args if args else str(msg),
                component=component,
                event_type=extra.get('event_type', 'general'),
                context=extra.copy(),
                correlation_id=extra.get('correlation_id'),
                session_id=extra.get('session_id')
            )
            
            # Add to aggregator
            self.log_aggregator.add_log(log_event)
            
            # Check for alerts
            self._check_alerts(log_event)
            
            # Call original log method
            return original_log(level, msg, args, exc_info, extra, stack_info, **kwargs)
        
        logger._log = enhanced_log
    
    async def _start_async_processing(self):
        """Start async log processing"""
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_logs())
    
    async def _process_logs(self):
        """Process logs asynchronously"""
        while self.is_running:
            try:
                # Process any queued operations
                if not self.log_queue.empty():
                    log_operation = await self.log_queue.get()
                    await self._execute_log_operation(log_operation)
                
                # Periodic maintenance
                await self._periodic_maintenance()
                
                await asyncio.sleep(0.1)  # Small delay
                
            except Exception as e:
                print(f"Error in log processing: {e}")
                await asyncio.sleep(1)
    
    async def _execute_log_operation(self, operation: Dict[str, Any]):
        """Execute a log operation"""
        op_type = operation.get("type")
        
        if op_type == "compress_old_logs":
            await self._compress_old_logs()
        elif op_type == "cleanup_old_logs":
            await self._cleanup_old_logs()
        elif op_type == "analyze_patterns":
            await self._analyze_log_patterns()
    
    async def _periodic_maintenance(self):
        """Perform periodic log maintenance"""
        current_time = time.time()
        
        # Check if compression is needed (daily)
        if not hasattr(self, '_last_compression') or current_time - self._last_compression > 86400:
            await self.log_queue.put({"type": "compress_old_logs"})
            self._last_compression = current_time
        
        # Check if cleanup is needed (weekly)
        if not hasattr(self, '_last_cleanup') or current_time - self._last_cleanup > 604800:
            await self.log_queue.put({"type": "cleanup_old_logs"})
            self._last_cleanup = current_time
    
    async def _compress_old_logs(self):
        """Compress old log files"""
        if not self.config.get("compression", True):
            return
        
        log_dir = Path(self.config["log_dir"])
        
        for log_file in log_dir.glob("*.log.*"):
            if not log_file.name.endswith('.gz'):
                try:
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(f"{log_file}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    log_file.unlink()  # Remove original file
                except Exception as e:
                    print(f"Error compressing {log_file}: {e}")
    
    async def _cleanup_old_logs(self):
        """Cleanup old compressed log files"""
        log_dir = Path(self.config["log_dir"])
        retention_days = self.config.get("retention_days", 30)
        cutoff_time = time.time() - (retention_days * 86400)
        
        for log_file in log_dir.glob("*.gz"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                except Exception as e:
                    print(f"Error deleting old log {log_file}: {e}")
    
    async def _analyze_log_patterns(self):
        """Analyze log patterns for insights"""
        # This would contain more sophisticated log analysis
        pass
    
    def get_logger(self, component: str) -> logging.Logger:
        """Get logger for a component"""
        if component not in self.loggers:
            # Create new logger if not exists
            logger = logging.getLogger(component)
            logger.setLevel(logging.DEBUG)
            
            for handler in self.handlers.values():
                logger.addHandler(handler)
            
            self._enhance_logger(logger, component)
            self.loggers[component] = logger
        
        return self.loggers[component]
    
    def log_performance(self, component: str, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        if not self.config.get("performance_tracking", True):
            return
        
        perf_data = {
            "component": component,
            "operation": operation,
            "duration": duration,
            "timestamp": time.time(),
            **kwargs
        }
        
        self.performance_metrics[component].append(perf_data)
        
        # Log to performance handler
        if "performance" in self.handlers:
            logger = self.get_logger("Performance")
            logger.info("Performance metric", extra={
                "event_type": "performance",
                "performance_data": perf_data
            })
    
    def start_correlation(self, operation: str) -> str:
        """Start correlation tracking"""
        correlation_id = f"{operation}_{int(time.time())}_{id(threading.current_thread())}"
        self.correlation_tracking[threading.current_thread().ident] = correlation_id
        return correlation_id
    
    def end_correlation(self):
        """End correlation tracking"""
        thread_id = threading.current_thread().ident
        if thread_id in self.correlation_tracking:
            del self.correlation_tracking[thread_id]
    
    def _get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID"""
        thread_id = threading.current_thread().ident
        return self.correlation_tracking.get(thread_id)
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for log alerts"""
        self.alert_callbacks.append(callback)
    
    def _check_alerts(self, log_event: LogEvent):
        """Check if log event should trigger alerts"""
        # Alert on critical errors
        if log_event.level == "CRITICAL":
            for callback in self.alert_callbacks:
                try:
                    callback(log_event)
                except Exception as e:
                    print(f"Error in alert callback: {e}")
        
        # Alert on high error rates
        if self.log_aggregator.metrics.error_rate > 0.1:  # 10% error rate
            for callback in self.alert_callbacks:
                try:
                    callback(log_event)
                except Exception as e:
                    print(f"Error in alert callback: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system logging metrics"""
        return {
            "log_metrics": self.log_aggregator.get_metrics_summary(),
            "performance_metrics": {
                component: len(metrics) for component, metrics in self.performance_metrics.items()
            },
            "active_loggers": len(self.loggers),
            "active_handlers": len(self.handlers),
            "correlation_tracking": len(self.correlation_tracking),
            "session_tracking": len(self.session_tracking),
            "alert_callbacks": len(self.alert_callbacks),
            "log_queue_size": self.log_queue.qsize() if hasattr(self.log_queue, 'qsize') else 0,
            "is_running": self.is_running
        }
    
    async def cleanup(self):
        """Cleanup logging system"""
        try:
            self.is_running = False
            
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            
            # Close all handlers
            for handler in self.handlers.values():
                handler.close()
            
            # Clear caches
            self.performance_metrics.clear()
            self.correlation_tracking.clear()
            self.log_buffer.clear()
            
            logger = self.get_logger("SystemLogger")
            logger.info("SystemLogger cleanup completed")
            
        except Exception as e:
            print(f"Error during SystemLogger cleanup: {e}") 