#!/usr/bin/env python3
"""
Comprehensive Trading Bot Logger

This system logs EVERYTHING your trading bot does so that analysis can be performed
to identify issues, optimize performance, and suggest improvements.

Features:
- Function call tracing with parameters and results
- API request/response logging
- Decision point tracking with full context
- Error capture with stack traces
- Performance timing for every operation
- Market condition snapshots
- Configuration change tracking
- Memory and resource usage monitoring
- Real-time behavioral analysis

Usage:
    from monitoring.comprehensive_bot_logger import BotActivityLogger
    
    # Initialize comprehensive logging
    logger = BotActivityLogger()
    
    # Wrap your bot class
    @logger.wrap_class
    class YourTradingBot:
        pass
    
    # Or use decorators on individual methods
    @logger.log_function
    async def your_trading_function():
        pass
"""

import asyncio
import functools
import inspect
import json
import logging
import sys
import time
import traceback
import psutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
import sqlite3
from contextlib import asynccontextmanager

@dataclass
class FunctionCall:
    """Detailed function call information"""
    timestamp: float
    function_name: str
    module_name: str
    class_name: Optional[str]
    args: str  # JSON string of arguments
    kwargs: str  # JSON string of keyword arguments
    result: Optional[str]  # JSON string of result
    execution_time_ms: float
    success: bool
    error_type: Optional[str]
    error_message: Optional[str]
    stack_trace: Optional[str]
    memory_usage_mb: float
    cpu_usage_percent: float

@dataclass
class APICall:
    """API call logging"""
    timestamp: float
    api_name: str
    endpoint: str
    method: str
    request_data: str
    response_data: str
    response_time_ms: float
    status_code: Optional[int]
    success: bool
    error_message: Optional[str]

@dataclass
class DecisionPoint:
    """Critical decision point in trading logic"""
    timestamp: float
    decision_type: str
    context: str  # JSON string of all relevant context
    inputs: str   # JSON string of inputs that led to decision
    output: str   # JSON string of decision output
    confidence: float
    reasoning: str
    alternatives_considered: str  # JSON string of other options
    risk_factors: str  # JSON string of risk considerations

@dataclass
class SystemEvent:
    """System-level events and state changes"""
    timestamp: float
    event_type: str
    component: str
    event_data: str  # JSON string
    severity: str
    impact_assessment: str

class BotActivityLogger:
    """Comprehensive activity logger for trading bots"""
    
    def __init__(self, bot_name: str = "trading_bot"):
        self.bot_name = bot_name
        self.setup_logging()
        self.setup_database()
        
        # Real-time tracking
        self.active_functions = {}
        self.performance_baseline = {}
        self.error_patterns = []
        self.behavioral_flags = []
        
        # Configuration tracking
        self.last_config_snapshot = {}
        self.config_changes = []
        
        # Analysis results
        self.issues_detected = []
        self.optimization_suggestions = []
        self.performance_anomalies = []
        
    def setup_logging(self):
        """Setup comprehensive logging infrastructure"""
        log_dir = Path("logs/comprehensive")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create specialized loggers
        self.function_logger = self._create_logger('function_calls', log_dir / "function_calls.log")
        self.api_logger = self._create_logger('api_calls', log_dir / "api_calls.log")
        self.decision_logger = self._create_logger('decisions', log_dir / "decision_points.log")
        self.system_logger = self._create_logger('system_events', log_dir / "system_events.log")
        self.analysis_logger = self._create_logger('analysis', log_dir / "bot_analysis.log")
        
    def _create_logger(self, name: str, log_file: Path) -> logging.Logger:
        """Create a specialized logger"""
        logger = logging.getLogger(f'{self.bot_name}_{name}')
        logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
            
        handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def setup_database(self):
        """Setup database for structured logging"""
        db_path = Path("data/comprehensive_logs.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
        
        # Create tables for different log types
        self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS function_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                function_name TEXT,
                module_name TEXT,
                class_name TEXT,
                args TEXT,
                kwargs TEXT,
                result TEXT,
                execution_time_ms REAL,
                success BOOLEAN,
                error_type TEXT,
                error_message TEXT,
                stack_trace TEXT,
                memory_usage_mb REAL,
                cpu_usage_percent REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                api_name TEXT,
                endpoint TEXT,
                method TEXT,
                request_data TEXT,
                response_data TEXT,
                response_time_ms REAL,
                status_code INTEGER,
                success BOOLEAN,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS decision_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                decision_type TEXT,
                context TEXT,
                inputs TEXT,
                output TEXT,
                confidence REAL,
                reasoning TEXT,
                alternatives_considered TEXT,
                risk_factors TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                event_type TEXT,
                component TEXT,
                event_data TEXT,
                severity TEXT,
                impact_assessment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS bot_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                analysis_type TEXT,
                findings TEXT,
                recommendations TEXT,
                priority TEXT,
                status TEXT DEFAULT 'new',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.db_connection.commit()
        
    def log_function(self, func: Callable) -> Callable:
        """Decorator to log function calls with full context"""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._log_function_call(func, args, kwargs, is_async=True)
            
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(self._log_function_call(func, args, kwargs, is_async=False))
            
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    async def _log_function_call(self, func: Callable, args: tuple, kwargs: dict, is_async: bool):
        """Log detailed function call information"""
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_before = psutil.cpu_percent()
        
        function_name = func.__name__
        module_name = func.__module__
        class_name = None
        
        # Extract class name if this is a method call
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            
        # Sanitize arguments for logging
        safe_args = self._sanitize_for_json(args)
        safe_kwargs = self._sanitize_for_json(kwargs)
        
        try:
            # Execute the function
            if is_async:
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            execution_time = (time.time() - start_time) * 1000
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_after = psutil.cpu_percent()
            
            # Log successful execution
            function_call = FunctionCall(
                timestamp=start_time,
                function_name=function_name,
                module_name=module_name,
                class_name=class_name,
                args=json.dumps(safe_args, default=str),
                kwargs=json.dumps(safe_kwargs, default=str),
                result=json.dumps(self._sanitize_for_json(result), default=str),
                execution_time_ms=execution_time,
                success=True,
                error_type=None,
                error_message=None,
                stack_trace=None,
                memory_usage_mb=memory_after - memory_before,
                cpu_usage_percent=cpu_after - cpu_before
            )
            
            await self._store_function_call(function_call)
            
            # Analyze for performance issues
            await self._analyze_function_performance(function_call)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Log failed execution
            function_call = FunctionCall(
                timestamp=start_time,
                function_name=function_name,
                module_name=module_name,
                class_name=class_name,
                args=json.dumps(safe_args, default=str),
                kwargs=json.dumps(safe_kwargs, default=str),
                result=None,
                execution_time_ms=execution_time,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                memory_usage_mb=memory_after - memory_before,
                cpu_usage_percent=0
            )
            
            await self._store_function_call(function_call)
            
            # Analyze error patterns
            await self._analyze_error_pattern(function_call)
            
            raise  # Re-raise the exception
            
    def wrap_class(self, cls):
        """Class decorator to wrap all methods with logging"""
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if callable(attr) and not attr_name.startswith('__'):
                setattr(cls, attr_name, self.log_function(attr))
        return cls
        
    async def log_api_call(self, api_name: str, endpoint: str, method: str, 
                          request_data: Any = None, response_data: Any = None,
                          response_time_ms: float = 0, status_code: int = None,
                          success: bool = True, error_message: str = None):
        """Log API call with full context"""
        api_call = APICall(
            timestamp=time.time(),
            api_name=api_name,
            endpoint=endpoint,
            method=method,
            request_data=json.dumps(self._sanitize_for_json(request_data), default=str),
            response_data=json.dumps(self._sanitize_for_json(response_data), default=str),
            response_time_ms=response_time_ms,
            status_code=status_code,
            success=success,
            error_message=error_message
        )
        
        # Store API call
        self.api_logger.info(json.dumps(asdict(api_call), default=str))
        
        self.db_connection.execute("""
            INSERT INTO api_calls 
            (timestamp, api_name, endpoint, method, request_data, response_data,
             response_time_ms, status_code, success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            api_call.timestamp, api_call.api_name, api_call.endpoint,
            api_call.method, api_call.request_data, api_call.response_data,
            api_call.response_time_ms, api_call.status_code, api_call.success,
            api_call.error_message
        ))
        self.db_connection.commit()
        
        # Analyze API performance
        await self._analyze_api_performance(api_call)
        
    async def log_decision_point(self, decision_type: str, context: Dict[str, Any],
                                inputs: Dict[str, Any], output: Any, confidence: float,
                                reasoning: str, alternatives: List[Any] = None,
                                risk_factors: Dict[str, Any] = None):
        """Log critical decision points in trading logic"""
        decision_point = DecisionPoint(
            timestamp=time.time(),
            decision_type=decision_type,
            context=json.dumps(self._sanitize_for_json(context), default=str),
            inputs=json.dumps(self._sanitize_for_json(inputs), default=str),
            output=json.dumps(self._sanitize_for_json(output), default=str),
            confidence=confidence,
            reasoning=reasoning,
            alternatives_considered=json.dumps(self._sanitize_for_json(alternatives or []), default=str),
            risk_factors=json.dumps(self._sanitize_for_json(risk_factors or {}), default=str)
        )
        
        # Store decision point
        self.decision_logger.info(json.dumps(asdict(decision_point), default=str))
        
        self.db_connection.execute("""
            INSERT INTO decision_points 
            (timestamp, decision_type, context, inputs, output, confidence,
             reasoning, alternatives_considered, risk_factors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision_point.timestamp, decision_point.decision_type,
            decision_point.context, decision_point.inputs, decision_point.output,
            decision_point.confidence, decision_point.reasoning,
            decision_point.alternatives_considered, decision_point.risk_factors
        ))
        self.db_connection.commit()
        
        # Analyze decision quality
        await self._analyze_decision_quality(decision_point)
        
    async def log_system_event(self, event_type: str, component: str,
                              event_data: Dict[str, Any], severity: str = "INFO",
                              impact_assessment: str = ""):
        """Log system-level events"""
        system_event = SystemEvent(
            timestamp=time.time(),
            event_type=event_type,
            component=component,
            event_data=json.dumps(self._sanitize_for_json(event_data), default=str),
            severity=severity,
            impact_assessment=impact_assessment
        )
        
        # Store system event
        self.system_logger.info(json.dumps(asdict(system_event), default=str))
        
        self.db_connection.execute("""
            INSERT INTO system_events 
            (timestamp, event_type, component, event_data, severity, impact_assessment)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            system_event.timestamp, system_event.event_type, system_event.component,
            system_event.event_data, system_event.severity, system_event.impact_assessment
        ))
        self.db_connection.commit()
        
    async def _store_function_call(self, function_call: FunctionCall):
        """Store function call in database and log file"""
        # Log to file
        self.function_logger.info(json.dumps(asdict(function_call), default=str))
        
        # Store in database
        self.db_connection.execute("""
            INSERT INTO function_calls 
            (timestamp, function_name, module_name, class_name, args, kwargs,
             result, execution_time_ms, success, error_type, error_message,
             stack_trace, memory_usage_mb, cpu_usage_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            function_call.timestamp, function_call.function_name,
            function_call.module_name, function_call.class_name,
            function_call.args, function_call.kwargs, function_call.result,
            function_call.execution_time_ms, function_call.success,
            function_call.error_type, function_call.error_message,
            function_call.stack_trace, function_call.memory_usage_mb,
            function_call.cpu_usage_percent
        ))
        self.db_connection.commit()
        
    async def _analyze_function_performance(self, function_call: FunctionCall):
        """Analyze function performance for issues"""
        function_key = f"{function_call.module_name}.{function_call.function_name}"
        
        # Track performance baseline
        if function_key not in self.performance_baseline:
            self.performance_baseline[function_key] = {
                'avg_time': function_call.execution_time_ms,
                'call_count': 1,
                'total_time': function_call.execution_time_ms
            }
        else:
            baseline = self.performance_baseline[function_key]
            baseline['call_count'] += 1
            baseline['total_time'] += function_call.execution_time_ms
            baseline['avg_time'] = baseline['total_time'] / baseline['call_count']
            
            # Check for performance regression
            if function_call.execution_time_ms > baseline['avg_time'] * 3:
                await self._flag_performance_issue(function_call, baseline)
                
    async def _analyze_error_pattern(self, function_call: FunctionCall):
        """Analyze error patterns for recurring issues"""
        error_signature = f"{function_call.function_name}:{function_call.error_type}"
        
        # Track error frequency
        recent_errors = [e for e in self.error_patterns if e['timestamp'] > time.time() - 3600]  # Last hour
        error_count = sum(1 for e in recent_errors if e['signature'] == error_signature)
        
        if error_count >= 3:  # 3 or more errors in an hour
            await self._flag_error_pattern(function_call, error_count)
            
        # Add to error tracking
        self.error_patterns.append({
            'timestamp': function_call.timestamp,
            'signature': error_signature,
            'function_call': function_call
        })
        
    async def _analyze_api_performance(self, api_call: APICall):
        """Analyze API call performance"""
        if api_call.response_time_ms > 5000:  # Slow API call
            await self._store_analysis_finding(
                "slow_api_call",
                f"Slow API response: {api_call.api_name} took {api_call.response_time_ms:.0f}ms",
                f"Consider optimizing {api_call.endpoint} or implementing caching",
                "MEDIUM"
            )
            
        if not api_call.success:
            await self._store_analysis_finding(
                "api_failure",
                f"API call failed: {api_call.api_name} - {api_call.error_message}",
                f"Implement retry logic or fallback for {api_call.endpoint}",
                "HIGH"
            )
            
    async def _analyze_decision_quality(self, decision_point: DecisionPoint):
        """Analyze decision point quality"""
        if decision_point.confidence < 0.5:
            await self._store_analysis_finding(
                "low_confidence_decision",
                f"Low confidence decision ({decision_point.confidence:.2f}) in {decision_point.decision_type}",
                "Review signal weights and decision logic",
                "MEDIUM"
            )
            
    async def _flag_performance_issue(self, function_call: FunctionCall, baseline: Dict):
        """Flag performance regression"""
        await self._store_analysis_finding(
            "performance_regression",
            f"Function {function_call.function_name} took {function_call.execution_time_ms:.0f}ms (avg: {baseline['avg_time']:.0f}ms)",
            f"Investigate why {function_call.function_name} is running slower than usual",
            "HIGH"
        )
        
    async def _flag_error_pattern(self, function_call: FunctionCall, error_count: int):
        """Flag recurring error pattern"""
        await self._store_analysis_finding(
            "recurring_error",
            f"Function {function_call.function_name} has failed {error_count} times with {function_call.error_type}",
            f"Fix the root cause of {function_call.error_type} in {function_call.function_name}",
            "CRITICAL"
        )
        
    async def _store_analysis_finding(self, analysis_type: str, findings: str, 
                                    recommendations: str, priority: str):
        """Store analysis finding in database"""
        self.analysis_logger.warning(f"{analysis_type.upper()}: {findings} | RECOMMENDATION: {recommendations}")
        
        self.db_connection.execute("""
            INSERT INTO bot_analysis 
            (timestamp, analysis_type, findings, recommendations, priority)
            VALUES (?, ?, ?, ?, ?)
        """, (time.time(), analysis_type, findings, recommendations, priority))
        self.db_connection.commit()
        
    def _sanitize_for_json(self, obj: Any) -> Any:
        """Sanitize object for JSON serialization"""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj[:10]]  # Limit size
        elif isinstance(obj, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in list(obj.items())[:20]}  # Limit size
        else:
            return str(obj)[:500]  # Truncate long strings
            
    @asynccontextmanager
    async def monitoring_session(self, session_name: str):
        """Context manager for monitoring sessions"""
        start_time = time.time()
        await self.log_system_event(
            "session_start", 
            "monitoring", 
            {"session_name": session_name},
            "INFO"
        )
        
        try:
            yield self
        except Exception as e:
            await self.log_system_event(
                "session_error",
                "monitoring",
                {
                    "session_name": session_name,
                    "error": str(e),
                    "duration": time.time() - start_time
                },
                "ERROR"
            )
            raise
        finally:
            await self.log_system_event(
                "session_end",
                "monitoring", 
                {
                    "session_name": session_name,
                    "duration": time.time() - start_time
                },
                "INFO"
            )
            
    async def generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report of bot behavior"""
        # Get recent analysis findings
        cursor = self.db_connection.cursor()
        cursor.execute("""
            SELECT analysis_type, findings, recommendations, priority, timestamp
            FROM bot_analysis 
            WHERE timestamp > ? 
            ORDER BY priority DESC, timestamp DESC
        """, (time.time() - 86400,))  # Last 24 hours
        
        findings = cursor.fetchall()
        
        # Categorize findings
        critical_issues = [f for f in findings if f[3] == 'CRITICAL']
        high_priority = [f for f in findings if f[3] == 'HIGH']
        medium_priority = [f for f in findings if f[3] == 'MEDIUM']
        
        # Get performance statistics
        cursor.execute("""
            SELECT function_name, AVG(execution_time_ms), COUNT(*), 
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count
            FROM function_calls 
            WHERE timestamp > ?
            GROUP BY function_name
            ORDER BY AVG(execution_time_ms) DESC
        """, (time.time() - 86400,))
        
        performance_stats = cursor.fetchall()
        
        report = {
            "analysis_timestamp": time.time(),
            "period": "last_24_hours",
            "summary": {
                "total_issues": len(findings),
                "critical_issues": len(critical_issues),
                "high_priority_issues": len(high_priority),
                "medium_priority_issues": len(medium_priority)
            },
            "critical_issues": [
                {
                    "type": f[0],
                    "description": f[1],
                    "recommendation": f[2],
                    "timestamp": f[4]
                } for f in critical_issues[:5]
            ],
            "performance_insights": [
                {
                    "function": stat[0],
                    "avg_execution_time_ms": stat[1],
                    "call_count": stat[2],
                    "success_rate": stat[3] / stat[2] if stat[2] > 0 else 0
                } for stat in performance_stats[:10]
            ],
            "recommendations": self._generate_priority_recommendations(findings)
        }
        
        return report
        
    def _generate_priority_recommendations(self, findings: List) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        # Critical and high priority recommendations first
        for finding in findings:
            if finding[3] in ['CRITICAL', 'HIGH']:
                recommendations.append(f"[{finding[3]}] {finding[2]}")
                
        return recommendations[:10]  # Top 10 recommendations

# Global logger instance
_global_bot_logger = None

def get_bot_logger(bot_name: str = "trading_bot") -> BotActivityLogger:
    """Get global bot logger instance"""
    global _global_bot_logger
    if _global_bot_logger is None:
        _global_bot_logger = BotActivityLogger(bot_name)
    return _global_bot_logger

# Convenience decorators
def log_all_methods(cls):
    """Class decorator to log all methods"""
    logger = get_bot_logger()
    return logger.wrap_class(cls)

def log_function_calls(func):
    """Function decorator to log calls"""
    logger = get_bot_logger()
    return logger.log_function(func) 