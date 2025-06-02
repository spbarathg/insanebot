"""
Production-Ready Error Handling and Recovery System

This module provides comprehensive error handling, automatic recovery mechanisms,
and resilience patterns for the trading bot.
"""

import asyncio
import logging
import time
import traceback
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import json
import functools

from .circuit_breaker import CircuitBreakerError, get_circuit_breaker

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    MANUAL_INTERVENTION = "manual_intervention"

@dataclass
class ErrorContext:
    """Context information for an error"""
    error: Exception
    timestamp: float
    severity: ErrorSeverity
    component: str
    operation: str
    attempt_count: int = 1
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryAction:
    """Definition of a recovery action"""
    strategy: RecoveryStrategy
    max_attempts: int = 3
    delay_between_attempts: float = 1.0
    exponential_backoff: bool = True
    fallback_function: Optional[Callable] = None
    circuit_breaker_name: Optional[str] = None

class TradingError(Exception):
    """Base exception for trading-related errors"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 component: str = "unknown", recoverable: bool = True):
        super().__init__(message)
        self.severity = severity
        self.component = component
        self.recoverable = recoverable

class CapitalError(TradingError):
    """Errors related to capital management"""
    pass

class ExecutionError(TradingError):
    """Errors related to trade execution"""
    pass

class NetworkError(TradingError):
    """Network-related errors"""
    pass

class DataError(TradingError):
    """Data validation or processing errors"""
    pass

class SecurityError(TradingError):
    """Security-related errors"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        kwargs.setdefault('recoverable', False)
        super().__init__(message, **kwargs)

class SystemError(TradingError):
    """System-level errors"""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)

class ErrorRecoverySystem:
    """
    Comprehensive error handling and recovery system
    
    Features:
    - Automatic retry mechanisms with exponential backoff
    - Circuit breaker integration
    - Fallback strategies
    - Error classification and severity assessment
    - Recovery metrics and monitoring
    - Graceful degradation
    """
    
    def __init__(self):
        self.error_handlers: Dict[type, RecoveryAction] = {}
        self.global_fallbacks: List[Callable] = []
        self.error_history: List[ErrorContext] = []
        self.recovery_metrics: Dict[str, int] = {
            "total_errors": 0,
            "recovered_errors": 0,
            "failed_recoveries": 0,
            "retry_attempts": 0,
            "fallback_executions": 0,
            "circuit_breaker_trips": 0
        }
        self._lock = threading.RLock()
        
        # Setup default error handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default error recovery strategies"""
        # Network errors - retry with exponential backoff
        self.register_handler(
            NetworkError,
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_attempts=5,
                delay_between_attempts=2.0,
                exponential_backoff=True,
                circuit_breaker_name="network"
            )
        )
        
        # Execution errors - retry with fallback
        self.register_handler(
            ExecutionError,
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_attempts=3,
                delay_between_attempts=1.0,
                fallback_function=self._execution_fallback
            )
        )
        
        # Capital errors - immediate fallback to safe mode
        self.register_handler(
            CapitalError,
            RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                fallback_function=self._capital_fallback
            )
        )
        
        # Security errors - no retry, immediate escalation
        self.register_handler(
            SecurityError,
            RecoveryAction(
                strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                max_attempts=1
            )
        )
        
        # System errors - circuit breaker with graceful degradation
        self.register_handler(
            SystemError,
            RecoveryAction(
                strategy=RecoveryStrategy.CIRCUIT_BREAKER,
                circuit_breaker_name="system",
                fallback_function=self._system_fallback
            )
        )
    
    def register_handler(self, error_type: type, action: RecoveryAction):
        """Register a recovery action for a specific error type"""
        with self._lock:
            self.error_handlers[error_type] = action
            logger.info(f"Registered recovery handler for {error_type.__name__}: {action.strategy.value}")
    
    def add_global_fallback(self, fallback_func: Callable):
        """Add a global fallback function"""
        self.global_fallbacks.append(fallback_func)
    
    async def handle_error_async(self, error: Exception, context: ErrorContext) -> Any:
        """Handle an error asynchronously with recovery strategies"""
        with self._lock:
            self.error_history.append(context)
            self.recovery_metrics["total_errors"] += 1
        
        logger.error(f"Handling error in {context.component}: {error}")
        
        # Find appropriate handler
        handler = self._find_handler(type(error))
        
        if not handler:
            logger.warning(f"No specific handler found for {type(error).__name__}, using default recovery")
            return await self._default_recovery(error, context)
        
        return await self._execute_recovery_strategy(error, context, handler)
    
    def handle_error(self, error: Exception, context: ErrorContext) -> Any:
        """Handle an error synchronously with recovery strategies"""
        return asyncio.run(self.handle_error_async(error, context))
    
    def _find_handler(self, error_type: type) -> Optional[RecoveryAction]:
        """Find the most specific handler for an error type"""
        # Look for exact match first
        if error_type in self.error_handlers:
            return self.error_handlers[error_type]
        
        # Look for parent class matches
        for registered_type, handler in self.error_handlers.items():
            if issubclass(error_type, registered_type):
                return handler
        
        return None
    
    async def _execute_recovery_strategy(self, error: Exception, context: ErrorContext, 
                                       handler: RecoveryAction) -> Any:
        """Execute the recovery strategy"""
        try:
            if handler.strategy == RecoveryStrategy.RETRY:
                return await self._retry_strategy(error, context, handler)
            elif handler.strategy == RecoveryStrategy.FALLBACK:
                return await self._fallback_strategy(error, context, handler)
            elif handler.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return await self._circuit_breaker_strategy(error, context, handler)
            elif handler.strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return await self._graceful_degradation_strategy(error, context, handler)
            elif handler.strategy == RecoveryStrategy.MANUAL_INTERVENTION:
                return await self._manual_intervention_strategy(error, context, handler)
            else:
                return await self._default_recovery(error, context)
        
        except Exception as recovery_error:
            logger.error(f"Recovery strategy failed: {recovery_error}")
            with self._lock:
                self.recovery_metrics["failed_recoveries"] += 1
            return await self._default_recovery(error, context)
    
    async def _retry_strategy(self, error: Exception, context: ErrorContext, 
                            handler: RecoveryAction) -> Any:
        """Implement retry strategy with exponential backoff"""
        delay = handler.delay_between_attempts
        
        for attempt in range(handler.max_attempts):
            if attempt > 0:
                if handler.exponential_backoff:
                    actual_delay = delay * (2 ** (attempt - 1))
                else:
                    actual_delay = delay
                
                logger.info(f"Retrying {context.operation} in {actual_delay}s (attempt {attempt + 1}/{handler.max_attempts})")
                await asyncio.sleep(actual_delay)
            
            try:
                with self._lock:
                    self.recovery_metrics["retry_attempts"] += 1
                
                # If circuit breaker is configured, use it
                if handler.circuit_breaker_name:
                    cb = get_circuit_breaker(handler.circuit_breaker_name)
                    return await cb.call_async(self._reattempt_operation, context)
                else:
                    return await self._reattempt_operation(context)
            
            except CircuitBreakerError:
                logger.warning(f"Circuit breaker open for {handler.circuit_breaker_name}")
                with self._lock:
                    self.recovery_metrics["circuit_breaker_trips"] += 1
                break
            except Exception as retry_error:
                if attempt == handler.max_attempts - 1:
                    # Last attempt failed, try fallback
                    if handler.fallback_function:
                        return await self._execute_fallback(handler.fallback_function, error, context)
                    else:
                        raise retry_error
                
                logger.warning(f"Retry attempt {attempt + 1} failed: {retry_error}")
        
        # All retries failed
        with self._lock:
            self.recovery_metrics["failed_recoveries"] += 1
        raise error
    
    async def _fallback_strategy(self, error: Exception, context: ErrorContext, 
                               handler: RecoveryAction) -> Any:
        """Implement fallback strategy"""
        if handler.fallback_function:
            return await self._execute_fallback(handler.fallback_function, error, context)
        else:
            return await self._default_fallback(error, context)
    
    async def _circuit_breaker_strategy(self, error: Exception, context: ErrorContext, 
                                      handler: RecoveryAction) -> Any:
        """Implement circuit breaker strategy"""
        if handler.circuit_breaker_name:
            cb = get_circuit_breaker(handler.circuit_breaker_name)
            try:
                return await cb.call_async(self._reattempt_operation, context)
            except CircuitBreakerError:
                logger.warning(f"Circuit breaker open, executing fallback")
                if handler.fallback_function:
                    return await self._execute_fallback(handler.fallback_function, error, context)
                else:
                    return await self._graceful_degradation_strategy(error, context, handler)
        else:
            raise ValueError("Circuit breaker strategy requires circuit_breaker_name")
    
    async def _graceful_degradation_strategy(self, error: Exception, context: ErrorContext, 
                                           handler: RecoveryAction) -> Any:
        """Implement graceful degradation"""
        logger.warning(f"Entering graceful degradation mode for {context.component}")
        
        # Return a safe default response based on the component
        if context.component == "trading":
            return {"status": "degraded", "action": "hold_position", "reason": str(error)}
        elif context.component == "data":
            return {"status": "degraded", "data": None, "cache_used": True}
        elif context.component == "risk":
            return {"status": "degraded", "risk_level": "high", "recommendation": "reduce_exposure"}
        else:
            return {"status": "degraded", "error": str(error)}
    
    async def _manual_intervention_strategy(self, error: Exception, context: ErrorContext, 
                                          handler: RecoveryAction) -> Any:
        """Handle errors requiring manual intervention"""
        logger.critical(f"Manual intervention required for {context.component}: {error}")
        
        # Log to special intervention log
        intervention_log = {
            "timestamp": datetime.now().isoformat(),
            "component": context.component,
            "operation": context.operation,
            "error": str(error),
            "severity": context.severity.value,
            "stack_trace": traceback.format_exc()
        }
        
        # This could send alerts, write to special logs, etc.
        self._escalate_for_manual_intervention(intervention_log)
        
        # Return safe state
        return {"status": "manual_intervention_required", "error": str(error)}
    
    async def _default_recovery(self, error: Exception, context: ErrorContext) -> Any:
        """Default recovery when no specific handler is found"""
        logger.warning(f"Using default recovery for {type(error).__name__}")
        
        # Try global fallbacks
        for fallback in self.global_fallbacks:
            try:
                return await self._execute_fallback(fallback, error, context)
            except Exception as fallback_error:
                logger.warning(f"Global fallback failed: {fallback_error}")
        
        # Last resort - graceful degradation
        return await self._graceful_degradation_strategy(error, context, RecoveryAction(RecoveryStrategy.GRACEFUL_DEGRADATION))
    
    async def _execute_fallback(self, fallback_func: Callable, error: Exception, 
                              context: ErrorContext) -> Any:
        """Execute a fallback function"""
        with self._lock:
            self.recovery_metrics["fallback_executions"] += 1
        
        try:
            if asyncio.iscoroutinefunction(fallback_func):
                return await fallback_func(error, context)
            else:
                return fallback_func(error, context)
        except Exception as fallback_error:
            logger.error(f"Fallback function failed: {fallback_error}")
            raise
    
    async def _reattempt_operation(self, context: ErrorContext) -> Any:
        """Reattempt the original operation (placeholder)"""
        # This would typically involve re-executing the original function
        # For now, we'll just return a success indicator
        logger.info(f"Reattempting operation: {context.operation}")
        return {"status": "reattempted", "operation": context.operation}
    
    def _execution_fallback(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Fallback for execution errors"""
        logger.warning(f"Using execution fallback for {context.operation}")
        return {
            "status": "fallback_executed",
            "original_error": str(error),
            "action": "position_held",
            "recommendation": "manual_review_required"
        }
    
    def _capital_fallback(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Fallback for capital errors"""
        logger.error(f"Capital safety fallback activated: {error}")
        return {
            "status": "capital_protection_active",
            "action": "stop_all_trading",
            "error": str(error),
            "safety_mode": True
        }
    
    def _system_fallback(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Fallback for system errors"""
        logger.warning(f"System fallback activated: {error}")
        return {
            "status": "system_degraded",
            "limited_functionality": True,
            "error": str(error)
        }
    
    def _default_fallback(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Default fallback when no specific fallback is available"""
        return {
            "status": "default_fallback",
            "error": str(error),
            "component": context.component,
            "safe_state": True
        }
    
    def _escalate_for_manual_intervention(self, intervention_log: Dict[str, Any]):
        """Escalate critical errors for manual intervention"""
        # Log to special file
        try:
            with open("logs/manual_intervention.log", "a") as f:
                f.write(json.dumps(intervention_log) + "\n")
        except Exception as log_error:
            logger.error(f"Failed to write intervention log: {log_error}")
        
        # Here you could also:
        # - Send alerts via email/SMS/Slack
        # - Create tickets in issue tracking systems
        # - Trigger emergency shutdown procedures
        # - Notify on-call personnel
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error and recovery statistics"""
        with self._lock:
            current_time = time.time()
            
            # Analyze recent errors (last hour)
            recent_errors = [
                ctx for ctx in self.error_history 
                if current_time - ctx.timestamp < 3600
            ]
            
            # Group by severity
            severity_counts = {}
            for severity in ErrorSeverity:
                severity_counts[severity.value] = len([
                    ctx for ctx in recent_errors 
                    if ctx.severity == severity
                ])
            
            # Group by component
            component_counts = {}
            for ctx in recent_errors:
                component_counts[ctx.component] = component_counts.get(ctx.component, 0) + 1
            
            return {
                "metrics": self.recovery_metrics.copy(),
                "recent_errors_count": len(recent_errors),
                "total_errors_count": len(self.error_history),
                "recovery_success_rate": (
                    self.recovery_metrics["recovered_errors"] / 
                    max(1, self.recovery_metrics["total_errors"])
                ),
                "severity_distribution": severity_counts,
                "component_distribution": component_counts,
                "timestamp": datetime.now().isoformat()
            }
    
    def reset_statistics(self):
        """Reset error statistics"""
        with self._lock:
            self.error_history.clear()
            self.recovery_metrics = {
                "total_errors": 0,
                "recovered_errors": 0,
                "failed_recoveries": 0,
                "retry_attempts": 0,
                "fallback_executions": 0,
                "circuit_breaker_trips": 0
            }

# Global error recovery system instance
error_recovery_system = ErrorRecoverySystem()

def with_error_recovery(component: str, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator to wrap functions with error recovery"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    error=e,
                    timestamp=time.time(),
                    severity=severity,
                    component=component,
                    operation=operation
                )
                return await error_recovery_system.handle_error_async(e, context)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    error=e,
                    timestamp=time.time(),
                    severity=severity,
                    component=component,
                    operation=operation
                )
                return error_recovery_system.handle_error(e, context)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator 