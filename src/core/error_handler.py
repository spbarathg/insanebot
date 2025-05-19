import asyncio
import time
import traceback
from datetime import datetime, timedelta
from typing import Callable, Any, Dict, List, Optional
from loguru import logger
from .config import CORE_CONFIG
from dataclasses import dataclass

@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    timestamp: float
    error_type: str
    message: str
    context: Dict[str, Any]

class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Attributes:
        failure_threshold: Number of failures before opening circuit
        reset_timeout: Time in seconds before attempting reset
        last_failure_time: Timestamp of last failure
        failure_count: Current failure count
        is_open: Whether circuit is open
    """
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.failure_count = 0
        self.is_open = False
    
    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning("Circuit breaker opened due to excessive failures")
    
    def record_success(self) -> None:
        """Record a success and reset failure count."""
        self.failure_count = 0
        if self.is_open:
            self.is_open = False
            logger.info("Circuit breaker closed after successful operation")
    
    def can_execute(self) -> bool:
        """
        Check if operation can be executed.
        
        Returns:
            bool: True if operation can be executed
        """
        if not self.is_open:
            return True
            
        # Check if reset timeout has elapsed
        if time.time() - self.last_failure_time >= self.reset_timeout:
            self.is_open = False
            self.failure_count = 0
            logger.info("Circuit breaker reset after timeout")
            return True
            
        return False

class ErrorHandler:
    """
    Enhanced error handling with circuit breakers and retry logic.
    
    Attributes:
        error_records: List of recent error records
        circuit_breakers: Dictionary of circuit breakers by error type
        max_errors: Maximum number of errors to store
        error_cooldown: Cooldown period in seconds
    """
    
    def __init__(
        self,
        max_errors: int = 100,
        error_cooldown: int = 300,
        circuit_breaker_config: Optional[Dict[str, Dict[str, int]]] = None
    ):
        self.error_records: List[ErrorRecord] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.max_errors = max_errors
        self.error_cooldown = error_cooldown
        
        # Initialize circuit breakers
        default_config = {
            "network": {"failure_threshold": 5, "reset_timeout": 60},
            "transaction": {"failure_threshold": 3, "reset_timeout": 30},
            "wallet": {"failure_threshold": 2, "reset_timeout": 300}
        }
        
        config = circuit_breaker_config or default_config
        for error_type, params in config.items():
            self.circuit_breakers[error_type] = CircuitBreaker(
                failure_threshold=params["failure_threshold"],
                reset_timeout=params["reset_timeout"]
            )
    
    def add_error(
        self,
        error: Exception,
        error_type: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an error record and update circuit breakers.
        
        Args:
            error: The exception that occurred
            error_type: Type of error for circuit breaker
            context: Additional error context
        """
        # Create error record
        record = ErrorRecord(
            timestamp=time.time(),
            error_type=error_type,
            message=str(error),
            context=context or {}
        )
        
        # Add to records
        self.error_records.append(record)
        if len(self.error_records) > self.max_errors:
            self.error_records.pop(0)
        
        # Update circuit breaker
        if error_type in self.circuit_breakers:
            self.circuit_breakers[error_type].record_failure()
        
        # Log error
        logger.error(
            f"Error occurred: {error_type} - {str(error)}",
            extra={"context": context}
        )
    
    def record_success(self, error_type: str = "general") -> None:
        """
        Record a successful operation.
        
        Args:
            error_type: Type of operation that succeeded
        """
        if error_type in self.circuit_breakers:
            self.circuit_breakers[error_type].record_success()
    
    def can_execute(self, error_type: str = "general") -> bool:
        """
        Check if operation can be executed.
        
        Args:
            error_type: Type of operation to check
            
        Returns:
            bool: True if operation can be executed
        """
        if error_type in self.circuit_breakers:
            return self.circuit_breakers[error_type].can_execute()
        return True
    
    def should_stop_trading(self) -> bool:
        """
        Check if trading should be stopped due to errors.
        
        Returns:
            bool: True if trading should be stopped
        """
        # Check if any critical circuit breakers are open
        critical_types = ["wallet", "transaction"]
        for error_type in critical_types:
            if error_type in self.circuit_breakers:
                if not self.circuit_breakers[error_type].can_execute():
                    return True
        
        # Check error frequency
        recent_errors = [
            record for record in self.error_records
            if time.time() - record.timestamp < self.error_cooldown
        ]
        
        return len(recent_errors) >= self.max_errors
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent errors.
        
        Returns:
            Dict[str, Any]: Error summary statistics
        """
        if not self.error_records:
            return {"total_errors": 0}
        
        # Calculate error statistics
        error_types = {}
        for record in self.error_records:
            error_types[record.error_type] = error_types.get(record.error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_records),
            "error_types": error_types,
            "latest_error": {
                "type": self.error_records[-1].error_type,
                "message": self.error_records[-1].message,
                "timestamp": self.error_records[-1].timestamp
            }
        }
    
    def clear_errors(self) -> None:
        """Clear all error records and reset circuit breakers."""
        self.error_records.clear()
        for breaker in self.circuit_breakers.values():
            breaker.failure_count = 0
            breaker.is_open = False
            breaker.last_failure_time = 0

    def cleanup_old_errors(self, max_age_hours: int = 24) -> None:
        """Remove old errors from the log."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        self.error_records = [e for e in self.error_records if e.timestamp > cutoff_time]

    def is_critical_error(self) -> bool:
        """Check if there's a critical error."""
        if not self.error_records:
            return False
        # Check for critical error types
        critical_phrases = ["insufficient_funds", "connection_error", "auth_error"]
        for record in self.error_records:
            for phrase in critical_phrases:
                if phrase in record.message.lower():
                    return True
        return False

    def reset_consecutive_errors(self) -> None:
        """Reset consecutive error counter."""
        pass

    def get_error_log(self) -> List[ErrorRecord]:
        """Get the full error log."""
        return self.error_records

    def get_error_category(self) -> str:
        """Categorize the most recent error."""
        if not self.error_records:
            return "none"
            
        error_str = self.error_records[-1].message.lower()
        
        if "network" in error_str or "connection" in error_str:
            return "network"
        elif "invalid input" in error_str or "valueerror" in error_str:
            return "validation"
        else:
            return "unknown"

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        error_types = {}
        critical_count = 0
        
        for record in self.error_records:
            error_type = record.error_type if hasattr(record.error_type, "__name__") else "unknown"
            error_types[error_type] = error_types.get(error_type, 0) + 1
            if self.is_critical_error():
                critical_count += 1
                
        return {
            "total_errors": len(self.error_records),
            "error_types": error_types,
            "critical_errors": critical_count,
            "consecutive_errors": 0,
            "time_since_last_error": time.time() - self.error_records[-1].timestamp
        }

    async def handle_operation(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Handle an operation with retry logic and error tracking.
        
        Args:
            operation: Async function to execute
            operation_name: Name of the operation for logging
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation
            
        Returns:
            Result of the operation if successful
            
        Raises:
            Exception: If operation fails after all retries
        """
        max_retries = CORE_CONFIG["monitoring"]["max_retries"]
        retry_delay = CORE_CONFIG["monitoring"]["retry_delay"]
        
        for attempt in range(max_retries):
            try:
                result = await operation(*args, **kwargs)
                self.record_success()
                return result
                
            except Exception as e:
                self.add_error(e)
                
                if self.should_stop_trading():
                    logger.error(f"Critical error occurred: {str(e)}")
                    raise
                
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Operation {operation_name} failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
                    )
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(
                        f"Operation {operation_name} failed after {max_retries} attempts: {str(e)}"
                    )
                    raise 