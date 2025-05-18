import asyncio
import time
import traceback
from datetime import datetime, timedelta
from typing import Callable, Any, Dict, List
from loguru import logger
from .config import CORE_CONFIG

class ErrorHandler:
    def __init__(self):
        self.errors: List[Dict[str, Any]] = []
        self.last_error_time = None
        self.consecutive_errors: int = 0
        self.last_error_reset: float = time.time()
        self.error_counts: Dict[str, int] = {}

    def add_error(self, error: Exception) -> None:
        """Add an error to the error log."""
        error_info = {
            "error": str(error),
            "timestamp": datetime.now(),
            "stack_trace": traceback.format_exc()
        }
        self.errors.append(error_info)
        self.last_error_time = datetime.now()
        self.consecutive_errors += 1

    def cleanup_old_errors(self, max_age_hours: int = 24) -> None:
        """Remove old errors from the log."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        self.errors = [e for e in self.errors if e["timestamp"] > cutoff_time]

    def is_critical_error(self) -> bool:
        """Check if there's a critical error."""
        if not self.errors:
            return False
        # Check for critical error types
        critical_phrases = ["insufficient_funds", "connection_error", "auth_error"]
        for error in self.errors:
            for phrase in critical_phrases:
                if phrase in error["error"].lower():
                    return True
        return False

    def should_stop_trading(self) -> bool:
        """Determine if trading should be stopped."""
        # Stop if there's a critical error
        if self.is_critical_error():
            return True
        
        # Stop if there are too many consecutive errors
        if self.consecutive_errors >= 5:  # This should match CORE_CONFIG
            return True
            
        return False

    def reset_consecutive_errors(self) -> None:
        """Reset consecutive error counter."""
        self.consecutive_errors = 0

    def get_error_log(self) -> List[Dict[str, Any]]:
        """Get the full error log."""
        return self.errors

    def get_error_category(self) -> str:
        """Categorize the most recent error."""
        if not self.errors:
            return "none"
            
        error_str = self.errors[-1]["error"].lower()
        
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
        
        for error in self.errors:
            error_type = error["error"].__class__.__name__ if hasattr(error["error"], "__class__") else "unknown"
            error_types[error_type] = error_types.get(error_type, 0) + 1
            if self.is_critical_error():
                critical_count += 1
                
        return {
            "total_errors": len(self.errors),
            "error_types": error_types,
            "critical_errors": critical_count,
            "consecutive_errors": self.consecutive_errors,
            "time_since_last_error": time.time() - self.last_error_reset
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
                self.reset_consecutive_errors()
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