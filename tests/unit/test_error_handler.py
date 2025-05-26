"""
Test suite for error handling functionality.
"""
import pytest
from datetime import datetime, timedelta
from src.core.error_handler import ErrorHandler

@pytest.fixture
def error_handler():
    """Create an error handler instance."""
    return ErrorHandler()

def test_error_handler_initialization(error_handler):
    """Test error handler initialization."""
    assert error_handler.errors == []
    assert error_handler.last_error_time is None
    assert error_handler.consecutive_errors == 0

def test_add_error(error_handler):
    """Test adding errors to the handler."""
    error = Exception("Test error")
    error_handler.add_error(error)
    
    assert len(error_handler.errors) == 1
    assert error_handler.errors[0]["error"] == str(error)
    assert error_handler.errors[0]["timestamp"] is not None
    assert error_handler.consecutive_errors == 1

def test_error_cooldown(error_handler):
    """Test error cooldown mechanism."""
    # Add first error
    error_handler.add_error(Exception("First error"))
    first_error_time = error_handler.last_error_time
    
    # Add second error immediately
    error_handler.add_error(Exception("Second error"))
    
    # Verify cooldown period
    assert error_handler.last_error_time > first_error_time
    assert error_handler.consecutive_errors == 2

def test_error_cleanup(error_handler):
    """Test error cleanup functionality."""
    # Add some old errors
    old_time = datetime.now() - timedelta(hours=2)
    error_handler.errors = [
        {"error": "Old error", "timestamp": old_time}
    ]
    
    # Clean up old errors
    error_handler.cleanup_old_errors(max_age_hours=1)
    
    assert len(error_handler.errors) == 0

def test_critical_error_handling(error_handler):
    """Test handling of critical errors."""
    # Add a critical error
    error_handler.add_error(Exception("insufficient_funds"))
    
    assert error_handler.is_critical_error()
    assert error_handler.should_stop_trading()

def test_error_threshold(error_handler):
    """Test error threshold mechanism."""
    # Add multiple errors
    for _ in range(5):
        error_handler.add_error(Exception("Test error"))
    
    assert error_handler.consecutive_errors == 5
    assert error_handler.should_stop_trading()

def test_error_recovery(error_handler):
    """Test error recovery mechanism."""
    # Add some errors
    for _ in range(3):
        error_handler.add_error(Exception("Test error"))
    
    # Simulate successful operation
    error_handler.reset_consecutive_errors()
    
    assert error_handler.consecutive_errors == 0
    assert not error_handler.should_stop_trading()

def test_error_logging(error_handler):
    """Test error logging functionality."""
    error = Exception("Test error")
    error_handler.add_error(error)
    
    # Verify error log format
    error_log = error_handler.get_error_log()
    assert len(error_log) == 1
    assert "error" in error_log[0]
    assert "timestamp" in error_log[0]
    assert "stack_trace" in error_log[0]

def test_error_categorization(error_handler):
    """Test error categorization."""
    # Test network error
    error_handler.add_error(Exception("network_error"))
    assert error_handler.get_error_category() == "network"
    
    # Test validation error
    error_handler.add_error(ValueError("Invalid input"))
    assert error_handler.get_error_category() == "validation"
    
    # Test unknown error
    error_handler.add_error(Exception("unknown_error"))
    assert error_handler.get_error_category() == "unknown"

def test_error_stats(error_handler):
    """Test error statistics."""
    # Add various types of errors
    error_handler.add_error(Exception("network_error"))
    error_handler.add_error(ValueError("Invalid input"))
    error_handler.add_error(Exception("insufficient_funds"))
    
    stats = error_handler.get_error_stats()
    assert "total_errors" in stats
    assert "error_types" in stats
    assert "critical_errors" in stats
    assert stats["total_errors"] == 3 