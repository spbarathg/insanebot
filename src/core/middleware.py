"""
Middleware for enforcing security measures and request validation.
"""
from typing import Callable, Dict, Any
import time
from functools import wraps
import logging
from .security import key_manager, ip_whitelist
import asyncio

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def check(self) -> bool:
        """Check if request is allowed"""
        now = time.time()
        # Remove old requests
        self.requests = [req for req in self.requests if now - req < self.time_window]
        if len(self.requests) >= self.max_requests:
            return False
        self.requests.append(now)
        return True

class ErrorHandler:
    """Centralized error handling"""
    def __init__(self):
        self.error_count = 0
        self.last_error_time = 0
    
    async def handle_error(self, error: Exception, context: str) -> None:
        """Handle and log errors"""
        self.error_count += 1
        self.last_error_time = time.time()
        logger.error(f"Error in {context}: {str(error)}")
        
        # Back off on network errors
        if isinstance(error, (ConnectionError, TimeoutError)):
            await asyncio.sleep(5)
    
    def should_continue(self) -> bool:
        """Check if we should continue after errors"""
        return self.error_count < 5

def handle_errors(func: Callable) -> Callable:
    """Decorator for error handling"""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_handler = ErrorHandler()
            await error_handler.handle_error(e, func.__name__)
            return None
    return wrapper

def rate_limit(max_requests: int = 10, time_window: int = 60) -> Callable:
    """Decorator for rate limiting"""
    limiter = RateLimiter(max_requests, time_window)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            if not await limiter.check():
                logger.warning(f"Rate limit exceeded for {func.__name__}")
                return None
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_signed_request(func: Callable) -> Callable:
    """Decorator to require signed requests."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = kwargs.get('request')
        if not request:
            raise ValueError("Request object not found in kwargs")

        # Extract signature from headers
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            raise ValueError("No authorization header found")

        try:
            timestamp, signature = auth_header.split(':')
            # Verify timestamp is within 5 minutes
            if abs(int(time.time()) - int(timestamp)) > 300:
                raise ValueError("Request timestamp expired")

            # Verify signature
            method = request.method
            path = request.url.path
            body = await request.json() if request.method in ['POST', 'PUT'] else None
            
            expected_signature = key_manager.sign_request(method, path, body)
            if signature != expected_signature.split(':')[1]:
                raise ValueError("Invalid request signature")

        except Exception as e:
            logger.error(f"Request validation failed: {str(e)}")
            raise

        return await func(*args, **kwargs)
    return wrapper

def require_ip_whitelist(func: Callable) -> Callable:
    """Decorator to require IP whitelist."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = kwargs.get('request')
        if not request:
            raise ValueError("Request object not found in kwargs")

        client_ip = request.client.host
        if not ip_whitelist.is_allowed(client_ip):
            logger.warning(f"Blocked request from non-whitelisted IP: {client_ip}")
            raise ValueError("IP not whitelisted")

        return await func(*args, **kwargs)
    return wrapper

def verify_transaction(func: Callable) -> Callable:
    """Decorator to verify transaction signatures."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        transaction = kwargs.get('transaction')
        if not transaction:
            raise ValueError("Transaction object not found in kwargs")

        try:
            # Verify transaction signature
            if not transaction.verify():
                raise ValueError("Invalid transaction signature")

            # Verify transaction hasn't expired
            if transaction.is_expired():
                raise ValueError("Transaction expired")

            # Verify transaction amount is within limits
            if not transaction.is_within_limits():
                raise ValueError("Transaction amount exceeds limits")

        except Exception as e:
            logger.error(f"Transaction verification failed: {str(e)}")
            raise

        return await func(*args, **kwargs)
    return wrapper

class SecurityMiddleware:
    """Middleware class for applying security measures."""
    
    def __init__(self, app):
        self.app = app

    async def __call__(self, request, call_next):
        # Apply IP whitelist check
        if not ip_whitelist.is_allowed(request.client.host):
            return {"error": "IP not whitelisted"}, 403

        # Apply request signing check for protected endpoints
        if request.url.path.startswith('/api/'):
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return {"error": "No authorization header"}, 401

            try:
                timestamp, signature = auth_header.split(':')
                if abs(int(time.time()) - int(timestamp)) > 300:
                    return {"error": "Request expired"}, 401

                expected_signature = key_manager.sign_request(
                    request.method,
                    request.url.path,
                    await request.json() if request.method in ['POST', 'PUT'] else None
                )
                
                if signature != expected_signature.split(':')[1]:
                    return {"error": "Invalid signature"}, 401

            except Exception as e:
                logger.error(f"Request validation failed: {str(e)}")
                return {"error": "Invalid request"}, 400

        response = await call_next(request)
        return response 

# Initialize global instances
error_handler = ErrorHandler()
rate_limiter = RateLimiter() 