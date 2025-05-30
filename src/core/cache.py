"""
Caching module for token and price data with optimized resource management.
"""
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
from functools import lru_cache
import asyncio
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Cache Configuration Constants
class CacheConstants:
    """Centralized cache configuration constants"""
    # Default cache sizes
    DEFAULT_TOKEN_CACHE_SIZE = 1000
    DEFAULT_PRICE_CACHE_SIZE = 1000
    
    # Default TTL values (seconds)
    DEFAULT_TOKEN_TTL = 300  # 5 minutes
    DEFAULT_PRICE_TTL = 60   # 1 minute
    
    # Cleanup intervals
    CLEANUP_INTERVAL_SECONDS = 60  # 1 minute
    
    # Performance thresholds
    MAX_CLEANUP_BATCH_SIZE = 100
    CACHE_WARNING_THRESHOLD = 0.9  # Warn when 90% full

class CacheEntry:
    """Cache entry with TTL support and validation"""
    
    def __init__(self, value: Any, ttl: int):
        if ttl <= 0:
            raise ValueError(f"TTL must be positive, got: {ttl}")
        
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.created_at > self.ttl
    
    def time_remaining(self) -> float:
        """Get remaining time before expiration"""
        return max(0, self.ttl - (time.time() - self.created_at))

class TokenCache:
    """Thread-safe LRU cache for token data with automatic cleanup"""
    
    def __init__(self, max_size: int = CacheConstants.DEFAULT_TOKEN_CACHE_SIZE, 
                 default_ttl: int = CacheConstants.DEFAULT_TOKEN_TTL):
        """Initialize token cache with size limit and default TTL."""
        if max_size <= 0:
            raise ValueError(f"Cache size must be positive, got: {max_size}")
        if default_ttl <= 0:
            raise ValueError(f"Default TTL must be positive, got: {default_ttl}")
            
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()
        
        # Cache statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.debug(f"TokenCache initialized: max_size={max_size}, default_ttl={default_ttl}s")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired with statistics tracking"""
        if not key:
            logger.warning("Cache get called with empty key")
            return None
            
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL and capacity management"""
        if not key:
            raise ValueError("Cache key cannot be empty")
        
        ttl = ttl or self._default_ttl
        if ttl <= 0:
            raise ValueError(f"TTL must be positive, got: {ttl}")
            
        async with self._lock:
            # Remove if exists
            if key in self._cache:
                del self._cache[key]

            # Remove oldest if at capacity
            while len(self._cache) >= self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug(f"Evicted cache entry: {evicted_key}")

            self._cache[key] = CacheEntry(value, ttl)
            
            # Warn if cache is getting full
            if len(self._cache) > self._max_size * CacheConstants.CACHE_WARNING_THRESHOLD:
                logger.warning(f"TokenCache is {len(self._cache)/self._max_size:.1%} full")

    async def delete(self, key: str) -> bool:
        """Delete value from cache, return True if key existed"""
        if not key:
            return False
            
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            logger.info("TokenCache cleared")

    async def cleanup(self) -> int:
        """Remove expired entries, return number of entries removed"""
        expired_count = 0
        async with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            
            # Batch cleanup to avoid memory spikes
            for i in range(0, len(expired_keys), CacheConstants.MAX_CLEANUP_BATCH_SIZE):
                batch = expired_keys[i:i + CacheConstants.MAX_CLEANUP_BATCH_SIZE]
                for key in batch:
                    if key in self._cache:  # Double-check in case of concurrent access
                        del self._cache[key]
                        expired_count += 1
        
        if expired_count > 0:
            logger.debug(f"TokenCache cleanup: removed {expired_count} expired entries")
        return expired_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "utilization": len(self._cache) / self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": hit_rate,
            "evictions": self._evictions
        }

class PriceCache:
    """Thread-safe cache for price data with exchange-specific storage"""
    
    def __init__(self, max_size: int = CacheConstants.DEFAULT_PRICE_CACHE_SIZE, 
                 default_ttl: int = CacheConstants.DEFAULT_PRICE_TTL):
        """Initialize price cache with size limit and default TTL."""
        if max_size <= 0:
            raise ValueError(f"Cache size must be positive, got: {max_size}")
        if default_ttl <= 0:
            raise ValueError(f"Default TTL must be positive, got: {default_ttl}")
            
        self._cache: Dict[str, Dict[str, CacheEntry]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()
        
        # Cache statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.debug(f"PriceCache initialized: max_size={max_size}, default_ttl={default_ttl}s")

    async def get_price(self, token_address: str, exchange: str) -> Optional[float]:
        """Get price from cache if not expired"""
        if not token_address or not exchange:
            logger.warning("PriceCache get_price called with empty token_address or exchange")
            return None
            
        async with self._lock:
            if token_address not in self._cache or exchange not in self._cache[token_address]:
                self._misses += 1
                return None

            entry = self._cache[token_address][exchange]
            if entry.is_expired():
                del self._cache[token_address][exchange]
                if not self._cache[token_address]:
                    del self._cache[token_address]
                self._misses += 1
                return None

            self._hits += 1
            return entry.value

    async def set_price(self, token_address: str, exchange: str, price: float, ttl: Optional[int] = None) -> None:
        """Set price in cache with optional TTL"""
        if not token_address or not exchange:
            raise ValueError("token_address and exchange cannot be empty")
        if price < 0:
            raise ValueError(f"Price cannot be negative, got: {price}")
            
        ttl = ttl or self._default_ttl
        
        async with self._lock:
            if token_address not in self._cache:
                self._cache[token_address] = {}

            self._cache[token_address][exchange] = CacheEntry(price, ttl)

            # Cleanup if too many tokens (LRU eviction)
            while len(self._cache) > self._max_size:
                oldest_token = next(iter(self._cache))
                del self._cache[oldest_token]
                self._evictions += 1
                logger.debug(f"Evicted price cache entry: {oldest_token}")

    async def get_all_prices(self, token_address: str) -> Dict[str, float]:
        """Get all exchange prices for a token"""
        if not token_address:
            return {}
            
        async with self._lock:
            if token_address not in self._cache:
                return {}

            prices = {}
            expired_exchanges = []
            
            for exchange, entry in self._cache[token_address].items():
                if entry.is_expired():
                    expired_exchanges.append(exchange)
                else:
                    prices[exchange] = entry.value

            # Cleanup expired entries
            for exchange in expired_exchanges:
                del self._cache[token_address][exchange]
            if not self._cache[token_address]:
                del self._cache[token_address]

            return prices

    async def cleanup(self) -> int:
        """Remove expired entries, return number of entries removed"""
        expired_count = 0
        async with self._lock:
            expired_tokens = []
            
            for token_address, exchanges in self._cache.items():
                expired_exchanges = [ex for ex, entry in exchanges.items() if entry.is_expired()]
                for exchange in expired_exchanges:
                    del exchanges[exchange]
                    expired_count += 1
                    
                if not exchanges:
                    expired_tokens.append(token_address)

            for token_address in expired_tokens:
                del self._cache[token_address]
        
        if expired_count > 0:
            logger.debug(f"PriceCache cleanup: removed {expired_count} expired entries")
        return expired_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "tokens_cached": len(self._cache),
            "max_tokens": self._max_size,
            "utilization": len(self._cache) / self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": hit_rate,
            "evictions": self._evictions
        }

# Initialize global instances with error handling
try:
    token_cache = TokenCache()
    price_cache = PriceCache()
    logger.info("Global cache instances initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize global cache instances: {str(e)}")
    raise

class CacheManager:
    """Manager for coordinated cache operations and monitoring"""
    
    def __init__(self):
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_requested = False
        
    async def start_cleanup_task(self):
        """Start periodic cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            logger.warning("Cleanup task already running")
            return
            
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("Cache cleanup task started")
    
    async def stop_cleanup_task(self):
        """Stop periodic cleanup task"""
        self._shutdown_requested = True
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Cache cleanup task stopped")
    
    async def _periodic_cleanup(self):
        """Periodic task to clean up expired cache entries"""
        while not self._shutdown_requested:
            try:
                token_cleaned = await token_cache.cleanup()
                price_cleaned = await price_cache.cleanup()
                
                if token_cleaned + price_cleaned > 0:
                    logger.debug(f"Cache cleanup completed: {token_cleaned} token entries, "
                               f"{price_cleaned} price entries removed")
                    
            except Exception as e:
                logger.error(f"Cache cleanup failed: {str(e)}")
                
            try:
                await asyncio.sleep(CacheConstants.CLEANUP_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        return {
            "token_cache": token_cache.get_stats(),
            "price_cache": price_cache.get_stats(),
            "cleanup_interval_seconds": CacheConstants.CLEANUP_INTERVAL_SECONDS
        }

# Global cache manager instance
cache_manager = CacheManager()

# Legacy compatibility functions with deprecation warnings
async def cleanup_caches():
    """Legacy function - use cache_manager.start_cleanup_task() instead"""
    logger.warning("cleanup_caches() is deprecated. Use cache_manager.start_cleanup_task() instead")
    await cache_manager.start_cleanup_task() 