"""
Caching module for token and price data.
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

class CacheEntry:
    def __init__(self, value: Any, ttl: int):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

class TokenCache:
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """Initialize token cache with size limit and default TTL."""
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        async with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        async with self._lock:
            # Remove if exists
            if key in self._cache:
                del self._cache[key]

            # Remove oldest if at capacity
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(value, ttl or self._default_ttl)

    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()

    async def cleanup(self) -> None:
        """Remove expired entries."""
        async with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            for key in expired_keys:
                del self._cache[key]

class PriceCache:
    def __init__(self, max_size: int = 1000, default_ttl: int = 60):
        """Initialize price cache with size limit and default TTL."""
        self._cache: Dict[str, Dict[str, CacheEntry]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()

    async def get_price(self, token_address: str, exchange: str) -> Optional[float]:
        """Get price from cache if not expired."""
        async with self._lock:
            if token_address not in self._cache or exchange not in self._cache[token_address]:
                return None

            entry = self._cache[token_address][exchange]
            if entry.is_expired():
                del self._cache[token_address][exchange]
                if not self._cache[token_address]:
                    del self._cache[token_address]
                return None

            return entry.value

    async def set_price(self, token_address: str, exchange: str, price: float, ttl: Optional[int] = None) -> None:
        """Set price in cache with optional TTL."""
        async with self._lock:
            if token_address not in self._cache:
                self._cache[token_address] = {}

            self._cache[token_address][exchange] = CacheEntry(price, ttl or self._default_ttl)

            # Cleanup if too many tokens
            if len(self._cache) > self._max_size:
                oldest_token = next(iter(self._cache))
                del self._cache[oldest_token]

    async def get_all_prices(self, token_address: str) -> Dict[str, float]:
        """Get all exchange prices for a token."""
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

    async def cleanup(self) -> None:
        """Remove expired entries."""
        async with self._lock:
            expired_tokens = []
            for token_address, exchanges in self._cache.items():
                expired_exchanges = [ex for ex, entry in exchanges.items() if entry.is_expired()]
                for exchange in expired_exchanges:
                    del exchanges[exchange]
                if not exchanges:
                    expired_tokens.append(token_address)

            for token_address in expired_tokens:
                del self._cache[token_address]

# Initialize global instances
token_cache = TokenCache()
price_cache = PriceCache()

# Periodic cleanup task
async def cleanup_caches():
    """Periodic task to clean up expired cache entries."""
    while True:
        try:
            await token_cache.cleanup()
            await price_cache.cleanup()
        except Exception as e:
            logger.error(f"Cache cleanup failed: {str(e)}")
        await asyncio.sleep(60)  # Run every minute 