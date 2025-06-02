"""
Helius service for Solana blockchain data with real API integration.
"""
import asyncio
import aiohttp
import json
import logging
import time
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass
from enum import Enum
from collections import deque

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout_seconds: int = 30   # Request timeout

class CircuitBreaker:
    """Circuit breaker for API resilience"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.failure_history = deque(maxlen=100)
        
    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("🔄 Circuit breaker moving to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs), 
                timeout=self.config.timeout_seconds
            )
            
            # Success handling
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("✅ Circuit breaker CLOSED - service recovered")
            
            return result
            
        except Exception as e:
            # Failure handling
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.failure_history.append({
                'timestamp': time.time(),
                'error': str(e)
            })
            
            if (self.state == CircuitState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                self.state = CircuitState.OPEN
                logger.error(f"🚨 Circuit breaker OPENED - {self.failure_count} failures")
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.error("🚨 Circuit breaker back to OPEN - recovery failed")
            
            raise

class HeliusAPIError(Exception):
    """Raised when Helius API calls fail"""
    pass

class HeliusService:
    """
    Real Helius service for Solana blockchain data.
    Supports both live API calls and simulation mode.
    """
    
    def __init__(self):
        """Initialize Helius service with REAL API integration only."""
        self.api_key = os.getenv("HELIUS_API_KEY", "")
        self.base_url = f"https://api.helius.xyz/v0"
        self.session = None
        self.max_retries = 3
        self.timeout = 30
        
        # Always running in real API mode - no simulation
        self.simulation_mode = False
        
        # Real-time cache for performance optimization
        self._price_cache = {}
        self._metadata_cache = {}
        self._cache_ttl = 10  # 10 seconds cache for real-time data
        
        # Check API key validity
        if not self.api_key or self.api_key in ["", "demo_key_for_testing", "your-helius-api-key"]:
            logger.warning("⚠️ No valid Helius API key provided - running in limited mode")
            logger.info("💡 To get full functionality, get a free API key at: https://helius.xyz/")
            self.api_key = None  # Set to None to clearly indicate no valid key
        else:
            logger.info("✅ Helius API key configured")
        
        logger.info("Helius Service initialized - REAL API MODE ONLY")
        
        # Circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig())
        
        # Connection management
        self.request_count = 0
        self.error_count = 0
        
        # Rate limiting
        self.rate_limit_requests = 100  # per minute
        self.rate_limit_window = 60
        self.request_timestamps = deque(maxlen=self.rate_limit_requests)
    
    async def _make_api_request(self, endpoint: str, params: Dict = None, method: str = "GET") -> Optional[Dict]:
        """Make API request to Helius with real data only."""
        try:
            # If no valid API key, return None gracefully
            if not self.api_key:
                logger.debug(f"Skipping Helius API call to {endpoint} - no valid API key")
                return None
                
            if not self.session:
                import aiohttp
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                headers = {"User-Agent": "Solana-Trading-Bot/1.0"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
            
            # Add API key to params if available
            if params is None:
                params = {}
            if self.api_key:
                params["api-key"] = self.api_key
            
            url = f"{self.base_url}/{endpoint}"
            
            for attempt in range(self.max_retries):
                try:
                    if method == "GET":
                        async with self.session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                logger.debug(f"Helius API success: {endpoint}")
                                return data
                            elif response.status == 429:  # Rate limited
                                wait_time = 2 ** attempt
                                logger.warning(f"Helius API rate limited, waiting {wait_time}s")
                                await asyncio.sleep(wait_time)
                                continue
                            elif response.status == 401:
                                logger.error("❌ Helius API authentication failed - check API key")
                                logger.info("💡 Get a free API key at: https://helius.xyz/")
                                break
                            elif response.status == 403:
                                logger.error("❌ Helius API access forbidden - check API key permissions")
                                break
                            else:
                                logger.warning(f"Helius API error {response.status} for {endpoint}")
                                break
                    elif method == "POST":
                        async with self.session.post(url, json=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                logger.debug(f"Helius API POST success: {endpoint}")
                                return data
                            elif response.status == 429:  # Rate limited
                                wait_time = 2 ** attempt
                                logger.warning(f"Helius API rate limited, waiting {wait_time}s")
                                await asyncio.sleep(wait_time)
                                continue
                            elif response.status == 401:
                                logger.error("❌ Helius API authentication failed - check API key")
                                break
                            else:
                                logger.warning(f"Helius API POST error {response.status} for {endpoint}")
                                break
                except asyncio.TimeoutError:
                    logger.warning(f"Helius API timeout on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                except Exception as e:
                    logger.error(f"Helius API request error: {str(e)}")
                    break
            
            logger.debug(f"Helius API request failed after {self.max_retries} attempts: {endpoint}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to make Helius API request: {str(e)}")
            return None
    
    async def get_token_metadata(self, token_address: str) -> Dict[str, Any]:
        """Get comprehensive token metadata."""
        try:
            # Check cache first
            cache_key = f"metadata_{token_address}"
            current_time = time.time()
            
            if cache_key in self._metadata_cache:
                cached_data = self._metadata_cache[cache_key]
                if (current_time - cached_data['timestamp']) < self._cache_ttl * 6:  # Cache metadata longer
                    return cached_data['data']
            
            # Method 1: Helius Metadata API
            try:
                response = await self._make_api_request(f"tokens/metadata", 
                                                      params={"mint": token_address}, 
                                                      method="GET")
                if response and isinstance(response, list) and len(response) > 0:
                    token_data = response[0]
                    
                    metadata = {
                        "address": token_address,
                        "name": token_data.get("account", {}).get("data", {}).get("name", "Unknown"),
                        "symbol": token_data.get("account", {}).get("data", {}).get("symbol", "UNKNOWN"),
                        "decimals": token_data.get("account", {}).get("data", {}).get("decimals", 9),
                        "supply": token_data.get("account", {}).get("data", {}).get("supply", 0),
                        "holders": 0,  # Helius doesn't provide this directly
                        "verified": False,  # Need to check verification
                        "timestamp": current_time
                    }
                    
                    # Cache the result
                    self._metadata_cache[cache_key] = {
                        'data': metadata,
                        'timestamp': current_time
                    }
                    
                    return metadata
            except Exception as e:
                logger.debug(f"Helius metadata API failed: {str(e)}")
            
            # Method 2: Get metadata from Jupiter token list
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    url = f"https://quote-api.jup.ag/v6/tokens"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                        if response.status == 200:
                            tokens = await response.json()
                            # Jupiter returns array of token addresses (strings), not objects
                            if isinstance(tokens, list) and token_address in tokens:
                                # Create minimal metadata for address found in Jupiter
                                metadata = {
                                    "address": token_address,
                                    "name": f"Token {token_address[:8]}",
                                    "symbol": f"TOKEN_{token_address[:8]}",
                                    "decimals": 9,
                                    "supply": 0,
                                    "holders": 0,
                                    "verified": True,  # If it's on Jupiter, it's somewhat verified
                                    "logo_uri": "",
                                    "timestamp": current_time
                                }
                                
                                # Cache the result
                                self._metadata_cache[cache_key] = {
                                    'data': metadata,
                                    'timestamp': current_time
                                }
                                
                                return metadata
            except Exception as e:
                logger.debug(f"Jupiter token list failed: {str(e)}")
            
            # Fallback: minimal metadata
            return {
                "address": token_address,
                "name": "Unknown Token",
                "symbol": "UNKNOWN",
                "decimals": 9,
                "supply": 0,
                "holders": 0,
                "verified": False,
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get token metadata for {token_address}: {str(e)}")
            return {}
    
    async def get_token_price(self, token_address: str) -> Dict[str, Any]:
        """Get current token price data."""
        try:
            # Check cache first for performance
            cache_key = f"price_{token_address}"
            current_time = time.time()
            
            if cache_key in self._price_cache:
                cached_data = self._price_cache[cache_key]
                if (current_time - cached_data['timestamp']) < self._cache_ttl:
                    return cached_data['data']
            
            # Method 1: Helius Token API (if available)
            try:
                response = await self._make_api_request(f"tokens/{token_address}")
                if response:
                    price_data = {
                        "address": token_address,
                        "price_usd": response.get("price", {}).get("usd", 0),
                        "price_change_1h": response.get("price_change_1h", 0),
                        "price_change_24h": response.get("price_change_24h", 0),
                        "volume_24h": response.get("volume_24h", 0),
                        "market_cap": response.get("market_cap", 0),
                        "liquidity": response.get("liquidity", 0),
                        "timestamp": current_time
                    }
                    
                    # Cache the result
                    self._price_cache[cache_key] = {
                        'data': price_data,
                        'timestamp': current_time
                    }
                    
                    return price_data
            except Exception as e:
                logger.debug(f"Helius token API failed: {str(e)}")
            
            # Method 2: Use Jupiter pricing as fallback
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    # Try multiple Jupiter API endpoints for DNS failover
                    jupiter_endpoints = [
                        "https://price.jup.ag/v4/price",
                        "https://quote-api.jup.ag/v6/price", 
                        "https://api.jup.ag/price/v1"  # Alternative endpoint
                    ]
                    
                    for endpoint in jupiter_endpoints:
                        try:
                            params = {"ids": token_address}
                            async with session.get(endpoint, params=params, timeout=aiohttp.ClientTimeout(total=3.0)) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    token_data = data.get("data", {}).get(token_address)
                                    if token_data:
                                        price_data = {
                                            "address": token_address,
                                            "price_usd": token_data.get("price", 0),
                                            "price_change_1h": 0,  # Jupiter doesn't provide change data
                                            "price_change_24h": 0,
                                            "volume_24h": 0,
                                            "market_cap": 0,
                                            "liquidity": 0,
                                            "timestamp": current_time
                                        }
                                        
                                        # Cache the result
                                        self._price_cache[cache_key] = {
                                            'data': price_data,
                                            'timestamp': current_time
                                        }
                                        
                                        return price_data
                        except Exception as endpoint_error:
                            logger.debug(f"Jupiter endpoint {endpoint} failed: {str(endpoint_error)}")
                            continue  # Try next endpoint
                            
            except Exception as e:
                logger.debug(f"Jupiter price API failed: {str(e)}")
            
            # Method 3: CoinGecko API for major tokens
            try:
                # Map token addresses to CoinGecko IDs for major tokens
                coingecko_mapping = {
                    "So11111111111111111111111111111111111111112": "solana",
                    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "usd-coin",
                    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "tether",
                    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": "bonk"
                }
                
                coingecko_id = coingecko_mapping.get(token_address)
                if coingecko_id:
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}"
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                            if response.status == 200:
                                data = await response.json()
                                market_data = data.get("market_data", {})
                                
                                price_data = {
                                    "address": token_address,
                                    "price_usd": market_data.get("current_price", {}).get("usd", 0),
                                    "price_change_1h": market_data.get("price_change_percentage_1h_in_currency", {}).get("usd", 0),
                                    "price_change_24h": market_data.get("price_change_percentage_24h_in_currency", {}).get("usd", 0),
                                    "volume_24h": market_data.get("total_volume", {}).get("usd", 0),
                                    "market_cap": market_data.get("market_cap", {}).get("usd", 0),
                                    "liquidity": market_data.get("total_volume", {}).get("usd", 0),  # Use volume as liquidity proxy
                                    "timestamp": current_time
                                }
                                
                                # Cache the result
                                self._price_cache[cache_key] = {
                                    'data': price_data,
                                    'timestamp': current_time
                                }
                                
                                return price_data
            except Exception as e:
                logger.debug(f"CoinGecko API failed: {str(e)}")
            
            # If all methods fail, return empty data
            logger.warning(f"Failed to get price data for token {token_address}")
            return {
                "address": token_address,
                "price_usd": 0,
                "price_change_1h": 0,
                "price_change_24h": 0,
                "volume_24h": 0,
                "market_cap": 0,
                "liquidity": 0,
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get token price for {token_address}: {str(e)}")
            return {}
    
    async def get_token_liquidity(self, token_address: str) -> Dict[str, Any]:
        """Get token liquidity information."""
        try:
            # Method 1: Helius DeFi APIs
            try:
                response = await self._make_api_request(f"tokens/{token_address}/liquidity")
                if response:
                    return {
                        "address": token_address,
                        "total_liquidity_usd": response.get("total_liquidity", 0),
                        "liquidity_pools": response.get("pools", []),
                        "timestamp": time.time()
                    }
            except Exception as e:
                logger.debug(f"Helius liquidity API failed: {str(e)}")
            
            # Method 2: Get liquidity from Jupiter pool data
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    # Get Jupiter pool info
                    url = f"https://quote-api.jup.ag/v6/tokens"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                        if response.status == 200:
                            tokens = await response.json()
                            # Jupiter returns array of token addresses (strings), not objects
                            if isinstance(tokens, list) and token_address in tokens:
                                # Estimate liquidity from Jupiter data
                                return {
                                    "address": token_address,
                                    "total_liquidity_usd": 0,  # Jupiter doesn't provide direct liquidity
                                    "liquidity_pools": [
                                        {
                                            "dex": "Jupiter_Aggregated",
                                            "liquidity_usd": 0,
                                            "volume_24h": 0
                                        }
                                    ],
                                    "timestamp": time.time()
                                }
            except Exception as e:
                logger.debug(f"Jupiter liquidity check failed: {str(e)}")
            
            # Method 3: Estimate from price data
            price_data = await self.get_token_price(token_address)
            estimated_liquidity = price_data.get("volume_24h", 0) * 2  # Rough estimate
            
            return {
                "address": token_address,
                "total_liquidity_usd": estimated_liquidity,
                "liquidity_pools": [
                    {
                        "dex": "estimated",
                        "liquidity_usd": estimated_liquidity,
                        "volume_24h": price_data.get("volume_24h", 0)
                    }
                ],
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get token liquidity for {token_address}: {str(e)}")
            return {}
    
    async def get_new_tokens(self, limit: int = 10, min_age_minutes: int = 5) -> List[Dict]:
        """Get recently created tokens."""
        try:
            if self.simulation_mode:
                return [
                    {
                        "address": f"SIM{i:010d}NewToken{int(time.time())}",
                        "symbol": f"NEW{i}",
                        "name": f"New Token {i}",
                        "decimals": 9,
                        "supply": 1000000,
                        "created_at": int(time.time()) - (i * 300),
                        "creator": f"Creator{i}Address",
                        "initial_liquidity": 1000 + (i * 100)
                    }
                    for i in range(1, limit + 1)
                ]
            
            params = {
                "limit": limit,
                "min_age_minutes": min_age_minutes
            }
            response = await self._make_api_request("tokens/new", params=params)
            return response.get("tokens", [])
            
        except Exception as e:
            logger.error(f"Failed to get new tokens: {str(e)}")
            return []
    
    async def get_token_holders(self, token_address: str, limit: int = 100) -> List[Dict]:
        """Get token holders information."""
        try:
            if self.simulation_mode:
                return [
                    {
                        "address": f"SIMHolder{i:010d}Address",
                        "balance": 1000000 - (i * 10000),
                        "percentage": round(10.0 / (i + 1), 2),
                        "rank": i + 1
                    }
                    for i in range(min(limit, 50))
                ]
            
            params = {"limit": limit}
            response = await self._make_api_request(f"tokens/{token_address}/holders", params=params)
            
            # Fix: Handle None response properly
            if response is None:
                logger.debug(f"No holder data available for token {token_address}")
                return []
            
            # Fix: Handle different response formats
            if isinstance(response, dict):
                holders = response.get("holders", [])
            elif isinstance(response, list):
                holders = response
            else:
                logger.warning(f"Unexpected response format for holders: {type(response)}")
                return []
            
            return holders if holders else []
            
        except Exception as e:
            logger.error(f"Failed to get token holders for {token_address}: {str(e)}")
            return []
    
    async def get_token_transactions(self, token_address: str, limit: int = 50) -> List[Dict]:
        """Get recent token transactions."""
        try:
            if self.simulation_mode:
                return [
                    {
                        "signature": f"SIMTransaction{i:010d}Signature",
                        "timestamp": int(time.time()) - (i * 60),
                        "type": "swap" if i % 2 == 0 else "transfer",
                        "amount": 1000 + (i * 100),
                        "price_usd": 0.01 + (i * 0.001),
                        "from_address": f"FromAddress{i}",
                        "to_address": f"ToAddress{i}"
                    }
                    for i in range(1, limit + 1)
                ]
            
            params = {"limit": limit}
            response = await self._make_api_request(f"tokens/{token_address}/transactions", params=params)
            return response.get("transactions", [])
            
        except Exception as e:
            logger.error(f"Failed to get token transactions for {token_address}: {str(e)}")
            return []
    
    async def analyze_token_security(self, token_address: str) -> Dict[str, Any]:
        """Analyze token for security risks."""
        try:
            if self.simulation_mode:
                return {
                    "address": token_address,
                    "is_mintable": False,
                    "is_freezable": False,
                    "has_mint_authority": False,
                    "has_freeze_authority": False,
                    "top_holder_percentage": 5.5,
                    "is_rug_risk": False,
                    "liquidity_locked": True,
                    "security_score": 85,
                    "risk_level": "low"
                }
            
            # Get token metadata and holders
            metadata = await self.get_token_metadata(token_address)
            holders = await self.get_token_holders(token_address, 10)
            
            # Analyze security factors
            has_mint_authority = metadata.get("mint_authority") is not None
            has_freeze_authority = metadata.get("freeze_authority") is not None
            top_holder_percentage = holders[0]["percentage"] if holders else 100
            
            # Calculate security score
            security_score = 100
            if has_mint_authority:
                security_score -= 20
            if has_freeze_authority:
                security_score -= 15
            if top_holder_percentage > 50:
                security_score -= 30
            elif top_holder_percentage > 20:
                security_score -= 15
            
            # Determine risk level
            if security_score >= 80:
                risk_level = "low"
            elif security_score >= 60:
                risk_level = "medium"
            elif security_score >= 40:
                risk_level = "high"
            else:
                risk_level = "extreme"
            
            return {
                "address": token_address,
                "is_mintable": has_mint_authority,
                "is_freezable": has_freeze_authority,
                "has_mint_authority": has_mint_authority,
                "has_freeze_authority": has_freeze_authority,
                "top_holder_percentage": top_holder_percentage,
                "is_rug_risk": security_score < 40,
                "security_score": security_score,
                "risk_level": risk_level
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze token security for {token_address}: {str(e)}")
            return {
                "address": token_address,
                "security_score": 0,
                "risk_level": "unknown",
                "error": str(e)
            }
    
    async def search_tokens(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for tokens by name or symbol."""
        try:
            if self.simulation_mode:
                return [
                    {
                        "address": f"SIMSearch{i:010d}Token",
                        "symbol": f"{query.upper()}{i}",
                        "name": f"{query} Token {i}",
                        "decimals": 9,
                        "price_usd": 0.01 + (i * 0.01),
                        "market_cap": 10000 + (i * 1000),
                        "volume_24h": 500 + (i * 50)
                    }
                    for i in range(1, min(limit, 5) + 1)
                ]
            
            params = {"q": query, "limit": limit}
            response = await self._make_api_request("tokens/search", params=params)
            return response.get("tokens", [])
            
        except Exception as e:
            logger.error(f"Failed to search tokens for query '{query}': {str(e)}")
            return []
    
    async def get_trending_tokens(self, time_range: str = "1h", limit: int = 20) -> List[Dict]:
        """Get trending tokens by volume or price change."""
        try:
            if self.simulation_mode:
                return [
                    {
                        "address": f"SIMTrending{i:010d}Token",
                        "symbol": f"TREND{i}",
                        "name": f"Trending Token {i}",
                        "price_usd": 0.1 + (i * 0.05),
                        "price_change": 50 - (i * 5),  # Decreasing trend
                        "volume_24h": 10000 - (i * 500),
                        "market_cap": 100000 - (i * 5000),
                        "rank": i
                    }
                    for i in range(1, min(limit, 10) + 1)
                ]
            
            params = {"time_range": time_range, "limit": limit}
            response = await self._make_api_request("tokens/trending", params=params)
            return response.get("tokens", [])
            
        except Exception as e:
            logger.error(f"Failed to get trending tokens: {str(e)}")
            return []
    
    async def close(self) -> None:
        """Close HTTP session and cleanup."""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            
            logger.info("Helius service closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing Helius service: {str(e)}")
    
    def __del__(self):
        """Ensure session is closed on object destruction."""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            # Create new event loop if necessary for cleanup
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.session.close())
                else:
                    loop.run_until_complete(self.session.close())
            except Exception:
                pass  # Ignore cleanup errors 

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "state": self.circuit_breaker.state.value,
            "failure_count": self.circuit_breaker.failure_count,
            "success_count": self.circuit_breaker.success_count,
            "last_failure_time": self.circuit_breaker.last_failure_time,
            "recent_failures": len(self.circuit_breaker.failure_history)
        }
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            logger.info("🧹 Helius Service cleaned up")
    
    async def get_network_metrics(self) -> Dict[str, Any]:
        """Get comprehensive network performance metrics"""
        try:
            current_time = time.time()
            
            # Circuit breaker metrics
            cb_status = self.get_circuit_breaker_status()
            
            # Calculate average latency from recent requests
            recent_latencies = [
                h.get('latency', 0) for h in self.failure_history 
                if current_time - h.get('timestamp', 0) < 300  # Last 5 minutes
            ]
            
            avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 100
            
            # Network health score based on circuit breaker state and performance
            if cb_status['state'] == 'closed':
                health_score = 100 - (cb_status['failure_count'] * 5)  # Reduce by 5% per failure
            elif cb_status['state'] == 'half_open':
                health_score = 50  # Moderate score during testing
            else:  # open
                health_score = 10  # Very low score when circuit is open
            
            health_score = max(0, min(100, health_score))
            
            return {
                "service_name": "helius",
                "circuit_breaker_state": cb_status['state'],
                "health_score": health_score,
                "average_latency_ms": avg_latency,
                "failure_count": cb_status['failure_count'],
                "success_count": cb_status['success_count'],
                "recent_failures": cb_status['recent_failures'],
                "api_key_configured": self.api_key is not None,
                "last_failure_time": cb_status['last_failure_time'],
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting network metrics: {str(e)}")
            return {
                "service_name": "helius",
                "circuit_breaker_state": "unknown",
                "health_score": 0,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def test_connectivity(self) -> Dict[str, Any]:
        """Test connectivity and measure response time"""
        try:
            start_time = time.time()
            
            # Use circuit breaker for the connectivity test
            result = await self.circuit_breaker.call(self._test_api_endpoint)
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                "connected": True,
                "response_time_ms": response_time,
                "api_accessible": result.get('accessible', False),
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "response_time_ms": 0,
                "api_accessible": False,
                "timestamp": time.time()
            }
    
    async def _test_api_endpoint(self) -> Dict[str, Any]:
        """Test a simple API endpoint to verify connectivity"""
        try:
            # Test with a simple endpoint that doesn't require authentication
            test_response = await self._make_api_request("ping", method="GET")
            
            if test_response is not None:
                return {"accessible": True, "response": test_response}
            else:
                # Try alternative health check
                if self.api_key:
                    # Test with authentication if we have a key
                    auth_response = await self._make_api_request("health", method="GET")
                    return {"accessible": auth_response is not None, "response": auth_response}
                else:
                    return {"accessible": False, "reason": "No API key configured"}
                    
        except Exception as e:
            return {"accessible": False, "error": str(e)} 