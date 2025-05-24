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

class HeliusAPIError(Exception):
    """Raised when Helius API calls fail"""
    pass

class HeliusService:
    """
    Real Helius service for Solana blockchain data.
    Supports both live API calls and simulation mode.
    """
    
    def __init__(self):
        """Initialize Helius service with real API configuration."""
        self.simulation_mode = os.getenv("SIMULATION_MODE", "true").lower() == "true"
        self.api_key = os.getenv("HELIUS_API_KEY", "")
        self.base_url = "https://api.helius.xyz/v0"
        self.rpc_url = f"https://mainnet.helius-rpc.com/?api-key={self.api_key}"
        self.websocket_url = f"wss://mainnet.helius-rpc.com/?api-key={self.api_key}"
        
        # Rate limiting
        self.max_requests_per_second = 10
        self.request_interval = 1.0 / self.max_requests_per_second
        self.last_request_time = 0
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=30)
        
        # Validation
        self._validate_configuration()
        
        logger.info(f"Helius service initialized in {'simulation' if self.simulation_mode else 'live'} mode")
    
    def _validate_configuration(self) -> None:
        """Validate Helius service configuration."""
        if not self.simulation_mode:
            if not self.api_key or self.api_key in ["", "abc123example_replace_with_real_api_key", "demo_key_for_testing"]:
                logger.error("Invalid or missing HELIUS_API_KEY for live mode")
                raise HeliusAPIError("HELIUS_API_KEY must be set for live mode")
            
            logger.info(f"Using Helius API key: {self.api_key[:8]}...")
        else:
            logger.info("Running in simulation mode - using mock data")
    
    async def _ensure_session(self) -> None:
        """Ensure HTTP session is created."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={"User-Agent": "Solana-Trading-Bot/1.0"}
            )
    
    async def _rate_limit(self) -> None:
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_interval:
            sleep_time = self.request_interval - time_since_last_request
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _make_api_request(self, endpoint: str, method: str = "GET", params: Dict = None, data: Dict = None) -> Dict:
        """Make authenticated API request to Helius."""
        if self.simulation_mode:
            # Return simulated data
            return self._get_simulation_data(endpoint)
        
        await self._ensure_session()
        await self._rate_limit()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data
            ) as response:
                
                if response.status == 429:
                    # Rate limited, wait and retry
                    await asyncio.sleep(1)
                    return await self._make_api_request(endpoint, method, params, data)
                
                if response.status >= 400:
                    error_text = await response.text()
                    raise HeliusAPIError(f"API request failed: {response.status} - {error_text}")
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            raise HeliusAPIError(f"Network error: {str(e)}")
        except Exception as e:
            raise HeliusAPIError(f"Unexpected error: {str(e)}")
    
    def _get_simulation_data(self, endpoint: str) -> Dict:
        """Generate realistic simulation data based on endpoint."""
        if "tokens" in endpoint or "token" in endpoint:
            return {
                "tokens": [
                    {
                        "address": f"SIM{i:010d}TokenAddress{int(time.time())}",
                        "symbol": f"SIM{i}",
                        "name": f"Simulation Token {i}",
                        "decimals": 9,
                        "supply": 1000000,
                        "price_usd": round(0.001 + (i * 0.01), 6),
                        "market_cap": 1000 + (i * 100),
                        "volume_24h": 500 + (i * 50),
                        "holders": 100 + (i * 10),
                        "created_at": int(time.time()) - (i * 3600)
                    }
                    for i in range(1, 6)
                ]
            }
        elif "price" in endpoint:
            return {
                "price_usd": round(0.01 + (time.time() % 100) * 0.001, 6),
                "price_change_24h": round((time.time() % 20) - 10, 2),
                "volume_24h": 1000 + (time.time() % 5000),
                "market_cap": 50000 + (time.time() % 100000)
            }
        elif "holders" in endpoint:
            return {
                "holders": [
                    {
                        "address": f"SIMHolder{i:010d}Address{int(time.time())}",
                        "balance": 1000 + (i * 100),
                        "percentage": round(5.0 / (i + 1), 2)
                    }
                    for i in range(10)
                ]
            }
        else:
            return {"simulated": True, "endpoint": endpoint, "timestamp": time.time()}
    
    async def get_token_metadata(self, token_address: str) -> Dict[str, Any]:
        """Get comprehensive token metadata."""
        try:
            if self.simulation_mode:
                return {
                    "address": token_address,
                    "symbol": "SIMTOKEN",
                    "name": "Simulation Token",
                    "decimals": 9,
                    "supply": 1000000,
                    "mint_authority": None,
                    "freeze_authority": None,
                    "is_initialized": True,
                    "created_at": int(time.time()) - 3600
                }
            
            response = await self._make_api_request(f"tokens/{token_address}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get token metadata for {token_address}: {str(e)}")
            raise
    
    async def get_token_price(self, token_address: str) -> Dict[str, Any]:
        """Get current token price data."""
        try:
            if self.simulation_mode:
                return {
                    "address": token_address,
                    "price_usd": round(0.01 + (time.time() % 100) * 0.001, 6),
                    "price_change_1h": round((time.time() % 10) - 5, 2),
                    "price_change_24h": round((time.time() % 20) - 10, 2),
                    "volume_24h": 1000 + (time.time() % 5000),
                    "market_cap": 50000 + (time.time() % 100000),
                    "liquidity": 10000 + (time.time() % 20000)
                }
            
            response = await self._make_api_request(f"tokens/{token_address}/price")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get token price for {token_address}: {str(e)}")
            raise
    
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
            return response.get("holders", [])
            
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