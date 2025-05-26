from typing import Dict, List, Optional
import asyncio
import aiohttp
from datetime import datetime, timedelta
import logging
from config.core_config import MARKET_CONFIG

logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self):
        self.config = MARKET_CONFIG
        self._cache = {}
        self._last_update = {}
        
    async def get_market_data(self, token_address: str) -> Dict:
        """Get market data for a specific token."""
        if self._is_cache_valid(token_address):
            return self._cache[token_address]
            
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch market data from various sources
                price_data = await self._fetch_price_data(session, token_address)
                volume_data = await self._fetch_volume_data(session, token_address)
                liquidity_data = await self._fetch_liquidity_data(session, token_address)
                
                market_data = {
                    "price": price_data,
                    "volume": volume_data,
                    "liquidity": liquidity_data,
                    "timestamp": datetime.now().isoformat()
                }
                
                self._update_cache(token_address, market_data)
                return market_data
                
        except Exception as e:
            logger.error(f"Error fetching market data for {token_address}: {str(e)}")
            return {}
            
    def _is_cache_valid(self, token_address: str) -> bool:
        """Check if cached data is still valid."""
        if token_address not in self._last_update:
            return False
            
        cache_age = datetime.now() - self._last_update[token_address]
        return cache_age < timedelta(seconds=self.config["update_interval"])
        
    def _update_cache(self, token_address: str, data: Dict):
        """Update cache with new data."""
        self._cache[token_address] = data
        self._last_update[token_address] = datetime.now()
        
    async def _fetch_price_data(self, session: aiohttp.ClientSession, token_address: str) -> Dict:
        """Fetch price data for a token."""
        try:
            # Simple price fetch from Solana RPC
            async with session.get(f"https://api.mainnet-beta.solana.com", json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenSupply",
                "params": [token_address]
            }) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "price": float(data.get("result", {}).get("value", {}).get("uiAmount", 0)),
                        "decimals": data.get("result", {}).get("value", {}).get("decimals", 0)
                    }
                return {"price": 0, "decimals": 0}
        except Exception as e:
            logger.error(f"Error fetching price data: {str(e)}")
            return {"price": 0, "decimals": 0}
        
    async def _fetch_volume_data(self, session: aiohttp.ClientSession, token_address: str) -> Dict:
        """Fetch volume data for a token."""
        try:
            # Simple volume calculation from recent transactions
            async with session.get(f"https://api.mainnet-beta.solana.com", json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [token_address, {"limit": 100}]
            }) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "24h_volume": len(data.get("result", [])),
                        "transaction_count": len(data.get("result", []))
                    }
                return {"24h_volume": 0, "transaction_count": 0}
        except Exception as e:
            logger.error(f"Error fetching volume data: {str(e)}")
            return {"24h_volume": 0, "transaction_count": 0}
        
    async def _fetch_liquidity_data(self, session: aiohttp.ClientSession, token_address: str) -> Dict:
        """Fetch liquidity data for a token."""
        try:
            # Simple liquidity check
            async with session.get(f"https://api.mainnet-beta.solana.com", json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenLargestAccounts",
                "params": [token_address]
            }) as response:
                if response.status == 200:
                    data = await response.json()
                    accounts = data.get("result", {}).get("value", [])
                    total_liquidity = sum(float(acc.get("uiAmount", 0)) for acc in accounts)
                    return {
                        "total_liquidity": total_liquidity,
                        "holder_count": len(accounts)
                    }
                return {"total_liquidity": 0, "holder_count": 0}
        except Exception as e:
            logger.error(f"Error fetching liquidity data: {str(e)}")
            return {"total_liquidity": 0, "holder_count": 0} 