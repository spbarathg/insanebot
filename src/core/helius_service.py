"""
Helius API service for Solana token data and transactions.
"""
import aiohttp
import asyncio
import logging
from typing import Dict, List, Optional, Any
from ..utils.config import settings

logger = logging.getLogger(__name__)

class HeliusService:
    def __init__(self):
        self.api_key = settings.HELIUS_API_KEY
        self.api_url = f"https://api.helius.xyz/v0"
        self.session = None
        self._cache = {}
        
    async def initialize(self) -> bool:
        """Initialize the Helius service."""
        try:
            self.session = aiohttp.ClientSession()
            logger.info("Helius service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Helius service: {str(e)}")
            return False
            
    async def close(self) -> None:
        """Close the Helius service."""
        if self.session:
            await self.session.close()
            
    async def get_token_price(self, token_address: str) -> Optional[Dict]:
        """Get token price from Helius."""
        try:
            if not self.session:
                await self.initialize()
                
            url = f"{self.api_url}/token-metadata"
            params = {
                "api-key": self.api_key,
                "mintAccounts": [token_address]
            }
            
            async with self.session.post(url, json=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        token_data = data[0]
                        price_info = {
                            "price": token_data.get("price", 0),
                            "pricePerSol": token_data.get("pricePerSol", 0)
                        }
                        return price_info
                    else:
                        return None
                else:
                    logger.error(f"Failed to get token price: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting token price: {str(e)}")
            return None
            
    async def get_token_metadata(self, token_address: str) -> Optional[Dict]:
        """Get token metadata from Helius."""
        try:
            if not self.session:
                await self.initialize()
                
            url = f"{self.api_url}/token-metadata"
            params = {
                "api-key": self.api_key,
                "mintAccounts": [token_address]
            }
            
            async with self.session.post(url, json=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        return data[0]
                    else:
                        return None
                else:
                    logger.error(f"Failed to get token metadata: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting token metadata: {str(e)}")
            return None
            
    async def get_token_balances(self, wallet_address: str) -> Optional[Dict]:
        """Get token balances for a wallet."""
        try:
            if not self.session:
                await self.initialize()
                
            url = f"{self.api_url}/balances"
            params = {
                "api-key": self.api_key,
                "wallet": wallet_address
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Failed to get token balances: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting token balances: {str(e)}")
            return None
            
    async def get_token_holders(self, token_address: str, limit: int = 100) -> Optional[List[Dict]]:
        """Get token holders data."""
        try:
            if not self.session:
                await self.initialize()
                
            url = f"{self.api_url}/token-holders"
            params = {
                "api-key": self.api_key,
                "mintAccount": token_address,
                "limit": limit
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("holders", [])
                else:
                    logger.error(f"Failed to get token holders: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting token holders: {str(e)}")
            return None
            
    async def get_token_liquidity(self, token_address: str) -> Optional[Dict]:
        """Get token liquidity data."""
        try:
            # This is a placeholder as Helius doesn't have a direct liquidity endpoint
            # In a real implementation, you might calculate this from DEX data
            # or use a different service
            
            # For now, get token metadata which might have some liquidity info
            metadata = await self.get_token_metadata(token_address)
            if metadata:
                # Extract any liquidity information available
                # This is a simplified example
                liquidity_data = {
                    "liquidity": metadata.get("volumeUsd24h", 0) / 10,  # Rough estimate
                    "timestamp": metadata.get("lastUpdatedAt", 0)
                }
                return liquidity_data
            return None
        except Exception as e:
            logger.error(f"Error getting token liquidity: {str(e)}")
            return None 