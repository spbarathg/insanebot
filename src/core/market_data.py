"""
Market data service for Solana trading bot.
"""
import logging
import time
import asyncio
from typing import Dict, List, Optional, Any
from src.services.helius_service import HeliusService
from src.services.jupiter_service import JupiterService

logger = logging.getLogger(__name__)

class MarketData:
    """
    Market data service that provides token prices, liquidity information,
    and other market metrics by combining data from Helius and Jupiter.
    """
    
    def __init__(self):
        self.helius = HeliusService()
        self.jupiter = JupiterService()
        self._price_cache = {}
        self._liquidity_cache = {}
        self._token_cache = {}
        self._cache_ttl = 60  # Cache TTL in seconds
        self.wsol_mint = "So11111111111111111111111111111111111111112"
        self.usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        
    async def initialize(self) -> bool:
        """Initialize market data service."""
        try:
            logger.info("Initializing market data service...")
            
            # Initialize Helius
            if not await self.helius.initialize():
                logger.error("Failed to initialize Helius service")
                return False
                
            # Initialize Jupiter
            if not await self.jupiter.initialize():
                logger.error("Failed to initialize Jupiter service")
                return False
                
            logger.info("Market data service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing market data service: {str(e)}")
            return False
            
    async def close(self) -> None:
        """Close market data service."""
        try:
            await self.helius.close()
            await self.jupiter.close()
            logger.info("Market data service closed")
        except Exception as e:
            logger.error(f"Error closing market data service: {str(e)}")
            
    async def get_token_price(self, token_address: str) -> Optional[Dict]:
        """
        Get token price data with caching.
        
        Combines data from Helius and Jupiter for comprehensive price metrics.
        """
        try:
            # Check cache first
            cache_key = f"price_{token_address}"
            if cache_key in self._price_cache:
                cached_data = self._price_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self._cache_ttl:
                    return cached_data['data']
                    
            # Get price from Helius
            helius_price = await self.helius.get_token_price(token_address)
            
            # Get price from Jupiter as a backup
            jupiter_price = await self.jupiter.get_price(token_address, "USDC")
            
            # Combine data
            price_data = {
                'address': token_address,
                'price_usd': None,
                'price_sol': None,
                'timestamp': time.time()
            }
            
            # Use Helius data if available
            if helius_price:
                price_data['price_usd'] = helius_price.get('price', 0)
                # Convert to SOL price if available
                if 'pricePerSol' in helius_price:
                    price_data['price_sol'] = helius_price['pricePerSol']
                
            # If no Helius data, try Jupiter data
            if not price_data['price_usd'] and jupiter_price:
                price_data['price_usd'] = jupiter_price
                
            # If we have price data, cache it
            if price_data['price_usd'] or price_data['price_sol']:
                self._price_cache[cache_key] = {
                    'timestamp': time.time(),
                    'data': price_data
                }
                return price_data
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting token price: {str(e)}")
            return None
            
    async def get_token_liquidity(self, token_address: str) -> Optional[Dict]:
        """
        Get token liquidity data with caching.
        """
        try:
            # Check cache first
            cache_key = f"liquidity_{token_address}"
            if cache_key in self._liquidity_cache:
                cached_data = self._liquidity_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self._cache_ttl:
                    return cached_data['data']
                    
            # Get liquidity from Helius
            helius_liquidity = await self.helius.get_token_liquidity(token_address)
            
            # Use Helius data if available
            if helius_liquidity:
                liquidity_data = {
                    'address': token_address,
                    'liquidity_usd': helius_liquidity.get('liquidity', 0),
                    'liquidity_sol': 0,  # Default to 0
                    'timestamp': time.time()
                }
                
                # Cache the data
                self._liquidity_cache[cache_key] = {
                    'timestamp': time.time(),
                    'data': liquidity_data
                }
                
                return liquidity_data
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting token liquidity: {str(e)}")
            return None
            
    async def get_token_metadata(self, token_address: str) -> Optional[Dict]:
        """
        Get token metadata with caching.
        """
        try:
            # Check cache first
            cache_key = f"token_{token_address}"
            if cache_key in self._token_cache:
                cached_data = self._token_cache[cache_key]
                # Token metadata doesn't change often, use longer TTL
                if time.time() - cached_data['timestamp'] < self._cache_ttl * 10:
                    return cached_data['data']
                    
            # Get metadata from Helius
            helius_metadata = await self.helius.get_token_metadata(token_address)
            
            # Use Helius data if available
            if helius_metadata:
                # Cache the data
                self._token_cache[cache_key] = {
                    'timestamp': time.time(),
                    'data': helius_metadata
                }
                
                return helius_metadata
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting token metadata: {str(e)}")
            return None
            
    async def get_token_full_data(self, token_address: str) -> Optional[Dict]:
        """
        Get comprehensive token data including price, liquidity, and metadata.
        """
        try:
            tasks = [
                self.get_token_price(token_address),
                self.get_token_liquidity(token_address),
                self.get_token_metadata(token_address)
            ]
            
            results = await asyncio.gather(*tasks)
            price_data, liquidity_data, metadata = results
            
            if not metadata:
                logger.warning(f"No metadata found for token {token_address}")
                return None
                
            # Combine all data
            full_data = {
                'address': token_address,
                'symbol': metadata.get('symbol', ''),
                'name': metadata.get('name', ''),
                'decimals': metadata.get('decimals', 9),
                'price_usd': price_data.get('price_usd', 0) if price_data else 0,
                'price_sol': price_data.get('price_sol', 0) if price_data else 0,
                'liquidity_usd': liquidity_data.get('liquidity_usd', 0) if liquidity_data else 0,
                'liquidity_sol': liquidity_data.get('liquidity_sol', 0) if liquidity_data else 0,
                'timestamp': time.time()
            }
            
            return full_data
            
        except Exception as e:
            logger.error(f"Error getting token full data: {str(e)}")
            return None
            
    async def get_token_holders(self, token_address: str, limit: int = 100) -> Optional[List[Dict]]:
        """
        Get token holders using Helius.
        """
        try:
            # Get holders from Helius
            holders = await self.helius.get_token_holders(token_address, limit)
            return holders
            
        except Exception as e:
            logger.error(f"Error getting token holders: {str(e)}")
            return None
            
    async def get_swap_quote(self, input_token: str, output_token: str, amount: float) -> Optional[Dict]:
        """
        Get swap quote from Jupiter.
        """
        try:
            # Get quote from Jupiter
            quote = await self.jupiter.get_swap_quote(input_token, output_token, amount)
            return quote
            
        except Exception as e:
            logger.error(f"Error getting swap quote: {str(e)}")
            return None
            
    async def get_price_impact(self, input_token: str, output_token: str, amount: float) -> float:
        """
        Calculate price impact for a trade.
        """
        try:
            # Get quote from Jupiter
            quote = await self.jupiter.get_swap_quote(input_token, output_token, amount)
            
            if not quote:
                return 1.0  # Default to high impact if no quote
                
            # Extract price impact
            price_impact = float(quote.get('priceImpactPct', 0))
            
            return price_impact
            
        except Exception as e:
            logger.error(f"Error calculating price impact: {str(e)}")
            return 1.0  # Default to high impact
            
    async def get_top_tokens(self, limit: int = 100) -> List[Dict]:
        """
        Get list of top tokens by volume.
        
        This is a placeholder since Helius doesn't provide a direct API for this.
        In a real implementation, you might want to aggregate data from multiple sources.
        """
        # This is a placeholder implementation
        try:
            # Get tokens from Jupiter
            jupiter_tokens = await self.jupiter.get_tokens()
            
            if not jupiter_tokens:
                return []
                
            # Sort by volume if available
            tokens = sorted(
                jupiter_tokens,
                key=lambda x: x.get('volume24h', 0),
                reverse=True
            )[:limit]
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error getting top tokens: {str(e)}")
            return [] 