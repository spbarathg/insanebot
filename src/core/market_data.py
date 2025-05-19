import aiohttp
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from ..utils.config import settings
from ..utils.logging_config import (
    market_logger, error_logger, handle_errors, log_performance,
    MarketDataError, NetworkError
)
from .middleware import rate_limit, error_handler
from .cache import market_data_cache, token_cache, price_cache

logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.token_cache = {}
        self.last_update = {}
        self.price_cache = {}
        self.liquidity_cache = {}
        self.cache_ttl = 30  # seconds
        self.update_interval = 60  # 1 minute
        market_logger.info("MarketData instance initialized")

    @handle_errors(market_logger)
    @log_performance(market_logger)
    async def initialize(self) -> bool:
        """Initialize the market data service"""
        try:
            self.session = aiohttp.ClientSession()
            market_logger.info("Market data service initialized successfully")
            return True
        except Exception as e:
            error_msg = f"Failed to initialize market data: {str(e)}"
            market_logger.error(error_msg)
            raise MarketDataError(error_msg)

    @handle_errors(market_logger)
    async def close(self) -> None:
        """Close the market data service"""
        try:
            if self.session:
                await self.session.close()
            market_logger.info("Market data service closed successfully")
        except Exception as e:
            error_msg = f"Error closing market data: {str(e)}"
            market_logger.error(error_msg)
            raise MarketDataError(error_msg)

    @handle_errors(market_logger)
    @rate_limit(max_requests=30, time_window=60)  # 30 requests per minute
    async def get_token_data(self, token_address: str) -> Optional[Dict]:
        """Get token data with caching"""
        try:
            # Check cache first
            cached_data = token_cache.get(token_address)
            if cached_data:
                return cached_data

            # Get fresh data
            async with self.session.get(
                f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and 'pairs' in data:
                        token_data = self._process_token_data(data['pairs'][0])
                        if token_data:
                            token_cache.set(token_address, token_data)
                        return token_data
                return None

        except Exception as e:
            await error_handler.handle_error(e, "get_token_data")
            return None

    @handle_errors(market_logger)
    @rate_limit(max_requests=10, time_window=60)  # 10 requests per minute
    async def get_price_data(self, token_address: str) -> Optional[Dict]:
        """Get price data with caching"""
        try:
            # Check cache first
            cached_data = price_cache.get(token_address)
            if cached_data:
                return cached_data

            # Get fresh data
            async with self.session.get(
                f"https://public-api.birdeye.so/public/price?address={token_address}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and 'data' in data:
                        price_data = self._process_price_data(data['data'])
                        if price_data:
                            price_cache.set(token_address, price_data)
                        return price_data
                return None

        except Exception as e:
            await error_handler.handle_error(e, "get_price_data")
            return None

    def _process_token_data(self, data: Dict) -> Optional[Dict]:
        """Process raw token data"""
        try:
            return {
                'address': data['baseToken']['address'],
                'symbol': data['baseToken']['symbol'],
                'price': float(data['priceUsd']),
                'volume_24h': float(data['volume']['h24']),
                'liquidity': float(data['liquidity']['usd']),
                'market_cap': float(data['marketCap']),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing token data: {e}")
            return None

    def _process_price_data(self, data: Dict) -> Optional[Dict]:
        """Process raw price data"""
        try:
            return {
                'price': float(data['value']),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing price data: {e}")
            return None

    @handle_errors(market_logger)
    @log_performance(market_logger)
    async def get_liquidity_info(self, token_address: str) -> Optional[Dict]:
        """Get liquidity information from DexScreener"""
        try:
            if not self.session:
                await self.initialize()

            # Check cache first
            if token_address in self.liquidity_cache:
                cache_time, data = self.liquidity_cache[token_address]
                if asyncio.get_event_loop().time() - cache_time < self.cache_ttl:
                    market_logger.debug(f"Returning cached liquidity info for {token_address}")
                    return data

            url = f"{settings.DEXSCREENER_API_URL}/tokens/{token_address}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get("pairs", [])
                    if pairs:
                        # Get the pair with highest liquidity
                        pair = max(pairs, key=lambda x: float(x.get("liquidity", {}).get("usd", 0)))
                        liquidity_info = {
                            "liquidity_usd": float(pair.get("liquidity", {}).get("usd", 0)),
                            "volume_24h": float(pair.get("volume", {}).get("h24", 0)),
                            "price_usd": float(pair.get("priceUsd", 0)),
                            "price_change_24h": float(pair.get("priceChange", {}).get("h24", 0)),
                            "holders": int(pair.get("holders", 0)),
                            "created_at": pair.get("createdAt", 0)
                        }
                        self.liquidity_cache[token_address] = (asyncio.get_event_loop().time(), liquidity_info)
                        market_logger.info(f"Updated liquidity info for {token_address}")
                        return liquidity_info
                else:
                    error_msg = f"Failed to get liquidity info for {token_address}: {response.status}"
                    market_logger.error(error_msg)
                    raise NetworkError(error_msg)

        except Exception as e:
            error_msg = f"Error getting liquidity info for {token_address}: {str(e)}"
            market_logger.error(error_msg)
            raise MarketDataError(error_msg)

    @handle_errors(market_logger)
    @log_performance(market_logger)
    async def get_swap_quote(self, input_token: str, output_token: str, amount: float) -> Optional[Dict]:
        """Get swap quote from Jupiter API"""
        try:
            if not self.session:
                await self.initialize()

            url = f"{settings.JUPITER_API_URL}/quote"
            params = {
                "inputMint": input_token,
                "outputMint": output_token,
                "amount": str(int(amount * 1e9)),  # Convert to lamports
                "slippageBps": int(settings.MIN_SLIPPAGE * 10000)
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    quote_data = await response.json()
                    market_logger.info(f"Got swap quote for {input_token} -> {output_token}")
                    return quote_data
                else:
                    error_msg = f"Failed to get swap quote: {response.status}"
                    market_logger.error(error_msg)
                    raise NetworkError(error_msg)

        except Exception as e:
            error_msg = f"Error getting swap quote: {str(e)}"
            market_logger.error(error_msg)
            raise MarketDataError(error_msg)

    @handle_errors(market_logger)
    @log_performance(market_logger)
    async def check_volatility(self, token_address: str) -> Tuple[bool, float]:
        """Check token volatility"""
        try:
            liquidity_info = await self.get_liquidity_info(token_address)
            if not liquidity_info:
                market_logger.warning(f"No liquidity info available for {token_address}")
                return True, 1.0  # Assume high volatility if no data

            price_change = abs(liquidity_info["price_change_24h"])
            is_volatile = price_change > settings.VOLATILITY_THRESHOLD
            market_logger.info(f"Volatility check for {token_address}: {is_volatile} ({price_change}%)")
            return is_volatile, price_change

        except Exception as e:
            error_msg = f"Error checking volatility for {token_address}: {str(e)}"
            market_logger.error(error_msg)
            raise MarketDataError(error_msg)

    @handle_errors(market_logger)
    @log_performance(market_logger)
    async def check_rug_risk(self, token_address: str) -> Tuple[bool, str]:
        """Check for rug pull risks"""
        try:
            liquidity_info = await self.get_liquidity_info(token_address)
            if not liquidity_info:
                market_logger.warning(f"No liquidity info available for {token_address}")
                return True, "No liquidity data available"

            # Check liquidity
            if liquidity_info["liquidity_usd"] < settings.MIN_LIQUIDITY:
                risk_msg = f"Insufficient liquidity: ${liquidity_info['liquidity_usd']:.2f}"
                market_logger.warning(f"Rug risk for {token_address}: {risk_msg}")
                return True, risk_msg

            # Check holder count
            if liquidity_info["holders"] < settings.MIN_HOLDER_COUNT:
                risk_msg = f"Low holder count: {liquidity_info['holders']}"
                market_logger.warning(f"Rug risk for {token_address}: {risk_msg}")
                return True, risk_msg

            # Check token age
            current_time = asyncio.get_event_loop().time()
            token_age = current_time - liquidity_info["created_at"]
            if token_age < settings.MAX_TOKEN_AGE:
                risk_msg = f"Token too new: {token_age:.0f} seconds old"
                market_logger.warning(f"Rug risk for {token_address}: {risk_msg}")
                return True, risk_msg

            market_logger.info(f"No rug risks detected for {token_address}")
            return False, "No significant risks detected"

        except Exception as e:
            error_msg = f"Error checking rug risk for {token_address}: {str(e)}"
            market_logger.error(error_msg)
            raise MarketDataError(error_msg)

    async def cleanup(self) -> None:
        """Clean up expired cache entries"""
        market_data_cache.cleanup()
        token_cache.cleanup()
        price_cache.cleanup() 