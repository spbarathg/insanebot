import aiohttp
import asyncio
from typing import Dict, Optional, Tuple
from ..utils.config import settings
from ..utils.logging_config import (
    market_logger, error_logger, handle_errors, log_performance,
    MarketDataError, NetworkError
)

class MarketData:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.token_cache = {}
        self.last_update = {}
        self.price_cache = {}
        self.liquidity_cache = {}
        self.cache_ttl = 30  # seconds
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
    @log_performance(market_logger)
    async def get_token_data(self, token_address: str) -> Optional[Dict]:
        """Get detailed data for a specific token"""
        try:
            # Check cache
            current_time = asyncio.get_event_loop().time()
            if (token_address in self.token_cache and 
                current_time - self.last_update.get(token_address, 0) < settings.DATA_REFRESH_INTERVAL):
                market_logger.debug(f"Returning cached data for token {token_address}")
                return self.token_cache[token_address]

            # Fetch data from multiple sources
            async with asyncio.TaskGroup() as group:
                birdeye_task = group.create_task(self._fetch_birdeye_token_data(token_address))
                dexscreener_task = group.create_task(self._fetch_dexscreener_token_data(token_address))

            # Get results
            birdeye_data = await birdeye_task
            dexscreener_data = await dexscreener_task

            # Combine data
            token_data = self._combine_token_data(birdeye_data, dexscreener_data)
            if token_data:
                self.token_cache[token_address] = token_data
                self.last_update[token_address] = current_time
                market_logger.info(f"Updated token data for {token_address}")
            else:
                market_logger.warning(f"No data available for token {token_address}")

            return token_data

        except Exception as e:
            error_msg = f"Error getting token data for {token_address}: {str(e)}"
            market_logger.error(error_msg)
            raise MarketDataError(error_msg)

    @handle_errors(market_logger)
    @log_performance(market_logger)
    async def _fetch_birdeye_token_data(self, token_address: str) -> Optional[Dict]:
        """Fetch token data from Birdeye"""
        try:
            headers = {
                'X-API-KEY': settings.BIRDEYE_API_KEY,
                'Accept': 'application/json'
            }
            
            async with self.session.get(
                f"{settings.BIRDEYE_API_URL}/tokens/{token_address}",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    market_logger.debug(f"Successfully fetched Birdeye data for {token_address}")
                    return self._process_birdeye_token_data(data)
                else:
                    error_msg = f"Birdeye API error: {response.status}"
                    market_logger.error(error_msg)
                    raise NetworkError(error_msg)

        except Exception as e:
            error_msg = f"Error fetching Birdeye token data for {token_address}: {str(e)}"
            market_logger.error(error_msg)
            raise MarketDataError(error_msg)

    @handle_errors(market_logger)
    @log_performance(market_logger)
    async def _fetch_dexscreener_token_data(self, token_address: str) -> Optional[Dict]:
        """Fetch token data from DexScreener"""
        try:
            async with self.session.get(
                f"{settings.DEXSCREENER_API_URL}/tokens/{token_address}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    market_logger.debug(f"Successfully fetched DexScreener data for {token_address}")
                    return self._process_dexscreener_token_data(data)
                else:
                    error_msg = f"DexScreener API error: {response.status}"
                    market_logger.error(error_msg)
                    raise NetworkError(error_msg)

        except Exception as e:
            error_msg = f"Error fetching DexScreener token data for {token_address}: {str(e)}"
            market_logger.error(error_msg)
            raise MarketDataError(error_msg)

    @handle_errors(market_logger)
    def _process_birdeye_token_data(self, data: Dict) -> Optional[Dict]:
        """Process raw Birdeye token data"""
        try:
            token = data.get('token', {})
            processed_data = {
                'address': token.get('address'),
                'symbol': token.get('symbol'),
                'price': float(token.get('price', 0)),
                'volume_24h': float(token.get('volume24h', 0)),
                'liquidity': float(token.get('liquidity', 0)),
                'market_cap': float(token.get('marketCap', 0)),
                'holders': int(token.get('holders', 0)),
                'transactions_24h': int(token.get('transactions24h', 0)),
                'source': 'birdeye'
            }
            market_logger.debug(f"Processed Birdeye data: {processed_data}")
            return processed_data
        except Exception as e:
            error_msg = f"Error processing Birdeye token data: {str(e)}"
            market_logger.error(error_msg)
            raise MarketDataError(error_msg)

    @handle_errors(market_logger)
    def _process_dexscreener_token_data(self, data: Dict) -> Optional[Dict]:
        """Process raw DexScreener token data"""
        try:
            pair = data.get('pair', {})
            token = pair.get('baseToken', {})
            processed_data = {
                'address': token.get('address'),
                'symbol': token.get('symbol'),
                'price': float(pair.get('priceUsd', 0)),
                'volume_24h': float(pair.get('volume24h', 0)),
                'liquidity': float(pair.get('liquidity', {}).get('usd', 0)),
                'market_cap': float(pair.get('marketCap', 0)),
                'holders': int(pair.get('holders', 0)),
                'transactions_24h': int(pair.get('transactions24h', 0)),
                'source': 'dexscreener'
            }
            market_logger.debug(f"Processed DexScreener data: {processed_data}")
            return processed_data
        except Exception as e:
            error_msg = f"Error processing DexScreener token data: {str(e)}"
            market_logger.error(error_msg)
            raise MarketDataError(error_msg)

    @handle_errors(market_logger)
    def _combine_token_data(self, birdeye_data: Optional[Dict], dexscreener_data: Optional[Dict]) -> Optional[Dict]:
        """Combine data from multiple sources"""
        try:
            if not birdeye_data and not dexscreener_data:
                market_logger.warning("No data available from either source")
                return None

            # Use Birdeye data as base if available
            if birdeye_data:
                base_data = birdeye_data.copy()
            else:
                base_data = dexscreener_data.copy()

            # Add DexScreener data if available
            if dexscreener_data:
                # Average prices if both sources available
                if birdeye_data:
                    base_data['price'] = (birdeye_data['price'] + dexscreener_data['price']) / 2
                    base_data['volume_24h'] = (birdeye_data['volume_24h'] + dexscreener_data['volume_24h']) / 2
                    base_data['liquidity'] = (birdeye_data['liquidity'] + dexscreener_data['liquidity']) / 2
                    base_data['market_cap'] = (birdeye_data['market_cap'] + dexscreener_data['market_cap']) / 2
                    base_data['holders'] = max(birdeye_data['holders'], dexscreener_data['holders'])
                    base_data['transactions_24h'] = max(birdeye_data['transactions_24h'], dexscreener_data['transactions_24h'])
                else:
                    base_data.update(dexscreener_data)

            market_logger.debug(f"Combined token data: {base_data}")
            return base_data

        except Exception as e:
            error_msg = f"Error combining token data: {str(e)}"
            market_logger.error(error_msg)
            raise MarketDataError(error_msg)

    @handle_errors(market_logger)
    @log_performance(market_logger)
    async def get_token_price(self, token_address: str) -> Optional[float]:
        """Get token price from Birdeye API"""
        try:
            if not self.session:
                await self.initialize()

            # Check cache first
            if token_address in self.price_cache:
                cache_time, price = self.price_cache[token_address]
                if asyncio.get_event_loop().time() - cache_time < self.cache_ttl:
                    market_logger.debug(f"Returning cached price for {token_address}: {price}")
                    return price

            headers = {"X-API-KEY": settings.BIRDEYE_API_KEY}
            url = f"{settings.BIRDEYE_API_URL}/public/price?address={token_address}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    price = float(data.get("data", {}).get("value", 0))
                    self.price_cache[token_address] = (asyncio.get_event_loop().time(), price)
                    market_logger.info(f"Updated price for {token_address}: {price}")
                    return price
                else:
                    error_msg = f"Failed to get price for {token_address}: {response.status}"
                    market_logger.error(error_msg)
                    raise NetworkError(error_msg)

        except Exception as e:
            error_msg = f"Error getting token price for {token_address}: {str(e)}"
            market_logger.error(error_msg)
            raise MarketDataError(error_msg)

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