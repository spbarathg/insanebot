import asyncio
import json
import time
from typing import Dict, List, Optional
from loguru import logger
import aiohttp
import websockets
from config import settings
from solana.publickey import PublicKey

class DataIngestion:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.sentiment_cache: Dict[str, float] = {}
        self.whale_cache: Dict[str, Dict] = {}
        self.whale_whitelist: List[str] = []
        self._load_whale_whitelist()
        self.running = False
        self.market_data_cache = {}
        self.last_update = 0

    def _load_whale_whitelist(self):
        """Load whitelist of high-win-rate whale wallets"""
        try:
            with open(settings.WHALE_LOG_FILE, 'r') as f:
                whale_data = json.load(f)
                # Sort by win rate and take top 10
                self.whale_whitelist = [
                    w['address'] for w in sorted(
                        whale_data,
                        key=lambda x: x.get('win_rate', 0),
                        reverse=True
                    )[:10]
                ]
        except FileNotFoundError:
            self.whale_whitelist = []

    async def start(self) -> bool:
        """Start the data ingestion service"""
        try:
            self.session = aiohttp.ClientSession()
            self.running = True
            logger.info("Data ingestion service started")
            await self._connect_websocket()
            return True
        except Exception as e:
            logger.error(f"Failed to start data ingestion: {str(e)}")
            return False

    async def close(self) -> None:
        """Close the data ingestion service"""
        try:
            self.running = False
            if self.session:
                await self.session.close()
            if self.ws:
                await self.ws.close()
            logger.info("Data ingestion service closed")
        except Exception as e:
            logger.error(f"Error closing data ingestion: {str(e)}")

    async def _connect_websocket(self):
        """Connect to X WebSocket for real-time sentiment"""
        try:
            self.ws = await websockets.connect(
                'wss://api.twitter.com/2/tweets/search/stream',
                extra_headers={
                    'Authorization': f'Bearer {settings.X_API_KEY}'
                }
            )
            # Start sentiment processing
            asyncio.create_task(self._process_sentiment_stream())
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            await asyncio.sleep(5)
            await self._connect_websocket()

    async def _process_sentiment_stream(self):
        """Process real-time sentiment stream"""
        try:
            while True:
                if not self.ws:
                    await asyncio.sleep(5)
                    continue

                message = await self.ws.recv()
                data = json.loads(message)

                # Extract token mentions and sentiment
                for tweet in data.get('data', []):
                    text = tweet.get('text', '').lower()
                    if 'pump' in text or 'moon' in text:
                        # Extract token address if present
                        token = self._extract_token_address(text)
                        if token:
                            # Get sentiment from Grok API
                            sentiment = await self._get_sentiment(text)
                            if sentiment > 0:
                                self.sentiment_cache[token] = {
                                    'score': sentiment,
                                    'timestamp': time.time()
                                }

        except Exception as e:
            logger.error(f"Process sentiment stream error: {e}")
            await asyncio.sleep(5)
            await self._process_sentiment_stream()

    def _extract_token_address(self, text: str) -> Optional[str]:
        """Extract Solana token address from text"""
        try:
            # Regular expression for Solana addresses (base58 encoded, 32-44 characters)
            import re
            pattern = r'[1-9A-HJ-NP-Za-km-z]{32,44}'
            matches = re.findall(pattern, text)
            
            if not matches:
                return None
                
            # Validate each potential address
            for address in matches:
                try:
                    # Check if it's a valid public key
                    pubkey = PublicKey(address)
                    
                    # Check if it's a token mint
                    if self._is_token_mint(pubkey):
                        return str(pubkey)
                except:
                    continue
                    
            return None
            
        except Exception as e:
            logger.error(f"Error extracting token address: {e}")
            return None
            
    def _is_token_mint(self, pubkey: PublicKey) -> bool:
        """Check if a public key is a token mint"""
        try:
            # Get account info
            response = self.session.get_account_info(pubkey)
            if not response["result"]["value"]:
                return False
                
            # Check if it's a token mint account
            data = response["result"]["value"]["data"]
            return (
                data["owner"] == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA" and
                len(data["data"]) == 82  # Size of a mint account
            )
            
        except Exception as e:
            logger.error(f"Error checking token mint: {e}")
            return False

    async def _get_sentiment(self, text: str) -> float:
        """Get sentiment score from Grok API"""
        try:
            async with self.session.post(
                'https://api.grok.ai/v1/sentiment',
                json={'text': text},
                headers={'Authorization': f'Bearer {settings.GROK_API_KEY}'}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('score', 0)
                return 0
        except Exception as e:
            logger.error(f"Get sentiment error: {e}")
            return 0

    async def get_pump_data(self) -> Optional[Dict]:
        """Get data from pump.fun"""
        try:
            async with self.session.get(
                f'https://api.quicknode.com/v1/pump',
                headers={'Authorization': f'Bearer {settings.QUICKNODE_API_KEY}'}
            ) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error(f"Get pump data error: {e}")
            return None

    def get_sentiment(self, token: str) -> float:
        """Get cached sentiment score for token"""
        cache_entry = self.sentiment_cache.get(token)
        if cache_entry and time.time() - cache_entry['timestamp'] < 300:  # 5 minutes
            return cache_entry['score']
        return 0

    def get_whale_activity(self, token: str) -> Dict:
        """Get whale activity for token"""
        cache_entry = self.whale_cache.get(token)
        if cache_entry and time.time() - cache_entry['timestamp'] < settings.WHALE_WINDOW:
            return cache_entry['data']
        return {'buy_count': 0, 'total_buy_volume': 0}

    async def _update_whale_activity(self, token: str):
        """Update whale activity for token"""
        try:
            async with self.session.get(
                f'https://api.quicknode.com/v1/whales/{token}',
                headers={'Authorization': f'Bearer {settings.QUICKNODE_API_KEY}'}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.whale_cache[token] = {
                        'data': data,
                        'timestamp': time.time()
                    }
        except Exception as e:
            logger.error(f"Update whale activity error: {e}")

    def _is_whitelisted_whale(self, address: str) -> bool:
        """Check if address is in whale whitelist"""
        return address in self.whale_whitelist

    async def _update_whale_whitelist(self):
        """Update whale whitelist based on performance"""
        try:
            with open(settings.WHALE_LOG_FILE, 'r') as f:
                whale_data = json.load(f)

            # Calculate win rates
            for whale in whale_data:
                trades = whale.get('trades', [])
                if trades:
                    wins = sum(1 for t in trades if t.get('profit', 0) > 0)
                    whale['win_rate'] = wins / len(trades)

            # Sort by win rate and take top 10
            self.whale_whitelist = [
                w['address'] for w in sorted(
                    whale_data,
                    key=lambda x: x.get('win_rate', 0),
                    reverse=True
                )[:10]
            ]

            # Save updated whitelist
            with open(settings.WHALE_LOG_FILE, 'w') as f:
                json.dump(whale_data, f)

        except Exception as e:
            logger.error(f"Update whale whitelist error: {e}")

    async def get_market_data(self) -> List[Dict]:
        """Get market data for all tokens"""
        try:
            # Check cache
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_update < settings.DATA_REFRESH_INTERVAL:
                return list(self.market_data_cache.values())

            # Fetch data from multiple sources
            async with asyncio.TaskGroup() as group:
                birdeye_task = group.create_task(self._fetch_birdeye_data())
                dexscreener_task = group.create_task(self._fetch_dexscreener_data())

            # Combine and deduplicate data
            market_data = {}
            
            # Process Birdeye data
            birdeye_data = await birdeye_task
            for token in birdeye_data:
                market_data[token['address']] = {
                    'address': token['address'],
                    'symbol': token['symbol'],
                    'price': token['price'],
                    'volume_24h': token['volume_24h'],
                    'liquidity': token['liquidity'],
                    'market_cap': token['market_cap'],
                    'created_at': token['created_at'],
                    'source': 'birdeye'
                }

            # Process DexScreener data
            dexscreener_data = await dexscreener_task
            for token in dexscreener_data:
                if token['address'] not in market_data:
                    market_data[token['address']] = {
                        'address': token['address'],
                        'symbol': token['symbol'],
                        'price': token['price'],
                        'volume_24h': token['volume_24h'],
                        'liquidity': token['liquidity'],
                        'market_cap': token['market_cap'],
                        'created_at': token['created_at'],
                        'source': 'dexscreener'
                    }

            # Update cache
            self.market_data_cache = market_data
            self.last_update = current_time

            return list(market_data.values())

        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return []

    async def _fetch_birdeye_data(self) -> List[Dict]:
        """Fetch data from Birdeye API"""
        try:
            headers = {
                'X-API-KEY': settings.BIRDEYE_API_KEY,
                'Accept': 'application/json'
            }
            
            async with self.session.get(
                f"{settings.BIRDEYE_API_URL}/tokens/list",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_birdeye_data(data)
                else:
                    logger.error(f"Birdeye API error: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error fetching Birdeye data: {str(e)}")
            return []

    async def _fetch_dexscreener_data(self) -> List[Dict]:
        """Fetch data from DexScreener API"""
        try:
            async with self.session.get(
                f"{settings.DEXSCREENER_API_URL}/tokens/solana"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_dexscreener_data(data)
                else:
                    logger.error(f"DexScreener API error: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error fetching DexScreener data: {str(e)}")
            return []

    def _process_birdeye_data(self, data: Dict) -> List[Dict]:
        """Process raw Birdeye data"""
        try:
            tokens = []
            for token in data.get('tokens', []):
                tokens.append({
                    'address': token['address'],
                    'symbol': token['symbol'],
                    'price': float(token['price']),
                    'volume_24h': float(token['volume24h']),
                    'liquidity': float(token['liquidity']),
                    'market_cap': float(token['marketCap']),
                    'created_at': int(token['createdAt'])
                })
            return tokens
        except Exception as e:
            logger.error(f"Error processing Birdeye data: {str(e)}")
            return []

    def _process_dexscreener_data(self, data: Dict) -> List[Dict]:
        """Process raw DexScreener data"""
        try:
            tokens = []
            for pair in data.get('pairs', []):
                token = pair['baseToken']
                tokens.append({
                    'address': token['address'],
                    'symbol': token['symbol'],
                    'price': float(pair['priceUsd']),
                    'volume_24h': float(pair['volume24h']),
                    'liquidity': float(pair['liquidity']['usd']),
                    'market_cap': float(pair['marketCap']),
                    'created_at': int(pair['createdAt'])
                })
            return tokens
        except Exception as e:
            logger.error(f"Error processing DexScreener data: {str(e)}")
            return [] 