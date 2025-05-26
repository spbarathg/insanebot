import asyncio
import json
import time
from typing import Dict, List, Optional
from loguru import logger
import aiohttp
import websockets
from src.utils.config import settings
from solders.pubkey import Pubkey

class DataIngestion:
    def __init__(self, quicknode_service=None, helius_service=None, jupiter_service=None):
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        
        # API Services (QuickNode Primary + Helius Backup + Jupiter DEX)
        self.quicknode_service = quicknode_service
        self.helius_service = helius_service
        self.jupiter_service = jupiter_service
        
        # Cache systems
        self.sentiment_cache: Dict[str, float] = {}
        self.whale_cache: Dict[str, Dict] = {}
        self.whale_whitelist: List[str] = []
        self.market_data_cache = {}
        self.last_update = 0
        
        # Service status
        self.running = False
        self.primary_service_active = False
        
        logger.info("Data Ingestion initialized with QuickNode Primary + Helius Backup + Jupiter DEX")
        self._load_whale_whitelist()

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
            
            # Check service availability
            self._check_service_availability()
            
            logger.info("✅ Data ingestion service started")
            await self._connect_websocket()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start data ingestion: {str(e)}")
            return False

    def _check_service_availability(self):
        """Check which API services are available"""
        if self.quicknode_service and self.quicknode_service.endpoint_url:
            self.primary_service_active = True
            logger.info("🚀 QuickNode primary service active for data ingestion")
        elif self.helius_service and self.helius_service.api_key:
            logger.info("🔄 Using Helius backup for data ingestion")
        else:
            logger.warning("⚠️ No premium API services available for data ingestion")

    async def close(self) -> None:
        """Close the data ingestion service"""
        try:
            self.running = False
            if self.session:
                await self.session.close()
            if self.ws:
                await self.ws.close()
            logger.info("✅ Data ingestion service closed")
        except Exception as e:
            logger.error(f"❌ Error closing data ingestion: {str(e)}")

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
                    pubkey = Pubkey(address)
                    
                    # Check if it's a token mint
                    if self._is_token_mint(pubkey):
                        return str(pubkey)
                except:
                    continue
                    
            return None
            
        except Exception as e:
            logger.error(f"Error extracting token address: {e}")
            return None
            
    def _is_token_mint(self, pubkey: Pubkey) -> bool:
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
        if cache_entry and time.time() - cache_entry['timestamp'] < 300:  # 5 minutes
            return cache_entry['data']
        return {'buy_count': 0, 'total_buy_volume': 0}

    async def update_whale_activity(self, token: str):
        """Update whale activity using available services"""
        try:
            if self.primary_service_active:
                # Use QuickNode to get recent transactions
                transactions = await self.quicknode_service.get_token_transactions(token, limit=20)
                whale_activity = self._analyze_whale_transactions(transactions)
            elif self.helius_service:
                # Use Helius to get transactions
                transactions = await self.helius_service.get_token_transactions(token, limit=20)
                whale_activity = self._analyze_whale_transactions(transactions)
            else:
                whale_activity = {'buy_count': 0, 'total_buy_volume': 0}
            
            self.whale_cache[token] = {
                'data': whale_activity,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error updating whale activity: {str(e)}")

    def _analyze_whale_transactions(self, transactions: List[Dict]) -> Dict:
        """Analyze transactions for whale activity"""
        try:
            whale_buys = 0
            total_volume = 0
            
            for tx in transactions:
                # Simple whale detection (transactions > 10 SOL)
                if tx.get("amount", 0) > 10:
                    whale_buys += 1
                    total_volume += tx.get("amount", 0)
            
            return {
                'buy_count': whale_buys,
                'total_buy_volume': total_volume
            }
            
        except Exception as e:
            logger.error(f"Error analyzing whale transactions: {str(e)}")
            return {'buy_count': 0, 'total_buy_volume': 0}

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
        """Get market data using QuickNode primary with Helius fallback"""
        try:
            # Check cache first
            current_time = time.time()
            if (current_time - self.last_update) < 30:  # 30 second cache
                return self.market_data_cache.get("tokens", [])
            
            # Try QuickNode first (primary)
            if self.primary_service_active:
                market_data = await self._get_market_data_quicknode()
                if market_data:
                    self.market_data_cache["tokens"] = market_data
                    self.last_update = current_time
                    return market_data
            
            # Fallback to Helius
            if self.helius_service and self.helius_service.api_key:
                logger.debug("Using Helius fallback for market data")
                market_data = await self._get_market_data_helius()
                if market_data:
                    self.market_data_cache["tokens"] = market_data
                    self.last_update = current_time
                    return market_data
            
            # Return cached data if available
            return self.market_data_cache.get("tokens", [])
            
        except Exception as e:
            logger.error(f"❌ Error getting market data: {str(e)}")
            return self.market_data_cache.get("tokens", [])

    async def _get_market_data_quicknode(self) -> List[Dict]:
        """Get market data using QuickNode"""
        try:
            market_data = []
            
            # Get trending tokens from Jupiter (for discovery)
            if self.jupiter_service:
                tokens = await self.jupiter_service.get_random_tokens(count=10)
            else:
                # Fallback token list
                tokens = [
                    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                    "So11111111111111111111111111111111111111112",   # SOL
                ]
            
            # Get detailed data for each token using QuickNode
            for token_address in tokens[:5]:  # Limit to 5 for performance
                try:
                    # Get metadata from QuickNode
                    metadata = await self.quicknode_service.get_token_metadata(token_address)
                    
                    # Get price data from QuickNode
                    price_data = await self.quicknode_service.get_token_price_from_dex_pools(token_address)
                    
                    # Get holder data from QuickNode
                    holders = await self.quicknode_service.get_token_holders(token_address, limit=10)
                    
                    # Combine data
                    token_data = {
                        "address": token_address,
                        "symbol": metadata.get("symbol", "UNKNOWN"),
                        "name": metadata.get("name", "Unknown Token"),
                        "price": price_data.get("price", 0),
                        "price_usd": price_data.get("price_usd", 0),
                        "liquidity": price_data.get("liquidity", 0),
                        "volume_24h": price_data.get("volume_24h", 0),
                        "holders": len(holders),
                        "market_cap": metadata.get("supply", 0) * price_data.get("price", 0),
                        "verified": metadata.get("verified", False),
                        "source": "quicknode_primary",
                        "timestamp": time.time()
                    }
                    
                    market_data.append(token_data)
                    
                except Exception as e:
                    logger.debug(f"Error processing token {token_address}: {str(e)}")
                    continue
            
            logger.info(f"🚀 QuickNode: Retrieved {len(market_data)} tokens")
            return market_data
            
        except Exception as e:
            logger.error(f"❌ QuickNode market data error: {str(e)}")
            return []

    async def _get_market_data_helius(self) -> List[Dict]:
        """Get market data using Helius backup"""
        try:
            # Get new tokens from Helius
            tokens = await self.helius_service.get_new_tokens(limit=5)
            
            market_data = []
            for token_data in tokens:
                try:
                    # Get additional data from Helius
                    price_data = await self.helius_service.get_token_price(token_data.get("address"))
                    
                    # Format data
                    formatted_data = {
                        "address": token_data.get("address"),
                        "symbol": token_data.get("symbol", "UNKNOWN"),
                        "name": token_data.get("name", "Unknown Token"),
                        "price": price_data.get("price", 0),
                        "price_usd": price_data.get("price", 0) * 100,  # Assuming SOL = $100
                        "liquidity": 0,  # Not available from Helius
                        "volume_24h": 0,  # Not available from Helius
                        "holders": token_data.get("holders", 0),
                        "market_cap": 0,  # Calculate if supply available
                        "verified": token_data.get("verified", False),
                        "source": "helius_backup",
                        "timestamp": time.time()
                    }
                    
                    market_data.append(formatted_data)
                    
                except Exception as e:
                    logger.debug(f"Error processing Helius token: {str(e)}")
                    continue
            
            logger.info(f"🔄 Helius: Retrieved {len(market_data)} tokens")
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Helius market data error: {str(e)}")
            return [] 