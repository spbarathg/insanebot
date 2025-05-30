"""
Data Ingestion System with Optimized Resource Management

This module handles real-time data collection from multiple sources:
- QuickNode (Primary API service)
- Helius (Backup API service) 
- Jupiter (DEX data)
- X/Twitter (Sentiment analysis)
- Whale activity monitoring

Features:
- Intelligent service failover
- Caching for performance optimization
- Real-time sentiment processing
- Whale activity tracking
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Set
from loguru import logger
import aiohttp
import websockets
import os
from solders.pubkey import Pubkey
import re

# Data Ingestion Configuration Constants
class DataConstants:
    """Centralized constants for data ingestion system"""
    # Cache durations (seconds)
    SENTIMENT_CACHE_TTL = 300        # 5 minutes
    WHALE_CACHE_TTL = 300           # 5 minutes
    MARKET_DATA_CACHE_TTL = 30      # 30 seconds
    METADATA_CACHE_TTL = 3600       # 1 hour
    
    # API limits and timeouts
    REQUEST_TIMEOUT_SECONDS = 10    # Request timeout
    WEBSOCKET_RETRY_DELAY = 5       # WebSocket reconnection delay
    MAX_RETRIES = 3                 # Maximum retry attempts
    
    # Data processing limits
    MAX_TOKENS_PER_REQUEST = 10     # Maximum tokens to process per request
    MAX_TRANSACTION_LIMIT = 20      # Maximum transactions to analyze
    WHALE_THRESHOLD_SOL = 10.0      # Minimum SOL amount to consider whale
    
    # Whale management
    MAX_WHALE_WHITELIST_SIZE = 10   # Maximum whitelisted whales
    MIN_TRADES_FOR_WHALE_RATING = 5 # Minimum trades to rate whale
    
    # Token validation
    SOLANA_ADDRESS_MIN_LENGTH = 32  # Minimum Solana address length
    SOLANA_ADDRESS_MAX_LENGTH = 44  # Maximum Solana address length
    TOKEN_MINT_DATA_SIZE = 82       # Size of token mint account
    
    # Default fallback tokens
    DEFAULT_TOKENS = [
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        "So11111111111111111111111111111111111111112",   # SOL (wrapped)
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",   # USDT
    ]
    
    # Sentiment keywords
    BULLISH_KEYWORDS = ['pump', 'moon', 'rocket', 'bullish', 'up', 'rise']
    BEARISH_KEYWORDS = ['dump', 'crash', 'bear', 'down', 'fall', 'rug']

class DataIngestion:
    """Optimized data ingestion system with intelligent failover and caching"""
    
    def __init__(self, quicknode_service=None, helius_service=None, jupiter_service=None):
        # Core components
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        
        # API Services (QuickNode Primary + Helius Backup + Jupiter DEX)
        self.quicknode_service = quicknode_service
        self.helius_service = helius_service
        self.jupiter_service = jupiter_service
        
        # Enhanced cache systems with TTL tracking
        self.sentiment_cache: Dict[str, Dict] = {}
        self.whale_cache: Dict[str, Dict] = {}
        self.market_data_cache: Dict[str, Dict] = {}
        self.metadata_cache: Dict[str, Dict] = {}
        
        # Whale management with performance tracking
        self.whale_whitelist: List[str] = []
        self.whale_performance: Dict[str, Dict] = {}
        
        # Service status and monitoring
        self.running = False
        self.primary_service_active = False
        self.service_stats = {
            "requests_made": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "last_error": None
        }
        
        logger.info("Data Ingestion initialized with enhanced caching and failover")
        self._load_whale_whitelist()

    def _load_whale_whitelist(self):
        """Load whitelist of high-win-rate whale wallets with validation"""
        try:
            whale_file_path = "whale.log"
            if not os.path.exists(whale_file_path):
                logger.warning("Whale log file not found, creating empty whitelist")
                self.whale_whitelist = []
                return
                
            with open(whale_file_path, 'r') as f:
                whale_data = json.load(f)
                
            if not isinstance(whale_data, list):
                logger.error("Invalid whale data format, expected list")
                self.whale_whitelist = []
                return
            
            # Filter and sort whales by performance
            valid_whales = []
            for whale in whale_data:
                if (isinstance(whale, dict) and 
                    whale.get('address') and 
                    isinstance(whale.get('trades', []), list) and
                    len(whale.get('trades', [])) >= DataConstants.MIN_TRADES_FOR_WHALE_RATING):
                    
                    trades = whale['trades']
                    wins = sum(1 for t in trades if t.get('profit', 0) > 0)
                    whale['win_rate'] = wins / len(trades) if trades else 0
                    valid_whales.append(whale)
            
            # Sort by win rate and take top performers
            self.whale_whitelist = [
                whale['address'] for whale in sorted(
                    valid_whales,
                    key=lambda x: x.get('win_rate', 0),
                    reverse=True
                )[:DataConstants.MAX_WHALE_WHITELIST_SIZE]
            ]
            
            logger.info(f"Loaded {len(self.whale_whitelist)} whales from whitelist")
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading whale whitelist: {str(e)}")
            self.whale_whitelist = []
        except Exception as e:
            logger.error(f"Unexpected error loading whale whitelist: {str(e)}")
            self.whale_whitelist = []

    async def start(self) -> bool:
        """Start the data ingestion service with comprehensive initialization"""
        try:
            # Create session with timeout configuration
            timeout = aiohttp.ClientTimeout(total=DataConstants.REQUEST_TIMEOUT_SECONDS)
            self.session = aiohttp.ClientSession(timeout=timeout)
            self.running = True
            
            # Check service availability with validation
            await self._check_service_availability()
            
            # Initialize websocket connection for real-time data
            await self._connect_websocket()
            
            logger.info("✅ Data ingestion service started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start data ingestion: {str(e)}")
            self.service_stats["errors"] += 1
            self.service_stats["last_error"] = str(e)
            return False

    async def _check_service_availability(self):
        """Check which API services are available with health checks"""
        try:
            # Check QuickNode service
            if self.quicknode_service and hasattr(self.quicknode_service, 'endpoint_url'):
                if await self._test_service_health(self.quicknode_service, "QuickNode"):
                    self.primary_service_active = True
                    logger.info("🚀 QuickNode primary service active and healthy")
                else:
                    logger.warning("⚠️ QuickNode service health check failed")
            
            # Check Helius backup service
            if self.helius_service and hasattr(self.helius_service, 'api_key'):
                if await self._test_service_health(self.helius_service, "Helius"):
                    logger.info("🔄 Helius backup service available and healthy")
                else:
                    logger.warning("⚠️ Helius service health check failed")
            
            # Check Jupiter service
            if self.jupiter_service:
                if await self._test_service_health(self.jupiter_service, "Jupiter"):
                    logger.info("📊 Jupiter DEX service available and healthy")
                else:
                    logger.warning("⚠️ Jupiter service health check failed")
            
            # Fallback warning if no services available
            if not self.primary_service_active and not (self.helius_service and self.helius_service.api_key):
                logger.warning("⚠️ No premium API services available - limited functionality")
                
        except Exception as e:
            logger.error(f"Service availability check failed: {str(e)}")

    async def _test_service_health(self, service, service_name: str) -> bool:
        """Test service health with timeout protection"""
        try:
            # Basic health check based on service type
            if hasattr(service, 'health_check'):
                return await asyncio.wait_for(
                    service.health_check(), 
                    timeout=DataConstants.REQUEST_TIMEOUT_SECONDS
                )
            elif hasattr(service, 'api_key') and service.api_key:
                return True  # Consider available if has API key
            elif hasattr(service, 'endpoint_url') and service.endpoint_url:
                return True  # Consider available if has endpoint
            else:
                return False
                
        except asyncio.TimeoutError:
            logger.warning(f"{service_name} health check timeout")
            return False
        except Exception as e:
            logger.debug(f"{service_name} health check error: {str(e)}")
            return False

    async def stop(self) -> None:
        """Gracefully stop the data ingestion service"""
        try:
            self.running = False
            
            # Close WebSocket connection
            if self.ws:
                await self.ws.close()
                self.ws = None
                logger.debug("WebSocket connection closed")
            
            # Close HTTP session
            if self.session:
                await self.session.close()
                self.session = None
                logger.debug("HTTP session closed")
            
            # Save whale performance data
            await self._save_whale_performance()
            
            logger.info("✅ Data ingestion service stopped gracefully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping data ingestion: {str(e)}")

    # Legacy method for backward compatibility
    async def close(self) -> None:
        """Legacy method - use stop() instead"""
        logger.warning("close() is deprecated, use stop() instead")
        await self.stop()

    async def _connect_websocket(self):
        """Connect to X WebSocket for real-time sentiment with retry logic"""
        max_retries = DataConstants.MAX_RETRIES
        retry_count = 0
        
        while retry_count < max_retries and self.running:
            try:
                x_api_key = os.getenv("X_API_KEY", "")
                if not x_api_key:
                    logger.warning("X API key not configured, skipping WebSocket connection")
                    return
                
                self.ws = await websockets.connect(
                    'wss://api.twitter.com/2/tweets/search/stream',
                    extra_headers={'Authorization': f'Bearer {x_api_key}'},
                    ping_interval=30,  # Keep connection alive
                    ping_timeout=10
                )
                
                logger.info("✅ WebSocket connected for real-time sentiment")
                
                # Start sentiment processing task
                asyncio.create_task(self._process_sentiment_stream())
                return
                
            except Exception as e:
                retry_count += 1
                logger.error(f"WebSocket connection attempt {retry_count} failed: {str(e)}")
                
                if retry_count < max_retries:
                    await asyncio.sleep(DataConstants.WEBSOCKET_RETRY_DELAY * retry_count)
                else:
                    logger.error("Max WebSocket connection retries exceeded")

    async def _process_sentiment_stream(self):
        """Process real-time sentiment stream with enhanced error handling"""
        try:
            while self.running and self.ws:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(
                        self.ws.recv(), 
                        timeout=DataConstants.REQUEST_TIMEOUT_SECONDS
                    )
                    
                    data = json.loads(message)
                    await self._process_sentiment_data(data)
                    
                except asyncio.TimeoutError:
                    logger.debug("WebSocket receive timeout, checking connection...")
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed, attempting reconnection...")
                    break
                except json.JSONDecodeError as e:
                    logger.debug(f"Invalid JSON received: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Sentiment stream processing error: {str(e)}")
            
        # Attempt reconnection if still running
        if self.running:
            await asyncio.sleep(DataConstants.WEBSOCKET_RETRY_DELAY)
            await self._connect_websocket()

    async def _process_sentiment_data(self, data: Dict):
        """Process individual sentiment data with token extraction"""
        try:
            tweets = data.get('data', [])
            if not isinstance(tweets, list):
                return
                
            for tweet in tweets:
                if not isinstance(tweet, dict):
                    continue
                    
                text = tweet.get('text', '').lower()
                if not text:
                    continue
                
                # Check for relevant keywords
                has_bullish = any(keyword in text for keyword in DataConstants.BULLISH_KEYWORDS)
                has_bearish = any(keyword in text for keyword in DataConstants.BEARISH_KEYWORDS)
                
                if has_bullish or has_bearish:
                    # Extract token address if present
                    token_address = self._extract_token_address(text)
                    if token_address:
                        # Calculate sentiment score
                        sentiment_score = await self._calculate_sentiment_score(text, has_bullish, has_bearish)
                        
                        if abs(sentiment_score) > 0.1:  # Only store significant sentiment
                            self._cache_sentiment(token_address, sentiment_score)
                            
        except Exception as e:
            logger.debug(f"Error processing sentiment data: {str(e)}")

    def _extract_token_address(self, text: str) -> Optional[str]:
        """Extract Solana token address from text with validation"""
        try:
            # Regular expression for Solana addresses (base58 encoded)
            pattern = f'[1-9A-HJ-NP-Za-km-z]{{{DataConstants.SOLANA_ADDRESS_MIN_LENGTH},{DataConstants.SOLANA_ADDRESS_MAX_LENGTH}}}'
            matches = re.findall(pattern, text)
            
            if not matches:
                return None
                
            # Validate each potential address
            for address in matches:
                try:
                    # Check if it's a valid public key
                    pubkey = Pubkey(address)
                    
                    # Additional validation could be added here
                    # For now, return first valid pubkey
                    return str(pubkey)
                    
                except Exception:
                    continue
                    
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting token address: {str(e)}")
            return None

    async def _calculate_sentiment_score(self, text: str, has_bullish: bool, has_bearish: bool) -> float:
        """Calculate sentiment score from text analysis"""
        try:
            # Basic sentiment scoring
            bullish_count = sum(1 for keyword in DataConstants.BULLISH_KEYWORDS if keyword in text)
            bearish_count = sum(1 for keyword in DataConstants.BEARISH_KEYWORDS if keyword in text)
            
            # Calculate net sentiment (-1 to 1)
            total_keywords = bullish_count + bearish_count
            if total_keywords == 0:
                return 0.0
                
            net_sentiment = (bullish_count - bearish_count) / total_keywords
            
            # Apply additional weighting based on text length and context
            text_length_factor = min(1.0, len(text) / 280)  # Normalize to tweet length
            sentiment_score = net_sentiment * text_length_factor
            
            return max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1, 1]
            
        except Exception as e:
            logger.debug(f"Error calculating sentiment score: {str(e)}")
            return 0.0

    def _cache_sentiment(self, token_address: str, sentiment_score: float):
        """Cache sentiment score with timestamp and aggregation"""
        try:
            current_time = time.time()
            
            # Get existing cache entry or create new one
            cache_entry = self.sentiment_cache.get(token_address, {
                'scores': [],
                'average_score': 0.0,
                'timestamp': current_time
            })
            
            # Add new score to rolling window
            cache_entry['scores'].append({
                'score': sentiment_score,
                'timestamp': current_time
            })
            
            # Keep only recent scores (within TTL)
            ttl_cutoff = current_time - DataConstants.SENTIMENT_CACHE_TTL
            cache_entry['scores'] = [
                score for score in cache_entry['scores'] 
                if score['timestamp'] > ttl_cutoff
            ]
            
            # Calculate weighted average (more recent scores have higher weight)
            if cache_entry['scores']:
                total_weight = 0
                weighted_sum = 0
                
                for score_entry in cache_entry['scores']:
                    age = current_time - score_entry['timestamp']
                    weight = max(0.1, 1.0 - (age / DataConstants.SENTIMENT_CACHE_TTL))
                    weighted_sum += score_entry['score'] * weight
                    total_weight += weight
                
                cache_entry['average_score'] = weighted_sum / total_weight if total_weight > 0 else 0.0
                cache_entry['timestamp'] = current_time
                
                self.sentiment_cache[token_address] = cache_entry
                
                logger.debug(f"Cached sentiment for {token_address[:8]}...: {cache_entry['average_score']:.3f}")
                
        except Exception as e:
            logger.debug(f"Error caching sentiment: {str(e)}")

    def get_sentiment(self, token: str) -> float:
        """Get cached sentiment score for token with TTL validation"""
        try:
            cache_entry = self.sentiment_cache.get(token)
            if not cache_entry:
                self.service_stats["cache_misses"] += 1
                return 0.0
                
            # Check if cache entry is still valid
            if time.time() - cache_entry['timestamp'] > DataConstants.SENTIMENT_CACHE_TTL:
                del self.sentiment_cache[token]
                self.service_stats["cache_misses"] += 1
                return 0.0
                
            self.service_stats["cache_hits"] += 1
            return cache_entry.get('average_score', 0.0)
            
        except Exception as e:
            logger.debug(f"Error getting sentiment: {str(e)}")
            self.service_stats["cache_misses"] += 1
            return 0.0

    def get_whale_activity(self, token: str) -> Dict:
        """Get whale activity for token with enhanced caching"""
        try:
            cache_entry = self.whale_cache.get(token)
            if not cache_entry:
                self.service_stats["cache_misses"] += 1
                return {'buy_count': 0, 'total_buy_volume': 0.0, 'whale_addresses': []}
                
            # Check cache validity
            if time.time() - cache_entry['timestamp'] > DataConstants.WHALE_CACHE_TTL:
                del self.whale_cache[token]
                self.service_stats["cache_misses"] += 1
                return {'buy_count': 0, 'total_buy_volume': 0.0, 'whale_addresses': []}
                
            self.service_stats["cache_hits"] += 1
            return cache_entry.get('data', {'buy_count': 0, 'total_buy_volume': 0.0, 'whale_addresses': []})
            
        except Exception as e:
            logger.debug(f"Error getting whale activity: {str(e)}")
            self.service_stats["cache_misses"] += 1
            return {'buy_count': 0, 'total_buy_volume': 0.0, 'whale_addresses': []}

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
            with open("whale.log", 'r') as f:
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
            with open("whale.log", 'w') as f:
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
                # Extract addresses from token objects if needed
                if tokens and isinstance(tokens[0], dict):
                    token_addresses = [token.get("address", token) for token in tokens if token.get("address")]
                else:
                    token_addresses = tokens
            else:
                # Fallback token list
                token_addresses = [
                    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                    "So11111111111111111111111111111111111111112",   # SOL
                ]
            
            # Get detailed data for each token using QuickNode
            for token_address in token_addresses[:5]:  # Limit to 5 for performance
                try:
                    # Ensure token_address is a string, not a dict
                    if isinstance(token_address, dict):
                        token_address = token_address.get("address")
                    
                    if not token_address or not isinstance(token_address, str):
                        continue
                        
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

    def _load_whale_data(self):
        """Load whale data from file"""
        try:
            whale_log_file = "whale.log"
            with open(whale_log_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return [] 