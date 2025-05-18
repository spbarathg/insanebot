import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple
from loguru import logger
import aiohttp
from datetime import datetime, timedelta
from ..utils.config import settings
from collections import defaultdict

class GrokEngine:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.decision_cache: Dict[str, Dict] = {}
        self.sentiment_cache: Dict[str, Dict] = {}
        self.feature_weights = settings.FEATURE_WEIGHTS.copy()
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        self._load_learning_data()
        self._setup_logging()
        self.twitter_cache_ttl = 300  # 5 minutes
        self.decision_cache_ttl = 30  # 30 seconds

    def _setup_logging(self):
        logger.add(settings.DEBUG_LOG_FILE, rotation="100 MB")

    def _load_learning_data(self):
        """Load learning data from file"""
        try:
            with open('learning_data.json', 'r') as f:
                data = json.load(f)
                self.feature_weights = data.get('feature_weights', self.feature_weights)
                self.confidence_threshold = data.get('confidence_threshold', self.confidence_threshold)
        except FileNotFoundError:
            pass

    def _save_learning_data(self):
        """Save learning data to file"""
        with open('learning_data.json', 'w') as f:
            json.dump({
                'feature_weights': self.feature_weights,
                'confidence_threshold': self.confidence_threshold
            }, f)

    async def start(self):
        """Start Grok engine"""
        self.session = aiohttp.ClientSession()
        # Start background tasks
        asyncio.create_task(self._cleanup_cache())

    async def close(self):
        """Close Grok engine"""
        if self.session:
            await self.session.close()
        self._save_learning_data()

    async def _cleanup_cache(self):
        """Periodically clean up expired cache entries"""
        while True:
            try:
                current_time = time.time()
                
                # Clean sentiment cache
                expired_sentiments = [
                    token for token, data in self.sentiment_cache.items()
                    if current_time - data['timestamp'] > self.twitter_cache_ttl
                ]
                for token in expired_sentiments:
                    del self.sentiment_cache[token]
                
                # Clean decision cache
                expired_decisions = [
                    token for token, data in self.decision_cache.items()
                    if current_time - data['timestamp'] > self.decision_cache_ttl
                ]
                for token in expired_decisions:
                    del self.decision_cache[token]
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)

    async def get_twitter_sentiment(self, token: str) -> float:
        """Get real-time Twitter sentiment for a token"""
        try:
            # Check cache first
            cache_entry = self.sentiment_cache.get(token)
            if cache_entry and time.time() - cache_entry['timestamp'] < self.twitter_cache_ttl:
                return cache_entry['sentiment']

            # Get tweets from Twitter API
            async with self.session.get(
                f"https://api.twitter.com/2/tweets/search/recent",
                params={
                    "query": f"#{token} OR ${token}",
                    "tweet.fields": "created_at,public_metrics,lang",
                    "max_results": 100
                },
                headers={"Authorization": f"Bearer {settings.TWITTER_API_KEY}"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    sentiment = self._analyze_twitter_sentiment(data)
                    
                    # Cache result
                    self.sentiment_cache[token] = {
                        'sentiment': sentiment,
                        'timestamp': time.time()
                    }
                    
                    return sentiment
                    
            return 0.0

        except Exception as e:
            logger.error(f"Twitter sentiment error: {e}")
            return 0.0

    def _analyze_twitter_sentiment(self, data: Dict) -> float:
        """Analyze sentiment from Twitter data"""
        try:
            if not data.get('data'):
                return 0.0
                
            tweets = data['data']
            total_sentiment = 0.0
            total_weight = 0.0
            
            for tweet in tweets:
                # Calculate tweet weight based on engagement
                metrics = tweet.get('public_metrics', {})
                weight = (
                    metrics.get('retweet_count', 0) * 0.3 +
                    metrics.get('reply_count', 0) * 0.2 +
                    metrics.get('like_count', 0) * 0.3 +
                    metrics.get('quote_count', 0) * 0.2
                )
                
                # Get sentiment from Grok API
                sentiment = self._get_grok_sentiment(tweet['text'])
                
                total_sentiment += sentiment * weight
                total_weight += weight
            
            return total_sentiment / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Twitter sentiment analysis error: {e}")
            return 0.0

    async def _get_grok_sentiment(self, text: str) -> float:
        """Get sentiment from Grok API"""
        try:
            async with self.session.post(
                'https://api.grok.ai/v1/sentiment',
                json={'text': text},
                headers={'Authorization': f'Bearer {settings.GROK_API_KEY}'}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get('sentiment', 0))
                return 0.0
        except Exception as e:
            logger.error(f"Grok sentiment error: {e}")
            return 0.0

    async def get_trading_decision(
        self, token: str, sentiment: float, whale_activity: Dict, market_data: Dict
    ) -> Tuple[str, float, float]:
        """Get trading decision from Grok API"""
        try:
            # Check cache
            cache_entry = self.decision_cache.get(token)
            if cache_entry and time.time() - cache_entry['timestamp'] < self.decision_cache_ttl:
                return (
                    cache_entry['action'],
                    cache_entry['confidence'],
                    cache_entry['trade_size']
                )

            # Get Twitter sentiment
            twitter_sentiment = await self.get_twitter_sentiment(token)
            
            # Prepare input data
            input_data = {
                'token': token,
                'sentiment': sentiment,
                'twitter_sentiment': twitter_sentiment,
                'whale_activity': whale_activity,
                'market_data': market_data,
                'feature_weights': self.feature_weights
            }

            # Get decision from Grok API
            async with self.session.post(
                'https://api.grok.ai/v1/trading/decision',
                json=input_data,
                headers={'Authorization': f'Bearer {settings.GROK_API_KEY}'}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache decision
                    self.decision_cache[token] = {
                        'action': data['action'],
                        'confidence': data['confidence'],
                        'trade_size': data['trade_size'],
                        'timestamp': time.time()
                    }

                    return (
                        data['action'],
                        data['confidence'],
                        data['trade_size']
                    )

                return 'hold', 0.0, 0.0

        except Exception as e:
            logger.error(f"Get trading decision error: {e}")
            return 'hold', 0.0, 0.0

    def should_enter_trade(
        self, action: str, confidence: float, sentiment: float,
        whale_activity: Dict, market_data: Dict
    ) -> bool:
        """Check if we should enter a trade"""
        try:
            if action != 'buy' or confidence < self.confidence_threshold:
                return False

            # Calculate weighted score
            score = (
                self.feature_weights['sentiment'] * sentiment +
                self.feature_weights['whale_activity'] * (
                    whale_activity.get('buy_count', 0) / settings.MIN_WHALE_BUYS
                ) +
                self.feature_weights['price_momentum'] * (
                    market_data.get('price_change_1m', 0) / 0.05
                )
            )

            # Check basic criteria
            if not self._check_basic_criteria(market_data):
                return False

            # Check for high-confidence signals
            if self._check_high_confidence_signals(sentiment, whale_activity, market_data):
                return True

            # Check weighted score
            return score > 0.7

        except Exception as e:
            logger.error(f"Should enter trade error: {e}")
            return False

    def should_exit_trade(
        self, token: str, current_price: float, entry_price: float,
        sentiment: float, whale_activity: Dict
    ) -> bool:
        """Check if we should exit a trade"""
        try:
            # Calculate profit/loss
            pnl = (current_price - entry_price) / entry_price

            # Check kill switch
            if pnl < -settings.KILL_SWITCH_THRESHOLD:
                return True

            # Check profit targets
            for target, portion in settings.TIERED_EXITS.items():
                if pnl >= target:
                    return True

            # Check sentiment drop
            if sentiment < settings.MIN_SENTIMENT_SCORE * 0.7:
                return True

            # Check whale sells
            if whale_activity.get('sell_count', 0) >= 2:
                return True

            return False

        except Exception as e:
            logger.error(f"Should exit trade error: {e}")
            return False

    def _check_basic_criteria(self, market_data: Dict) -> bool:
        """Check basic trading criteria"""
        try:
            return (
                market_data.get('liquidity', 0) >= settings.MIN_LIQUIDITY and
                market_data.get('holders', 0) >= settings.MIN_HOLDERS and
                market_data.get('age', 0) <= settings.MAX_TOKEN_AGE
            )
        except Exception as e:
            logger.error(f"Check basic criteria error: {e}")
            return False

    def _check_high_confidence_signals(
        self, sentiment: float, whale_activity: Dict, market_data: Dict
    ) -> bool:
        """Check for high-confidence trading signals"""
        try:
            # Check sentiment spike
            sentiment_spike = (
                sentiment > settings.MIN_SENTIMENT_SCORE and
                sentiment - market_data.get('prev_sentiment', 0) > settings.SENTIMENT_SPIKE_THRESHOLD
            )

            # Check whale activity
            whale_signal = (
                whale_activity.get('buy_count', 0) >= settings.MIN_WHALE_BUYS and
                whale_activity.get('total_buy_volume', 0) >= settings.MIN_WHALE_BUY_SIZE
            )

            # Check price momentum
            price_momentum = (
                market_data.get('price_change_1m', 0) > 0.05 and
                market_data.get('volume_change_1m', 0) > 0.1
            )

            return sentiment_spike or (whale_signal and price_momentum)

        except Exception as e:
            logger.error(f"Check high confidence signals error: {e}")
            return False

    def update_learning(self, trade_result: Dict):
        """Update learning parameters based on trade result"""
        try:
            # Update feature weights
            if trade_result['profit'] > 0:
                # Increase weights for successful features
                for feature, value in trade_result['features'].items():
                    if value > 0.5:
                        self.feature_weights[feature] *= 1.1
            else:
                # Decrease weights for unsuccessful features
                for feature, value in trade_result['features'].items():
                    if value > 0.5:
                        self.feature_weights[feature] *= 0.9

            # Normalize weights
            total_weight = sum(self.feature_weights.values())
            if total_weight > 0:
                for feature in self.feature_weights:
                    self.feature_weights[feature] /= total_weight

            # Update confidence threshold
            if trade_result['profit'] > 0:
                self.confidence_threshold *= 0.95  # Lower threshold for successful trades
            else:
                self.confidence_threshold *= 1.05  # Raise threshold for failed trades

            # Save learning data
            self._save_learning_data()

        except Exception as e:
            logger.error(f"Update learning error: {e}") 