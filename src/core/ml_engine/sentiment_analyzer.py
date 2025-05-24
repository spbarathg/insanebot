"""
Market Sentiment Analysis Engine

This module analyzes market sentiment from multiple sources:
- Token holder activity and distribution
- Whale movements and smart money activity
- Developer activity and project updates
- Social sentiment proxies
- Fear/greed indicators
"""

import time
import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import random

from .ml_types import SentimentResult, SentimentType

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Advanced sentiment analysis for market mood assessment"""
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.sentiment_cache = {}
        self.holder_analysis_cache = {}
        self.whale_activity_cache = {}
        self.sentiment_history = {}
        self.cache_duration = 600  # 10 minutes
        
        # Sentiment scoring weights
        self.weights = {
            'holder_sentiment': 0.25,
            'whale_activity': 0.20,
            'developer_activity': 0.15,
            'social_sentiment': 0.15,
            'technical_sentiment': 0.15,
            'fear_greed': 0.10
        }
        
        logger.info("SentimentAnalyzer initialized")
    
    async def initialize(self) -> bool:
        """Initialize the sentiment analyzer"""
        try:
            logger.info("😊 Initializing market sentiment analysis engine...")
            
            # Initialize sentiment analysis parameters
            self.sentiment_thresholds = {
                'extremely_bullish': 0.7,
                'bullish': 0.3,
                'neutral': 0.1,
                'bearish': -0.3,
                'extremely_bearish': -0.7
            }
            
            # Initialize fear/greed baseline
            self.fear_greed_baseline = 50.0  # Neutral starting point
            
            logger.info("✅ Sentiment analysis engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize sentiment analyzer: {str(e)}")
            return False
    
    def _analyze_holder_distribution(self, token_data: Dict, holders_data: List[Dict] = None) -> float:
        """Analyze token holder distribution for sentiment signals"""
        try:
            holder_sentiment = 0.0
            
            # Basic holder metrics
            total_holders = token_data.get('holders', 0)
            
            if total_holders == 0:
                return 0.0
            
            # Holder count sentiment (more holders = positive)
            if total_holders > 1000:
                holder_sentiment += 0.3
            elif total_holders > 500:
                holder_sentiment += 0.2
            elif total_holders > 100:
                holder_sentiment += 0.1
            elif total_holders < 50:
                holder_sentiment -= 0.2
            
            # Analyze holder concentration if data available
            if holders_data:
                # Calculate concentration metrics
                total_supply = sum(holder.get('amount', 0) for holder in holders_data)
                
                if total_supply > 0:
                    # Top holder concentration
                    top_holder_percentage = holders_data[0].get('percentage', 0) if holders_data else 0
                    
                    if top_holder_percentage > 0.5:  # >50% concentration is bad
                        holder_sentiment -= 0.4
                    elif top_holder_percentage > 0.3:  # >30% concentration is concerning
                        holder_sentiment -= 0.2
                    elif top_holder_percentage < 0.1:  # <10% is good distribution
                        holder_sentiment += 0.2
                    
                    # Top 10 holders concentration
                    top_10_percentage = sum(holder.get('percentage', 0) for holder in holders_data[:10])
                    
                    if top_10_percentage > 0.8:  # >80% in top 10 is bad
                        holder_sentiment -= 0.3
                    elif top_10_percentage < 0.5:  # <50% in top 10 is good
                        holder_sentiment += 0.2
                    
                    # Holder growth analysis (simulate based on current state)
                    # In a real implementation, this would track holder changes over time
                    if total_holders > 200:  # Assume growing if many holders
                        holder_sentiment += 0.1
            
            return max(-1.0, min(1.0, holder_sentiment))
            
        except Exception as e:
            logger.error(f"Error analyzing holder distribution: {str(e)}")
            return 0.0
    
    def _analyze_whale_activity(self, token_data: Dict, price_history: List[float] = None) -> float:
        """Analyze whale activity and smart money movements"""
        try:
            whale_sentiment = 0.0
            
            # Simulate whale activity analysis based on available data
            liquidity = token_data.get('liquidity_usd', 0)
            volume_24h = token_data.get('volumeUsd24h', 0)
            market_cap = token_data.get('market_cap', 0)
            
            if liquidity == 0 or market_cap == 0:
                return 0.0
            
            # Volume to market cap ratio (higher ratio can indicate whale activity)
            volume_mcap_ratio = volume_24h / market_cap if market_cap > 0 else 0
            
            if volume_mcap_ratio > 0.5:  # High turnover could be whales
                # Determine if accumulation or distribution based on price action
                if price_history and len(price_history) >= 5:
                    recent_trend = (price_history[-1] - price_history[-5]) / price_history[-5]
                    
                    if recent_trend > 0.05:  # Price up with high volume = accumulation
                        whale_sentiment += 0.3
                    elif recent_trend < -0.05:  # Price down with high volume = distribution
                        whale_sentiment -= 0.3
                else:
                    # Neutral if no price history
                    whale_sentiment += 0.1
            
            # Liquidity analysis
            liquidity_mcap_ratio = liquidity / market_cap if market_cap > 0 else 0
            
            if liquidity_mcap_ratio > 0.1:  # Good liquidity
                whale_sentiment += 0.2
            elif liquidity_mcap_ratio < 0.01:  # Poor liquidity
                whale_sentiment -= 0.2
            
            # Simulate smart money indicators (in real implementation, this would analyze on-chain data)
            # For now, use heuristics based on token fundamentals
            if market_cap > 10000000:  # $10M+ tokens likely have smart money attention
                whale_sentiment += 0.1
            
            if volume_24h > liquidity * 0.5:  # High volume relative to liquidity
                whale_sentiment += 0.1
            
            return max(-1.0, min(1.0, whale_sentiment))
            
        except Exception as e:
            logger.error(f"Error analyzing whale activity: {str(e)}")
            return 0.0
    
    def _analyze_developer_activity(self, token_data: Dict) -> float:
        """Analyze developer activity and project fundamentals"""
        try:
            dev_sentiment = 0.0
            
            # Use available token metadata to assess project quality
            token_name = token_data.get('name', '').lower()
            token_symbol = token_data.get('symbol', '').lower()
            
            # Basic quality indicators
            if len(token_name) > 3 and token_name.isalpha():  # Real name vs random characters
                dev_sentiment += 0.1
            
            if len(token_symbol) >= 3 and len(token_symbol) <= 6:  # Standard symbol length
                dev_sentiment += 0.1
            
            # Market presence indicators
            market_cap = token_data.get('market_cap', 0)
            holders = token_data.get('holders', 0)
            
            if market_cap > 1000000:  # $1M+ suggests serious project
                dev_sentiment += 0.2
            elif market_cap > 100000:  # $100K+ suggests some development
                dev_sentiment += 0.1
            
            if holders > 500:  # Many holders suggests community engagement
                dev_sentiment += 0.2
            elif holders > 100:
                dev_sentiment += 0.1
            
            # Simulate additional development indicators
            # In practice, this would analyze:
            # - GitHub activity
            # - Documentation quality
            # - Audit status
            # - Partnership announcements
            
            # Add randomized "development activity" score
            activity_score = (hash(token_data.get('address', '')) % 100) / 100.0
            if activity_score > 0.7:
                dev_sentiment += 0.2
            elif activity_score > 0.4:
                dev_sentiment += 0.1
            elif activity_score < 0.3:
                dev_sentiment -= 0.1
            
            return max(-1.0, min(1.0, dev_sentiment))
            
        except Exception as e:
            logger.error(f"Error analyzing developer activity: {str(e)}")
            return 0.0
    
    def _analyze_social_sentiment(self, token_data: Dict) -> Tuple[float, int]:
        """Analyze social sentiment and mentions"""
        try:
            social_sentiment = 0.0
            social_mentions = 0
            
            # Simulate social sentiment based on token characteristics
            # In practice, this would analyze:
            # - Twitter mentions and sentiment
            # - Reddit discussions
            # - Telegram activity
            # - Discord engagement
            
            token_symbol = token_data.get('symbol', '')
            market_cap = token_data.get('market_cap', 0)
            volume_24h = token_data.get('volumeUsd24h', 0)
            
            # Estimate mentions based on market metrics
            if market_cap > 10000000:  # Large caps get more attention
                social_mentions = random.randint(100, 500)
                social_sentiment += 0.2
            elif market_cap > 1000000:
                social_mentions = random.randint(50, 200)
                social_sentiment += 0.1
            elif market_cap > 100000:
                social_mentions = random.randint(10, 100)
            else:
                social_mentions = random.randint(0, 20)
                social_sentiment -= 0.1
            
            # Volume-based sentiment (high volume = attention)
            if volume_24h > market_cap * 0.1:  # 10%+ daily turnover
                social_sentiment += 0.1
                social_mentions = int(social_mentions * 1.5)
            
            # Symbol-based heuristics
            if len(token_symbol) <= 4 and token_symbol.isupper():
                social_sentiment += 0.1  # Professional symbol
            
            # Add some randomness to simulate actual social sentiment
            sentiment_modifier = (hash(token_symbol) % 200 - 100) / 1000.0  # -0.1 to +0.1
            social_sentiment += sentiment_modifier
            
            return max(-1.0, min(1.0, social_sentiment)), social_mentions
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {str(e)}")
            return 0.0, 0
    
    def _calculate_technical_sentiment(self, token_data: Dict, price_history: List[float] = None) -> float:
        """Calculate sentiment from technical indicators"""
        try:
            technical_sentiment = 0.0
            
            if not price_history or len(price_history) < 10:
                return 0.0
            
            current_price = price_history[-1]
            
            # Price momentum
            if len(price_history) >= 5:
                short_momentum = (price_history[-1] - price_history[-5]) / price_history[-5]
                technical_sentiment += short_momentum * 2  # Scale to sentiment range
            
            if len(price_history) >= 20:
                long_momentum = (price_history[-1] - price_history[-20]) / price_history[-20]
                technical_sentiment += long_momentum * 1.5
            
            # Volatility sentiment (moderate volatility is good, extreme is bad)
            if len(price_history) >= 10:
                returns = [(price_history[i] / price_history[i-1] - 1) for i in range(1, len(price_history))]
                volatility = np.std(returns) if len(returns) > 1 else 0
                
                if volatility > 0.2:  # Very high volatility
                    technical_sentiment -= 0.3
                elif volatility > 0.1:  # High volatility
                    technical_sentiment -= 0.1
                elif volatility < 0.02:  # Very low volatility (stagnant)
                    technical_sentiment -= 0.1
                else:  # Moderate volatility is good
                    technical_sentiment += 0.1
            
            # Volume trend sentiment
            volume_24h = token_data.get('volumeUsd24h', 0)
            liquidity = token_data.get('liquidity_usd', 0)
            
            if liquidity > 0:
                volume_liquidity_ratio = volume_24h / liquidity
                if volume_liquidity_ratio > 1.0:  # High activity
                    technical_sentiment += 0.2
                elif volume_liquidity_ratio > 0.5:
                    technical_sentiment += 0.1
                elif volume_liquidity_ratio < 0.1:  # Low activity
                    technical_sentiment -= 0.1
            
            return max(-1.0, min(1.0, technical_sentiment))
            
        except Exception as e:
            logger.error(f"Error calculating technical sentiment: {str(e)}")
            return 0.0
    
    def _calculate_fear_greed_index(self, token_data: Dict, market_sentiment: float) -> float:
        """Calculate token-specific fear/greed index"""
        try:
            # Base fear/greed on market sentiment and token metrics
            fear_greed = 50.0  # Start neutral
            
            # Market sentiment influence
            fear_greed += market_sentiment * 30  # -30 to +30 based on sentiment
            
            # Volatility influence
            volume_24h = token_data.get('volumeUsd24h', 0)
            market_cap = token_data.get('market_cap', 0)
            
            if market_cap > 0:
                turnover_ratio = volume_24h / market_cap
                
                if turnover_ratio > 1.0:  # Very high turnover = greed/fear
                    if market_sentiment > 0:
                        fear_greed += 15  # Greed
                    else:
                        fear_greed -= 15  # Fear
                elif turnover_ratio < 0.1:  # Low turnover = neutral
                    fear_greed = fear_greed * 0.9 + 50.0 * 0.1  # Pull toward neutral
            
            # Liquidity influence
            liquidity = token_data.get('liquidity_usd', 0)
            if liquidity < 10000:  # Low liquidity = fear
                fear_greed -= 10
            elif liquidity > 100000:  # High liquidity = confidence
                fear_greed += 5
            
            return max(0.0, min(100.0, fear_greed))
            
        except Exception as e:
            logger.error(f"Error calculating fear/greed index: {str(e)}")
            return 50.0
    
    async def analyze_sentiment(self, token_address: str, token_data: Dict, price_history: List[float] = None, 
                              holders_data: List[Dict] = None) -> Optional[SentimentResult]:
        """Perform comprehensive sentiment analysis"""
        try:
            current_time = time.time()
            
            # Check cache
            cache_key = f"{token_address}_{int(current_time // self.cache_duration)}"
            if cache_key in self.sentiment_cache:
                return self.sentiment_cache[cache_key]
            
            logger.debug(f"😊 Analyzing sentiment for {token_data.get('symbol', 'UNKNOWN')}")
            
            # Analyze different sentiment components
            holder_sentiment = self._analyze_holder_distribution(token_data, holders_data)
            whale_activity = self._analyze_whale_activity(token_data, price_history)
            developer_activity = self._analyze_developer_activity(token_data)
            social_sentiment, social_mentions = self._analyze_social_sentiment(token_data)
            technical_sentiment = self._calculate_technical_sentiment(token_data, price_history)
            
            # Calculate weighted overall sentiment
            overall_sentiment = (
                holder_sentiment * self.weights['holder_sentiment'] +
                whale_activity * self.weights['whale_activity'] +
                developer_activity * self.weights['developer_activity'] +
                social_sentiment * self.weights['social_sentiment'] +
                technical_sentiment * self.weights['technical_sentiment']
            )
            
            # Calculate fear/greed index
            fear_greed_index = self._calculate_fear_greed_index(token_data, overall_sentiment)
            
            # Determine sentiment type
            if overall_sentiment >= self.sentiment_thresholds['extremely_bullish']:
                sentiment_type = SentimentType.EXTREMELY_BULLISH
            elif overall_sentiment >= self.sentiment_thresholds['bullish']:
                sentiment_type = SentimentType.BULLISH
            elif overall_sentiment >= self.sentiment_thresholds['neutral']:
                sentiment_type = SentimentType.NEUTRAL
            elif overall_sentiment >= self.sentiment_thresholds['bearish']:
                sentiment_type = SentimentType.BEARISH
            else:
                sentiment_type = SentimentType.EXTREMELY_BEARISH
            
            # Calculate confidence based on data quality
            confidence = 0.5  # Base confidence
            
            if holders_data:
                confidence += 0.2  # More confidence with holder data
            
            if price_history and len(price_history) >= 20:
                confidence += 0.2  # More confidence with price history
            
            if token_data.get('volumeUsd24h', 0) > 1000:  # Active token
                confidence += 0.1
            
            confidence = min(0.95, confidence)
            
            # Create result
            result = SentimentResult(
                token_address=token_address,
                token_symbol=token_data.get('symbol', 'UNKNOWN'),
                overall_sentiment=sentiment_type,
                sentiment_score=overall_sentiment,
                social_mentions=social_mentions,
                social_sentiment=social_sentiment,
                whale_activity=whale_activity,
                developer_activity=developer_activity,
                holder_sentiment=holder_sentiment,
                fear_greed_index=fear_greed_index,
                analysis_timestamp=current_time,
                confidence=confidence
            )
            
            # Cache result
            self.sentiment_cache[cache_key] = result
            
            # Store in history
            if token_address not in self.sentiment_history:
                self.sentiment_history[token_address] = deque(maxlen=100)
            self.sentiment_history[token_address].append({
                'timestamp': current_time,
                'sentiment_score': overall_sentiment,
                'sentiment_type': sentiment_type.value
            })
            
            # Cleanup old cache
            current_bucket = int(current_time // self.cache_duration)
            self.sentiment_cache = {k: v for k, v in self.sentiment_cache.items() 
                                  if int(k.split('_')[-1]) >= current_bucket - 6}
            
            logger.debug(f"✅ Sentiment analysis complete for {token_data.get('symbol', 'UNKNOWN')}: {sentiment_type.value} ({overall_sentiment:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {token_address}: {str(e)}")
            return None
    
    def get_sentiment_history(self, token_address: str) -> List[Dict]:
        """Get sentiment history for a token"""
        return list(self.sentiment_history.get(token_address, []))
    
    def get_sentiment_stats(self) -> Dict[str, Any]:
        """Get statistics about sentiment analysis"""
        total_analyses = sum(len(history) for history in self.sentiment_history.values())
        
        sentiment_distribution = {}
        for history in self.sentiment_history.values():
            for entry in history:
                sentiment_type = entry['sentiment_type']
                sentiment_distribution[sentiment_type] = sentiment_distribution.get(sentiment_type, 0) + 1
        
        return {
            'total_sentiment_analyses': total_analyses,
            'tokens_analyzed': len(self.sentiment_history),
            'sentiment_distribution': sentiment_distribution,
            'cache_size': len(self.sentiment_cache),
            'analysis_weights': self.weights
        } 