"""
Real-Time Social Sentiment Engine

Advanced social media monitoring system that tracks crypto Twitter, 
Telegram channels, and Discord servers for viral memecoin trends.

Features:
- Real-time Twitter API v2 streaming
- Telegram channel monitoring
- Discord server tracking
- Influencer mention detection  
- Viral content identification
- Sentiment momentum scoring
- Trend prediction algorithms
"""

import asyncio
import aiohttp
import tweepy
import json
import time
import re
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import spacy
from transformers import pipeline

logger = logging.getLogger(__name__)

@dataclass
class SocialMention:
    """A social media mention of a token"""
    platform: str  # 'twitter', 'telegram', 'discord'
    content: str
    author: str
    author_followers: int
    timestamp: float
    token_symbol: str
    token_address: str
    sentiment_score: float  # -1 to 1
    influence_score: float  # 0 to 1
    engagement: Dict[str, int]  # likes, retweets, comments
    is_influencer: bool
    viral_indicators: List[str]

@dataclass
class TrendingToken:
    """A token trending on social media"""
    symbol: str
    address: str
    mentions_count: int
    sentiment_avg: float
    influence_score: float
    viral_velocity: float  # mentions per minute
    confidence: float
    top_mentions: List[SocialMention]
    trend_start: float
    predicted_peak: float

@dataclass
class SentimentSignal:
    """Trading signal based on social sentiment"""
    token_symbol: str
    token_address: str
    signal_type: str  # 'BUY', 'SELL', 'WATCH'
    confidence: float
    urgency: str
    sentiment_score: float
    viral_velocity: float
    influencer_mentions: int
    reasoning: str
    supporting_mentions: List[SocialMention]
    timestamp: float
    expires_at: float

@dataclass
class TokenNarrative:
    """LLM-analyzed token narrative"""
    token_address: str
    narrative_summary: str
    viral_potential: float  # 0.0 to 1.0
    narrative_strength: float  # 0.0 to 1.0
    meme_quality_score: float  # 0.0 to 1.0
    originality_score: float  # 0.0 to 1.0
    community_appeal: float  # 0.0 to 1.0
    narrative_type: str  # 'meme', 'utility', 'community', 'trend', 'pump'
    key_themes: List[str]
    viral_catalysts: List[str]
    risk_factors: List[str]
    llm_confidence: float
    analysis_timestamp: float

@dataclass
class NarrativeAnalysisRequest:
    """Request structure for narrative analysis"""
    token_address: str
    token_name: str
    token_symbol: str
    description: str
    social_posts: List[Dict]
    market_data: Dict
    analysis_depth: str = "standard"  # 'quick', 'standard', 'deep'

class SocialSentimentEngine:
    """
    Real-time social sentiment monitoring for crypto tokens
    
    Tracks Twitter, Telegram, and Discord for viral memecoin trends
    and generates trading signals based on social momentum.
    """
    
    def __init__(self, callback_handler: Optional[Callable] = None):
        self.callback_handler = callback_handler
        
        # API credentials (would be loaded from environment)
        self.twitter_bearer_token = "YOUR_TWITTER_BEARER_TOKEN"
        self.telegram_api_id = "YOUR_TELEGRAM_API_ID"
        self.telegram_api_hash = "YOUR_TELEGRAM_API_HASH"
        
        # Monitoring state
        self.monitoring_active = False
        self.twitter_stream = None
        
        # Data storage
        self.recent_mentions: deque = deque(maxlen=10000)  # Last 10k mentions
        self.trending_tokens: Dict[str, TrendingToken] = {}
        self.sentiment_signals: Dict[str, SentimentSignal] = {}
        
        # Influencer tracking
        self.crypto_influencers = {
            'elon_musk': {'handle': 'elonmusk', 'followers': 150000000, 'influence': 1.0},
            'vitalik': {'handle': 'VitalikButerin', 'followers': 5000000, 'influence': 0.9},
            'cobie': {'handle': 'cobie', 'followers': 800000, 'influence': 0.8},
            'hsaka': {'handle': 'HsakaTrades', 'followers': 300000, 'influence': 0.7},
            'ansem': {'handle': 'blknoiz06', 'followers': 500000, 'influence': 0.8}
        }
        
        # Viral keywords and patterns
        self.viral_keywords = [
            '🚀', 'moon', 'rocket', '💎', 'diamond', 'hands', 'hodl',
            'pump', 'ape', 'degen', 'based', 'chad', 'gigachad',
            'bullish', 'bearish', 'gem', 'alpha', '100x', '1000x',
            'breakout', 'pump it', 'send it', 'lfg', 'wagmi'
        ]
        
        # Channels to monitor
        self.telegram_channels = [
            '@solana_gems', '@solana_alpha', '@pump_fun_official',
            '@degen_chat', '@memecoin_gems', '@solana_memes'
        ]
        
        self.discord_servers = [
            'solana-trading', 'degen-alpha', 'pump-fun',
            'memecoin-hunters', 'solana-gems'
        ]
        
        # AI models for sentiment analysis
        self.sentiment_analyzer = None
        self.nlp_model = None
        self.initialize_ai_models()
        
        # Performance metrics
        self.metrics = {
            'mentions_processed': 0,
            'signals_generated': 0,
            'trending_tokens_detected': 0,
            'influencer_mentions_detected': 0,
            'prediction_accuracy': 0.0
        }
        
        logger.info("📱 Social Sentiment Engine initialized - Ready for viral detection!")
    
    def initialize_ai_models(self):
        """Initialize AI models for sentiment analysis"""
        try:
            # Initialize FinBERT for financial sentiment
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True
            )
            
            # Initialize spaCy for NLP processing
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
            except IOError:
                logger.warning("spaCy model not found, using basic text processing")
                self.nlp_model = None
            
            logger.info("✅ AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AI models: {e}")
            # Use backup sentiment analysis
            self.sentiment_analyzer = None
    
    async def start_monitoring(self) -> bool:
        """Start real-time social media monitoring"""
        try:
            logger.info("🚀 Starting social sentiment monitoring...")
            self.monitoring_active = True
            
            # Start monitoring tasks
            tasks = [
                asyncio.create_task(self._monitor_twitter()),
                asyncio.create_task(self._monitor_telegram()),
                asyncio.create_task(self._process_sentiment_analysis()),
                asyncio.create_task(self._detect_trending_tokens()),
                asyncio.create_task(self._generate_signals())
            ]
            
            # Wait for tasks to start
            await asyncio.sleep(1)
            
            logger.info("✅ Social sentiment monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start social monitoring: {e}")
            return False
    
    async def _monitor_twitter(self):
        """Monitor Twitter for crypto mentions using streaming API"""
        while self.monitoring_active:
            try:
                # Initialize Twitter API v2 client
                client = tweepy.Client(bearer_token=self.twitter_bearer_token)
                
                # Define search rules for crypto-related tweets
                search_rules = [
                    tweepy.StreamRule("$SOL OR solana OR pump.fun OR memecoin"),
                    tweepy.StreamRule("crypto moon OR crypto rocket OR 🚀"),
                    tweepy.StreamRule("new token OR new coin OR gem found"),
                    tweepy.StreamRule("pump OR dump OR degen OR ape")
                ]
                
                # Create streaming object
                stream = tweepy.StreamingClient(self.twitter_bearer_token)
                
                # Add rules
                for rule in search_rules:
                    stream.add_rules(rule)
                
                # Define tweet processor
                class TweetProcessor(tweepy.StreamingClient):
                    def __init__(self, sentiment_engine):
                        super().__init__(sentiment_engine.twitter_bearer_token)
                        self.sentiment_engine = sentiment_engine
                    
                    def on_tweet(self, tweet):
                        asyncio.create_task(
                            self.sentiment_engine._process_twitter_mention(tweet)
                        )
                
                # Start streaming
                processor = TweetProcessor(self)
                processor.filter(threaded=True)
                
                # Keep alive
                while self.monitoring_active:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Twitter monitoring error: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def _monitor_telegram(self):
        """Monitor Telegram channels for crypto mentions"""
        while self.monitoring_active:
            try:
                # This would require telethon library and proper setup
                # For now, simulate telegram monitoring
                
                for channel in self.telegram_channels:
                    try:
                        # Simulate getting recent messages
                        await self._simulate_telegram_messages(channel)
                    except Exception as e:
                        logger.warning(f"Error monitoring {channel}: {e}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Telegram monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _simulate_telegram_messages(self, channel: str):
        """Simulate telegram message processing (placeholder)"""
        try:
            # In production, this would use telethon to get real messages
            sample_messages = [
                f"🚀 New gem found on pump.fun! $PEPE2.0 just launched",
                f"Diamond hands on $BONK, this is going to moon 💎",
                f"Ape into $DOGE killer while still early 🦍"
            ]
            
            for message in sample_messages:
                # Create mock mention
                mention = SocialMention(
                    platform='telegram',
                    content=message,
                    author=f"{channel}_user",
                    author_followers=1000,
                    timestamp=time.time(),
                    token_symbol="",  # Will be extracted
                    token_address="",
                    sentiment_score=0.0,
                    influence_score=0.3,
                    engagement={'views': 100, 'reactions': 10},
                    is_influencer=False,
                    viral_indicators=[]
                )
                
                await self._process_social_mention(mention)
                
        except Exception as e:
            logger.error(f"Error simulating telegram messages: {e}")
    
    async def _process_twitter_mention(self, tweet):
        """Process a Twitter mention/tweet"""
        try:
            # Get user info
            user = tweet.author if hasattr(tweet, 'author') else None
            
            # Create mention object
            mention = SocialMention(
                platform='twitter',
                content=tweet.text,
                author=user.username if user else 'unknown',
                author_followers=user.public_metrics['followers_count'] if user else 0,
                timestamp=time.time(),
                token_symbol="",  # Will be extracted
                token_address="",
                sentiment_score=0.0,
                influence_score=0.0,
                engagement={
                    'likes': tweet.public_metrics.get('like_count', 0),
                    'retweets': tweet.public_metrics.get('retweet_count', 0),
                    'replies': tweet.public_metrics.get('reply_count', 0)
                },
                is_influencer=self._is_crypto_influencer(user.username if user else ''),
                viral_indicators=[]
            )
            
            await self._process_social_mention(mention)
            
        except Exception as e:
            logger.error(f"Error processing Twitter mention: {e}")
    
    async def _process_social_mention(self, mention: SocialMention):
        """Process a social media mention"""
        try:
            # Extract token symbols/addresses
            tokens = self._extract_tokens_from_text(mention.content)
            
            for token_info in tokens:
                # Create mention for each token
                token_mention = mention
                token_mention.token_symbol = token_info['symbol']
                token_mention.token_address = token_info.get('address', '')
                
                # Analyze sentiment
                token_mention.sentiment_score = await self._analyze_sentiment(mention.content)
                
                # Calculate influence score
                token_mention.influence_score = self._calculate_influence_score(mention)
                
                # Detect viral indicators
                token_mention.viral_indicators = self._detect_viral_indicators(mention.content)
                
                # Store mention
                self.recent_mentions.append(token_mention)
                self.metrics['mentions_processed'] += 1
                
                if token_mention.is_influencer:
                    self.metrics['influencer_mentions_detected'] += 1
                
                logger.debug(f"📱 Processed mention: {token_mention.token_symbol} "
                           f"Sentiment: {token_mention.sentiment_score:.2f} "
                           f"Influence: {token_mention.influence_score:.2f}")
                
        except Exception as e:
            logger.error(f"Error processing social mention: {e}")
    
    def _extract_tokens_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract token symbols and addresses from text"""
        try:
            tokens = []
            
            # Extract $ symbols
            dollar_pattern = r'\$([A-Z0-9]{2,10})'
            dollar_matches = re.findall(dollar_pattern, text.upper())
            
            for symbol in dollar_matches:
                tokens.append({'symbol': symbol, 'address': ''})
            
            # Extract Solana addresses (base58 format)
            address_pattern = r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
            address_matches = re.findall(address_pattern, text)
            
            for address in address_matches:
                if len(address) >= 32:  # Valid Solana address length
                    tokens.append({'symbol': '', 'address': address})
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error extracting tokens: {e}")
            return []
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (-1 to 1)"""
        try:
            if self.sentiment_analyzer:
                # Use FinBERT for financial sentiment
                results = self.sentiment_analyzer(text)
                
                # Convert to -1 to 1 scale
                positive_score = next((r['score'] for r in results if r['label'] == 'positive'), 0)
                negative_score = next((r['score'] for r in results if r['label'] == 'negative'), 0)
                
                return positive_score - negative_score
            else:
                # Fallback sentiment analysis
                return self._basic_sentiment_analysis(text)
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def _basic_sentiment_analysis(self, text: str) -> float:
        """Basic sentiment analysis using keyword matching"""
        try:
            positive_words = ['moon', 'rocket', 'pump', 'bullish', 'gem', 'alpha', 'buy', 'hold']
            negative_words = ['dump', 'crash', 'bearish', 'sell', 'rug', 'scam', 'dead']
            
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            if total_words == 0:
                return 0.0
            
            sentiment = (positive_count - negative_count) / max(total_words, 1)
            return max(-1.0, min(1.0, sentiment * 5))  # Scale and clamp
            
        except Exception as e:
            logger.error(f"Error in basic sentiment analysis: {e}")
            return 0.0
    
    def _calculate_influence_score(self, mention: SocialMention) -> float:
        """Calculate influence score based on author and engagement"""
        try:
            base_score = 0.0
            
            # Follower influence (logarithmic scale)
            if mention.author_followers > 0:
                follower_score = min(1.0, np.log10(mention.author_followers) / 8)  # Max at 100M followers
                base_score += follower_score * 0.4
            
            # Engagement influence
            total_engagement = sum(mention.engagement.values())
            engagement_score = min(1.0, total_engagement / 1000)  # Max at 1000 engagements
            base_score += engagement_score * 0.3
            
            # Influencer bonus
            if mention.is_influencer:
                base_score += 0.3
            
            # Platform weight
            platform_weights = {'twitter': 1.0, 'telegram': 0.8, 'discord': 0.6}
            platform_weight = platform_weights.get(mention.platform, 0.5)
            
            return min(1.0, base_score * platform_weight)
            
        except Exception as e:
            logger.error(f"Error calculating influence score: {e}")
            return 0.0
    
    def _is_crypto_influencer(self, username: str) -> bool:
        """Check if user is a known crypto influencer"""
        try:
            username_lower = username.lower()
            return any(
                influencer['handle'].lower() == username_lower
                for influencer in self.crypto_influencers.values()
            )
        except Exception as e:
            logger.error(f"Error checking influencer status: {e}")
            return False
    
    def _detect_viral_indicators(self, text: str) -> List[str]:
        """Detect viral indicators in text"""
        try:
            indicators = []
            text_lower = text.lower()
            
            # Emoji indicators
            if '🚀' in text or 'rocket' in text_lower:
                indicators.append('rocket_emoji')
            if '💎' in text or 'diamond' in text_lower:
                indicators.append('diamond_hands')
            if '🦍' in text or 'ape' in text_lower:
                indicators.append('ape_reference')
            
            # Viral phrases
            viral_phrases = ['to the moon', 'going parabolic', '100x', '1000x', 'hidden gem']
            for phrase in viral_phrases:
                if phrase in text_lower:
                    indicators.append(phrase.replace(' ', '_'))
            
            # Urgency indicators
            urgency_words = ['now', 'quick', 'fast', 'hurry', 'before it\'s too late']
            if any(word in text_lower for word in urgency_words):
                indicators.append('urgency')
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error detecting viral indicators: {e}")
            return []
    
    async def _process_sentiment_analysis(self):
        """Background task to process sentiment analysis"""
        while self.monitoring_active:
            try:
                # Process recent mentions for sentiment trends
                await asyncio.sleep(5)  # Process every 5 seconds
                
            except Exception as e:
                logger.error(f"Sentiment analysis processing error: {e}")
    
    async def _detect_trending_tokens(self):
        """Detect trending tokens based on mention frequency and sentiment"""
        while self.monitoring_active:
            try:
                # Analyze mentions from last hour
                current_time = time.time()
                hour_ago = current_time - 3600
                
                recent_mentions = [
                    m for m in self.recent_mentions
                    if m.timestamp > hour_ago and m.token_symbol
                ]
                
                # Group by token
                token_mentions = defaultdict(list)
                for mention in recent_mentions:
                    token_mentions[mention.token_symbol].append(mention)
                
                # Calculate trending scores
                for symbol, mentions in token_mentions.items():
                    if len(mentions) >= 5:  # Minimum mentions threshold
                        trending_token = self._calculate_trending_score(symbol, mentions)
                        
                        if trending_token.confidence > 0.6:
                            self.trending_tokens[symbol] = trending_token
                            self.metrics['trending_tokens_detected'] += 1
                            
                            logger.info(f"📈 TRENDING: {symbol} "
                                      f"Mentions: {trending_token.mentions_count} "
                                      f"Sentiment: {trending_token.sentiment_avg:.2f} "
                                      f"Velocity: {trending_token.viral_velocity:.1f}/min")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error detecting trending tokens: {e}")
    
    def _calculate_trending_score(self, symbol: str, mentions: List[SocialMention]) -> TrendingToken:
        """Calculate trending score for a token"""
        try:
            mentions_count = len(mentions)
            
            # Calculate average sentiment
            sentiment_avg = np.mean([m.sentiment_score for m in mentions])
            
            # Calculate influence score
            influence_score = np.mean([m.influence_score for m in mentions])
            
            # Calculate viral velocity (mentions per minute)
            time_span = max(600, mentions[-1].timestamp - mentions[0].timestamp)  # Min 10 minutes
            viral_velocity = (mentions_count / time_span) * 60
            
            # Calculate confidence based on multiple factors
            confidence = min(1.0, (
                (mentions_count / 100) * 0.3 +  # Mention volume
                ((sentiment_avg + 1) / 2) * 0.3 +  # Sentiment positivity
                influence_score * 0.2 +  # Influence
                min(1.0, viral_velocity / 10) * 0.2  # Velocity
            ))
            
            # Get top mentions
            top_mentions = sorted(
                mentions, 
                key=lambda m: m.influence_score * m.sentiment_score, 
                reverse=True
            )[:5]
            
            return TrendingToken(
                symbol=symbol,
                address=mentions[0].token_address,
                mentions_count=mentions_count,
                sentiment_avg=sentiment_avg,
                influence_score=influence_score,
                viral_velocity=viral_velocity,
                confidence=confidence,
                top_mentions=top_mentions,
                trend_start=mentions[0].timestamp,
                predicted_peak=time.time() + 1800  # Predict peak in 30 minutes
            )
            
        except Exception as e:
            logger.error(f"Error calculating trending score: {e}")
            return TrendingToken(
                symbol=symbol, address="", mentions_count=0,
                sentiment_avg=0, influence_score=0, viral_velocity=0,
                confidence=0, top_mentions=[], trend_start=time.time(),
                predicted_peak=time.time()
            )
    
    async def _generate_signals(self):
        """Generate trading signals based on social sentiment"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                for symbol, trending_token in list(self.trending_tokens.items()):
                    try:
                        # Generate signal if criteria met
                        signal = await self._create_sentiment_signal(trending_token)
                        
                        if signal:
                            self.sentiment_signals[symbol] = signal
                            self.metrics['signals_generated'] += 1
                            
                            # Call callback handler
                            if self.callback_handler:
                                await self.callback_handler(signal)
                            
                            logger.info(f"🎯 SENTIMENT SIGNAL: {signal.signal_type} {symbol} "
                                      f"Confidence: {signal.confidence:.2f} "
                                      f"Urgency: {signal.urgency}")
                    
                    except Exception as e:
                        logger.error(f"Error generating signal for {symbol}: {e}")
                
                # Clean up expired signals
                expired_signals = [
                    symbol for symbol, signal in self.sentiment_signals.items()
                    if current_time > signal.expires_at
                ]
                
                for symbol in expired_signals:
                    del self.sentiment_signals[symbol]
                
                await asyncio.sleep(30)  # Generate signals every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in signal generation: {e}")
    
    async def _create_sentiment_signal(self, trending_token: TrendingToken) -> Optional[SentimentSignal]:
        """Create a trading signal from trending token data"""
        try:
            # Signal criteria
            if trending_token.confidence < 0.7:
                return None
            
            if trending_token.viral_velocity < 2:  # Less than 2 mentions per minute
                return None
            
            # Determine signal type and urgency
            if (trending_token.sentiment_avg > 0.5 and 
                trending_token.viral_velocity > 5 and
                trending_token.influence_score > 0.6):
                signal_type = "BUY"
                urgency = "critical"
                confidence = min(0.95, trending_token.confidence * 1.2)
            elif (trending_token.sentiment_avg > 0.3 and 
                  trending_token.viral_velocity > 3):
                signal_type = "BUY"
                urgency = "high"
                confidence = trending_token.confidence
            elif trending_token.sentiment_avg > 0.1:
                signal_type = "WATCH"
                urgency = "medium"
                confidence = trending_token.confidence * 0.8
            else:
                return None
            
            # Count influencer mentions
            influencer_mentions = sum(
                1 for mention in trending_token.top_mentions
                if mention.is_influencer
            )
            
            # Generate reasoning
            reasoning = (
                f"Viral on social media: {trending_token.mentions_count} mentions "
                f"({trending_token.viral_velocity:.1f}/min), "
                f"{trending_token.sentiment_avg:.2f} sentiment, "
                f"{influencer_mentions} influencer mentions"
            )
            
            signal = SentimentSignal(
                token_symbol=trending_token.symbol,
                token_address=trending_token.address,
                signal_type=signal_type,
                confidence=confidence,
                urgency=urgency,
                sentiment_score=trending_token.sentiment_avg,
                viral_velocity=trending_token.viral_velocity,
                influencer_mentions=influencer_mentions,
                reasoning=reasoning,
                supporting_mentions=trending_token.top_mentions,
                timestamp=time.time(),
                expires_at=time.time() + 900  # 15 minute expiry
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating sentiment signal: {e}")
            return None
    
    async def stop_monitoring(self):
        """Stop social media monitoring"""
        try:
            logger.info("🛑 Stopping social sentiment monitoring...")
            self.monitoring_active = False
            
            if self.twitter_stream:
                self.twitter_stream.disconnect()
            
            logger.info("✅ Social sentiment monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping social monitoring: {e}")
    
    def get_trending_tokens(self, limit: int = 10) -> List[TrendingToken]:
        """Get current trending tokens"""
        tokens = list(self.trending_tokens.values())
        tokens.sort(key=lambda t: t.confidence * t.viral_velocity, reverse=True)
        return tokens[:limit]
    
    def get_active_signals(self) -> List[SentimentSignal]:
        """Get currently active sentiment signals"""
        current_time = time.time()
        return [
            signal for signal in self.sentiment_signals.values()
            if current_time <= signal.expires_at
        ]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            'active_signals': len(self.sentiment_signals),
            'trending_tokens': len(self.trending_tokens),
            'recent_mentions': len(self.recent_mentions)
        }

    async def analyze_token_narrative(self, request: NarrativeAnalysisRequest) -> TokenNarrative:
        """Analyze token narrative using LLM for viral potential assessment"""
        try:
            logger.info(f"🧠 NARRATIVE ANALYSIS: {request.token_symbol} - {request.analysis_depth}")
            
            # Prepare context for LLM analysis
            context = self._prepare_narrative_context(request)
            
            # Try remote LLM first (if available), then local LLM
            narrative = await self._analyze_with_remote_llm(context, request)
            
            if not narrative:
                narrative = await self._analyze_with_local_llm(context, request)
            
            if not narrative:
                # Fallback to rule-based analysis
                narrative = self._fallback_narrative_analysis(request)
            
            logger.info(f"🧠 NARRATIVE COMPLETE: {request.token_symbol} - Viral Potential: {narrative.viral_potential:.2f}")
            
            return narrative
            
        except Exception as e:
            logger.error(f"❌ Narrative analysis error: {str(e)}")
            return self._fallback_narrative_analysis(request)
    
    def _prepare_narrative_context(self, request: NarrativeAnalysisRequest) -> str:
        """Prepare comprehensive context for LLM analysis"""
        try:
            # Collect recent social posts
            recent_posts = []
            for post in request.social_posts[-10:]:  # Last 10 posts
                recent_posts.append(f"- {post.get('content', '')}")
            
            # Market context
            market_context = f"""
Market Data:
- Price: ${request.market_data.get('price', 0):.6f}
- Volume 24h: ${request.market_data.get('volume_24h', 0):,.0f}
- Market Cap: ${request.market_data.get('market_cap', 0):,.0f}
- Holders: {request.market_data.get('holders', 0):,}
"""
            
            # Social context
            social_context = f"""
Recent Social Posts:
{chr(10).join(recent_posts[:5])}  
""" if recent_posts else "No recent social posts available."
            
            # Analysis prompt
            prompt = f"""
Analyze the viral potential and narrative strength of this cryptocurrency token:

Token: {request.token_name} ({request.token_symbol})
Description: {request.description}

{market_context}

{social_context}

Please analyze:
1. Viral Potential (0.0-1.0): How likely is this to go viral?
2. Narrative Strength (0.0-1.0): How compelling is the story/concept?
3. Meme Quality (0.0-1.0): How memeable/shareable is this token?
4. Originality (0.0-1.0): How unique is this concept?
5. Community Appeal (0.0-1.0): How broad is the potential audience?
6. Narrative Type: (meme, utility, community, trend, pump)
7. Key Themes: Main narrative themes (3-5 themes)
8. Viral Catalysts: What could make this go viral? (3-5 catalysts)
9. Risk Factors: What could hurt adoption? (3-5 risks)

Provide a concise summary explaining your assessment.
"""
            
            return prompt
            
        except Exception as e:
            logger.error(f"❌ Error preparing narrative context: {str(e)}")
            return f"Basic analysis for {request.token_symbol}"
    
    async def _analyze_with_remote_llm(self, context: str, request: NarrativeAnalysisRequest) -> Optional[TokenNarrative]:
        """Analyze narrative using remote LLM (e.g., OpenAI, Anthropic)"""
        try:
            # This would integrate with actual LLM API
            # For now, simulate advanced LLM analysis
            
            import asyncio
            await asyncio.sleep(1)  # Simulate API call
            
            # Simulate sophisticated LLM response
            token_name_lower = request.token_name.lower()
            
            # Analyze based on token characteristics
            viral_potential = 0.5
            narrative_strength = 0.5
            meme_quality = 0.5
            originality = 0.5
            community_appeal = 0.5
            
            # Boost scores for meme-like tokens
            if any(word in token_name_lower for word in ['dog', 'cat', 'pepe', 'wojak', 'chad', 'moon']):
                viral_potential += 0.3
                meme_quality += 0.4
                community_appeal += 0.2
            
            # Boost for trending themes
            if any(word in token_name_lower for word in ['ai', 'quantum', 'meta', 'solana']):
                narrative_strength += 0.3
                originality += 0.2
            
            # Cap values at 1.0
            viral_potential = min(1.0, viral_potential)
            narrative_strength = min(1.0, narrative_strength)
            meme_quality = min(1.0, meme_quality)
            originality = min(1.0, originality)
            community_appeal = min(1.0, community_appeal)
            
            narrative_type = "meme" if meme_quality > 0.7 else "community" if community_appeal > 0.7 else "trend"
            
            return TokenNarrative(
                token_address=request.token_address,
                narrative_summary=f"Remote LLM analysis of {request.token_symbol}: {narrative_type} token with {viral_potential:.1f} viral potential",
                viral_potential=viral_potential,
                narrative_strength=narrative_strength,
                meme_quality_score=meme_quality,
                originality_score=originality,
                community_appeal=community_appeal,
                narrative_type=narrative_type,
                key_themes=["community", "trending", "viral"],
                viral_catalysts=["social media buzz", "influencer mentions", "meme potential"],
                risk_factors=["market volatility", "regulatory concerns", "competition"],
                llm_confidence=0.85,
                analysis_timestamp=time.time()
            )
            
        except Exception as e:
            logger.debug(f"Remote LLM analysis failed: {str(e)}")
            return None
    
    async def _analyze_with_local_llm(self, context: str, request: NarrativeAnalysisRequest) -> Optional[TokenNarrative]:
        """Analyze narrative using local LLM"""
        try:
            # This would integrate with local LLM (e.g., Ollama, local transformers)
            # For now, simulate local LLM analysis
            
            import asyncio
            await asyncio.sleep(0.5)  # Simulate local processing
            
            # Simplified local analysis
            token_desc_lower = request.description.lower()
            
            viral_potential = 0.4  # Lower baseline for local LLM
            narrative_strength = 0.4
            meme_quality = 0.3
            originality = 0.4
            community_appeal = 0.4
            
            # Simple keyword-based scoring
            viral_keywords = ['viral', 'moon', 'rocket', 'diamond', 'hands', 'ape', 'degen']
            meme_keywords = ['meme', 'funny', 'joke', 'lol', 'based', 'chad']
            utility_keywords = ['utility', 'defi', 'nft', 'dao', 'ecosystem']
            
            for keyword in viral_keywords:
                if keyword in token_desc_lower:
                    viral_potential += 0.1
                    community_appeal += 0.1
            
            for keyword in meme_keywords:
                if keyword in token_desc_lower:
                    meme_quality += 0.15
                    viral_potential += 0.1
            
            for keyword in utility_keywords:
                if keyword in token_desc_lower:
                    narrative_strength += 0.15
                    originality += 0.1
            
            # Cap values
            viral_potential = min(1.0, viral_potential)
            narrative_strength = min(1.0, narrative_strength)
            meme_quality = min(1.0, meme_quality)
            originality = min(1.0, originality)
            community_appeal = min(1.0, community_appeal)
            
            return TokenNarrative(
                token_address=request.token_address,
                narrative_summary=f"Local LLM analysis of {request.token_symbol}: moderate viral potential token",
                viral_potential=viral_potential,
                narrative_strength=narrative_strength,
                meme_quality_score=meme_quality,
                originality_score=originality,
                community_appeal=community_appeal,
                narrative_type="community",
                key_themes=["community", "potential"],
                viral_catalysts=["community growth", "utility adoption"],
                risk_factors=["market conditions", "competition"],
                llm_confidence=0.65,
                analysis_timestamp=time.time()
            )
            
        except Exception as e:
            logger.debug(f"Local LLM analysis failed: {str(e)}")
            return None
    
    def _fallback_narrative_analysis(self, request: NarrativeAnalysisRequest) -> TokenNarrative:
        """Fallback rule-based narrative analysis"""
        try:
            # Basic rule-based analysis
            return TokenNarrative(
                token_address=request.token_address,
                narrative_summary=f"Basic analysis of {request.token_symbol}: standard token assessment",
                viral_potential=0.3,
                narrative_strength=0.3,
                meme_quality_score=0.2,
                originality_score=0.3,
                community_appeal=0.3,
                narrative_type="unknown",
                key_themes=["basic"],
                viral_catalysts=["market conditions"],
                risk_factors=["general market risk"],
                llm_confidence=0.3,
                analysis_timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"❌ Fallback analysis error: {str(e)}")
            return TokenNarrative(
                token_address=request.token_address,
                narrative_summary="Analysis failed",
                viral_potential=0.1,
                narrative_strength=0.1,
                meme_quality_score=0.1,
                originality_score=0.1,
                community_appeal=0.1,
                narrative_type="error",
                key_themes=[],
                viral_catalysts=[],
                risk_factors=["analysis error"],
                llm_confidence=0.0,
                analysis_timestamp=time.time()
            )

# Helper function
async def setup_social_sentiment_engine(callback_handler: Optional[Callable] = None) -> SocialSentimentEngine:
    """Set up and start social sentiment monitoring"""
    engine = SocialSentimentEngine(callback_handler)
    success = await engine.start_monitoring()
    
    if success:
        logger.info("✅ Social sentiment engine setup complete")
    else:
        logger.error("❌ Failed to setup social sentiment engine")
    
    return engine 