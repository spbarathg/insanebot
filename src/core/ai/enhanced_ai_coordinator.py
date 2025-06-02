"""
Enhanced AI Coordinator with Real Machine Learning Integration

This module coordinates AI analysis across multiple real models:
- Real sentiment analysis using FinBERT/DistilBERT
- Online learning with River framework  
- Local LLM reasoning with Ollama
- Ensemble decision making with confidence scoring

Replaces the previous mock AI with production-ready machine learning.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Import the real AI coordinator
try:
    from .real_ai_coordinator import RealAICoordinator, AIDecision
    REAL_AI_AVAILABLE = True
except ImportError:
    REAL_AI_AVAILABLE = False
    logging.warning("Real AI Coordinator not available - using fallback")

logger = logging.getLogger(__name__)

@dataclass
class AIAnalysisResult:
    """Result of AI analysis with all components"""
    action: str  # "buy", "sell", "hold"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    sentiment_score: float
    risk_score: float
    ml_confidence: float
    llm_reasoning: str
    processing_time: float
    timestamp: float

class AICoordinator:
    """
    Enhanced AI Coordinator with Real Machine Learning
    
    Coordinates multiple AI engines:
    - Online learning for continuous adaptation
    - Local LLM for market reasoning
    - Real sentiment analysis
    - Risk assessment and ensemble decisions
    """
    
    def __init__(self):
        self.real_ai = None
        self.fallback_mode = False
        
        if REAL_AI_AVAILABLE:
            try:
                self.real_ai = RealAICoordinator()
                logger.info("✅ Enhanced AI Coordinator initialized with Real AI")
            except Exception as e:
                logger.error(f"Failed to initialize Real AI: {e}")
                self.fallback_mode = True
        else:
            logger.warning("⚠️ Real AI not available - using fallback mode")
            self.fallback_mode = True
        
        # Fallback components for when real AI isn't available
        self.decision_history = []
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_predictions': 0,
            'accuracy': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize the AI coordinator and all engines"""
        try:
            if self.real_ai:
                success = await self.real_ai.initialize()
                if success:
                    logger.info("🧠 Real AI Coordinator initialized successfully")
                    return True
                else:
                    logger.warning("Real AI initialization failed, falling back")
                    self.fallback_mode = True
            
            logger.info("🔄 AI Coordinator running in fallback mode")
            return True
            
        except Exception as e:
            logger.error(f"AI Coordinator initialization failed: {e}")
            self.fallback_mode = True
            return True  # Continue with fallback
    
    async def analyze_market_opportunity(self, token_address: str, market_data: Dict, 
                                       social_data: Optional[Dict] = None) -> AIAnalysisResult:
        """
        Comprehensive AI analysis of trading opportunity
        
        Args:
            token_address: Token contract address
            market_data: Market metrics (price, volume, liquidity, etc.)
            social_data: Optional social sentiment data
            
        Returns:
            AIAnalysisResult with decision, confidence, and reasoning
        """
        start_time = time.time()
        
        try:
            if self.real_ai and not self.fallback_mode:
                # Use real AI system
                ai_decision = await self.real_ai.analyze_trading_opportunity(token_address, market_data)
                
                return AIAnalysisResult(
                    action=ai_decision.action,
                    confidence=ai_decision.confidence,
                    reasoning=ai_decision.reasoning,
                    sentiment_score=ai_decision.factors.get('sentiment_score', 0.0),
                    risk_score=ai_decision.factors.get('risk_score', 0.5),
                    ml_confidence=ai_decision.factors.get('ml_confidence', 0.5),
                    llm_reasoning=ai_decision.reasoning,
                    processing_time=time.time() - start_time,
                    timestamp=ai_decision.timestamp
                )
            else:
                # Fallback to simple heuristics
                return await self._fallback_analysis(token_address, market_data, start_time)
                
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return await self._fallback_analysis(token_address, market_data, start_time)
    
    async def _fallback_analysis(self, token_address: str, market_data: Dict, start_time: float) -> AIAnalysisResult:
        """Fallback analysis using simple heuristics when real AI isn't available"""
        try:
            # Simple heuristic-based analysis
            confidence = 0.3  # Low confidence for fallback
            
            # Basic sentiment from price movement
            price_change = market_data.get('price_change_24h', 0)
            if price_change > 10:
                sentiment_score = 0.6
                action = "buy"
                confidence = 0.4
                reasoning = f"Fallback: Strong price momentum (+{price_change:.1f}%)"
            elif price_change < -20:
                sentiment_score = -0.6
                action = "sell"
                confidence = 0.4
                reasoning = f"Fallback: Significant price decline ({price_change:.1f}%)"
            else:
                sentiment_score = 0.0
                action = "hold"
                reasoning = "Fallback: Neutral market conditions"
            
            # Basic risk assessment
            market_cap = market_data.get('market_cap', 0)
            liquidity = market_data.get('liquidity', 0)
            
            risk_score = 0.7  # High risk default
            if market_cap > 1000000 and liquidity > 50000:
                risk_score = 0.4
                confidence *= 1.2
            elif market_cap < 100000 or liquidity < 10000:
                risk_score = 0.9
                confidence *= 0.8
            
            return AIAnalysisResult(
                action=action,
                confidence=min(0.5, confidence),  # Cap fallback confidence
                reasoning=reasoning + " (Fallback Mode)",
                sentiment_score=sentiment_score,
                risk_score=risk_score,
                ml_confidence=0.0,  # No ML in fallback
                llm_reasoning="LLM not available in fallback mode",
                processing_time=time.time() - start_time,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return AIAnalysisResult(
                action="hold",
                confidence=0.1,
                reasoning=f"Analysis error: {str(e)}",
                sentiment_score=0.0,
                risk_score=0.8,
                ml_confidence=0.0,
                llm_reasoning="Error in analysis",
                processing_time=time.time() - start_time,
                timestamp=time.time()
            )
    
    async def learn_from_trade_outcome(self, trade_data: Dict, outcome: Dict) -> bool:
        """
        Learn from trade outcome to improve future decisions
        
        Args:
            trade_data: Original trade data and market conditions
            outcome: Trade result (success, profit/loss, etc.)
            
        Returns:
            bool: True if learning was successful
        """
        try:
            if self.real_ai and not self.fallback_mode:
                # Use real AI learning
                return await self.real_ai.learn_from_trade_outcome(trade_data, outcome)
            else:
                # Fallback learning (simple tracking)
                self.performance_metrics['total_analyses'] += 1
                if outcome.get('success', False):
                    self.performance_metrics['successful_predictions'] += 1
                
                # Update accuracy
                self.performance_metrics['accuracy'] = (
                    self.performance_metrics['successful_predictions'] / 
                    self.performance_metrics['total_analyses']
                )
                
                logger.debug(f"Fallback learning: accuracy={self.performance_metrics['accuracy']:.3f}")
                return True
                
        except Exception as e:
            logger.error(f"Learning from outcome failed: {e}")
            return False
    
    async def analyze_market_sentiment(self, token_address: str, social_data: Dict = None,
                                     news_data: List[str] = None) -> Dict[str, Any]:
        """
        Analyze market sentiment from various sources
        
        Args:
            token_address: Token to analyze
            social_data: Social media sentiment data
            news_data: List of news articles/descriptions
            
        Returns:
            Dict with sentiment analysis results
        """
        try:
            if self.real_ai and not self.fallback_mode:
                # Use real sentiment analysis
                sentiment_score = 0.0
                
                # Analyze text data if available
                if news_data:
                    text_sentiments = []
                    for text in news_data[:5]:  # Limit to 5 items
                        score, details = self.real_ai.sentiment_engine.analyze_text_sentiment(text)
                        text_sentiments.append(score)
                    
                    if text_sentiments:
                        sentiment_score = sum(text_sentiments) / len(text_sentiments)
                
                return {
                    'sentiment_score': sentiment_score,
                    'confidence': 0.8 if news_data else 0.3,
                    'source': 'real_ai' if news_data else 'market_based',
                    'details': {
                        'text_items_analyzed': len(news_data) if news_data else 0,
                        'social_signals': bool(social_data)
                    }
                }
            else:
                # Fallback sentiment analysis
                return {
                    'sentiment_score': 0.0,
                    'confidence': 0.2,
                    'source': 'fallback',
                    'details': {
                        'text_items_analyzed': 0,
                        'social_signals': False
                    }
                }
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                'sentiment_score': 0.0,
                'confidence': 0.1,
                'source': 'error',
                'details': {'error': str(e)}
            }
    
    async def analyze_technical_indicators(self, token_address: str, price_history: List[float],
                                         volume_history: List[float] = None) -> Dict[str, Any]:
        """
        Analyze technical indicators and patterns
        
        Args:
            token_address: Token to analyze
            price_history: Historical price data
            volume_history: Optional historical volume data
            
        Returns:
            Dict with technical analysis results
        """
        try:
            if not price_history or len(price_history) < 5:
                return {
                    'signal': 'hold',
                    'confidence': 0.1,
                    'reasoning': 'Insufficient price history',
                    'indicators': {}
                }
            
            # Basic technical analysis (works in both real and fallback mode)
            recent_prices = price_history[-10:]
            
            # Simple trend analysis
            if len(recent_prices) >= 3:
                trend_direction = 1 if recent_prices[-1] > recent_prices[0] else -1
                price_volatility = self._calculate_volatility(recent_prices)
                
                # Simple momentum
                momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
                
                # Determine signal
                if momentum > 5 and price_volatility < 50:
                    signal = 'buy'
                    confidence = min(0.7, abs(momentum) / 20)
                elif momentum < -10:
                    signal = 'sell'
                    confidence = min(0.7, abs(momentum) / 20)
                else:
                    signal = 'hold'
                    confidence = 0.4
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'reasoning': f"Momentum: {momentum:.1f}%, Volatility: {price_volatility:.1f}%",
                    'indicators': {
                        'momentum': momentum,
                        'volatility': price_volatility,
                        'trend_direction': trend_direction,
                        'data_points': len(recent_prices)
                    }
                }
            
            return {
                'signal': 'hold',
                'confidence': 0.2,
                'reasoning': 'Insufficient data for analysis',
                'indicators': {}
            }
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {
                'signal': 'hold',
                'confidence': 0.1,
                'reasoning': f'Analysis error: {str(e)}',
                'indicators': {}
            }
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate simple price volatility"""
        try:
            if len(prices) < 2:
                return 0.0
            
            # Calculate price changes
            changes = []
            for i in range(1, len(prices)):
                change = (prices[i] - prices[i-1]) / prices[i-1] * 100
                changes.append(abs(change))
            
            # Return average absolute change as volatility proxy
            return sum(changes) / len(changes) if changes else 0.0
            
        except Exception:
            return 0.0
    
    def get_ai_status(self) -> Dict[str, Any]:
        """Get comprehensive AI system status and performance metrics"""
        try:
            if self.real_ai and not self.fallback_mode:
                # Get real AI status
                real_status = self.real_ai.get_ai_status()
                real_status['mode'] = 'real_ai'
                return real_status
            else:
                # Fallback status
                return {
                    'mode': 'fallback',
                    'engines': {
                        'online_learning': False,
                        'local_llm': False,
                        'sentiment_analysis': False
                    },
                    'performance': self.performance_metrics,
                    'ml_metrics': {},
                    'decision_history_count': len(self.decision_history),
                    'ensemble_weights': {}
                }
                
        except Exception as e:
            logger.error(f"Error getting AI status: {e}")
            return {
                'mode': 'error',
                'error': str(e)
            }
    
    def is_real_ai_available(self) -> bool:
        """Check if real AI system is available and working"""
        return self.real_ai is not None and not self.fallback_mode
    
    async def update_ensemble_weights(self, new_weights: Dict[str, float]) -> bool:
        """Update ensemble model weights based on performance"""
        try:
            if self.real_ai and not self.fallback_mode:
                # Update real AI ensemble weights
                total_weight = sum(new_weights.values())
                if abs(total_weight - 1.0) < 0.1:  # Weights should sum to ~1.0
                    self.real_ai.ensemble_weights.update(new_weights)
                    logger.info(f"Updated ensemble weights: {new_weights}")
                    return True
                else:
                    logger.warning(f"Invalid weights (sum={total_weight}), not updating")
                    return False
            else:
                logger.warning("Cannot update weights in fallback mode")
                return False
                
        except Exception as e:
            logger.error(f"Error updating ensemble weights: {e}")
            return False 