"""
Enhanced AI Coordination System for Ant Bot - PROFIT-MAXIMIZED VERSION

This module implements ultra-aggressive AI collaboration optimized for high-risk, high-reward trading:
- Profit-first decision making with aggressive confidence scaling
- Dynamic momentum detection for explosive opportunities
- Enhanced risk-reward calculations for maximum profitability
- Compound learning that increases aggression with success
"""

import asyncio
import json
import logging
import time
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pickle
import os

from .grok_engine import GrokEngine
from ..local_llm import LocalLLM

logger = logging.getLogger(__name__)

class AIModelRole(Enum):
    """Distinct roles for different AI models"""
    SENTIMENT_ANALYZER = "sentiment_analyzer"  # Grok - Twitter/hype analysis
    TECHNICAL_ANALYST = "technical_analyst"    # Local LLM - price/technical analysis
    RISK_ASSESSOR = "risk_assessor"           # Combined - risk evaluation
    DECISION_MAKER = "decision_maker"         # Ensemble - final decisions

class PromptStrategy(Enum):
    """Different prompt engineering strategies"""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    ULTRA_AGGRESSIVE = "ultra_aggressive"  # New ultra-aggressive mode
    PROFIT_MAXIMIZER = "profit_maximizer"  # Profit-focused mode
    ADAPTIVE = "adaptive"

@dataclass
class AIDecision:
    """Structured AI decision with full context"""
    model_role: AIModelRole
    confidence: float
    decision: str
    reasoning: str
    risk_score: float
    profit_potential: float = 0.0  # New: estimated profit potential
    momentum_score: float = 0.0    # New: market momentum score
    conviction_level: str = "MEDIUM"  # New: LOW, MEDIUM, HIGH, EXTREME
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class LearningFeedback:
    """Feedback from trade outcomes for model improvement"""
    trade_id: str
    original_decision: AIDecision
    actual_outcome: Dict[str, Any]
    profit_loss: float
    success: bool
    profit_multiplier: float = 1.0  # New: how much profit exceeded expectations
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    lesson_learned: str = ""
    timestamp: float = field(default_factory=time.time)

@dataclass
class ModelPerformance:
    """Track performance metrics for each AI model"""
    model_role: AIModelRole
    total_decisions: int = 0
    correct_decisions: int = 0
    total_profit: float = 0.0
    best_decision: float = 0.0
    worst_decision: float = 0.0
    average_confidence: float = 0.0
    average_profit_multiplier: float = 1.0  # New: average profit multiplier
    recent_accuracy: List[bool] = field(default_factory=list)
    profit_track_record: List[float] = field(default_factory=list)  # New: profit tracking
    
    @property
    def accuracy(self) -> float:
        return self.correct_decisions / self.total_decisions if self.total_decisions > 0 else 0.0
    
    @property
    def recent_accuracy_rate(self) -> float:
        if not self.recent_accuracy:
            return 0.0
        return sum(self.recent_accuracy) / len(self.recent_accuracy)
    
    @property
    def profit_efficiency(self) -> float:
        """Calculate profit efficiency score"""
        if not self.profit_track_record:
            return 1.0
        positive_profits = [p for p in self.profit_track_record if p > 0]
        if not positive_profits:
            return 0.5
        return min(3.0, sum(positive_profits) / len(positive_profits))  # Cap at 3x
    
    def update_performance(self, correct: bool, profit: float, confidence: float, profit_multiplier: float = 1.0):
        """Update performance metrics with profit tracking"""
        self.total_decisions += 1
        self.total_profit += profit
        self.average_confidence = (self.average_confidence * (self.total_decisions - 1) + confidence) / self.total_decisions
        self.average_profit_multiplier = (self.average_profit_multiplier * (self.total_decisions - 1) + profit_multiplier) / self.total_decisions
        
        if correct:
            self.correct_decisions += 1
        
        if profit > self.best_decision:
            self.best_decision = profit
        if profit < self.worst_decision:
            self.worst_decision = profit
        
        # Update recent accuracy (keep last 20 decisions)
        self.recent_accuracy.append(correct)
        if len(self.recent_accuracy) > 20:
            self.recent_accuracy.pop(0)
        
        # Update profit track record (keep last 50 trades)
        self.profit_track_record.append(profit)
        if len(self.profit_track_record) > 50:
            self.profit_track_record.pop(0)

class AggressivePromptEngineer:
    """Ultra-aggressive prompt engineering for maximum profitability"""
    
    def __init__(self):
        self.base_prompts = {
            AIModelRole.SENTIMENT_ANALYZER: {
                PromptStrategy.ULTRA_AGGRESSIVE: "HUNT FOR EXPLOSIVE OPPORTUNITIES: Identify tokens with viral potential, massive hype, and community excitement. Focus on 10x-100x moonshot candidates.",
                PromptStrategy.PROFIT_MAXIMIZER: "MAXIMIZE PROFIT POTENTIAL: Analyze sentiment for maximum profit opportunities. Look for early stage hype, whale activity, and viral indicators that signal massive price movements.",
                PromptStrategy.AGGRESSIVE: "Find high-potential opportunities with strong community sentiment and viral indicators. Target tokens with explosive growth potential.",
                PromptStrategy.ADAPTIVE: "Adapt analysis based on market conditions and recent profit performance."
            },
            AIModelRole.TECHNICAL_ANALYST: {
                PromptStrategy.ULTRA_AGGRESSIVE: "IDENTIFY BREAKOUT MONSTERS: Find explosive breakout patterns, momentum acceleration, and technical setups for massive price pumps. Target 20%+ moves minimum.",
                PromptStrategy.PROFIT_MAXIMIZER: "TECHNICAL PROFIT HUNTING: Analyze charts for maximum profit potential. Focus on breakouts, momentum signals, and patterns that historically generate 50%+ returns.",
                PromptStrategy.AGGRESSIVE: "Identify aggressive breakout patterns and momentum signals for high-reward entries. Focus on explosive technical setups.",
                PromptStrategy.ADAPTIVE: "Adjust technical analysis based on recent market behavior and profit outcomes."
            }
        }
        self.current_strategy = PromptStrategy.PROFIT_MAXIMIZER  # Start with profit-maximizing mode
        self.performance_history = []
        self.aggression_multiplier = 1.5  # Start with 1.5x aggression
    
    def get_aggressive_prompt(self, model_role: AIModelRole, market_data: Dict, performance_context: ModelPerformance, profit_target: float = 0.2) -> str:
        """Generate ultra-aggressive prompts focused on profit maximization"""
        try:
            base_prompt = self.base_prompts[model_role][self.current_strategy]
            
            # Profit-based modifications
            if performance_context.profit_efficiency > 2.0:
                # Excellent profit performance - go even more aggressive
                profit_modifier = f" Your recent profit performance is EXCEPTIONAL (efficiency: {performance_context.profit_efficiency:.2f}x). INCREASE AGGRESSION and target even bigger opportunities. You're in the zone - trust your instincts and go for MASSIVE gains."
                self.aggression_multiplier = min(3.0, self.aggression_multiplier * 1.1)
            elif performance_context.profit_efficiency > 1.5:
                # Good performance - maintain aggression
                profit_modifier = f" You're performing well (efficiency: {performance_context.profit_efficiency:.2f}x). Maintain aggressive stance and continue hunting for high-profit opportunities."
            elif performance_context.profit_efficiency < 0.8:
                # Poor performance - recalibrate but stay aggressive
                profit_modifier = f" Recent performance needs improvement (efficiency: {performance_context.profit_efficiency:.2f}x). Recalibrate strategy but STAY AGGRESSIVE. Look for clearer, higher-conviction opportunities."
                self.aggression_multiplier = max(1.0, self.aggression_multiplier * 0.95)
            else:
                profit_modifier = " Maintain current aggressive approach while seeking higher-conviction opportunities."
            
            # Market volatility enhancement
            volatility = market_data.get("volatility", 0.5)
            if volatility > 0.7:
                volatility_modifier = f" EXTREME VOLATILITY DETECTED ({volatility:.2f}) - This is PRIME TIME for massive gains. Increase position sizes and target explosive moves. High volatility = HIGH PROFIT POTENTIAL."
            elif volatility > 0.5:
                volatility_modifier = f" High volatility ({volatility:.2f}) provides excellent profit opportunities. Target momentum plays and breakouts."
            else:
                volatility_modifier = f" Lower volatility ({volatility:.2f}) - hunt for accumulation patterns and breakout setups that will explode when volatility returns."
            
            # Profit target integration
            target_modifier = f" TARGET MINIMUM {profit_target*100:.0f}% GAINS on this analysis. Don't settle for small moves - we're hunting for MASSIVE PROFITS."
            
            # Conviction level guidance
            conviction_modifier = " Rate your conviction: EXTREME (90%+ confidence in massive gains), HIGH (70%+ confidence in significant gains), MEDIUM (50%+ confidence), LOW (uncertain). Only recommend EXTREME and HIGH conviction plays."
            
            return base_prompt + profit_modifier + volatility_modifier + target_modifier + conviction_modifier
            
        except Exception as e:
            logger.error(f"Aggressive prompt generation error: {str(e)}")
            return f"Analyze aggressively for maximum profit potential targeting {profit_target*100:.0f}%+ gains."
    
    def update_strategy_based_on_profits(self, recent_profit_performance: float, win_streak: int):
        """Update strategy based on profit performance"""
        if recent_profit_performance > 0.5 and win_streak > 3:
            # Excellent performance - go ultra-aggressive
            self.current_strategy = PromptStrategy.ULTRA_AGGRESSIVE
            self.aggression_multiplier = min(3.0, self.aggression_multiplier * 1.2)
        elif recent_profit_performance > 0.2:
            # Good performance - stay profit-focused
            self.current_strategy = PromptStrategy.PROFIT_MAXIMIZER
        elif recent_profit_performance < -0.2:
            # Poor performance - adaptive mode
            self.current_strategy = PromptStrategy.ADAPTIVE
            self.aggression_multiplier = max(1.0, self.aggression_multiplier * 0.9)
        else:
            # Maintain aggressive approach
            self.current_strategy = PromptStrategy.AGGRESSIVE

class ProfitMaximizedAICoordinator:
    """AI Coordinator optimized for maximum profitability"""
    
    def __init__(self):
        self.grok_engine = None
        self.local_llm = None
        self.prompt_engineer = AggressivePromptEngineer()
        
        # Performance tracking with profit focus
        self.model_performances = {
            AIModelRole.SENTIMENT_ANALYZER: ModelPerformance(AIModelRole.SENTIMENT_ANALYZER),
            AIModelRole.TECHNICAL_ANALYST: ModelPerformance(AIModelRole.TECHNICAL_ANALYST),
            AIModelRole.RISK_ASSESSOR: ModelPerformance(AIModelRole.RISK_ASSESSOR),
            AIModelRole.DECISION_MAKER: ModelPerformance(AIModelRole.DECISION_MAKER)
        }
        
        # Profit-optimized model weights
        self.model_weights = {
            AIModelRole.SENTIMENT_ANALYZER: 0.3,  # Reduced - sentiment can be manipulated
            AIModelRole.TECHNICAL_ANALYST: 0.7   # Increased - technicals more reliable for entries
        }
        
        # Profit tracking
        self.profit_history = []
        self.win_streak = 0
        self.total_realized_profit = 0.0
        self.best_trade_profit = 0.0
        
        # Learning system with profit focus
        self.feedback_history: List[LearningFeedback] = []
        self.learned_profit_patterns = {}
        self.decision_history: List[AIDecision] = []
        
        # Aggression settings
        self.base_confidence_multiplier = 1.3  # Start 30% more confident
        self.profit_target_minimum = 0.15      # 15% minimum profit target
        
    async def initialize(self) -> bool:
        """Initialize AI coordinator with profit-maximized settings"""
        try:
            self.grok_engine = GrokEngine()
            self.local_llm = LocalLLM()
            
            await self.grok_engine.initialize()
            await self.local_llm.initialize()
            
            logger.info("✅ PROFIT-MAXIMIZED AI Coordinator initialized successfully!")
            logger.info(f"🎯 Base confidence multiplier: {self.base_confidence_multiplier}x")
            logger.info(f"💰 Minimum profit target: {self.profit_target_minimum*100}%")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI coordinator: {str(e)}")
            return False

    async def analyze_for_maximum_profit(self, token_address: str, market_data: Dict, portfolio_context: Dict = None) -> Dict[str, Any]:
        """Analyze token with focus on maximum profit potential"""
        try:
            logger.info(f"🎯 PROFIT-FOCUSED ANALYSIS: {token_address}")
            
            # Enhanced market data with profit indicators
            enhanced_market_data = self._enhance_market_data_for_profit(market_data, portfolio_context)
            
            # Get sentiment analysis with profit focus
            sentiment_analysis = await self._get_profit_focused_sentiment(token_address, enhanced_market_data)
            
            # Get technical analysis with breakout focus
            technical_analysis = await self._get_profit_focused_technical(token_address, enhanced_market_data)
            
            # Calculate profit potential
            profit_assessment = await self._calculate_profit_potential(
                token_address, enhanced_market_data, sentiment_analysis, technical_analysis
            )
            
            # Make final profit-maximized decision
            final_decision = await self._make_profit_maximized_decision(
                sentiment_analysis, technical_analysis, profit_assessment, enhanced_market_data
            )
            
            # Log decision with profit metrics
            self._log_profit_decision(final_decision, profit_assessment)
            
            return {
                'sentiment': sentiment_analysis,
                'technical': technical_analysis,
                'profit_assessment': profit_assessment,
                'final_decision': final_decision,
                'profit_target': profit_assessment.get('target_profit', self.profit_target_minimum),
                'conviction_level': final_decision.get('conviction_level', 'MEDIUM'),
                'recommended_position_size': self._calculate_position_size_from_conviction(final_decision),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: AI decision making completely failed: {str(e)}")
            logger.error("🧠 AI brain malfunction detected - cannot make ANY trading decisions")
            logger.error("🛑 ALL TRADING OPERATIONS HALTED - AI brain is required for safe operation")
            raise Exception(f"AI decision making failure: {str(e)} - Bot cannot operate without functioning AI brain")

    def _enhance_market_data_for_profit(self, market_data: Dict, portfolio_context: Dict = None) -> Dict:
        """Enhance market data with profit-focused indicators"""
        enhanced = market_data.copy()
        
        # Add profit indicators
        price = market_data.get('price', 0)
        volume = market_data.get('volume_24h', 0)
        
        # Calculate profit indicators
        enhanced['profit_momentum'] = min(2.0, volume / max(price * 1000000, 1))  # Volume/price ratio
        enhanced['volatility_opportunity'] = market_data.get('volatility', 0.5)
        enhanced['breakout_potential'] = self._calculate_breakout_potential(market_data)
        
        # Portfolio context for aggressive sizing
        if portfolio_context:
            enhanced['portfolio_profit_streak'] = portfolio_context.get('win_streak', 0)
            enhanced['compound_multiplier'] = portfolio_context.get('compound_multiplier', 1.0)
            enhanced['portfolio_momentum'] = portfolio_context.get('total_return_percent', 0) / 100
        
        return enhanced

    def _calculate_breakout_potential(self, market_data: Dict) -> float:
        """Calculate breakout potential score"""
        try:
            volatility = market_data.get('volatility', 0.5)
            volume = market_data.get('volume_24h', 0)
            liquidity = market_data.get('liquidity', 0)
            
            # Higher volatility + volume = higher breakout potential
            breakout_score = (volatility * 0.6) + (min(volume/1000000, 1.0) * 0.4)
            
            # Liquidity factor (enough to trade but not too much resistance)
            if liquidity > 0:
                liquidity_factor = min(1.0, 100000 / max(liquidity, 10000))
                breakout_score *= (0.7 + 0.3 * liquidity_factor)
            
            return min(1.0, breakout_score)
            
        except Exception as e:
            logger.error(f"Error calculating breakout potential: {e}")
            return 0.5

    async def _get_profit_focused_sentiment(self, token_address: str, market_data: Dict) -> AIDecision:
        """Get sentiment analysis focused on profit potential"""
        try:
            performance = self.model_performances[AIModelRole.SENTIMENT_ANALYZER]
            prompt = self.prompt_engineer.get_aggressive_prompt(
                AIModelRole.SENTIMENT_ANALYZER, 
                market_data, 
                performance,
                self.profit_target_minimum
            )
            
            # Get sentiment from Grok with profit focus
            sentiment_result = await self.grok_engine.analyze_sentiment_for_profit(
                token_address, market_data, prompt
            )
            
            # Enhance confidence based on profit indicators
            base_confidence = sentiment_result.get('confidence', 0.5)
            profit_enhanced_confidence = self._enhance_confidence_for_profit(
                base_confidence, market_data, 'sentiment'
            )
            
            decision = AIDecision(
                model_role=AIModelRole.SENTIMENT_ANALYZER,
                confidence=profit_enhanced_confidence,
                decision=sentiment_result.get('decision', 'HOLD'),
                reasoning=sentiment_result.get('reasoning', 'No clear sentiment signals'),
                risk_score=sentiment_result.get('risk_score', 0.5),
                profit_potential=sentiment_result.get('profit_potential', 0.1),
                momentum_score=market_data.get('profit_momentum', 0.5),
                conviction_level=self._determine_conviction_level(profit_enhanced_confidence),
                supporting_data=sentiment_result
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Sentiment analysis failed: {str(e)}")
            logger.error("🧠 AI brain malfunction detected - cannot make trading decisions without sentiment analysis")
            raise Exception(f"AI sentiment analysis failure: {str(e)} - Bot cannot operate without AI brain")

    async def _get_profit_focused_technical(self, token_address: str, market_data: Dict) -> AIDecision:
        """Get technical analysis focused on profit opportunities"""
        try:
            performance = self.model_performances[AIModelRole.TECHNICAL_ANALYST]
            prompt = self.prompt_engineer.get_aggressive_prompt(
                AIModelRole.TECHNICAL_ANALYST, 
                market_data, 
                performance,
                self.profit_target_minimum
            )
            
            # Get technical analysis with profit focus
            technical_result = await self.local_llm.analyze_technical_for_profit(
                token_address, market_data, prompt
            )
            
            # Enhance confidence based on breakout potential
            base_confidence = technical_result.get('confidence', 0.5)
            profit_enhanced_confidence = self._enhance_confidence_for_profit(
                base_confidence, market_data, 'technical'
            )
            
            decision = AIDecision(
                model_role=AIModelRole.TECHNICAL_ANALYST,
                confidence=profit_enhanced_confidence,
                decision=technical_result.get('decision', 'HOLD'),
                reasoning=technical_result.get('reasoning', 'No clear technical signals'),
                risk_score=technical_result.get('risk_score', 0.5),
                profit_potential=technical_result.get('profit_potential', 0.1),
                momentum_score=market_data.get('breakout_potential', 0.5),
                conviction_level=self._determine_conviction_level(profit_enhanced_confidence),
                supporting_data=technical_result
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Technical analysis failed: {str(e)}")
            logger.error("🧠 AI brain malfunction detected - cannot make trading decisions without technical analysis")
            raise Exception(f"AI technical analysis failure: {str(e)} - Bot cannot operate without AI brain")

    def _enhance_confidence_for_profit(self, base_confidence: float, market_data: Dict, analysis_type: str) -> float:
        """Enhance confidence based on profit indicators"""
        enhanced_confidence = base_confidence * self.base_confidence_multiplier
        
        # Portfolio momentum boost
        portfolio_momentum = market_data.get('portfolio_momentum', 0)
        if portfolio_momentum > 0.2:  # 20%+ portfolio gains
            enhanced_confidence *= 1.2
        elif portfolio_momentum > 0.1:  # 10%+ portfolio gains
            enhanced_confidence *= 1.1
        
        # Win streak boost
        win_streak = market_data.get('portfolio_profit_streak', 0)
        if win_streak > 3:
            enhanced_confidence *= (1 + win_streak * 0.05)  # 5% per win in streak
        
        # Volatility opportunity boost
        if analysis_type == 'technical':
            breakout_potential = market_data.get('breakout_potential', 0.5)
            enhanced_confidence *= (0.8 + 0.4 * breakout_potential)
        
        # Market momentum boost
        profit_momentum = market_data.get('profit_momentum', 0.5)
        enhanced_confidence *= (0.9 + 0.2 * profit_momentum)
        
        # Cap at reasonable levels but allow aggressive confidence
        return min(0.95, max(0.1, enhanced_confidence))

    def _determine_conviction_level(self, confidence: float) -> str:
        """Determine conviction level based on confidence"""
        if confidence >= 0.85:
            return "EXTREME"
        elif confidence >= 0.70:
            return "HIGH"
        elif confidence >= 0.55:
            return "MEDIUM"
        else:
            return "LOW"

    async def _calculate_profit_potential(self, token_address: str, market_data: Dict, 
                                        sentiment: AIDecision, technical: AIDecision) -> Dict[str, Any]:
        """Calculate comprehensive profit potential"""
        try:
            # Base profit calculation
            sentiment_profit = sentiment.profit_potential
            technical_profit = technical.profit_potential
            
            # Weighted average based on model weights
            base_profit_potential = (
                sentiment_profit * self.model_weights[AIModelRole.SENTIMENT_ANALYZER] +
                technical_profit * self.model_weights[AIModelRole.TECHNICAL_ANALYST]
            )
            
            # Market enhancement factors
            volatility_multiplier = 1 + market_data.get('volatility_opportunity', 0.5)
            momentum_multiplier = 1 + market_data.get('profit_momentum', 0.5) * 0.5
            breakout_multiplier = 1 + market_data.get('breakout_potential', 0.5) * 0.3
            
            # Enhanced profit potential
            enhanced_profit_potential = (base_profit_potential * volatility_multiplier * 
                                       momentum_multiplier * breakout_multiplier)
            
            # Risk-adjusted profit (higher risk can mean higher reward)
            avg_risk = (sentiment.risk_score + technical.risk_score) / 2
            risk_reward_multiplier = 1 + (avg_risk * 0.5)  # Higher risk = higher potential reward
            
            final_profit_potential = enhanced_profit_potential * risk_reward_multiplier
            
            # Calculate target profit (minimum vs potential)
            target_profit = max(self.profit_target_minimum, final_profit_potential * 0.8)
            
            return {
                'base_potential': base_profit_potential,
                'enhanced_potential': enhanced_profit_potential,
                'final_potential': final_profit_potential,
                'target_profit': target_profit,
                'risk_reward_ratio': final_profit_potential / max(avg_risk, 0.1),
                'volatility_factor': volatility_multiplier,
                'momentum_factor': momentum_multiplier,
                'breakout_factor': breakout_multiplier,
                'profit_confidence': (sentiment.confidence + technical.confidence) / 2
            }
            
        except Exception as e:
            logger.error(f"Error calculating profit potential: {str(e)}")
            return {
                'base_potential': self.profit_target_minimum,
                'enhanced_potential': self.profit_target_minimum,
                'final_potential': self.profit_target_minimum,
                'target_profit': self.profit_target_minimum,
                'risk_reward_ratio': 1.0,
                'profit_confidence': 0.5
            }

    async def _make_profit_maximized_decision(self, sentiment: AIDecision, technical: AIDecision, 
                                            profit_assessment: Dict, market_data: Dict) -> Dict[str, Any]:
        """Make final decision optimized for profit maximization"""
        try:
            # Calculate ensemble confidence with profit weighting
            sentiment_weight = self.model_weights[AIModelRole.SENTIMENT_ANALYZER]
            technical_weight = self.model_weights[AIModelRole.TECHNICAL_ANALYST]
            
            # Adjust weights based on profit performance
            sentiment_performance = self.model_performances[AIModelRole.SENTIMENT_ANALYZER].profit_efficiency
            technical_performance = self.model_performances[AIModelRole.TECHNICAL_ANALYST].profit_efficiency
            
            # Boost weights for better performing models
            if sentiment_performance > technical_performance:
                sentiment_weight *= 1.2
                technical_weight *= 0.9
            else:
                sentiment_weight *= 0.9
                technical_weight *= 1.2
            
            # Normalize weights
            total_weight = sentiment_weight + technical_weight
            sentiment_weight /= total_weight
            technical_weight /= total_weight
            
            # Ensemble confidence
            ensemble_confidence = (sentiment.confidence * sentiment_weight + 
                                 technical.confidence * technical_weight)
            
            # Profit potential consideration
            profit_potential = profit_assessment['final_potential']
            profit_confidence = profit_assessment['profit_confidence']
            
            # Final confidence with profit boost
            final_confidence = ensemble_confidence * (1 + profit_potential * 0.5)
            final_confidence = min(0.95, final_confidence)
            
            # Decision logic with profit priority
            if final_confidence >= 0.75 and profit_potential >= self.profit_target_minimum:
                decision = "STRONG_BUY"
            elif final_confidence >= 0.60 and profit_potential >= self.profit_target_minimum * 0.8:
                decision = "BUY"
            elif final_confidence >= 0.45 and profit_potential >= self.profit_target_minimum * 0.6:
                decision = "WEAK_BUY"
            else:
                decision = "HOLD"
            
            # Conviction level
            conviction_level = self._determine_conviction_level(final_confidence)
            
            # Risk assessment with profit consideration
            avg_risk = (sentiment.risk_score + technical.risk_score) / 2
            profit_adjusted_risk = avg_risk / max(profit_potential, 0.1)  # Lower effective risk for higher profit
            
            return {
                'decision': decision,
                'confidence': final_confidence,
                'conviction_level': conviction_level,
                'profit_potential': profit_potential,
                'target_profit': profit_assessment['target_profit'],
                'risk_reward_ratio': profit_assessment['risk_reward_ratio'],
                'effective_risk': profit_adjusted_risk,
                'reasoning': f"PROFIT-FOCUSED: {sentiment.decision} (sentiment) + {technical.decision} (technical) = {decision} with {profit_potential*100:.1f}% profit potential",
                'model_weights': {'sentiment': sentiment_weight, 'technical': technical_weight},
                'profit_confidence': profit_confidence,
                'market_factors': {
                    'volatility_opportunity': market_data.get('volatility_opportunity', 0.5),
                    'breakout_potential': market_data.get('breakout_potential', 0.5),
                    'profit_momentum': market_data.get('profit_momentum', 0.5)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Profit-maximized decision making failed: {str(e)}")
            logger.error("🧠 AI brain cannot make trading decisions - all trading operations halted")
            raise Exception(f"AI decision making failure: {str(e)} - Cannot trade without AI brain")

    def _calculate_position_size_from_conviction(self, decision: Dict) -> float:
        """Calculate recommended position size based on conviction level"""
        conviction = decision.get('conviction_level', 'MEDIUM')
        confidence = decision.get('confidence', 0.5)
        profit_potential = decision.get('profit_potential', 0.1)
        
        # Base position sizes by conviction
        base_sizes = {
            'EXTREME': 0.35,  # 35% for extreme conviction
            'HIGH': 0.25,     # 25% for high conviction
            'MEDIUM': 0.15,   # 15% for medium conviction
            'LOW': 0.05       # 5% for low conviction
        }
        
        base_size = base_sizes.get(conviction, 0.15)
        
        # Adjust for profit potential (higher profit = larger position)
        profit_multiplier = 1 + min(profit_potential, 0.5)  # Up to 50% increase
        
        # Adjust for confidence
        confidence_multiplier = 0.5 + confidence  # 0.5x to 1.5x based on confidence
        
        final_size = base_size * profit_multiplier * confidence_multiplier
        return min(0.4, max(0.05, final_size))  # Cap between 5% and 40%

    def _log_profit_decision(self, decision: Dict, profit_assessment: Dict):
        """Log decision with profit metrics"""
        try:
            profit_potential = decision.get('profit_potential', 0)
            conviction = decision.get('conviction_level', 'MEDIUM')
            confidence = decision.get('confidence', 0)
            
            logger.info(f"🎯 PROFIT DECISION: {decision.get('decision', 'HOLD')}")
            logger.info(f"   💰 Profit Potential: {profit_potential*100:.1f}%")
            logger.info(f"   🔥 Conviction: {conviction} ({confidence*100:.1f}% confidence)")
            logger.info(f"   📊 Risk/Reward: {decision.get('risk_reward_ratio', 1.0):.2f}")
            logger.info(f"   📈 Target Profit: {decision.get('target_profit', 0)*100:.1f}%")
            
        except Exception as e:
            logger.error(f"Error logging profit decision: {e}")

    def _get_safe_fallback_decision(self) -> Dict[str, Any]:
        """Get safe fallback decision when analysis fails"""
        return {
            'sentiment': AIDecision(
                model_role=AIModelRole.SENTIMENT_ANALYZER,
                confidence=0.3,
                decision='HOLD',
                reasoning='Analysis failed - fallback mode',
                risk_score=0.8,
                profit_potential=0.05
            ),
            'technical': AIDecision(
                model_role=AIModelRole.TECHNICAL_ANALYST,
                confidence=0.3,
                decision='HOLD',
                reasoning='Analysis failed - fallback mode',
                risk_score=0.8,
                profit_potential=0.05
            ),
            'final_decision': {
                'decision': 'HOLD',
                'confidence': 0.3,
                'conviction_level': 'LOW',
                'profit_potential': 0.05,
                'reasoning': 'Fallback decision due to analysis error'
            }
        }

    # Alias the new method to maintain compatibility
    async def analyze_comprehensive(self, token_address: str, market_data: Dict) -> Dict[str, Any]:
        """Alias for analyze_for_maximum_profit to maintain compatibility"""
        return await self.analyze_for_maximum_profit(token_address, market_data)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary with profit focus"""
        try:
            overall_profit = sum(perf.total_profit for perf in self.model_performances.values())
            total_decisions = sum(perf.total_decisions for perf in self.model_performances.values())
            
            return {
                'total_decisions': total_decisions,
                'total_profit': overall_profit,
                'average_profit_per_decision': overall_profit / max(total_decisions, 1),
                'win_streak': self.win_streak,
                'best_trade_profit': self.best_trade_profit,
                'overall_accuracy': sum(perf.accuracy for perf in self.model_performances.values()) / len(self.model_performances),
                'model_weights': self.model_weights,
                'model_performance': {
                    role.value: {
                        'accuracy': perf.accuracy,
                        'recent_accuracy_rate': perf.recent_accuracy_rate,
                        'total_profit': perf.total_profit,
                        'profit_efficiency': perf.profit_efficiency,
                        'average_profit_multiplier': perf.average_profit_multiplier
                    }
                    for role, perf in self.model_performances.items()
                },
                'aggression_multiplier': self.prompt_engineer.aggression_multiplier,
                'current_strategy': self.prompt_engineer.current_strategy.value
            }
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                'total_decisions': 0,
                'total_profit': 0.0,
                'overall_accuracy': 0.5,
                'model_weights': self.model_weights,
                'model_performance': {}
            }

# Alias for backward compatibility
AICoordinator = ProfitMaximizedAICoordinator 