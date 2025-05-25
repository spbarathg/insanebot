"""
Enhanced AI Coordination System for Ant Bot

This module implements sophisticated AI collaboration with:
- Clear role separation: Grok for sentiment/hype, Local LLM for technical decisions
- Learning feedback loops that adapt based on trading outcomes
- Dynamic prompt engineering based on performance
- Multi-model ensemble decision making
- Performance-based model weighting
"""

import asyncio
import json
import logging
import time
import numpy as np
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
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"

@dataclass
class AIDecision:
    """Structured AI decision with full context"""
    model_role: AIModelRole
    confidence: float
    decision: str
    reasoning: str
    risk_score: float
    supporting_data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

@dataclass
class LearningFeedback:
    """Feedback from trade outcomes for model improvement"""
    trade_id: str
    original_decision: AIDecision
    actual_outcome: Dict[str, Any]
    profit_loss: float
    success: bool
    market_conditions: Dict[str, Any]
    lesson_learned: str
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
    recent_accuracy: List[bool] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        return self.correct_decisions / self.total_decisions if self.total_decisions > 0 else 0.0
    
    @property
    def recent_accuracy_rate(self) -> float:
        if not self.recent_accuracy:
            return 0.0
        return sum(self.recent_accuracy) / len(self.recent_accuracy)
    
    def update_performance(self, correct: bool, profit: float, confidence: float):
        """Update performance metrics"""
        self.total_decisions += 1
        self.total_profit += profit
        self.average_confidence = (self.average_confidence * (self.total_decisions - 1) + confidence) / self.total_decisions
        
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

class PromptEngineer:
    """Dynamic prompt engineering based on performance feedback"""
    
    def __init__(self):
        self.base_prompts = {
            AIModelRole.SENTIMENT_ANALYZER: {
                PromptStrategy.CONSERVATIVE: "Analyze sentiment conservatively, focusing on verified information and established trends.",
                PromptStrategy.AGGRESSIVE: "Identify high-potential opportunities with strong community sentiment and viral indicators.",
                PromptStrategy.BALANCED: "Provide balanced sentiment analysis considering both hype and fundamentals.",
                PromptStrategy.ADAPTIVE: "Adapt analysis style based on current market conditions and recent performance."
            },
            AIModelRole.TECHNICAL_ANALYST: {
                PromptStrategy.CONSERVATIVE: "Focus on established technical patterns and strong support/resistance levels.",
                PromptStrategy.AGGRESSIVE: "Identify breakout patterns and momentum signals for quick entries.",
                PromptStrategy.BALANCED: "Combine multiple technical indicators for well-rounded analysis.",
                PromptStrategy.ADAPTIVE: "Adjust technical analysis approach based on recent market behavior."
            }
        }
        self.current_strategy = PromptStrategy.BALANCED
        self.performance_history = []
    
    def get_prompt(self, model_role: AIModelRole, market_data: Dict, performance_context: ModelPerformance) -> str:
        """Generate dynamic prompt based on model role and performance"""
        try:
            base_prompt = self.base_prompts[model_role][self.current_strategy]
            
            # Add performance-based modifications
            if performance_context.recent_accuracy_rate < 0.4:
                # Poor recent performance - be more conservative
                modification = " Be more cautious and focus on high-confidence signals only."
            elif performance_context.recent_accuracy_rate > 0.7:
                # Good performance - can be more aggressive
                modification = " You've been performing well - trust your analysis and consider slightly more aggressive positions."
            else:
                modification = " Maintain current analysis approach."
            
            # Add market context
            volatility = market_data.get("volatility", 0.5)
            if volatility > 0.8:
                market_context = " Market is highly volatile - prioritize risk management."
            elif volatility < 0.3:
                market_context = " Market is stable - look for gradual trend opportunities."
            else:
                market_context = " Market conditions are normal."
            
            return base_prompt + modification + market_context
            
        except Exception as e:
            logger.error(f"Prompt generation error: {str(e)}")
            return "Analyze the given market data and provide your assessment."
    
    def update_strategy(self, overall_performance: float):
        """Update prompt strategy based on overall performance"""
        if overall_performance < 0.3:
            self.current_strategy = PromptStrategy.CONSERVATIVE
        elif overall_performance > 0.7:
            self.current_strategy = PromptStrategy.AGGRESSIVE
        else:
            self.current_strategy = PromptStrategy.BALANCED

class AICoordinator:
    """Coordinates multiple AI models with learning and adaptation"""
    
    def __init__(self):
        self.grok_engine = None
        self.local_llm = None
        self.prompt_engineer = PromptEngineer()
        
        # Performance tracking
        self.model_performances = {
            AIModelRole.SENTIMENT_ANALYZER: ModelPerformance(AIModelRole.SENTIMENT_ANALYZER),
            AIModelRole.TECHNICAL_ANALYST: ModelPerformance(AIModelRole.TECHNICAL_ANALYST),
            AIModelRole.RISK_ASSESSOR: ModelPerformance(AIModelRole.RISK_ASSESSOR),
            AIModelRole.DECISION_MAKER: ModelPerformance(AIModelRole.DECISION_MAKER)
        }
        
        # Learning system
        self.feedback_history: List[LearningFeedback] = []
        self.learned_patterns = {}
        self.model_weights = {
            AIModelRole.SENTIMENT_ANALYZER: 0.4,
            AIModelRole.TECHNICAL_ANALYST: 0.6
        }
        
        # Decision history
        self.decision_history: List[AIDecision] = []
        
    async def initialize(self) -> bool:
        """Initialize AI coordinator with all models"""
        try:
            self.grok_engine = GrokEngine()
            self.local_llm = LocalLLM()
            
            await self.grok_engine.initialize()
            await self.local_llm.initialize()
            
            # Load learned patterns if available
            await self._load_learned_patterns()
            
            logger.info("AI Coordinator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"AI Coordinator initialization error: {str(e)}")
            return False
    
    async def analyze_comprehensive(self, token_address: str, market_data: Dict) -> Dict[str, Any]:
        """Perform comprehensive analysis using all AI models"""
        try:
            # Step 1: Sentiment Analysis (Grok)
            sentiment_decision = await self._get_sentiment_analysis(token_address, market_data)
            
            # Step 2: Technical Analysis (Local LLM)
            technical_decision = await self._get_technical_analysis(token_address, market_data)
            
            # Step 3: Risk Assessment (Combined)
            risk_decision = await self._get_risk_assessment(token_address, market_data, sentiment_decision, technical_decision)
            
            # Step 4: Final Decision (Ensemble)
            final_decision = await self._make_ensemble_decision(sentiment_decision, technical_decision, risk_decision, market_data)
            
            return {
                "sentiment_analysis": sentiment_decision,
                "technical_analysis": technical_decision,
                "risk_assessment": risk_decision,
                "final_decision": final_decision,
                "confidence": final_decision.confidence,
                "recommendation": final_decision.decision
            }
            
        except Exception as e:
            logger.error(f"Comprehensive analysis error: {str(e)}")
            return {"error": str(e)}
    
    async def _get_sentiment_analysis(self, token_address: str, market_data: Dict) -> AIDecision:
        """Get sentiment analysis from Grok AI"""
        try:
            # Generate dynamic prompt
            prompt = self.prompt_engineer.get_prompt(
                AIModelRole.SENTIMENT_ANALYZER,
                market_data,
                self.model_performances[AIModelRole.SENTIMENT_ANALYZER]
            )
            
            # Get Grok analysis with enhanced prompt
            enhanced_market_data = market_data.copy()
            enhanced_market_data["analysis_prompt"] = prompt
            
            grok_result = await self.grok_engine.analyze_market(enhanced_market_data)
            
            if not grok_result or "error" in grok_result:
                raise Exception("Grok analysis failed")
            
            # Extract sentiment metrics
            sentiment_score = grok_result.get("confidence", 0.0)
            hype_level = grok_result.get("hype_level", 0.0)
            community_sentiment = grok_result.get("community_sentiment", 0.0)
            
            # Combine metrics for overall sentiment confidence
            confidence = (sentiment_score * 0.5) + (hype_level * 0.3) + (community_sentiment * 0.2)
            
            decision = AIDecision(
                model_role=AIModelRole.SENTIMENT_ANALYZER,
                confidence=confidence,
                decision="bullish" if confidence > 0.6 else "bearish" if confidence < -0.6 else "neutral",
                reasoning=grok_result.get("reasoning", "Sentiment analysis based on social media activity"),
                risk_score=1.0 - abs(confidence),  # Higher confidence = lower risk
                supporting_data={
                    "sentiment_score": sentiment_score,
                    "hype_level": hype_level,
                    "community_sentiment": community_sentiment,
                    "social_volume": grok_result.get("social_volume", 0),
                    "trend_strength": grok_result.get("trend_strength", 0)
                }
            )
            
            self.decision_history.append(decision)
            return decision
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return AIDecision(
                model_role=AIModelRole.SENTIMENT_ANALYZER,
                confidence=0.0,
                decision="neutral",
                reasoning=f"Analysis failed: {str(e)}",
                risk_score=1.0,
                supporting_data={}
            )
    
    async def _get_technical_analysis(self, token_address: str, market_data: Dict) -> AIDecision:
        """Get technical analysis from Local LLM"""
        try:
            # Generate dynamic prompt
            prompt = self.prompt_engineer.get_prompt(
                AIModelRole.TECHNICAL_ANALYST,
                market_data,
                self.model_performances[AIModelRole.TECHNICAL_ANALYST]
            )
            
            # Get Local LLM analysis with enhanced prompt
            enhanced_market_data = market_data.copy()
            enhanced_market_data["analysis_prompt"] = prompt
            
            llm_result = await self.local_llm.analyze_market(enhanced_market_data)
            
            if not llm_result or "error" in llm_result:
                raise Exception("Local LLM analysis failed")
            
            # Extract technical metrics
            technical_score = llm_result.get("confidence", 0.0)
            trend_strength = llm_result.get("trend_strength", 0.0)
            momentum_score = llm_result.get("momentum_score", 0.0)
            
            # Combine metrics for overall technical confidence
            confidence = (technical_score * 0.6) + (trend_strength * 0.25) + (momentum_score * 0.15)
            
            decision = AIDecision(
                model_role=AIModelRole.TECHNICAL_ANALYST,
                confidence=confidence,
                decision="buy" if confidence > 0.6 else "sell" if confidence < -0.6 else "hold",
                reasoning=llm_result.get("reasoning", "Technical analysis based on price patterns and indicators"),
                risk_score=llm_result.get("risk_level", 0.5),
                supporting_data={
                    "technical_score": technical_score,
                    "trend_strength": trend_strength,
                    "momentum_score": momentum_score,
                    "support_resistance": llm_result.get("support_resistance", {}),
                    "indicators": llm_result.get("indicators", {})
                }
            )
            
            self.decision_history.append(decision)
            return decision
            
        except Exception as e:
            logger.error(f"Technical analysis error: {str(e)}")
            return AIDecision(
                model_role=AIModelRole.TECHNICAL_ANALYST,
                confidence=0.0,
                decision="hold",
                reasoning=f"Analysis failed: {str(e)}",
                risk_score=1.0,
                supporting_data={}
            )
    
    async def _get_risk_assessment(self, token_address: str, market_data: Dict, 
                                 sentiment: AIDecision, technical: AIDecision) -> AIDecision:
        """Perform combined risk assessment"""
        try:
            # Combine risk indicators from both models
            sentiment_risk = sentiment.risk_score
            technical_risk = technical.risk_score
            
            # Market-based risk factors
            liquidity = market_data.get("liquidity", 0)
            volume = market_data.get("volume_24h", 0)
            volatility = market_data.get("volatility", 0.5)
            holder_count = market_data.get("holder_count", 0)
            
            # Calculate risk components
            liquidity_risk = 1.0 if liquidity < 10000 else max(0.1, 1.0 - (liquidity / 100000))
            volume_risk = 1.0 if volume < 1000 else max(0.1, 1.0 - (volume / 50000))
            volatility_risk = min(1.0, volatility)
            holder_risk = 1.0 if holder_count < 50 else max(0.1, 1.0 - (holder_count / 1000))
            
            # Combined risk score
            overall_risk = np.mean([
                sentiment_risk * 0.2,
                technical_risk * 0.3,
                liquidity_risk * 0.2,
                volume_risk * 0.15,
                volatility_risk * 0.1,
                holder_risk * 0.05
            ])
            
            # Risk-based decision
            if overall_risk > 0.8:
                risk_decision = "high_risk"
                confidence = 0.9
            elif overall_risk > 0.6:
                risk_decision = "medium_risk"
                confidence = 0.7
            elif overall_risk > 0.3:
                risk_decision = "low_risk"
                confidence = 0.8
            else:
                risk_decision = "very_low_risk"
                confidence = 0.95
            
            decision = AIDecision(
                model_role=AIModelRole.RISK_ASSESSOR,
                confidence=confidence,
                decision=risk_decision,
                reasoning=f"Combined risk assessment: liquidity={liquidity_risk:.2f}, volume={volume_risk:.2f}, volatility={volatility_risk:.2f}",
                risk_score=overall_risk,
                supporting_data={
                    "sentiment_risk": sentiment_risk,
                    "technical_risk": technical_risk,
                    "liquidity_risk": liquidity_risk,
                    "volume_risk": volume_risk,
                    "volatility_risk": volatility_risk,
                    "holder_risk": holder_risk,
                    "overall_risk": overall_risk
                }
            )
            
            self.decision_history.append(decision)
            return decision
            
        except Exception as e:
            logger.error(f"Risk assessment error: {str(e)}")
            return AIDecision(
                model_role=AIModelRole.RISK_ASSESSOR,
                confidence=0.5,
                decision="high_risk",
                reasoning=f"Risk assessment failed: {str(e)}",
                risk_score=1.0,
                supporting_data={}
            )
    
    async def _make_ensemble_decision(self, sentiment: AIDecision, technical: AIDecision, 
                                    risk: AIDecision, market_data: Dict) -> AIDecision:
        """Make final ensemble decision combining all analyses"""
        try:
            # Dynamic model weighting based on recent performance
            sentiment_weight = self.model_weights[AIModelRole.SENTIMENT_ANALYZER]
            technical_weight = self.model_weights[AIModelRole.TECHNICAL_ANALYST]
            
            # Adjust weights based on recent accuracy
            sentiment_perf = self.model_performances[AIModelRole.SENTIMENT_ANALYZER]
            technical_perf = self.model_performances[AIModelRole.TECHNICAL_ANALYST]
            
            if sentiment_perf.recent_accuracy_rate > technical_perf.recent_accuracy_rate:
                sentiment_weight *= 1.2
                technical_weight *= 0.8
            elif technical_perf.recent_accuracy_rate > sentiment_perf.recent_accuracy_rate:
                technical_weight *= 1.2
                sentiment_weight *= 0.8
            
            # Normalize weights
            total_weight = sentiment_weight + technical_weight
            sentiment_weight /= total_weight
            technical_weight /= total_weight
            
            # Calculate weighted confidence
            sentiment_score = sentiment.confidence if sentiment.decision == "bullish" else -sentiment.confidence
            technical_score = technical.confidence if technical.decision == "buy" else -technical.confidence if technical.decision == "sell" else 0
            
            weighted_confidence = (sentiment_score * sentiment_weight) + (technical_score * technical_weight)
            
            # Risk adjustment
            risk_multiplier = 1.0 - (risk.risk_score * 0.5)  # Reduce confidence for high risk
            final_confidence = weighted_confidence * risk_multiplier
            
            # Final decision logic
            if final_confidence > 0.6 and risk.risk_score < 0.7:
                decision = "buy"
                action_confidence = abs(final_confidence)
            elif final_confidence < -0.6 and risk.risk_score < 0.8:
                decision = "sell"
                action_confidence = abs(final_confidence)
            else:
                decision = "hold"
                action_confidence = 1.0 - abs(final_confidence)
            
            # Position sizing based on confidence and risk
            position_size_multiplier = action_confidence * (1.0 - risk.risk_score)
            
            ensemble_decision = AIDecision(
                model_role=AIModelRole.DECISION_MAKER,
                confidence=action_confidence,
                decision=decision,
                reasoning=f"Ensemble: sentiment={sentiment_score:.2f}({sentiment_weight:.1f}), technical={technical_score:.2f}({technical_weight:.1f}), risk={risk.risk_score:.2f}",
                risk_score=risk.risk_score,
                supporting_data={
                    "weighted_confidence": weighted_confidence,
                    "final_confidence": final_confidence,
                    "position_size_multiplier": position_size_multiplier,
                    "sentiment_weight": sentiment_weight,
                    "technical_weight": technical_weight,
                    "risk_adjustment": risk_multiplier,
                    "component_decisions": {
                        "sentiment": sentiment.decision,
                        "technical": technical.decision,
                        "risk": risk.decision
                    }
                }
            )
            
            self.decision_history.append(ensemble_decision)
            return ensemble_decision
            
        except Exception as e:
            logger.error(f"Ensemble decision error: {str(e)}")
            return AIDecision(
                model_role=AIModelRole.DECISION_MAKER,
                confidence=0.0,
                decision="hold",
                reasoning=f"Ensemble decision failed: {str(e)}",
                risk_score=1.0,
                supporting_data={}
            )
    
    async def learn_from_outcome(self, trade_id: str, original_decision: AIDecision, 
                               trade_outcome: Dict[str, Any]) -> bool:
        """Learn from trade outcomes and update model performance"""
        try:
            profit_loss = trade_outcome.get("profit", 0.0)
            success = trade_outcome.get("success", False)
            
            # Create feedback record
            feedback = LearningFeedback(
                trade_id=trade_id,
                original_decision=original_decision,
                actual_outcome=trade_outcome,
                profit_loss=profit_loss,
                success=success,
                market_conditions=trade_outcome.get("market_conditions", {}),
                lesson_learned=self._extract_lesson(original_decision, trade_outcome)
            )
            
            self.feedback_history.append(feedback)
            
            # Update model performance
            model_perf = self.model_performances[original_decision.model_role]
            model_perf.update_performance(success, profit_loss, original_decision.confidence)
            
            # Update model weights based on performance
            await self._update_model_weights()
            
            # Update prompt strategy
            overall_performance = sum(p.recent_accuracy_rate for p in self.model_performances.values()) / len(self.model_performances)
            self.prompt_engineer.update_strategy(overall_performance)
            
            # Save learned patterns
            await self._save_learned_patterns()
            
            logger.info(f"Learning feedback processed for trade {trade_id}: {'success' if success else 'failure'}")
            return True
            
        except Exception as e:
            logger.error(f"Learning feedback error: {str(e)}")
            return False
    
    def _extract_lesson(self, decision: AIDecision, outcome: Dict[str, Any]) -> str:
        """Extract lesson learned from trade outcome"""
        try:
            if outcome.get("success", False):
                return f"High confidence {decision.decision} was correct - trust similar patterns"
            else:
                risk_score = decision.risk_score
                confidence = decision.confidence
                
                if risk_score > 0.7:
                    return "High risk trades should be avoided or position sizes reduced"
                elif confidence < 0.5:
                    return "Low confidence decisions should not be acted upon"
                else:
                    return f"Review {decision.model_role.value} analysis methodology"
                    
        except Exception as e:
            return f"Failed to extract lesson: {str(e)}"
    
    async def _update_model_weights(self):
        """Update model weights based on recent performance"""
        try:
            sentiment_perf = self.model_performances[AIModelRole.SENTIMENT_ANALYZER].recent_accuracy_rate
            technical_perf = self.model_performances[AIModelRole.TECHNICAL_ANALYST].recent_accuracy_rate
            
            # Calculate new weights based on performance
            total_perf = sentiment_perf + technical_perf
            if total_perf > 0:
                self.model_weights[AIModelRole.SENTIMENT_ANALYZER] = sentiment_perf / total_perf
                self.model_weights[AIModelRole.TECHNICAL_ANALYST] = technical_perf / total_perf
            
            logger.debug(f"Updated model weights: sentiment={self.model_weights[AIModelRole.SENTIMENT_ANALYZER]:.2f}, technical={self.model_weights[AIModelRole.TECHNICAL_ANALYST]:.2f}")
            
        except Exception as e:
            logger.error(f"Model weight update error: {str(e)}")
    
    async def _save_learned_patterns(self):
        """Save learned patterns to disk"""
        try:
            patterns_data = {
                "model_performances": {k.value: v.__dict__ for k, v in self.model_performances.items()},
                "model_weights": {k.value: v for k, v in self.model_weights.items()},
                "feedback_summary": [
                    {
                        "trade_id": f.trade_id,
                        "success": f.success,
                        "profit_loss": f.profit_loss,
                        "lesson": f.lesson_learned,
                        "timestamp": f.timestamp
                    }
                    for f in self.feedback_history[-100:]  # Keep last 100
                ]
            }
            
            os.makedirs("data/learned_patterns", exist_ok=True)
            with open("data/learned_patterns/ai_patterns.json", "w") as f:
                json.dump(patterns_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Save patterns error: {str(e)}")
    
    async def _load_learned_patterns(self):
        """Load learned patterns from disk"""
        try:
            patterns_file = "data/learned_patterns/ai_patterns.json"
            if os.path.exists(patterns_file):
                with open(patterns_file, "r") as f:
                    patterns_data = json.load(f)
                
                # Restore model weights
                if "model_weights" in patterns_data:
                    for role_str, weight in patterns_data["model_weights"].items():
                        role = AIModelRole(role_str)
                        self.model_weights[role] = weight
                
                logger.info("Loaded learned patterns successfully")
                
        except Exception as e:
            logger.error(f"Load patterns error: {str(e)}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            return {
                "model_performances": {
                    k.value: {
                        "accuracy": v.accuracy,
                        "recent_accuracy": v.recent_accuracy_rate,
                        "total_decisions": v.total_decisions,
                        "total_profit": v.total_profit,
                        "average_confidence": v.average_confidence
                    }
                    for k, v in self.model_performances.items()
                },
                "model_weights": {k.value: v for k, v in self.model_weights.items()},
                "current_strategy": self.prompt_engineer.current_strategy.value,
                "total_feedback": len(self.feedback_history),
                "recent_lessons": [f.lesson_learned for f in self.feedback_history[-5:]]
            }
            
        except Exception as e:
            logger.error(f"Performance summary error: {str(e)}")
            return {"error": str(e)} 