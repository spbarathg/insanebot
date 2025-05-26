"""
Intelligence Layer - AI learning compounding

This layer implements AI learning mechanisms that compound over time through:
- Knowledge accumulation that improves decision making
- Model performance enhancement through continuous learning
- Pattern recognition capability improvement
- Strategic intelligence evolution
- Cross-domain knowledge transfer and application
"""

import logging
import time
import asyncio
import json
import math
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeUnit:
    """Represents a unit of learned knowledge"""
    knowledge_id: str
    domain: str  # trading, market_analysis, risk_management, etc.
    content: Dict[str, Any]
    confidence: float = 0.5
    usefulness: float = 0.5
    creation_time: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    success_rate: float = 0.0
    
@dataclass  
class LearningPattern:
    """Patterns identified in learning processes"""
    pattern_id: str
    pattern_type: str  # trend, correlation, strategy, risk
    triggers: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    outcomes: List[str] = field(default_factory=list)
    confidence: float = 0.5
    frequency: int = 0
    effectiveness: float = 0.0
    
@dataclass
class IntelligenceMetrics:
    """Metrics for tracking intelligence development"""
    total_knowledge_units: int = 0
    total_patterns: int = 0
    avg_confidence: float = 0.0
    avg_usefulness: float = 0.0
    learning_rate: float = 0.0
    pattern_accuracy: float = 0.0
    decision_quality: float = 0.0
    cross_domain_transfers: int = 0
    intelligence_score: float = 1.0
    
@dataclass
class LearningSession:
    """Represents a learning session"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    domain: str = "general"
    learning_type: str = "experience"  # experience, pattern, strategic, cross_domain
    knowledge_gained: int = 0
    patterns_discovered: int = 0
    intelligence_improvement: float = 0.0
    success: bool = False
    
class IntelligenceLayer:
    """
    Layer 4: Intelligence compounding
    
    Handles AI learning mechanisms that compound over time,
    creating increasingly sophisticated decision-making capabilities.
    """
    
    def __init__(self):
        self.layer_id = "intelligence_layer"
        self.initialized = False
        
        # Knowledge management
        self.knowledge_base: Dict[str, KnowledgeUnit] = {}
        self.learning_patterns: Dict[str, LearningPattern] = {}
        self.intelligence_metrics = IntelligenceMetrics()
        
        # Learning tracking
        self.active_sessions: Dict[str, LearningSession] = {}
        self.completed_sessions: List[LearningSession] = []
        
        # Domain organization
        self.knowledge_domains = {
            "trading": [],
            "market_analysis": [],
            "risk_management": [],
            "pattern_recognition": [],
            "strategy_optimization": [],
            "cross_domain": []
        }
        
        # Intelligence compounding
        self.base_intelligence = 1.0
        self.compound_intelligence = 1.0
        self.learning_acceleration = 1.0
        self.pattern_recognition_multiplier = 1.0
        self.decision_accuracy_multiplier = 1.0
        
        # Compound rates by learning type
        self.learning_compound_rates = {
            "experience": 1.008,      # 0.8% per session
            "pattern": 1.012,         # 1.2% per session
            "strategic": 1.015,       # 1.5% per session
            "cross_domain": 1.020     # 2.0% per session
        }
        
        # Knowledge decay and refresh
        self.knowledge_decay_rate = 0.99995  # Very slow decay
        self.pattern_refresh_threshold = 0.7
        
        # Learning optimization
        self.optimal_learning_intervals = {
            "experience": 300,        # 5 minutes
            "pattern": 1800,          # 30 minutes
            "strategic": 7200,        # 2 hours
            "cross_domain": 21600     # 6 hours
        }
        
        logger.info(f"IntelligenceLayer {self.layer_id} created")
    
    async def initialize(self) -> bool:
        """Initialize the intelligence layer"""
        try:
            logger.info(f"Initializing IntelligenceLayer {self.layer_id}...")
            
            # Initialize base knowledge
            await self._initialize_base_knowledge()
            
            # Initialize learning patterns
            await self._initialize_learning_patterns()
            
            # Load historical intelligence data
            await self._load_intelligence_history()
            
            # Calculate current intelligence metrics
            await self._calculate_intelligence_metrics()
            
            self.initialized = True
            logger.info(f"IntelligenceLayer {self.layer_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing IntelligenceLayer {self.layer_id}: {e}")
            return False
    
    async def _initialize_base_knowledge(self):
        """Initialize foundational knowledge units"""
        base_knowledge = [
            {
                "knowledge_id": "trading_basics",
                "domain": "trading",
                "content": {
                    "concepts": ["buy_low_sell_high", "risk_management", "position_sizing"],
                    "rules": ["never_risk_more_than_2_percent", "use_stop_losses"],
                    "strategies": ["trend_following", "mean_reversion"]
                },
                "confidence": 0.9,
                "usefulness": 0.8
            },
            {
                "knowledge_id": "market_patterns",
                "domain": "market_analysis", 
                "content": {
                    "patterns": ["support_resistance", "trends", "volatility_cycles"],
                    "indicators": ["moving_averages", "rsi", "volume"],
                    "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
                },
                "confidence": 0.7,
                "usefulness": 0.9
            },
            {
                "knowledge_id": "risk_fundamentals",
                "domain": "risk_management",
                "content": {
                    "principles": ["diversification", "position_sizing", "correlation_analysis"],
                    "metrics": ["sharpe_ratio", "max_drawdown", "var"],
                    "tools": ["stop_losses", "hedging", "portfolio_balance"]
                },
                "confidence": 0.8,
                "usefulness": 0.95
            }
        ]
        
        current_time = time.time()
        for knowledge_data in base_knowledge:
            knowledge_unit = KnowledgeUnit(
                knowledge_id=knowledge_data["knowledge_id"],
                domain=knowledge_data["domain"],
                content=knowledge_data["content"],
                confidence=knowledge_data["confidence"],
                usefulness=knowledge_data["usefulness"],
                creation_time=current_time,
                last_accessed=current_time
            )
            
            self.knowledge_base[knowledge_unit.knowledge_id] = knowledge_unit
            self.knowledge_domains[knowledge_unit.domain].append(knowledge_unit.knowledge_id)
    
    async def _initialize_learning_patterns(self):
        """Initialize basic learning patterns"""
        base_patterns = [
            {
                "pattern_id": "profit_reinforcement",
                "pattern_type": "strategy",
                "triggers": ["profitable_trade"],
                "conditions": {"profit_margin": "> 0.03"},
                "outcomes": ["increase_position_confidence", "repeat_strategy"],
                "confidence": 0.8,
                "effectiveness": 0.7
            },
            {
                "pattern_id": "loss_adaptation",
                "pattern_type": "risk",
                "triggers": ["losing_trade"],
                "conditions": {"loss_margin": "> 0.05"},
                "outcomes": ["reduce_risk", "analyze_failure", "update_strategy"],
                "confidence": 0.9,
                "effectiveness": 0.8
            },
            {
                "pattern_id": "volume_correlation",
                "pattern_type": "trend",
                "triggers": ["high_volume", "price_movement"],
                "conditions": {"volume_ratio": "> 1.5", "price_change": "> 0.02"},
                "outcomes": ["trend_confirmation", "position_sizing_increase"],
                "confidence": 0.6,
                "effectiveness": 0.6
            }
        ]
        
        for pattern_data in base_patterns:
            pattern = LearningPattern(
                pattern_id=pattern_data["pattern_id"],
                pattern_type=pattern_data["pattern_type"],
                triggers=pattern_data["triggers"],
                conditions=pattern_data["conditions"],
                outcomes=pattern_data["outcomes"],
                confidence=pattern_data["confidence"],
                effectiveness=pattern_data["effectiveness"]
            )
            
            self.learning_patterns[pattern.pattern_id] = pattern
    
    async def _load_intelligence_history(self):
        """Load historical intelligence data"""
        # In a real implementation, this would load from persistent storage
        # For now, initialize with baseline intelligence
        self.compound_intelligence = 1.0
        self.learning_acceleration = 1.0
    
    async def learn_from_experience(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from trading experience and update knowledge"""
        try:
            session_id = f"exp_{int(time.time())}"
            session = LearningSession(
                session_id=session_id,
                start_time=time.time(),
                domain=experience_data.get("domain", "trading"),
                learning_type="experience"
            )
            
            self.active_sessions[session_id] = session
            
            # Extract learning opportunities
            knowledge_gained = await self._extract_knowledge_from_experience(experience_data)
            patterns_discovered = await self._discover_patterns_from_experience(experience_data)
            
            # Apply compound learning effects
            compound_multiplier = self.learning_acceleration * self.compound_intelligence
            effective_knowledge = knowledge_gained * compound_multiplier
            effective_patterns = patterns_discovered * compound_multiplier
            
            # Update session results
            session.knowledge_gained = int(effective_knowledge)
            session.patterns_discovered = int(effective_patterns)
            session.end_time = time.time()
            session.success = True
            
            # Calculate intelligence improvement
            intelligence_improvement = await self._calculate_intelligence_improvement(
                "experience", effective_knowledge, effective_patterns
            )
            session.intelligence_improvement = intelligence_improvement
            
            # Apply compound effect
            await self._apply_learning_compound_effect("experience", intelligence_improvement)
            
            # Update metrics
            await self._update_intelligence_metrics()
            
            # Move to completed sessions
            self.completed_sessions.append(session)
            del self.active_sessions[session_id]
            
            result = {
                "success": True,
                "session_id": session_id,
                "knowledge_gained": session.knowledge_gained,
                "patterns_discovered": session.patterns_discovered,
                "intelligence_improvement": intelligence_improvement,
                "compound_intelligence": self.compound_intelligence
            }
            
            logger.info(f"Learning session {session_id}: +{effective_knowledge:.2f} knowledge, +{effective_patterns} patterns")
            return result
            
        except Exception as e:
            logger.error(f"Error in learning from experience: {e}")
            return {"success": False, "error": str(e)}
    
    async def _extract_knowledge_from_experience(self, experience_data: Dict[str, Any]) -> float:
        """Extract knowledge units from experience data"""
        try:
            knowledge_score = 0.0
            
            # Trade outcome analysis
            if "trade_result" in experience_data:
                trade_result = experience_data["trade_result"]
                profit_loss = trade_result.get("profit_loss", 0)
                
                # More knowledge from significant outcomes
                if abs(profit_loss) > 0.05:  # 5% or more movement
                    knowledge_score += 1.0
                    
                    # Create new knowledge unit
                    await self._create_knowledge_unit(
                        domain="trading",
                        content={
                            "trade_data": trade_result,
                            "outcome": "profitable" if profit_loss > 0 else "loss",
                            "magnitude": abs(profit_loss),
                            "lessons": await self._extract_lessons(trade_result)
                        },
                        confidence=min(0.9, abs(profit_loss) * 5),  # Higher confidence for bigger moves
                        usefulness=0.7
                    )
            
            # Market condition analysis
            if "market_conditions" in experience_data:
                market_data = experience_data["market_conditions"]
                volatility = market_data.get("volatility", 0)
                volume = market_data.get("volume", 0)
                
                if volatility > 0.03 or volume > 1.5:  # Unusual conditions
                    knowledge_score += 0.5
                    
                    await self._create_knowledge_unit(
                        domain="market_analysis",
                        content={
                            "market_state": market_data,
                            "volatility_level": "high" if volatility > 0.03 else "normal",
                            "volume_level": "high" if volume > 1.5 else "normal"
                        },
                        confidence=0.6,
                        usefulness=0.8
                    )
            
            return knowledge_score
            
        except Exception as e:
            logger.error(f"Error extracting knowledge: {e}")
            return 0.0
    
    async def _discover_patterns_from_experience(self, experience_data: Dict[str, Any]) -> int:
        """Discover new patterns from experience"""
        try:
            patterns_found = 0
            
            # Look for correlations in the data
            if "trade_result" in experience_data and "market_conditions" in experience_data:
                trade = experience_data["trade_result"]
                market = experience_data["market_conditions"]
                
                # Check for volume-profit correlation
                volume = market.get("volume", 0)
                profit = trade.get("profit_loss", 0)
                
                if volume > 1.2 and profit > 0.02:
                    # Strong pattern: high volume + profit
                    await self._update_or_create_pattern(
                        pattern_type="correlation",
                        triggers=["high_volume"],
                        conditions={"volume_ratio": "> 1.2"},
                        outcomes=["increased_profit_probability"],
                        effectiveness=abs(profit)
                    )
                    patterns_found += 1
                
                # Check for volatility-risk correlation
                volatility = market.get("volatility", 0)
                if volatility > 0.04 and abs(profit) > 0.03:
                    await self._update_or_create_pattern(
                        pattern_type="risk",
                        triggers=["high_volatility"],
                        conditions={"volatility": "> 0.04"},
                        outcomes=["high_movement_probability"],
                        effectiveness=abs(profit)
                    )
                    patterns_found += 1
            
            return patterns_found
            
        except Exception as e:
            logger.error(f"Error discovering patterns: {e}")
            return 0
    
    async def _create_knowledge_unit(self, domain: str, content: Dict[str, Any], 
                                   confidence: float, usefulness: float) -> str:
        """Create a new knowledge unit"""
        knowledge_id = f"{domain}_{int(time.time())}_{len(self.knowledge_base)}"
        
        knowledge_unit = KnowledgeUnit(
            knowledge_id=knowledge_id,
            domain=domain,
            content=content,
            confidence=confidence,
            usefulness=usefulness,
            creation_time=time.time(),
            last_accessed=time.time()
        )
        
        self.knowledge_base[knowledge_id] = knowledge_unit
        self.knowledge_domains[domain].append(knowledge_id)
        
        return knowledge_id
    
    async def _update_or_create_pattern(self, pattern_type: str, triggers: List[str],
                                      conditions: Dict[str, Any], outcomes: List[str],
                                      effectiveness: float):
        """Update existing pattern or create new one"""
        # Generate pattern signature for matching
        pattern_signature = f"{pattern_type}_{sorted(triggers)}_{sorted(outcomes)}"
        
        # Look for existing similar pattern
        existing_pattern = None
        for pattern in self.learning_patterns.values():
            if (pattern.pattern_type == pattern_type and 
                set(pattern.triggers) == set(triggers) and
                set(pattern.outcomes) == set(outcomes)):
                existing_pattern = pattern
                break
        
        if existing_pattern:
            # Update existing pattern
            existing_pattern.frequency += 1
            existing_pattern.effectiveness = (
                existing_pattern.effectiveness * 0.9 + effectiveness * 0.1
            )
            existing_pattern.confidence = min(0.95, existing_pattern.confidence + 0.05)
        else:
            # Create new pattern
            pattern_id = f"pattern_{pattern_type}_{int(time.time())}"
            new_pattern = LearningPattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                triggers=triggers,
                conditions=conditions,
                outcomes=outcomes,
                confidence=0.3,  # Start with low confidence
                frequency=1,
                effectiveness=effectiveness
            )
            
            self.learning_patterns[pattern_id] = new_pattern
    
    async def _extract_lessons(self, trade_result: Dict[str, Any]) -> List[str]:
        """Extract lessons from trade result"""
        lessons = []
        
        profit_loss = trade_result.get("profit_loss", 0)
        duration = trade_result.get("duration", 0)
        entry_price = trade_result.get("entry_price", 0)
        exit_price = trade_result.get("exit_price", 0)
        
        if profit_loss > 0.05:
            lessons.append("strategy_effective_for_current_conditions")
            lessons.append("position_sizing_appropriate")
        elif profit_loss < -0.03:
            lessons.append("reevaluate_entry_criteria")
            lessons.append("consider_earlier_exit_strategy")
        
        if duration > 3600:  # More than 1 hour
            lessons.append("consider_shorter_timeframe_strategies")
        elif duration < 300:  # Less than 5 minutes
            lessons.append("scalping_opportunity_identified")
        
        return lessons
    
    async def _calculate_intelligence_improvement(self, learning_type: str, 
                                                 knowledge_gained: float, 
                                                 patterns_discovered: int) -> float:
        """Calculate intelligence improvement from learning session"""
        try:
            base_improvement = 0.0
            
            # Knowledge contribution
            knowledge_contribution = knowledge_gained * 0.02  # 2% per knowledge unit
            
            # Pattern contribution (patterns are more valuable)
            pattern_contribution = patterns_discovered * 0.05  # 5% per pattern
            
            # Learning type multiplier
            type_multipliers = {
                "experience": 1.0,
                "pattern": 1.5,
                "strategic": 2.0,
                "cross_domain": 3.0
            }
            
            type_multiplier = type_multipliers.get(learning_type, 1.0)
            
            base_improvement = (knowledge_contribution + pattern_contribution) * type_multiplier
            
            return base_improvement
            
        except Exception as e:
            logger.error(f"Error calculating intelligence improvement: {e}")
            return 0.0
    
    async def _apply_learning_compound_effect(self, learning_type: str, improvement: float):
        """Apply compounding effect to intelligence improvement"""
        try:
            # Get compound rate for learning type
            compound_rate = self.learning_compound_rates.get(learning_type, 1.008)
            
            # Apply compound effect to intelligence
            self.compound_intelligence *= (1 + improvement)
            
            # Apply compound rate to future learning
            self.learning_acceleration *= compound_rate
            
            # Update pattern recognition multiplier
            if learning_type == "pattern":
                self.pattern_recognition_multiplier *= 1.01
            
            # Update decision accuracy multiplier
            if learning_type in ["strategic", "cross_domain"]:
                self.decision_accuracy_multiplier *= 1.008
            
        except Exception as e:
            logger.error(f"Error applying compound effect: {e}")
    
    async def _calculate_intelligence_metrics(self):
        """Calculate current intelligence metrics"""
        try:
            # Basic counts
            self.intelligence_metrics.total_knowledge_units = len(self.knowledge_base)
            self.intelligence_metrics.total_patterns = len(self.learning_patterns)
            
            # Average confidence and usefulness
            if self.knowledge_base:
                total_confidence = sum(k.confidence for k in self.knowledge_base.values())
                total_usefulness = sum(k.usefulness for k in self.knowledge_base.values())
                
                self.intelligence_metrics.avg_confidence = total_confidence / len(self.knowledge_base)
                self.intelligence_metrics.avg_usefulness = total_usefulness / len(self.knowledge_base)
            
            # Pattern accuracy
            if self.learning_patterns:
                total_effectiveness = sum(p.effectiveness for p in self.learning_patterns.values())
                self.intelligence_metrics.pattern_accuracy = total_effectiveness / len(self.learning_patterns)
            
            # Learning rate (based on recent sessions)
            recent_sessions = [s for s in self.completed_sessions[-10:] if s.success]
            if recent_sessions:
                avg_improvement = sum(s.intelligence_improvement for s in recent_sessions) / len(recent_sessions)
                self.intelligence_metrics.learning_rate = avg_improvement
            
            # Overall intelligence score
            self.intelligence_metrics.intelligence_score = self.compound_intelligence
            
        except Exception as e:
            logger.error(f"Error calculating intelligence metrics: {e}")
    
    async def _update_intelligence_metrics(self):
        """Update intelligence metrics after learning"""
        await self._calculate_intelligence_metrics()
    
    async def query_knowledge(self, domain: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Query the knowledge base for relevant information"""
        try:
            relevant_knowledge = []
            
            # Get knowledge units from domain
            domain_knowledge_ids = self.knowledge_domains.get(domain, [])
            
            for knowledge_id in domain_knowledge_ids:
                knowledge_unit = self.knowledge_base.get(knowledge_id)
                if knowledge_unit:
                    # Simple relevance scoring (in real implementation, use embeddings/semantic search)
                    relevance_score = await self._calculate_relevance(knowledge_unit, query)
                    
                    if relevance_score > 0.3:  # Relevance threshold
                        relevant_knowledge.append({
                            "knowledge_id": knowledge_unit.knowledge_id,
                            "content": knowledge_unit.content,
                            "confidence": knowledge_unit.confidence,
                            "usefulness": knowledge_unit.usefulness,
                            "relevance": relevance_score
                        })
                        
                        # Update access metrics
                        knowledge_unit.last_accessed = time.time()
                        knowledge_unit.access_count += 1
            
            # Sort by relevance * confidence * usefulness
            relevant_knowledge.sort(
                key=lambda x: x["relevance"] * x["confidence"] * x["usefulness"],
                reverse=True
            )
            
            return {
                "success": True,
                "domain": domain,
                "relevant_knowledge": relevant_knowledge[:10],  # Top 10 results
                "total_found": len(relevant_knowledge)
            }
            
        except Exception as e:
            logger.error(f"Error querying knowledge: {e}")
            return {"success": False, "error": str(e)}
    
    async def _calculate_relevance(self, knowledge_unit: KnowledgeUnit, query: Dict[str, Any]) -> float:
        """Calculate relevance score between knowledge unit and query"""
        try:
            relevance = 0.0
            
            # Simple keyword matching (in real implementation, use semantic similarity)
            query_terms = set()
            for value in query.values():
                if isinstance(value, str):
                    query_terms.update(value.lower().split())
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            query_terms.update(item.lower().split())
            
            knowledge_terms = set()
            for value in knowledge_unit.content.values():
                if isinstance(value, str):
                    knowledge_terms.update(value.lower().split())
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            knowledge_terms.update(item.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(query_terms.intersection(knowledge_terms))
            union = len(query_terms.union(knowledge_terms))
            
            if union > 0:
                relevance = intersection / union
            
            return relevance
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.0
    
    async def get_intelligence_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI recommendations based on accumulated intelligence"""
        try:
            recommendations = []
            
            # Use patterns to generate recommendations
            for pattern in self.learning_patterns.values():
                if await self._pattern_matches_context(pattern, context):
                    confidence = pattern.confidence * self.decision_accuracy_multiplier
                    
                    recommendations.append({
                        "type": "pattern_based",
                        "pattern_id": pattern.pattern_id,
                        "pattern_type": pattern.pattern_type,
                        "recommendations": pattern.outcomes,
                        "confidence": min(0.99, confidence),
                        "effectiveness": pattern.effectiveness
                    })
            
            # Use knowledge to generate recommendations
            relevant_domains = ["trading", "market_analysis", "risk_management"]
            for domain in relevant_domains:
                knowledge_query = await self.query_knowledge(domain, context)
                if knowledge_query["success"] and knowledge_query["relevant_knowledge"]:
                    top_knowledge = knowledge_query["relevant_knowledge"][0]
                    
                    recommendations.append({
                        "type": "knowledge_based",
                        "domain": domain,
                        "knowledge_id": top_knowledge["knowledge_id"],
                        "recommendations": await self._extract_recommendations_from_knowledge(top_knowledge),
                        "confidence": top_knowledge["confidence"] * self.compound_intelligence,
                        "usefulness": top_knowledge["usefulness"]
                    })
            
            # Sort by confidence * effectiveness/usefulness
            recommendations.sort(
                key=lambda x: x["confidence"] * x.get("effectiveness", x.get("usefulness", 0.5)),
                reverse=True
            )
            
            return {
                "success": True,
                "recommendations": recommendations[:5],  # Top 5 recommendations
                "intelligence_score": self.compound_intelligence,
                "total_recommendations": len(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {"success": False, "error": str(e)}
    
    async def _pattern_matches_context(self, pattern: LearningPattern, context: Dict[str, Any]) -> bool:
        """Check if pattern triggers match current context"""
        try:
            for trigger in pattern.triggers:
                if trigger.lower() in str(context).lower():
                    return True
            
            # More sophisticated matching could be implemented here
            return False
            
        except Exception as e:
            logger.error(f"Error matching pattern to context: {e}")
            return False
    
    async def _extract_recommendations_from_knowledge(self, knowledge: Dict[str, Any]) -> List[str]:
        """Extract actionable recommendations from knowledge unit"""
        recommendations = []
        
        content = knowledge.get("content", {})
        
        if "strategies" in content:
            recommendations.extend([f"consider_{strategy}" for strategy in content["strategies"]])
        
        if "rules" in content:
            recommendations.extend([f"apply_rule_{rule}" for rule in content["rules"]])
        
        if "lessons" in content:
            recommendations.extend([f"remember_{lesson}" for lesson in content["lessons"]])
        
        return recommendations[:3]  # Limit to top 3
    
    def get_layer_metrics(self) -> Dict[str, Any]:
        """Get comprehensive intelligence layer metrics"""
        return {
            "layer_id": self.layer_id,
            "initialized": self.initialized,
            "compound_intelligence": self.compound_intelligence,
            "learning_acceleration": self.learning_acceleration,
            "pattern_recognition_multiplier": self.pattern_recognition_multiplier,
            "decision_accuracy_multiplier": self.decision_accuracy_multiplier,
            "total_knowledge_units": self.intelligence_metrics.total_knowledge_units,
            "total_patterns": self.intelligence_metrics.total_patterns,
            "avg_confidence": self.intelligence_metrics.avg_confidence,
            "avg_usefulness": self.intelligence_metrics.avg_usefulness,
            "learning_rate": self.intelligence_metrics.learning_rate,
            "pattern_accuracy": self.intelligence_metrics.pattern_accuracy,
            "intelligence_score": self.intelligence_metrics.intelligence_score,
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.completed_sessions),
            "knowledge_domains": {domain: len(knowledge_ids) for domain, knowledge_ids in self.knowledge_domains.items()}
        }
    
    def get_compound_effects(self) -> Dict[str, Any]:
        """Get current compounding effects"""
        return {
            "compound_intelligence": self.compound_intelligence,
            "learning_acceleration": self.learning_acceleration,
            "pattern_recognition_multiplier": self.pattern_recognition_multiplier,
            "decision_accuracy_multiplier": self.decision_accuracy_multiplier,
            "compound_rates": self.learning_compound_rates,
            "knowledge_retention": 1.0 - (1.0 - self.knowledge_decay_rate) * 1000,  # Effective retention
            "learning_efficiency": self.learning_acceleration * self.compound_intelligence
        } 