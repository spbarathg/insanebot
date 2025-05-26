"""
Ant Drone - AI Coordination Hub

Contains Grok AI and Local LLM working synchronously to:
- Monitor Twitter for trending meme coins
- Learn from outcomes and adjust strategies
- Direct Worker Ants with evolved intelligence
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .base_ant import BaseAnt, AntRole, AntStatus
from ..intelligence.grok_integration import GrokIntegration
from ..intelligence.local_llm import LocalLLM
from ..intelligence.learning_engine import LearningEngine
from ..flywheel.feedback_loops import FeedbackLoops

logger = logging.getLogger(__name__)

@dataclass
class AIIntelligence:
    """AI intelligence and learning metrics"""
    grok_insights_count: int = 0
    llm_decisions_count: int = 0
    successful_predictions: int = 0
    total_predictions: int = 0
    learning_iterations: int = 0
    strategy_evolution_count: int = 0
    sync_cycles_completed: int = 0
    last_sync: float = 0.0
    intelligence_score: float = 0.5
    
    @property
    def prediction_accuracy(self) -> float:
        """Calculate prediction accuracy percentage"""
        return (self.successful_predictions / self.total_predictions * 100) if self.total_predictions > 0 else 0.0
    
    def update_prediction_result(self, success: bool):
        """Update prediction tracking"""
        self.total_predictions += 1
        if success:
            self.successful_predictions += 1
        
        # Update intelligence score based on recent performance
        accuracy = self.prediction_accuracy
        if accuracy > 70:
            self.intelligence_score = min(1.0, self.intelligence_score + 0.01)
        elif accuracy < 40:
            self.intelligence_score = max(0.1, self.intelligence_score - 0.01)

class AntDrone(BaseAnt):
    """AI coordination hub that manages Grok AI and Local LLM synchronously"""
    
    def __init__(self, ant_id: str, parent_id: str):
        super().__init__(ant_id, AntRole.DRONE, parent_id)
        
        # AI components (initialized later)
        self.grok_ai: Optional[GrokIntegration] = None
        self.local_llm: Optional[LocalLLM] = None
        self.learning_engine: Optional[LearningEngine] = None
        self.feedback_loops: Optional[FeedbackLoops] = None
        
        # Intelligence metrics
        self.intelligence = AIIntelligence()
        
        # Sync configuration
        self.sync_interval = self.config["ai_sync_interval"]
        self.last_sync_time = 0.0
        
        # Market intelligence cache
        self.market_insights: Dict[str, Any] = {}
        self.trending_coins: List[Dict] = []
        self.strategy_recommendations: Dict[str, Any] = {}
        
        logger.info(f"AntDrone {ant_id} created for AI coordination")
    
    async def initialize(self) -> bool:
        """Initialize AI components and establish synchronization"""
        try:
            # Initialize Grok AI for Twitter monitoring
            self.grok_ai = GrokIntegration()
            grok_success = await self.grok_ai.initialize()
            if not grok_success:
                logger.warning("Grok AI initialization failed, continuing with limited intelligence")
            
            # Initialize Local LLM for strategy learning
            self.local_llm = LocalLLM()
            llm_success = await self.local_llm.initialize()
            if not llm_success:
                logger.error("Local LLM initialization failed")
                return False
            
            # Initialize learning engine
            self.learning_engine = LearningEngine()
            await self.learning_engine.initialize()
            
            # Initialize feedback loops for flywheel effect
            self.feedback_loops = FeedbackLoops()
            await self.feedback_loops.initialize()
            
            # Perform initial sync
            await self._sync_ai_systems()
            
            logger.info(f"AntDrone {self.ant_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AntDrone {self.ant_id}: {e}")
            self.status = AntStatus.ERROR
            return False
    
    async def execute_cycle(self) -> Dict[str, Any]:
        """Execute AI coordination cycle"""
        if self.status != AntStatus.ACTIVE:
            return {"status": "inactive", "reason": f"Drone status: {self.status.value}"}
        
        try:
            self.update_activity()
            
            # Check if we need to sync AI systems
            current_time = time.time()
            if current_time - self.last_sync_time >= self.sync_interval:
                await self._sync_ai_systems()
                self.last_sync_time = current_time
            
            # Gather intelligence from both AI systems
            intelligence_result = await self._gather_intelligence()
            
            # Process and learn from recent outcomes
            learning_result = await self._process_learning()
            
            # Generate strategy recommendations for Workers
            strategy_result = await self._generate_strategies()
            
            # Update flywheel mechanisms
            flywheel_result = await self._update_flywheel()
            
            return {
                "status": "active",
                "intelligence": intelligence_result,
                "learning": learning_result,
                "strategies": strategy_result,
                "flywheel": flywheel_result,
                "ai_metrics": self._get_ai_metrics()
            }
            
        except Exception as e:
            logger.error(f"Error in AntDrone {self.ant_id} cycle: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _sync_ai_systems(self):
        """Synchronize Grok AI and Local LLM for coordinated intelligence"""
        try:
            logger.debug(f"AntDrone {self.ant_id} syncing AI systems")
            
            # Get latest insights from Grok AI
            grok_insights = {}
            if self.grok_ai:
                grok_insights = await self.grok_ai.get_latest_insights()
                self.intelligence.grok_insights_count += len(grok_insights.get("trending_topics", []))
            
            # Process insights through Local LLM
            if self.local_llm and grok_insights:
                llm_processing = await self.local_llm.process_grok_insights(grok_insights)
                self.intelligence.llm_decisions_count += 1
                
                # Store processed intelligence
                self.market_insights.update(llm_processing.get("processed_insights", {}))
                self.trending_coins = llm_processing.get("coin_recommendations", [])
            
            # Update learning engine with synchronized data
            if self.learning_engine:
                await self.learning_engine.update_knowledge_base(
                    grok_data=grok_insights,
                    llm_processed_data=llm_processing if 'llm_processing' in locals() else {},
                    sync_timestamp=time.time()
                )
            
            self.intelligence.sync_cycles_completed += 1
            self.intelligence.last_sync = time.time()
            
            logger.debug(f"AntDrone {self.ant_id} AI sync completed")
            
        except Exception as e:
            logger.error(f"Error syncing AI systems in AntDrone {self.ant_id}: {e}")
    
    async def _gather_intelligence(self) -> Dict[str, Any]:
        """Gather market intelligence from both AI systems"""
        try:
            intelligence = {
                "trending_coins": [],
                "market_sentiment": {},
                "risk_alerts": [],
                "opportunities": []
            }
            
            # Gather from Grok AI if available
            if self.grok_ai:
                grok_data = await self.grok_ai.scan_twitter_trends()
                intelligence["trending_coins"].extend(grok_data.get("trending_coins", []))
                intelligence["market_sentiment"].update(grok_data.get("sentiment_analysis", {}))
                intelligence["risk_alerts"].extend(grok_data.get("risk_warnings", []))
            
            # Process through Local LLM for enhanced analysis
            if self.local_llm and intelligence["trending_coins"]:
                llm_analysis = await self.local_llm.analyze_market_opportunities(
                    trending_coins=intelligence["trending_coins"],
                    market_sentiment=intelligence["market_sentiment"]
                )
                intelligence["opportunities"] = llm_analysis.get("opportunities", [])
                
                # Update market insights cache
                self.market_insights["latest_analysis"] = llm_analysis
                self.market_insights["timestamp"] = time.time()
            
            return intelligence
            
        except Exception as e:
            logger.error(f"Error gathering intelligence in AntDrone {self.ant_id}: {e}")
            return {"error": str(e)}
    
    async def _process_learning(self) -> Dict[str, Any]:
        """Process learning from recent trading outcomes"""
        try:
            if not self.learning_engine:
                return {"status": "no_learning_engine"}
            
            # Get recent outcomes from feedback loops
            recent_outcomes = await self.feedback_loops.get_recent_outcomes()
            
            # Process through learning engine
            learning_result = await self.learning_engine.process_outcomes(recent_outcomes)
            
            # Update AI intelligence metrics
            if learning_result.get("successful_predictions"):
                for prediction in learning_result["successful_predictions"]:
                    self.intelligence.update_prediction_result(True)
            
            if learning_result.get("failed_predictions"):
                for prediction in learning_result["failed_predictions"]:
                    self.intelligence.update_prediction_result(False)
            
            # Apply learnings to both AI systems
            if learning_result.get("strategy_updates"):
                if self.grok_ai:
                    await self.grok_ai.update_monitoring_parameters(learning_result["strategy_updates"])
                
                if self.local_llm:
                    await self.local_llm.update_decision_parameters(learning_result["strategy_updates"])
            
            self.intelligence.learning_iterations += 1
            
            return learning_result
            
        except Exception as e:
            logger.error(f"Error processing learning in AntDrone {self.ant_id}: {e}")
            return {"error": str(e)}
    
    async def _generate_strategies(self) -> Dict[str, Any]:
        """Generate trading strategies for Worker Ants"""
        try:
            if not self.local_llm:
                return {"status": "no_llm_available"}
            
            # Generate strategies based on current intelligence
            strategies = await self.local_llm.generate_worker_strategies(
                market_insights=self.market_insights,
                trending_coins=self.trending_coins,
                intelligence_score=self.intelligence.intelligence_score
            )
            
            # Store strategies for Worker access
            self.strategy_recommendations = {
                "strategies": strategies,
                "generated_at": time.time(),
                "intelligence_score": self.intelligence.intelligence_score
            }
            
            self.intelligence.strategy_evolution_count += 1
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error generating strategies in AntDrone {self.ant_id}: {e}")
            return {"error": str(e)}
    
    async def _update_flywheel(self) -> Dict[str, Any]:
        """Update flywheel mechanisms for continuous improvement"""
        try:
            if not self.feedback_loops:
                return {"status": "no_feedback_system"}
            
            # Update feedback loops with current AI performance
            flywheel_update = await self.feedback_loops.update_ai_performance(
                grok_insights_count=self.intelligence.grok_insights_count,
                llm_decisions_count=self.intelligence.llm_decisions_count,
                prediction_accuracy=self.intelligence.prediction_accuracy,
                intelligence_score=self.intelligence.intelligence_score
            )
            
            # Apply flywheel improvements
            if flywheel_update.get("performance_improvements"):
                improvements = flywheel_update["performance_improvements"]
                
                # Update sync interval based on performance
                if improvements.get("sync_optimization"):
                    new_interval = improvements["sync_optimization"]["recommended_interval"]
                    self.sync_interval = max(30, min(300, new_interval))  # Between 30s and 5min
                
                # Update AI parameters based on flywheel feedback
                if improvements.get("ai_parameters") and self.local_llm:
                    await self.local_llm.update_parameters(improvements["ai_parameters"])
            
            return flywheel_update
            
        except Exception as e:
            logger.error(f"Error updating flywheel in AntDrone {self.ant_id}: {e}")
            return {"error": str(e)}
    
    async def get_trading_recommendations(self, worker_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get trading recommendations for a specific Worker Ant"""
        try:
            if not self.local_llm:
                return {"error": "No LLM available for recommendations"}
            
            # Get worker-specific recommendations
            recommendations = await self.local_llm.get_worker_recommendations(
                worker_context=worker_context,
                market_insights=self.market_insights,
                trending_coins=self.trending_coins,
                current_strategies=self.strategy_recommendations
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations from AntDrone {self.ant_id}: {e}")
            return {"error": str(e)}
    
    async def analyze_coin_opportunity(self, coin_address: str, worker_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a specific coin opportunity for a Worker Ant"""
        try:
            analysis = {"buy_signal": False, "sell_signal": False, "confidence": 0.0}
            
            # Get Grok analysis if available
            if self.grok_ai:
                grok_analysis = await self.grok_ai.analyze_coin(coin_address)
                analysis.update(grok_analysis)
            
            # Enhance with LLM analysis
            if self.local_llm:
                llm_analysis = await self.local_llm.analyze_coin_opportunity(
                    coin_address=coin_address,
                    grok_data=analysis,
                    worker_context=worker_context,
                    market_context=self.market_insights
                )
                analysis.update(llm_analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing coin in AntDrone {self.ant_id}: {e}")
            return {"error": str(e), "buy_signal": False, "sell_signal": False, "confidence": 0.0}
    
    async def export_learning_data(self) -> Dict[str, Any]:
        """Export learned data for inheritance by new ants"""
        try:
            learning_data = {
                "ai_intelligence": {
                    "intelligence_score": self.intelligence.intelligence_score,
                    "prediction_accuracy": self.intelligence.prediction_accuracy,
                    "successful_patterns": {},
                    "learned_parameters": {}
                },
                "strategy_evolution": self.strategy_recommendations,
                "market_insights": self.market_insights
            }
            
            # Export from learning engine
            if self.learning_engine:
                engine_data = await self.learning_engine.export_knowledge_base()
                learning_data["knowledge_base"] = engine_data
            
            # Export from LLM
            if self.local_llm:
                llm_data = await self.local_llm.export_learned_parameters()
                learning_data["ai_intelligence"]["learned_parameters"] = llm_data
            
            return learning_data
            
        except Exception as e:
            logger.error(f"Error exporting learning data from AntDrone {self.ant_id}: {e}")
            return {}
    
    def _get_ai_metrics(self) -> Dict[str, Any]:
        """Get comprehensive AI metrics"""
        return {
            "intelligence_score": self.intelligence.intelligence_score,
            "prediction_accuracy": self.intelligence.prediction_accuracy,
            "grok_insights_count": self.intelligence.grok_insights_count,
            "llm_decisions_count": self.intelligence.llm_decisions_count,
            "learning_iterations": self.intelligence.learning_iterations,
            "strategy_evolution_count": self.intelligence.strategy_evolution_count,
            "sync_cycles_completed": self.intelligence.sync_cycles_completed,
            "sync_interval": self.sync_interval,
            "last_sync": self.intelligence.last_sync,
            "market_insights_count": len(self.market_insights),
            "trending_coins_count": len(self.trending_coins)
        }
    
    async def cleanup(self):
        """Cleanup AI components and resources"""
        try:
            if self.grok_ai:
                await self.grok_ai.cleanup()
            
            if self.local_llm:
                await self.local_llm.cleanup()
            
            if self.learning_engine:
                await self.learning_engine.cleanup()
            
            if self.feedback_loops:
                await self.feedback_loops.cleanup()
            
            logger.info(f"AntDrone {self.ant_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during AntDrone {self.ant_id} cleanup: {e}") 