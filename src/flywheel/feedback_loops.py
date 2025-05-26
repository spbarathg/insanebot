"""
Feedback Loops - AI improvement through outcome analysis

Implements feedback loops that capture trading outcomes, AI predictions,
and system performance to continuously improve decision making.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class FeedbackEvent:
    """Represents a feedback event in the system"""
    event_id: str
    event_type: str  # "trade_outcome", "prediction_result", "performance_metric"
    source_id: str
    timestamp: float
    data: Dict[str, Any]
    processed: bool = False

@dataclass
class FeedbackMetrics:
    """Tracks feedback loop effectiveness"""
    total_events: int = 0
    processed_events: int = 0
    improvement_iterations: int = 0
    effectiveness_score: float = 0.5
    learning_rate: float = 0.1
    last_improvement: float = 0.0

class FeedbackLoops:
    """Manages AI feedback loops for continuous improvement"""
    
    def __init__(self):
        # Feedback tracking
        self.feedback_events: deque = deque(maxlen=10000)  # Keep last 10k events
        self.feedback_sources: Dict[str, Any] = {}
        self.metrics = FeedbackMetrics()
        
        # Processing configuration
        self.batch_size = 100
        self.processing_interval = 60.0  # Process every minute
        self.effectiveness_threshold = 0.7
        
        # Improvement tracking
        self.improvement_history: List[Dict] = []
        self.pattern_database: Dict[str, Any] = {}
        self.learning_models: Dict[str, Any] = {}
        
        logger.info("FeedbackLoops initialized")
    
    async def initialize(self) -> bool:
        """Initialize feedback loop system"""
        try:
            # Initialize pattern recognition
            await self._initialize_pattern_database()
            
            # Initialize learning models
            await self._initialize_learning_models()
            
            logger.info("FeedbackLoops initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FeedbackLoops: {e}")
            return False
    
    async def add_feedback_source(self, source_id: str, source_component: Any):
        """Add a component as a feedback source"""
        self.feedback_sources[source_id] = {
            "component": source_component,
            "events_contributed": 0,
            "last_feedback": 0.0
        }
        logger.debug(f"Added feedback source: {source_id}")
    
    async def record_feedback_event(
        self, 
        event_type: str, 
        source_id: str, 
        data: Dict[str, Any]
    ) -> str:
        """Record a new feedback event"""
        try:
            event_id = f"fb_{int(time.time())}_{len(self.feedback_events)}"
            
            event = FeedbackEvent(
                event_id=event_id,
                event_type=event_type,
                source_id=source_id,
                timestamp=time.time(),
                data=data
            )
            
            self.feedback_events.append(event)
            self.metrics.total_events += 1
            
            # Update source tracking
            if source_id in self.feedback_sources:
                self.feedback_sources[source_id]["events_contributed"] += 1
                self.feedback_sources[source_id]["last_feedback"] = time.time()
            
            logger.debug(f"Recorded feedback event: {event_type} from {source_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error recording feedback event: {e}")
            return ""
    
    async def process_feedback_cycle(self) -> Dict[str, Any]:
        """Process accumulated feedback events"""
        try:
            cycle_results = {
                "events_processed": 0,
                "patterns_discovered": 0,
                "improvements_applied": 0,
                "effectiveness_score": self.metrics.effectiveness_score
            }
            
            # Get unprocessed events
            unprocessed_events = [e for e in self.feedback_events if not e.processed]
            
            if not unprocessed_events:
                return cycle_results
            
            # Process events in batches
            for i in range(0, len(unprocessed_events), self.batch_size):
                batch = unprocessed_events[i:i + self.batch_size]
                batch_result = await self._process_event_batch(batch)
                
                cycle_results["events_processed"] += batch_result["processed_count"]
                cycle_results["patterns_discovered"] += batch_result["patterns_found"]
            
            # Apply improvements based on processed feedback
            improvements = await self._apply_feedback_improvements()
            cycle_results["improvements_applied"] = improvements["applied_count"]
            
            # Update effectiveness score
            await self._update_effectiveness_score()
            cycle_results["effectiveness_score"] = self.metrics.effectiveness_score
            
            return cycle_results
            
        except Exception as e:
            logger.error(f"Error in feedback processing cycle: {e}")
            return {"error": str(e)}
    
    async def _process_event_batch(self, events: List[FeedbackEvent]) -> Dict[str, Any]:
        """Process a batch of feedback events"""
        try:
            batch_result = {
                "processed_count": 0,
                "patterns_found": 0,
                "learning_updates": 0
            }
            
            for event in events:
                # Process different event types
                if event.event_type == "trade_outcome":
                    await self._process_trade_outcome(event)
                elif event.event_type == "prediction_result":
                    await self._process_prediction_result(event)
                elif event.event_type == "performance_metric":
                    await self._process_performance_metric(event)
                
                event.processed = True
                batch_result["processed_count"] += 1
                self.metrics.processed_events += 1
            
            # Look for patterns in this batch
            patterns_found = await self._analyze_batch_patterns(events)
            batch_result["patterns_found"] = len(patterns_found)
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Error processing event batch: {e}")
            return {"processed_count": 0, "patterns_found": 0}
    
    async def _process_trade_outcome(self, event: FeedbackEvent):
        """Process trade outcome feedback"""
        data = event.data
        profit = data.get("profit", 0.0)
        prediction_confidence = data.get("prediction_confidence", 0.5)
        strategy_used = data.get("strategy", "unknown")
        
        # Update strategy performance tracking
        if strategy_used not in self.pattern_database:
            self.pattern_database[strategy_used] = {
                "success_count": 0,
                "total_count": 0,
                "total_profit": 0.0,
                "avg_confidence": 0.0
            }
        
        strategy_data = self.pattern_database[strategy_used]
        strategy_data["total_count"] += 1
        strategy_data["total_profit"] += profit
        
        if profit > 0:
            strategy_data["success_count"] += 1
        
        # Update average confidence
        strategy_data["avg_confidence"] = (
            (strategy_data["avg_confidence"] * (strategy_data["total_count"] - 1) + prediction_confidence) /
            strategy_data["total_count"]
        )
    
    async def _process_prediction_result(self, event: FeedbackEvent):
        """Process AI prediction result feedback"""
        data = event.data
        predicted_outcome = data.get("predicted_outcome")
        actual_outcome = data.get("actual_outcome")
        model_used = data.get("model", "default")
        
        # Track prediction accuracy
        if model_used not in self.learning_models:
            self.learning_models[model_used] = {
                "correct_predictions": 0,
                "total_predictions": 0,
                "accuracy": 0.0,
                "confidence_calibration": []
            }
        
        model_data = self.learning_models[model_used]
        model_data["total_predictions"] += 1
        
        if predicted_outcome == actual_outcome:
            model_data["correct_predictions"] += 1
        
        model_data["accuracy"] = model_data["correct_predictions"] / model_data["total_predictions"]
    
    async def _process_performance_metric(self, event: FeedbackEvent):
        """Process system performance metric feedback"""
        data = event.data
        metric_name = data.get("metric_name")
        metric_value = data.get("metric_value")
        component = data.get("component")
        
        # Track component performance trends
        component_key = f"{component}_{metric_name}"
        
        if component_key not in self.pattern_database:
            self.pattern_database[component_key] = {
                "values": deque(maxlen=100),  # Keep last 100 values
                "trend": "stable",
                "improvement_rate": 0.0
            }
        
        component_data = self.pattern_database[component_key]
        component_data["values"].append({
            "value": metric_value,
            "timestamp": event.timestamp
        })
        
        # Calculate trend
        if len(component_data["values"]) >= 10:
            await self._calculate_performance_trend(component_data)
    
    async def _analyze_batch_patterns(self, events: List[FeedbackEvent]) -> List[Dict]:
        """Analyze patterns in a batch of events"""
        patterns = []
        
        # Group events by type and analyze
        event_groups = {}
        for event in events:
            if event.event_type not in event_groups:
                event_groups[event.event_type] = []
            event_groups[event.event_type].append(event)
        
        # Look for success patterns
        if "trade_outcome" in event_groups:
            trade_patterns = await self._find_trade_patterns(event_groups["trade_outcome"])
            patterns.extend(trade_patterns)
        
        return patterns
    
    async def _find_trade_patterns(self, trade_events: List[FeedbackEvent]) -> List[Dict]:
        """Find patterns in trading outcomes"""
        patterns = []
        
        # Group by strategy
        strategy_groups = {}
        for event in trade_events:
            strategy = event.data.get("strategy", "unknown")
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(event)
        
        # Analyze each strategy's performance
        for strategy, events in strategy_groups.items():
            if len(events) >= 5:  # Minimum events for pattern analysis
                profits = [e.data.get("profit", 0.0) for e in events]
                success_rate = len([p for p in profits if p > 0]) / len(profits)
                
                if success_rate > 0.7:  # High success rate pattern
                    patterns.append({
                        "type": "high_success_strategy",
                        "strategy": strategy,
                        "success_rate": success_rate,
                        "avg_profit": sum(profits) / len(profits),
                        "confidence": min(1.0, len(events) / 20.0)  # Higher confidence with more data
                    })
        
        return patterns
    
    async def _apply_feedback_improvements(self) -> Dict[str, Any]:
        """Apply improvements based on processed feedback"""
        try:
            improvements = {
                "applied_count": 0,
                "improvement_types": [],
                "effectiveness_gain": 0.0
            }
            
            # Apply strategy improvements
            strategy_improvements = await self._apply_strategy_improvements()
            improvements["applied_count"] += strategy_improvements["count"]
            improvements["improvement_types"].extend(strategy_improvements["types"])
            
            # Apply model improvements
            model_improvements = await self._apply_model_improvements()
            improvements["applied_count"] += model_improvements["count"]
            improvements["improvement_types"].extend(model_improvements["types"])
            
            # Update metrics
            if improvements["applied_count"] > 0:
                self.metrics.improvement_iterations += 1
                self.metrics.last_improvement = time.time()
                
                # Record improvement in history
                self.improvement_history.append({
                    "timestamp": time.time(),
                    "improvements": improvements,
                    "effectiveness_before": self.metrics.effectiveness_score
                })
            
            return improvements
            
        except Exception as e:
            logger.error(f"Error applying feedback improvements: {e}")
            return {"applied_count": 0, "improvement_types": []}
    
    async def _apply_strategy_improvements(self) -> Dict[str, Any]:
        """Apply improvements to trading strategies"""
        improvements = {"count": 0, "types": []}
        
        # Find best performing strategies
        best_strategies = []
        for strategy, data in self.pattern_database.items():
            if data.get("total_count", 0) >= 10:  # Minimum trades for confidence
                success_rate = data.get("success_count", 0) / data["total_count"]
                if success_rate > 0.6:  # 60% success rate threshold
                    best_strategies.append({
                        "strategy": strategy,
                        "success_rate": success_rate,
                        "total_profit": data.get("total_profit", 0.0)
                    })
        
        if best_strategies:
            # Sort by success rate
            best_strategies.sort(key=lambda x: x["success_rate"], reverse=True)
            
            # Apply improvements (simplified - in production would update AI models)
            improvements["count"] += len(best_strategies)
            improvements["types"].append("strategy_optimization")
            
            logger.info(f"Applied strategy improvements for {len(best_strategies)} strategies")
        
        return improvements
    
    async def _apply_model_improvements(self) -> Dict[str, Any]:
        """Apply improvements to AI models"""
        improvements = {"count": 0, "types": []}
        
        # Find underperforming models
        for model, data in self.learning_models.items():
            if data.get("total_predictions", 0) >= 20:  # Minimum predictions
                accuracy = data.get("accuracy", 0.0)
                if accuracy < 0.5:  # Below 50% accuracy
                    # Apply recalibration (simplified)
                    improvements["count"] += 1
                    improvements["types"].append("model_recalibration")
                    
                    # Update learning rate for this model
                    self.metrics.learning_rate = min(0.2, self.metrics.learning_rate * 1.1)
        
        return improvements
    
    async def _calculate_performance_trend(self, component_data: Dict):
        """Calculate performance trend for a component"""
        values = list(component_data["values"])
        if len(values) < 10:
            return
        
        # Simple trend calculation
        recent_values = [v["value"] for v in values[-10:]]
        older_values = [v["value"] for v in values[-20:-10]] if len(values) >= 20 else recent_values
        
        recent_avg = sum(recent_values) / len(recent_values)
        older_avg = sum(older_values) / len(older_values)
        
        if recent_avg > older_avg * 1.05:  # 5% improvement
            component_data["trend"] = "improving"
            component_data["improvement_rate"] = (recent_avg - older_avg) / older_avg
        elif recent_avg < older_avg * 0.95:  # 5% decline
            component_data["trend"] = "declining"
            component_data["improvement_rate"] = (recent_avg - older_avg) / older_avg
        else:
            component_data["trend"] = "stable"
            component_data["improvement_rate"] = 0.0
    
    async def _update_effectiveness_score(self):
        """Update overall feedback loop effectiveness score"""
        try:
            # Base effectiveness on processing rate and improvement frequency
            processing_rate = (
                self.metrics.processed_events / max(1, self.metrics.total_events)
            )
            
            # Recent improvement factor
            time_since_improvement = time.time() - self.metrics.last_improvement
            improvement_freshness = max(0.0, 1.0 - (time_since_improvement / 3600))  # Decay over 1 hour
            
            # Model accuracy factor
            avg_model_accuracy = 0.5
            if self.learning_models:
                accuracies = [m.get("accuracy", 0.5) for m in self.learning_models.values()]
                avg_model_accuracy = sum(accuracies) / len(accuracies)
            
            # Combined effectiveness score
            self.metrics.effectiveness_score = (
                processing_rate * 0.3 +
                improvement_freshness * 0.3 +
                avg_model_accuracy * 0.4
            )
            
            # Ensure score is within bounds
            self.metrics.effectiveness_score = max(0.0, min(1.0, self.metrics.effectiveness_score))
            
        except Exception as e:
            logger.error(f"Error updating effectiveness score: {e}")
    
    async def get_effectiveness_score(self) -> float:
        """Get current effectiveness score"""
        return self.metrics.effectiveness_score
    
    async def get_recent_outcomes(self) -> List[Dict]:
        """Get recent outcomes for learning engine"""
        recent_events = [
            e for e in self.feedback_events 
            if e.timestamp > (time.time() - 3600) and e.processed  # Last hour
        ]
        
        return [
            {
                "event_type": e.event_type,
                "source_id": e.source_id,
                "timestamp": e.timestamp,
                "data": e.data
            }
            for e in recent_events
        ]
    
    async def _initialize_pattern_database(self):
        """Initialize pattern recognition database"""
        self.pattern_database = {}
    
    async def _initialize_learning_models(self):
        """Initialize learning model tracking"""
        self.learning_models = {}
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get comprehensive feedback loop summary"""
        return {
            "metrics": {
                "total_events": self.metrics.total_events,
                "processed_events": self.metrics.processed_events,
                "improvement_iterations": self.metrics.improvement_iterations,
                "effectiveness_score": self.metrics.effectiveness_score,
                "learning_rate": self.metrics.learning_rate
            },
            "sources": {
                source_id: {
                    "events_contributed": data["events_contributed"],
                    "last_feedback_hours_ago": (time.time() - data["last_feedback"]) / 3600
                }
                for source_id, data in self.feedback_sources.items()
            },
            "patterns": {
                "total_patterns": len(self.pattern_database),
                "learning_models": len(self.learning_models),
                "improvement_history_length": len(self.improvement_history)
            }
        }
    
    async def cleanup(self):
        """Cleanup feedback loop resources"""
        try:
            # Clear large data structures
            self.feedback_events.clear()
            self.improvement_history.clear()
            
            logger.info("FeedbackLoops cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during FeedbackLoops cleanup: {e}") 