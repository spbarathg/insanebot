"""
Performance Amplification - Success multiplication through flywheel effects

Implements performance amplification mechanisms that create exponential growth
through success pattern recognition, momentum building, and multiplicative effects.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
import math

logger = logging.getLogger(__name__)

@dataclass
class SuccessPattern:
    """Represents an identified success pattern"""
    pattern_id: str
    pattern_type: str
    component: str
    conditions: Dict[str, Any]
    success_probability: float
    impact_multiplier: float
    confidence_score: float
    occurrences: int
    last_seen: float

@dataclass
class AmplificationTrigger:
    """Represents a trigger for performance amplification"""
    trigger_id: str
    trigger_type: str
    component: str
    threshold_value: float
    current_value: float
    amplification_factor: float
    is_active: bool
    activated_at: float = 0.0

@dataclass
class MomentumMetric:
    """Tracks momentum in a specific area"""
    metric_id: str
    component: str
    metric_name: str
    momentum_score: float
    velocity: float
    acceleration: float
    peak_momentum: float
    momentum_history: deque

class PerformanceAmplification:
    """Manages performance amplification through flywheel effects"""
    
    def __init__(self):
        # Amplification tracking
        self.success_patterns: Dict[str, SuccessPattern] = {}
        self.amplification_triggers: Dict[str, AmplificationTrigger] = {}
        self.momentum_metrics: Dict[str, MomentumMetric] = {}
        
        # Amplification configuration
        self.amplification_threshold = 1.2  # 20% improvement triggers amplification
        self.momentum_decay_rate = 0.95  # Momentum decay per cycle
        self.max_amplification_factor = 3.0  # Maximum amplification multiplier
        
        # Performance tracking
        self.total_amplifications = 0
        self.cumulative_amplification_effect = 1.0
        self.flywheel_momentum = 0.0
        self.amplification_efficiency = 0.5
        
        # Success tracking
        self.success_events: deque = deque(maxlen=1000)
        self.amplification_history: List[Dict] = []
        
        logger.info("PerformanceAmplification initialized")
    
    async def initialize(self) -> bool:
        """Initialize performance amplification system"""
        try:
            # Initialize success pattern recognition
            await self._initialize_pattern_recognition()
            
            # Initialize amplification triggers
            await self._initialize_amplification_triggers()
            
            # Initialize momentum tracking
            await self._initialize_momentum_tracking()
            
            logger.info("PerformanceAmplification initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PerformanceAmplification: {e}")
            return False
    
    async def execute_amplification_cycle(self) -> Dict[str, Any]:
        """Execute performance amplification cycle"""
        try:
            cycle_results = {
                "patterns_analyzed": 0,
                "triggers_activated": 0,
                "amplifications_applied": 0,
                "momentum_acceleration": 0.0,
                "flywheel_effect": 0.0
            }
            
            # Analyze success patterns
            pattern_analysis = await self._analyze_success_patterns()
            cycle_results["patterns_analyzed"] = pattern_analysis["patterns_found"]
            
            # Check amplification triggers
            trigger_check = await self._check_amplification_triggers()
            cycle_results["triggers_activated"] = trigger_check["triggers_activated"]
            
            # Apply amplification effects
            amplification_results = await self._apply_amplification_effects()
            cycle_results["amplifications_applied"] = amplification_results["amplifications_applied"]
            
            # Update momentum metrics
            momentum_update = await self._update_momentum_metrics()
            cycle_results["momentum_acceleration"] = momentum_update["average_acceleration"]
            
            # Calculate flywheel effect
            flywheel_effect = await self._calculate_flywheel_effect()
            cycle_results["flywheel_effect"] = flywheel_effect["current_momentum"]
            
            return cycle_results
            
        except Exception as e:
            logger.error(f"Error in performance amplification cycle: {e}")
            return {"error": str(e)}
    
    async def record_success_event(
        self, 
        component: str, 
        event_type: str, 
        performance_data: Dict[str, Any]
    ) -> str:
        """Record a success event for pattern analysis"""
        try:
            event_id = f"success_{int(time.time())}_{len(self.success_events)}"
            
            success_event = {
                "event_id": event_id,
                "component": component,
                "event_type": event_type,
                "timestamp": time.time(),
                "performance_data": performance_data,
                "analyzed": False
            }
            
            self.success_events.append(success_event)
            
            # Check for immediate amplification opportunities
            await self._check_immediate_amplification(success_event)
            
            logger.debug(f"Recorded success event: {event_type} from {component}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error recording success event: {e}")
            return ""
    
    async def _analyze_success_patterns(self) -> Dict[str, Any]:
        """Analyze success events for patterns"""
        try:
            analysis_results = {
                "patterns_found": 0,
                "pattern_confidence": 0.0,
                "new_patterns": [],
                "updated_patterns": []
            }
            
            # Get unanalyzed success events
            unanalyzed_events = [e for e in self.success_events if not e["analyzed"]]
            
            if len(unanalyzed_events) < 5:  # Need minimum events for pattern analysis
                return analysis_results
            
            # Group events by component and type
            event_groups = self._group_success_events(unanalyzed_events)
            
            for group_key, events in event_groups.items():
                if len(events) >= 3:  # Minimum for pattern
                    pattern = await self._identify_pattern(group_key, events)
                    if pattern:
                        if pattern.pattern_id in self.success_patterns:
                            # Update existing pattern
                            existing_pattern = self.success_patterns[pattern.pattern_id]
                            existing_pattern.occurrences += pattern.occurrences
                            existing_pattern.last_seen = pattern.last_seen
                            existing_pattern.confidence_score = min(1.0, existing_pattern.confidence_score + 0.1)
                            analysis_results["updated_patterns"].append(pattern.pattern_id)
                        else:
                            # Add new pattern
                            self.success_patterns[pattern.pattern_id] = pattern
                            analysis_results["new_patterns"].append(pattern.pattern_id)
                        
                        analysis_results["patterns_found"] += 1
            
            # Mark events as analyzed
            for event in unanalyzed_events:
                event["analyzed"] = True
            
            # Calculate average pattern confidence
            if self.success_patterns:
                total_confidence = sum(p.confidence_score for p in self.success_patterns.values())
                analysis_results["pattern_confidence"] = total_confidence / len(self.success_patterns)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing success patterns: {e}")
            return {"patterns_found": 0}
    
    def _group_success_events(self, events: List[Dict]) -> Dict[str, List[Dict]]:
        """Group success events by component and type"""
        groups = {}
        
        for event in events:
            group_key = f"{event['component']}_{event['event_type']}"
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(event)
        
        return groups
    
    async def _identify_pattern(self, group_key: str, events: List[Dict]) -> Optional[SuccessPattern]:
        """Identify a success pattern from grouped events"""
        try:
            component, event_type = group_key.split('_', 1)
            
            # Analyze performance data for patterns
            performance_values = []
            conditions = {}
            
            for event in events:
                perf_data = event["performance_data"]
                performance_values.append(perf_data.get("profit", 0.0))
                
                # Extract common conditions
                for key, value in perf_data.items():
                    if key not in conditions:
                        conditions[key] = []
                    conditions[key].append(value)
            
            # Calculate pattern metrics
            avg_performance = sum(performance_values) / len(performance_values)
            success_rate = len([p for p in performance_values if p > 0]) / len(performance_values)
            
            if success_rate < 0.6:  # Require 60% success rate minimum
                return None
            
            # Create pattern
            pattern_id = f"pattern_{component}_{event_type}_{int(time.time())}"
            
            pattern = SuccessPattern(
                pattern_id=pattern_id,
                pattern_type=event_type,
                component=component,
                conditions=self._extract_pattern_conditions(conditions),
                success_probability=success_rate,
                impact_multiplier=max(1.0, avg_performance),
                confidence_score=min(1.0, len(events) / 10.0),  # Higher confidence with more events
                occurrences=len(events),
                last_seen=max(e["timestamp"] for e in events)
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error identifying pattern: {e}")
            return None
    
    def _extract_pattern_conditions(self, conditions: Dict[str, List]) -> Dict[str, Any]:
        """Extract common conditions from success events"""
        pattern_conditions = {}
        
        for key, values in conditions.items():
            if isinstance(values[0], (int, float)):
                # Numerical condition - find range
                pattern_conditions[key] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values)
                }
            elif isinstance(values[0], str):
                # Categorical condition - find most common
                from collections import Counter
                most_common = Counter(values).most_common(1)[0]
                if most_common[1] / len(values) > 0.5:  # Appears in >50% of cases
                    pattern_conditions[key] = most_common[0]
        
        return pattern_conditions
    
    async def _check_amplification_triggers(self) -> Dict[str, Any]:
        """Check if amplification triggers should be activated"""
        try:
            trigger_results = {
                "triggers_checked": 0,
                "triggers_activated": 0,
                "newly_activated": [],
                "deactivated": []
            }
            
            for trigger_id, trigger in self.amplification_triggers.items():
                trigger_results["triggers_checked"] += 1
                
                # Check if trigger condition is met
                should_activate = trigger.current_value >= trigger.threshold_value
                
                if should_activate and not trigger.is_active:
                    # Activate trigger
                    trigger.is_active = True
                    trigger.activated_at = time.time()
                    trigger_results["triggers_activated"] += 1
                    trigger_results["newly_activated"].append(trigger_id)
                    
                    logger.info(f"Amplification trigger activated: {trigger_id}")
                
                elif not should_activate and trigger.is_active:
                    # Deactivate trigger
                    trigger.is_active = False
                    trigger_results["deactivated"].append(trigger_id)
            
            return trigger_results
            
        except Exception as e:
            logger.error(f"Error checking amplification triggers: {e}")
            return {"triggers_activated": 0}
    
    async def _apply_amplification_effects(self) -> Dict[str, Any]:
        """Apply amplification effects based on active triggers and patterns"""
        try:
            amplification_results = {
                "amplifications_applied": 0,
                "total_amplification_factor": 1.0,
                "components_amplified": [],
                "amplification_breakdown": {}
            }
            
            # Apply amplification for each active trigger
            for trigger_id, trigger in self.amplification_triggers.items():
                if trigger.is_active:
                    amplification_effect = await self._calculate_amplification_effect(trigger)
                    
                    if amplification_effect > 1.0:
                        amplification_results["amplifications_applied"] += 1
                        amplification_results["total_amplification_factor"] *= amplification_effect
                        amplification_results["components_amplified"].append(trigger.component)
                        amplification_results["amplification_breakdown"][trigger.component] = amplification_effect
                        
                        # Record amplification in history
                        await self._record_amplification(trigger_id, amplification_effect)
            
            # Update cumulative amplification effect
            self.cumulative_amplification_effect *= amplification_results["total_amplification_factor"]
            
            # Cap amplification to prevent infinite growth
            self.cumulative_amplification_effect = min(
                self.cumulative_amplification_effect, 
                self.max_amplification_factor
            )
            
            return amplification_results
            
        except Exception as e:
            logger.error(f"Error applying amplification effects: {e}")
            return {"amplifications_applied": 0}
    
    async def _calculate_amplification_effect(self, trigger: AmplificationTrigger) -> float:
        """Calculate amplification effect for a specific trigger"""
        try:
            # Base amplification from trigger
            base_amplification = trigger.amplification_factor
            
            # Boost based on momentum
            momentum_boost = 1.0
            if trigger.component in self.momentum_metrics:
                momentum = self.momentum_metrics[trigger.component].momentum_score
                momentum_boost = 1.0 + (momentum * 0.5)  # Up to 50% boost from momentum
            
            # Boost based on success patterns
            pattern_boost = 1.0
            relevant_patterns = [
                p for p in self.success_patterns.values() 
                if p.component == trigger.component and p.confidence_score > 0.7
            ]
            
            if relevant_patterns:
                avg_pattern_multiplier = sum(p.impact_multiplier for p in relevant_patterns) / len(relevant_patterns)
                pattern_boost = 1.0 + (avg_pattern_multiplier - 1.0) * 0.3  # 30% of pattern impact
            
            # Calculate total amplification
            total_amplification = base_amplification * momentum_boost * pattern_boost
            
            # Apply diminishing returns
            total_amplification = 1.0 + (total_amplification - 1.0) * math.exp(-self.total_amplifications * 0.1)
            
            return min(total_amplification, self.max_amplification_factor)
            
        except Exception as e:
            logger.error(f"Error calculating amplification effect: {e}")
            return 1.0
    
    async def _update_momentum_metrics(self) -> Dict[str, Any]:
        """Update momentum metrics for all components"""
        try:
            momentum_update = {
                "metrics_updated": 0,
                "average_momentum": 0.0,
                "average_acceleration": 0.0,
                "peak_momentum_achieved": []
            }
            
            total_momentum = 0.0
            total_acceleration = 0.0
            
            for metric_id, momentum_metric in self.momentum_metrics.items():
                # Update momentum based on recent performance
                new_momentum = await self._calculate_momentum(momentum_metric)
                
                # Calculate velocity (change in momentum)
                old_momentum = momentum_metric.momentum_score
                momentum_metric.velocity = new_momentum - old_momentum
                
                # Calculate acceleration (change in velocity)
                momentum_metric.acceleration = momentum_metric.velocity - (
                    momentum_metric.momentum_history[-1]["velocity"] if momentum_metric.momentum_history else 0.0
                )
                
                momentum_metric.momentum_score = new_momentum
                
                # Record in history
                momentum_metric.momentum_history.append({
                    "timestamp": time.time(),
                    "momentum": new_momentum,
                    "velocity": momentum_metric.velocity
                })
                
                # Check for peak momentum
                if new_momentum > momentum_metric.peak_momentum:
                    momentum_metric.peak_momentum = new_momentum
                    momentum_update["peak_momentum_achieved"].append(metric_id)
                
                total_momentum += new_momentum
                total_acceleration += momentum_metric.acceleration
                momentum_update["metrics_updated"] += 1
            
            if momentum_update["metrics_updated"] > 0:
                momentum_update["average_momentum"] = total_momentum / momentum_update["metrics_updated"]
                momentum_update["average_acceleration"] = total_acceleration / momentum_update["metrics_updated"]
            
            return momentum_update
            
        except Exception as e:
            logger.error(f"Error updating momentum metrics: {e}")
            return {"metrics_updated": 0}
    
    async def _calculate_momentum(self, momentum_metric: MomentumMetric) -> float:
        """Calculate current momentum for a specific metric"""
        try:
            # Get recent performance data (this would integrate with actual component data)
            # For now, simulate momentum calculation
            
            # Apply momentum decay
            current_momentum = momentum_metric.momentum_score * self.momentum_decay_rate
            
            # Add momentum from recent success patterns
            recent_patterns = [
                p for p in self.success_patterns.values()
                if (p.component == momentum_metric.component and 
                    time.time() - p.last_seen < 3600)  # Within last hour
            ]
            
            if recent_patterns:
                pattern_momentum = sum(p.impact_multiplier for p in recent_patterns) / len(recent_patterns)
                current_momentum += (pattern_momentum - 1.0) * 0.1  # 10% momentum boost
            
            # Ensure momentum stays within bounds
            return max(0.0, min(2.0, current_momentum))
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return momentum_metric.momentum_score * self.momentum_decay_rate
    
    async def _calculate_flywheel_effect(self) -> Dict[str, Any]:
        """Calculate overall flywheel momentum effect"""
        try:
            # Calculate flywheel momentum from all components
            total_momentum = sum(m.momentum_score for m in self.momentum_metrics.values())
            avg_momentum = total_momentum / max(1, len(self.momentum_metrics))
            
            # Factor in success patterns
            pattern_effect = 1.0
            if self.success_patterns:
                avg_pattern_confidence = sum(p.confidence_score for p in self.success_patterns.values()) / len(self.success_patterns)
                pattern_effect = 1.0 + avg_pattern_confidence * 0.5
            
            # Factor in amplification history
            amplification_effect = min(2.0, self.cumulative_amplification_effect)
            
            # Calculate flywheel momentum
            self.flywheel_momentum = avg_momentum * pattern_effect * amplification_effect
            
            return {
                "current_momentum": self.flywheel_momentum,
                "momentum_components": total_momentum,
                "pattern_effect": pattern_effect,
                "amplification_effect": amplification_effect
            }
            
        except Exception as e:
            logger.error(f"Error calculating flywheel effect: {e}")
            return {"current_momentum": 0.0}
    
    async def _check_immediate_amplification(self, success_event: Dict[str, Any]):
        """Check for immediate amplification opportunities from success event"""
        try:
            performance_data = success_event["performance_data"]
            component = success_event["component"]
            
            # Check if this success event triggers any amplification
            profit = performance_data.get("profit", 0.0)
            
            if profit > 1.5:  # Significant success
                # Update relevant amplification triggers
                for trigger in self.amplification_triggers.values():
                    if trigger.component == component:
                        trigger.current_value = max(trigger.current_value, profit)
                
                # Boost momentum for this component
                if component in self.momentum_metrics:
                    momentum_metric = self.momentum_metrics[component]
                    momentum_boost = min(0.2, (profit - 1.0) * 0.1)  # Up to 20% boost
                    momentum_metric.momentum_score = min(2.0, momentum_metric.momentum_score + momentum_boost)
            
        except Exception as e:
            logger.error(f"Error checking immediate amplification: {e}")
    
    async def _record_amplification(self, trigger_id: str, amplification_effect: float):
        """Record an amplification event in history"""
        try:
            amplification_record = {
                "trigger_id": trigger_id,
                "timestamp": time.time(),
                "amplification_effect": amplification_effect,
                "cumulative_effect": self.cumulative_amplification_effect
            }
            
            self.amplification_history.append(amplification_record)
            self.total_amplifications += 1
            
            # Keep only last 1000 amplification records
            if len(self.amplification_history) > 1000:
                self.amplification_history = self.amplification_history[-1000:]
            
        except Exception as e:
            logger.error(f"Error recording amplification: {e}")
    
    async def _initialize_pattern_recognition(self):
        """Initialize success pattern recognition system"""
        self.success_patterns = {}
    
    async def _initialize_amplification_triggers(self):
        """Initialize amplification triggers for key components"""
        trigger_configs = [
            {"component": "worker_ants", "trigger_type": "profit_threshold", "threshold": 1.3, "amplification": 1.2},
            {"component": "ant_drone", "trigger_type": "accuracy_threshold", "threshold": 0.75, "amplification": 1.15},
            {"component": "compounding_system", "trigger_type": "compound_rate", "threshold": 1.25, "amplification": 1.3},
            {"component": "feedback_loops", "trigger_type": "effectiveness", "threshold": 0.8, "amplification": 1.1}
        ]
        
        for config in trigger_configs:
            trigger_id = f"trigger_{config['component']}_{config['trigger_type']}"
            
            self.amplification_triggers[trigger_id] = AmplificationTrigger(
                trigger_id=trigger_id,
                trigger_type=config["trigger_type"],
                component=config["component"],
                threshold_value=config["threshold"],
                current_value=0.0,
                amplification_factor=config["amplification"],
                is_active=False
            )
    
    async def _initialize_momentum_tracking(self):
        """Initialize momentum tracking for key components"""
        components = ["worker_ants", "ant_drone", "compounding_system", "feedback_loops"]
        
        for component in components:
            metric_id = f"momentum_{component}"
            
            self.momentum_metrics[metric_id] = MomentumMetric(
                metric_id=metric_id,
                component=component,
                metric_name="overall_momentum",
                momentum_score=0.5,  # Start with neutral momentum
                velocity=0.0,
                acceleration=0.0,
                peak_momentum=0.5,
                momentum_history=deque(maxlen=100)
            )
    
    def get_amplification_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance amplification summary"""
        return {
            "amplification_stats": {
                "total_amplifications": self.total_amplifications,
                "cumulative_amplification_effect": self.cumulative_amplification_effect,
                "flywheel_momentum": self.flywheel_momentum,
                "amplification_efficiency": self.amplification_efficiency
            },
            "pattern_analysis": {
                "success_patterns_identified": len(self.success_patterns),
                "average_pattern_confidence": sum(p.confidence_score for p in self.success_patterns.values()) / max(1, len(self.success_patterns)),
                "success_events_recorded": len(self.success_events)
            },
            "momentum_overview": {
                "components_tracked": len(self.momentum_metrics),
                "average_momentum": sum(m.momentum_score for m in self.momentum_metrics.values()) / max(1, len(self.momentum_metrics)),
                "peak_momentum_components": len([m for m in self.momentum_metrics.values() if m.momentum_score > 1.5])
            },
            "trigger_status": {
                "total_triggers": len(self.amplification_triggers),
                "active_triggers": len([t for t in self.amplification_triggers.values() if t.is_active]),
                "trigger_activation_rate": len([t for t in self.amplification_triggers.values() if t.is_active]) / max(1, len(self.amplification_triggers))
            }
        }
    
    def get_flywheel_metrics(self) -> Dict[str, Any]:
        """Get detailed flywheel effect metrics"""
        return {
            "current_flywheel_momentum": self.flywheel_momentum,
            "momentum_by_component": {
                m.component: {
                    "momentum_score": m.momentum_score,
                    "velocity": m.velocity,
                    "acceleration": m.acceleration,
                    "peak_momentum": m.peak_momentum
                }
                for m in self.momentum_metrics.values()
            },
            "active_amplifications": {
                t.trigger_id: {
                    "component": t.component,
                    "threshold": t.threshold_value,
                    "current_value": t.current_value,
                    "amplification_factor": t.amplification_factor,
                    "active_since": (time.time() - t.activated_at) / 3600 if t.activated_at > 0 else None
                }
                for t in self.amplification_triggers.values() if t.is_active
            }
        }
    
    async def cleanup(self):
        """Cleanup performance amplification resources"""
        try:
            # Clear large data structures
            self.success_events.clear()
            self.amplification_history.clear()
            
            # Clear momentum history
            for momentum_metric in self.momentum_metrics.values():
                momentum_metric.momentum_history.clear()
            
            logger.info("PerformanceAmplification cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during PerformanceAmplification cleanup: {e}") 