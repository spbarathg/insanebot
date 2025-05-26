"""
Compounding Logic - Implementation of 5-layer compounding system

Implements the compounding layers:
1. Monetary Layer - Capital compounding and growth
2. Worker Layer - Worker ant multiplication and management
3. Carwash Layer - Reset and cleanup mechanisms
4. Intelligence Layer - AI learning and improvement compounding
5. Data Layer - Trading data accumulation and pattern learning
"""

import asyncio
import logging
import math
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class CompoundingLayer(Enum):
    """The five compounding layers"""
    MONETARY = "monetary"
    WORKER = "worker"
    CARWASH = "carwash"
    INTELLIGENCE = "intelligence"
    DATA = "data"

@dataclass
class MonetaryCompounding:
    """Monetary layer compounding metrics"""
    initial_capital: float = 0.0
    current_capital: float = 0.0
    total_profit: float = 0.0
    compound_rate: float = 1.0
    reinvestment_rate: float = 0.8
    growth_cycles: int = 0
    last_compound: float = 0.0
    
    def calculate_growth_rate(self) -> float:
        """Calculate current growth rate"""
        if self.initial_capital > 0:
            return (self.current_capital / self.initial_capital) - 1.0
        return 0.0
    
    def apply_compound(self, profit: float) -> float:
        """Apply compounding to profit"""
        compounded_profit = profit * self.compound_rate
        self.total_profit += compounded_profit
        self.current_capital += compounded_profit
        self.growth_cycles += 1
        self.last_compound = time.time()
        
        # Increase compound rate based on success
        if profit > 0:
            self.compound_rate = min(2.0, self.compound_rate * 1.01)  # Max 2x multiplier
        else:
            self.compound_rate = max(0.8, self.compound_rate * 0.99)  # Min 0.8x multiplier
        
        return compounded_profit

@dataclass
class WorkerCompounding:
    """Worker layer compounding metrics"""
    active_workers: int = 1
    max_workers: int = 50
    split_threshold: float = 2.0
    merge_threshold: float = 0.1
    worker_efficiency: float = 1.0
    collective_performance: float = 0.0
    split_events: int = 0
    merge_events: int = 0
    
    def should_create_workers(self, capital: float) -> int:
        """Calculate how many new workers should be created"""
        if capital >= self.split_threshold and self.active_workers < self.max_workers:
            new_workers = min(5, (self.max_workers - self.active_workers))
            return new_workers
        return 0
    
    def calculate_worker_multiplier(self) -> float:
        """Calculate multiplier based on worker count and efficiency"""
        base_multiplier = 1.0 + (self.active_workers - 1) * 0.1  # 10% per additional worker
        efficiency_multiplier = self.worker_efficiency
        return base_multiplier * efficiency_multiplier

@dataclass
class CarwashCompounding:
    """Carwash layer - reset and cleanup compounding"""
    cleanup_cycles: int = 0
    reset_events: int = 0
    efficiency_gain: float = 0.0
    last_cleanup: float = 0.0
    cleanup_interval: float = 3600.0  # 1 hour
    memory_optimization: float = 1.0
    
    def should_cleanup(self) -> bool:
        """Determine if cleanup should be performed"""
        return (time.time() - self.last_cleanup) >= self.cleanup_interval
    
    def perform_cleanup(self) -> Dict[str, Any]:
        """Perform cleanup and return efficiency gains"""
        self.cleanup_cycles += 1
        self.last_cleanup = time.time()
        
        # Increase efficiency slightly with each cleanup
        self.efficiency_gain += 0.02  # 2% improvement per cleanup
        self.memory_optimization = min(1.5, self.memory_optimization * 1.01)
        
        return {
            "efficiency_gain": self.efficiency_gain,
            "memory_optimization": self.memory_optimization,
            "cleanup_cycle": self.cleanup_cycles
        }

@dataclass 
class IntelligenceCompounding:
    """Intelligence layer compounding metrics"""
    learning_iterations: int = 0
    intelligence_score: float = 0.5
    pattern_recognition: float = 0.5
    decision_accuracy: float = 0.5
    adaptation_rate: float = 0.1
    knowledge_base_size: int = 0
    successful_predictions: int = 0
    
    def apply_learning(self, outcome_success: bool, prediction_confidence: float) -> float:
        """Apply learning from trading outcome"""
        self.learning_iterations += 1
        
        if outcome_success:
            self.successful_predictions += 1
            # Improve intelligence based on confidence and success
            improvement = prediction_confidence * self.adaptation_rate
            self.intelligence_score = min(1.0, self.intelligence_score + improvement)
            self.decision_accuracy = min(1.0, self.decision_accuracy + improvement * 0.5)
        else:
            # Slight penalty for failures, but still learning
            penalty = prediction_confidence * self.adaptation_rate * 0.3
            self.intelligence_score = max(0.1, self.intelligence_score - penalty)
        
        # Pattern recognition improves with each iteration
        self.pattern_recognition = min(1.0, self.pattern_recognition + 0.001)
        self.knowledge_base_size += 1
        
        return self.intelligence_score

@dataclass
class DataCompounding:
    """Data layer compounding metrics"""
    total_data_points: int = 0
    pattern_database_size: int = 0
    successful_patterns: int = 0
    data_quality_score: float = 0.5
    prediction_models_count: int = 0
    data_correlation_strength: float = 0.0
    last_data_update: float = 0.0
    
    def add_data_point(self, trade_data: Dict[str, Any], success: bool) -> Dict[str, Any]:
        """Add new trading data point and update compounding metrics"""
        self.total_data_points += 1
        self.last_data_update = time.time()
        
        # Improve data quality with successful trades
        if success:
            self.successful_patterns += 1
            quality_improvement = 0.01 * (self.successful_patterns / self.total_data_points)
            self.data_quality_score = min(1.0, self.data_quality_score + quality_improvement)
        
        # Every 100 data points, create a new pattern
        if self.total_data_points % 100 == 0:
            self.pattern_database_size += 1
            self.prediction_models_count += 1
        
        # Calculate correlation strength
        success_rate = self.successful_patterns / self.total_data_points
        self.data_correlation_strength = success_rate * self.data_quality_score
        
        return {
            "data_points": self.total_data_points,
            "quality_score": self.data_quality_score,
            "correlation_strength": self.data_correlation_strength,
            "patterns_discovered": self.pattern_database_size
        }

class CompoundingLogic:
    """Main compounding logic coordinator for all 5 layers"""
    
    def __init__(self):
        # Initialize all compounding layers
        self.monetary = MonetaryCompounding()
        self.worker = WorkerCompounding()
        self.carwash = CarwashCompounding()
        self.intelligence = IntelligenceCompounding()
        self.data = DataCompounding()
        
        # Performance tracking
        self.compounding_metrics: Dict[str, Any] = {}
        self.layer_interactions: Dict[str, float] = {}
        
        logger.info("CompoundingLogic initialized with all 5 layers")
    
    async def calculate_multiplier(
        self, 
        win_rate: float, 
        profit_factor: float, 
        trades_completed: int,
        confidence: float
    ) -> float:
        """Calculate overall compounding multiplier from all layers"""
        try:
            # Layer 1: Monetary compounding
            monetary_multiplier = self._calculate_monetary_multiplier(profit_factor)
            
            # Layer 2: Worker compounding
            worker_multiplier = self.worker.calculate_worker_multiplier()
            
            # Layer 3: Carwash efficiency
            carwash_multiplier = self.carwash.memory_optimization
            
            # Layer 4: Intelligence compounding
            intelligence_multiplier = 1.0 + (self.intelligence.intelligence_score - 0.5)
            
            # Layer 5: Data compounding
            data_multiplier = 1.0 + (self.data.data_correlation_strength * 0.5)
            
            # Combine all multipliers with weights
            weights = {
                CompoundingLayer.MONETARY: 0.30,
                CompoundingLayer.WORKER: 0.25,
                CompoundingLayer.CARWASH: 0.15,
                CompoundingLayer.INTELLIGENCE: 0.20,
                CompoundingLayer.DATA: 0.10
            }
            
            final_multiplier = (
                monetary_multiplier * weights[CompoundingLayer.MONETARY] +
                worker_multiplier * weights[CompoundingLayer.WORKER] +
                carwash_multiplier * weights[CompoundingLayer.CARWASH] +
                intelligence_multiplier * weights[CompoundingLayer.INTELLIGENCE] +
                data_multiplier * weights[CompoundingLayer.DATA]
            )
            
            # Apply confidence scaling
            confidence_adjustment = 0.5 + (confidence * 0.5)  # Scale between 0.5 and 1.0
            final_multiplier *= confidence_adjustment
            
            # Cap the multiplier to reasonable bounds
            final_multiplier = max(0.1, min(3.0, final_multiplier))
            
            # Store metrics for analysis
            self.compounding_metrics.update({
                "monetary_multiplier": monetary_multiplier,
                "worker_multiplier": worker_multiplier,
                "carwash_multiplier": carwash_multiplier,
                "intelligence_multiplier": intelligence_multiplier,
                "data_multiplier": data_multiplier,
                "final_multiplier": final_multiplier,
                "confidence_adjustment": confidence_adjustment,
                "timestamp": time.time()
            })
            
            logger.debug(f"Calculated compounding multiplier: {final_multiplier:.3f}")
            return final_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating compounding multiplier: {e}")
            return 1.0  # Safe fallback
    
    def _calculate_monetary_multiplier(self, profit_factor: float) -> float:
        """Calculate monetary layer multiplier"""
        # Base multiplier from profit factor
        base = 1.0 + (profit_factor - 1.0) * 0.5  # Scale profit factor influence
        
        # Apply growth rate influence
        growth_rate = self.monetary.calculate_growth_rate()
        growth_influence = 1.0 + (growth_rate * 0.3)  # 30% influence from growth
        
        return base * growth_influence
    
    async def update_metrics(
        self, 
        ant_id: str, 
        trade_result: Dict[str, Any], 
        current_performance: Any
    ):
        """Update all compounding layers with new trade result"""
        try:
            profit = trade_result.get("profit", 0.0)
            success = profit > 0
            confidence = trade_result.get("confidence", 0.5)
            
            # Update Layer 1: Monetary
            if profit != 0:
                compounded_profit = self.monetary.apply_compound(profit)
                logger.debug(f"Applied monetary compounding: {profit} -> {compounded_profit}")
            
            # Update Layer 2: Worker (handled by parent Queen)
            # This is managed at the Queen level for worker creation/destruction
            
            # Update Layer 3: Carwash
            if self.carwash.should_cleanup():
                cleanup_result = self.carwash.perform_cleanup()
                logger.debug(f"Performed carwash cleanup: {cleanup_result}")
            
            # Update Layer 4: Intelligence
            intelligence_update = self.intelligence.apply_learning(success, confidence)
            
            # Update Layer 5: Data
            data_update = self.data.add_data_point(trade_result, success)
            
            # Calculate layer interactions
            await self._update_layer_interactions()
            
        except Exception as e:
            logger.error(f"Error updating compounding metrics: {e}")
    
    async def _update_layer_interactions(self):
        """Update interactions between compounding layers for flywheel effect"""
        try:
            # Intelligence-Data interaction
            intel_data_synergy = (
                self.intelligence.intelligence_score * self.data.data_quality_score
            )
            
            # Monetary-Worker interaction
            monetary_worker_synergy = (
                self.monetary.compound_rate * self.worker.worker_efficiency
            )
            
            # Carwash enhances all other layers
            carwash_enhancement = self.carwash.efficiency_gain
            
            self.layer_interactions = {
                "intelligence_data_synergy": intel_data_synergy,
                "monetary_worker_synergy": monetary_worker_synergy,
                "carwash_enhancement": carwash_enhancement,
                "overall_synergy": (intel_data_synergy + monetary_worker_synergy) * (1 + carwash_enhancement)
            }
            
        except Exception as e:
            logger.error(f"Error updating layer interactions: {e}")
    
    def get_worker_split_recommendation(self, current_capital: float) -> Dict[str, Any]:
        """Get recommendation for worker splitting based on compounding analysis"""
        new_workers = self.worker.should_create_workers(current_capital)
        
        if new_workers > 0:
            return {
                "should_split": True,
                "new_workers": new_workers,
                "capital_per_worker": current_capital / (new_workers + 1),
                "expected_multiplier": self.worker.calculate_worker_multiplier() * 1.2,
                "reasoning": f"Capital threshold reached, creating {new_workers} workers"
            }
        
        return {
            "should_split": False,
            "reasoning": "Capital threshold not met or max workers reached"
        }
    
    def get_compounding_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all compounding layers"""
        return {
            "layers": {
                "monetary": {
                    "growth_rate": self.monetary.calculate_growth_rate(),
                    "compound_rate": self.monetary.compound_rate,
                    "total_profit": self.monetary.total_profit,
                    "growth_cycles": self.monetary.growth_cycles
                },
                "worker": {
                    "active_workers": self.worker.active_workers,
                    "efficiency": self.worker.worker_efficiency,
                    "multiplier": self.worker.calculate_worker_multiplier(),
                    "split_events": self.worker.split_events
                },
                "carwash": {
                    "cleanup_cycles": self.carwash.cleanup_cycles,
                    "efficiency_gain": self.carwash.efficiency_gain,
                    "memory_optimization": self.carwash.memory_optimization,
                    "next_cleanup": self.carwash.last_cleanup + self.carwash.cleanup_interval
                },
                "intelligence": {
                    "score": self.intelligence.intelligence_score,
                    "learning_iterations": self.intelligence.learning_iterations,
                    "decision_accuracy": self.intelligence.decision_accuracy,
                    "successful_predictions": self.intelligence.successful_predictions
                },
                "data": {
                    "total_points": self.data.total_data_points,
                    "quality_score": self.data.data_quality_score,
                    "correlation_strength": self.data.data_correlation_strength,
                    "pattern_database_size": self.data.pattern_database_size
                }
            },
            "interactions": self.layer_interactions,
            "latest_metrics": self.compounding_metrics
        }
    
    async def reset_layer(self, layer: CompoundingLayer) -> bool:
        """Reset a specific compounding layer (carwash function)"""
        try:
            if layer == CompoundingLayer.MONETARY:
                # Partial reset - keep some progress
                self.monetary.compound_rate = max(1.0, self.monetary.compound_rate * 0.8)
                self.monetary.growth_cycles = 0
            
            elif layer == CompoundingLayer.WORKER:
                self.worker.worker_efficiency = 1.0
                self.worker.collective_performance = 0.0
            
            elif layer == CompoundingLayer.CARWASH:
                self.carwash.efficiency_gain = 0.0
                self.carwash.memory_optimization = 1.0
            
            elif layer == CompoundingLayer.INTELLIGENCE:
                # Retain some learning but reset performance metrics
                self.intelligence.intelligence_score *= 0.9
                self.intelligence.decision_accuracy *= 0.9
            
            elif layer == CompoundingLayer.DATA:
                # Archive old data and start fresh tracking
                self.data.data_quality_score = 0.5
                self.data.data_correlation_strength = 0.0
            
            self.carwash.reset_events += 1
            logger.info(f"Reset compounding layer: {layer.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting layer {layer.value}: {e}")
            return False 