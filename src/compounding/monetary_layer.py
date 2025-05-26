"""
Monetary Layer - Capital growth compounding

Implements monetary compounding effects that create exponential capital growth
through reinvestment, profit compounding, and adaptive compound rates.
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
class CompoundingRecord:
    """Represents a compounding event record"""
    record_id: str
    timestamp: float
    principal_amount: float
    compound_rate: float
    compounded_amount: float
    source_component: str
    compound_type: str  # "profit", "reinvestment", "bonus"

@dataclass
class MonetaryMetrics:
    """Tracks monetary compounding metrics"""
    total_principal: float = 0.0
    total_compounded: float = 0.0
    compound_rate: float = 1.05  # 5% default
    effective_compound_rate: float = 1.05
    compound_frequency: float = 1.0  # Daily by default
    total_compound_events: int = 0
    compound_efficiency: float = 1.0

class MonetaryLayer:
    """Implements monetary compounding layer of the system"""
    
    def __init__(self):
        # Compounding configuration
        self.base_compound_rate = 1.05  # 5% base rate
        self.max_compound_rate = 1.25   # 25% maximum rate
        self.min_compound_rate = 1.01   # 1% minimum rate
        self.adaptation_factor = 0.1    # Rate adaptation speed
        
        # Monetary tracking
        self.metrics = MonetaryMetrics()
        self.compounding_history: deque = deque(maxlen=1000)
        self.compound_pools: Dict[str, float] = {}
        
        # Adaptive compounding
        self.performance_scores: deque = deque(maxlen=100)
        self.compound_multipliers: Dict[str, float] = {}
        self.reinvestment_pools: Dict[str, float] = {}
        
        # Cycle management
        self.last_compound_time = 0.0
        self.compound_interval = 3600.0  # 1 hour
        self.total_capital_added = 0.0
        self.total_capital_compounded = 0.0
        
        logger.info("MonetaryLayer initialized")
    
    async def initialize(self) -> bool:
        """Initialize the monetary compounding layer"""
        try:
            # Initialize compound pools
            await self._initialize_compound_pools()
            
            # Initialize adaptive mechanisms
            await self._initialize_adaptive_compounding()
            
            # Set initial metrics
            self.metrics.compound_rate = self.base_compound_rate
            self.metrics.effective_compound_rate = self.base_compound_rate
            
            logger.info("MonetaryLayer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MonetaryLayer: {e}")
            return False
    
    async def compound_capital(
        self, 
        principal: float, 
        source_component: str,
        compound_type: str = "profit"
    ) -> float:
        """Compound capital from a specific source"""
        try:
            # Calculate adaptive compound rate for this component
            compound_rate = await self._calculate_adaptive_rate(source_component, principal)
            
            # Apply compounding
            compounded_amount = principal * compound_rate
            compound_gain = compounded_amount - principal
            
            # Record compounding event
            record_id = f"compound_{int(time.time())}_{source_component}"
            compound_record = CompoundingRecord(
                record_id=record_id,
                timestamp=time.time(),
                principal_amount=principal,
                compound_rate=compound_rate,
                compounded_amount=compounded_amount,
                source_component=source_component,
                compound_type=compound_type
            )
            
            self.compounding_history.append(compound_record)
            
            # Update metrics
            self.metrics.total_principal += principal
            self.metrics.total_compounded += compound_gain
            self.metrics.total_compound_events += 1
            
            # Update compound pools
            if source_component not in self.compound_pools:
                self.compound_pools[source_component] = 0.0
            self.compound_pools[source_component] += compound_gain
            
            # Track performance for adaptive compounding
            await self._track_compound_performance(compound_rate, compound_gain)
            
            logger.debug(f"Compounded {principal} SOL from {source_component} at {compound_rate:.3f}x rate")
            return compounded_amount
            
        except Exception as e:
            logger.error(f"Error compounding capital: {e}")
            return principal  # Return original amount on error
    
    async def execute_compound_cycle(self) -> Dict[str, Any]:
        """Execute a monetary compounding cycle"""
        try:
            cycle_results = {
                "cycle_timestamp": time.time(),
                "compounds_processed": 0,
                "total_compound_gain": 0.0,
                "adaptive_rate_updates": 0,
                "reinvestment_executed": 0.0
            }
            
            # Check if it's time for compounding cycle
            if time.time() - self.last_compound_time < self.compound_interval:
                return cycle_results
            
            # Process reinvestment pools
            reinvestment_results = await self._process_reinvestment_pools()
            cycle_results["reinvestment_executed"] = reinvestment_results["total_reinvested"]
            
            # Update adaptive compound rates
            rate_updates = await self._update_adaptive_rates()
            cycle_results["adaptive_rate_updates"] = rate_updates["rates_updated"]
            
            # Calculate compound efficiency
            efficiency_update = await self._calculate_compound_efficiency()
            
            # Update system-wide compound rate
            await self._update_system_compound_rate()
            
            self.last_compound_time = time.time()
            
            return cycle_results
            
        except Exception as e:
            logger.error(f"Error in monetary compound cycle: {e}")
            return {"error": str(e)}
    
    async def _calculate_adaptive_rate(self, component: str, principal: float) -> float:
        """Calculate adaptive compound rate for a component"""
        try:
            # Start with base rate
            adaptive_rate = self.base_compound_rate
            
            # Apply component-specific multiplier
            if component in self.compound_multipliers:
                adaptive_rate *= self.compound_multipliers[component]
            
            # Apply principal size bonus (larger amounts get slight bonus)
            principal_bonus = 1.0 + min(0.1, math.log10(max(1, principal)) * 0.02)
            adaptive_rate *= principal_bonus
            
            # Apply performance-based adjustment
            if self.performance_scores:
                avg_performance = sum(self.performance_scores) / len(self.performance_scores)
                performance_adjustment = 1.0 + (avg_performance - 1.0) * self.adaptation_factor
                adaptive_rate *= performance_adjustment
            
            # Apply compound frequency boost
            frequency_boost = 1.0 + (self.metrics.compound_frequency - 1.0) * 0.05
            adaptive_rate *= frequency_boost
            
            # Ensure rate stays within bounds
            adaptive_rate = max(self.min_compound_rate, min(self.max_compound_rate, adaptive_rate))
            
            return adaptive_rate
            
        except Exception as e:
            logger.error(f"Error calculating adaptive rate: {e}")
            return self.base_compound_rate
    
    async def _track_compound_performance(self, compound_rate: float, compound_gain: float):
        """Track compounding performance for adaptive improvements"""
        try:
            # Calculate performance score based on gain vs expected
            expected_gain = compound_gain / (compound_rate - 1.0) if compound_rate > 1.0 else 1.0
            performance_score = min(2.0, compound_gain / max(0.01, expected_gain))
            
            self.performance_scores.append(performance_score)
            
            # Update compound efficiency
            if self.metrics.total_principal > 0:
                self.metrics.compound_efficiency = (
                    self.metrics.total_compounded / self.metrics.total_principal
                )
            
        except Exception as e:
            logger.error(f"Error tracking compound performance: {e}")
    
    async def _process_reinvestment_pools(self) -> Dict[str, Any]:
        """Process accumulated reinvestment pools"""
        try:
            reinvestment_results = {
                "pools_processed": 0,
                "total_reinvested": 0.0,
                "reinvestment_breakdown": {}
            }
            
            for source, pool_amount in self.reinvestment_pools.items():
                if pool_amount >= 0.1:  # Minimum reinvestment threshold
                    # Compound the pool amount
                    compounded_amount = await self.compound_capital(
                        pool_amount, source, "reinvestment"
                    )
                    
                    gain = compounded_amount - pool_amount
                    reinvestment_results["total_reinvested"] += gain
                    reinvestment_results["reinvestment_breakdown"][source] = gain
                    reinvestment_results["pools_processed"] += 1
                    
                    # Reset pool
                    self.reinvestment_pools[source] = 0.0
            
            return reinvestment_results
            
        except Exception as e:
            logger.error(f"Error processing reinvestment pools: {e}")
            return {"pools_processed": 0, "total_reinvested": 0.0}
    
    async def _update_adaptive_rates(self) -> Dict[str, Any]:
        """Update adaptive compound rates based on performance"""
        try:
            rate_updates = {
                "rates_updated": 0,
                "average_adjustment": 0.0,
                "component_adjustments": {}
            }
            
            # Update component-specific multipliers based on recent performance
            component_performance = self._calculate_component_performance()
            
            total_adjustment = 0.0
            for component, performance in component_performance.items():
                if component not in self.compound_multipliers:
                    self.compound_multipliers[component] = 1.0
                
                # Adjust multiplier based on performance
                if performance > 1.1:  # Above 10% performance
                    adjustment = min(0.05, (performance - 1.0) * 0.1)  # Up to 5% boost
                    self.compound_multipliers[component] += adjustment
                elif performance < 0.9:  # Below 90% performance
                    adjustment = max(-0.05, (performance - 1.0) * 0.1)  # Up to 5% reduction
                    self.compound_multipliers[component] += adjustment
                else:
                    adjustment = 0.0
                
                # Keep multipliers within reasonable bounds
                self.compound_multipliers[component] = max(0.5, min(2.0, self.compound_multipliers[component]))
                
                if adjustment != 0:
                    rate_updates["rates_updated"] += 1
                    rate_updates["component_adjustments"][component] = adjustment
                    total_adjustment += abs(adjustment)
            
            if rate_updates["rates_updated"] > 0:
                rate_updates["average_adjustment"] = total_adjustment / rate_updates["rates_updated"]
            
            return rate_updates
            
        except Exception as e:
            logger.error(f"Error updating adaptive rates: {e}")
            return {"rates_updated": 0}
    
    def _calculate_component_performance(self) -> Dict[str, float]:
        """Calculate performance score for each component"""
        component_performance = {}
        
        # Group recent compounding records by component
        recent_records = [r for r in self.compounding_history if time.time() - r.timestamp < 3600]
        
        component_records = {}
        for record in recent_records:
            if record.source_component not in component_records:
                component_records[record.source_component] = []
            component_records[record.source_component].append(record)
        
        # Calculate performance for each component
        for component, records in component_records.items():
            if records:
                avg_rate = sum(r.compound_rate for r in records) / len(records)
                avg_gain_ratio = sum((r.compounded_amount - r.principal_amount) / r.principal_amount for r in records) / len(records)
                
                # Performance score combines rate effectiveness and gain efficiency
                performance_score = (avg_rate + avg_gain_ratio) / 2.0
                component_performance[component] = performance_score
        
        return component_performance
    
    async def _calculate_compound_efficiency(self) -> float:
        """Calculate overall compound efficiency"""
        try:
            if self.metrics.total_principal <= 0:
                return 1.0
            
            # Base efficiency from compound ratio
            base_efficiency = self.metrics.total_compounded / self.metrics.total_principal
            
            # Frequency efficiency boost
            frequency_efficiency = min(2.0, self.metrics.compound_frequency)
            
            # Recent performance efficiency
            recent_efficiency = 1.0
            if self.performance_scores:
                recent_efficiency = sum(self.performance_scores[-10:]) / len(self.performance_scores[-10:])
            
            # Combined efficiency
            self.metrics.compound_efficiency = base_efficiency * frequency_efficiency * recent_efficiency
            
            return self.metrics.compound_efficiency
            
        except Exception as e:
            logger.error(f"Error calculating compound efficiency: {e}")
            return 1.0
    
    async def _update_system_compound_rate(self):
        """Update system-wide effective compound rate"""
        try:
            if not self.compound_multipliers:
                self.metrics.effective_compound_rate = self.base_compound_rate
                return
            
            # Calculate weighted average of component multipliers
            total_weight = sum(self.compound_pools.values())
            if total_weight > 0:
                weighted_multiplier = sum(
                    mult * self.compound_pools.get(comp, 0) / total_weight
                    for comp, mult in self.compound_multipliers.items()
                )
            else:
                weighted_multiplier = sum(self.compound_multipliers.values()) / len(self.compound_multipliers)
            
            # Apply to base rate
            self.metrics.effective_compound_rate = self.base_compound_rate * weighted_multiplier
            
            # Update metrics
            self.metrics.compound_rate = self.metrics.effective_compound_rate
            
        except Exception as e:
            logger.error(f"Error updating system compound rate: {e}")
    
    async def add_to_reinvestment_pool(self, component: str, amount: float):
        """Add capital to reinvestment pool for later compounding"""
        if component not in self.reinvestment_pools:
            self.reinvestment_pools[component] = 0.0
        
        self.reinvestment_pools[component] += amount
        self.total_capital_added += amount
    
    async def _initialize_compound_pools(self):
        """Initialize compound pools for different components"""
        self.compound_pools = {
            "worker_ants": 0.0,
            "ant_queen": 0.0,
            "ant_drone": 0.0,
            "compounding_system": 0.0,
            "flywheel_effects": 0.0
        }
    
    async def _initialize_adaptive_compounding(self):
        """Initialize adaptive compounding mechanisms"""
        self.compound_multipliers = {
            "worker_ants": 1.0,
            "ant_queen": 1.05,    # Slight boost for Queen
            "ant_drone": 1.02,    # Small boost for AI
            "compounding_system": 1.1,  # Higher boost for compounding
            "flywheel_effects": 1.15    # Highest boost for flywheel
        }
        
        self.reinvestment_pools = {
            component: 0.0 for component in self.compound_pools.keys()
        }
    
    def get_monetary_summary(self) -> Dict[str, Any]:
        """Get comprehensive monetary layer summary"""
        return {
            "compound_metrics": {
                "total_principal": self.metrics.total_principal,
                "total_compounded": self.metrics.total_compounded,
                "compound_rate": self.metrics.compound_rate,
                "effective_compound_rate": self.metrics.effective_compound_rate,
                "compound_efficiency": self.metrics.compound_efficiency,
                "total_compound_events": self.metrics.total_compound_events
            },
            "compound_pools": self.compound_pools.copy(),
            "reinvestment_pools": self.reinvestment_pools.copy(),
            "component_multipliers": self.compound_multipliers.copy(),
            "performance_metrics": {
                "recent_performance_scores": list(self.performance_scores)[-10:],
                "total_capital_added": self.total_capital_added,
                "compound_history_length": len(self.compounding_history)
            }
        }
    
    def get_compound_rate_for_component(self, component: str) -> float:
        """Get current compound rate for a specific component"""
        base_rate = self.metrics.effective_compound_rate
        multiplier = self.compound_multipliers.get(component, 1.0)
        return base_rate * multiplier
    
    async def cleanup(self):
        """Cleanup monetary layer resources"""
        try:
            # Clear large data structures
            self.compounding_history.clear()
            self.performance_scores.clear()
            
            logger.info("MonetaryLayer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during MonetaryLayer cleanup: {e}") 