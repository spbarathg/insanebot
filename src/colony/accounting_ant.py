"""
Accounting Ant - Capital tracking and hedging

Manages capital tracking, hedging, and financial oversight for the Queen.
Provides risk management and capital preservation functions.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .base_ant import BaseAnt, AntRole, AntStatus

logger = logging.getLogger(__name__)

@dataclass
class HedgePosition:
    """Represents a hedge position"""
    position_id: str
    amount: float
    entry_price: float
    hedge_type: str  # "long", "short", "stablecoin"
    created_at: float
    target_hedge_ratio: float
    current_value: float = 0.0

class AccountingAnt(BaseAnt):
    """Capital tracking and hedging for risk management"""
    
    def __init__(self, ant_id: str, parent_id: str):
        super().__init__(ant_id, AntRole.ACCOUNTING, parent_id)
        
        # Accounting-specific attributes
        self.tracked_capital: Dict[str, float] = {}  # Track capital by worker/source
        self.hedge_positions: Dict[str, HedgePosition] = {}
        self.capital_history: List[Dict] = []
        
        # Risk management settings
        self.hedge_percentage = self.config["hedge_percentage"]  # 10% default
        self.risk_thresholds = {
            "max_drawdown": 0.2,  # 20% max drawdown
            "hedge_trigger": 0.1,  # Hedge when 10% loss
            "rebalance_threshold": 0.05  # Rebalance when 5% off target
        }
        
        # Accounting metrics
        self.total_tracked_capital: float = 0.0
        self.total_hedged_amount: float = 0.0
        self.hedge_effectiveness: float = 0.0
        self.capital_at_risk: float = 0.0
        
        logger.info(f"AccountingAnt {ant_id} created for capital tracking")
    
    async def initialize(self) -> bool:
        """Initialize the Accounting Ant"""
        try:
            # Initialize capital tracking
            await self._initialize_capital_tracking()
            
            # Initialize hedge tracking
            await self._initialize_hedge_system()
            
            logger.info(f"AccountingAnt {self.ant_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AccountingAnt {self.ant_id}: {e}")
            self.status = AntStatus.ERROR
            return False
    
    async def execute_cycle(self) -> Dict[str, Any]:
        """Execute accounting and hedging cycle"""
        if self.status != AntStatus.ACTIVE:
            return {"status": "inactive", "reason": f"Accounting Ant status: {self.status.value}"}
        
        try:
            self.update_activity()
            
            # Update capital tracking
            capital_update = await self._update_capital_tracking()
            
            # Assess hedging needs
            hedge_assessment = await self._assess_hedging_needs()
            
            # Execute hedging operations if needed
            hedge_operations = await self._execute_hedging_operations(hedge_assessment)
            
            # Update risk metrics
            risk_update = await self._update_risk_metrics()
            
            # Record capital history
            await self._record_capital_snapshot()
            
            return {
                "status": "active",
                "capital_tracking": capital_update,
                "hedge_assessment": hedge_assessment,
                "hedge_operations": hedge_operations,
                "risk_metrics": risk_update,
                "accounting_summary": self._get_accounting_summary()
            }
            
        except Exception as e:
            logger.error(f"Error in AccountingAnt {self.ant_id} cycle: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _initialize_capital_tracking(self):
        """Initialize capital tracking system"""
        self.tracked_capital = {
            "queen_capital": 0.0,
            "worker_capital": 0.0,
            "hedged_capital": 0.0,
            "total_profit": 0.0,
            "unrealized_pnl": 0.0
        }
    
    async def _initialize_hedge_system(self):
        """Initialize hedging system"""
        # For now, we'll track hedge positions in memory
        # In production, this would connect to actual hedging mechanisms
        self.hedge_positions = {}
    
    async def _update_capital_tracking(self) -> Dict[str, Any]:
        """Update capital tracking across all sources"""
        try:
            # This would be called by the Queen to update capital amounts
            # For now, we'll maintain the tracking structure
            
            self.total_tracked_capital = sum(self.tracked_capital.values())
            
            return {
                "total_tracked": self.total_tracked_capital,
                "breakdown": self.tracked_capital.copy(),
                "tracking_accuracy": 99.5  # Placeholder for tracking accuracy
            }
            
        except Exception as e:
            logger.error(f"Error updating capital tracking: {e}")
            return {"error": str(e)}
    
    async def _assess_hedging_needs(self) -> Dict[str, Any]:
        """Assess if hedging is needed based on risk thresholds"""
        try:
            assessment = {
                "hedge_needed": False,
                "current_hedge_ratio": 0.0,
                "target_hedge_ratio": self.hedge_percentage,
                "risk_level": "low",
                "recommendations": []
            }
            
            # Calculate current hedge ratio
            if self.total_tracked_capital > 0:
                assessment["current_hedge_ratio"] = self.total_hedged_amount / self.total_tracked_capital
            
            # Check if we need to hedge more
            if assessment["current_hedge_ratio"] < self.hedge_percentage:
                assessment["hedge_needed"] = True
                assessment["recommendations"].append("Increase hedge positions")
            
            # Assess risk level based on capital at risk
            risk_ratio = self.capital_at_risk / max(1.0, self.total_tracked_capital)
            if risk_ratio > self.risk_thresholds["max_drawdown"]:
                assessment["risk_level"] = "high"
                assessment["recommendations"].append("Immediate hedging required")
            elif risk_ratio > self.risk_thresholds["hedge_trigger"]:
                assessment["risk_level"] = "medium"
                assessment["recommendations"].append("Consider additional hedging")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing hedging needs: {e}")
            return {"error": str(e)}
    
    async def _execute_hedging_operations(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hedging operations based on assessment"""
        try:
            operations_result = {
                "operations_executed": 0,
                "new_hedge_positions": [],
                "closed_positions": [],
                "total_hedged_amount": self.total_hedged_amount
            }
            
            if assessment.get("hedge_needed", False):
                # Calculate hedge amount needed
                target_hedge_amount = self.total_tracked_capital * self.hedge_percentage
                additional_hedge_needed = target_hedge_amount - self.total_hedged_amount
                
                if additional_hedge_needed > 0.01:  # Minimum hedge amount
                    hedge_position = await self._create_hedge_position(additional_hedge_needed)
                    if hedge_position:
                        operations_result["new_hedge_positions"].append(hedge_position.position_id)
                        operations_result["operations_executed"] += 1
                        self.total_hedged_amount += hedge_position.amount
            
            # Check for positions that need rebalancing
            rebalance_operations = await self._rebalance_hedge_positions()
            operations_result["operations_executed"] += rebalance_operations
            
            operations_result["total_hedged_amount"] = self.total_hedged_amount
            
            return operations_result
            
        except Exception as e:
            logger.error(f"Error executing hedging operations: {e}")
            return {"error": str(e)}
    
    async def _create_hedge_position(self, amount: float) -> Optional[HedgePosition]:
        """Create a new hedge position"""
        try:
            position_id = f"hedge_{int(time.time())}_{len(self.hedge_positions)}"
            
            # For this implementation, we'll create a conceptual hedge position
            # In production, this would execute actual hedging trades
            hedge_position = HedgePosition(
                position_id=position_id,
                amount=amount,
                entry_price=100.0,  # Placeholder SOL price
                hedge_type="stablecoin",  # Simplified hedge type
                created_at=time.time(),
                target_hedge_ratio=self.hedge_percentage,
                current_value=amount
            )
            
            self.hedge_positions[position_id] = hedge_position
            
            logger.info(f"AccountingAnt {self.ant_id} created hedge position {position_id} for {amount} SOL")
            return hedge_position
            
        except Exception as e:
            logger.error(f"Error creating hedge position: {e}")
            return None
    
    async def _rebalance_hedge_positions(self) -> int:
        """Rebalance existing hedge positions"""
        try:
            rebalanced_count = 0
            
            for position in self.hedge_positions.values():
                # Check if position needs rebalancing
                current_hedge_ratio = position.current_value / max(1.0, self.total_tracked_capital)
                target_ratio = position.target_hedge_ratio
                
                if abs(current_hedge_ratio - target_ratio) > self.risk_thresholds["rebalance_threshold"]:
                    # Rebalance position (simplified)
                    new_target_amount = self.total_tracked_capital * target_ratio
                    position.current_value = new_target_amount
                    rebalanced_count += 1
                    
                    logger.debug(f"Rebalanced hedge position {position.position_id}")
            
            return rebalanced_count
            
        except Exception as e:
            logger.error(f"Error rebalancing hedge positions: {e}")
            return 0
    
    async def _update_risk_metrics(self) -> Dict[str, Any]:
        """Update risk management metrics"""
        try:
            # Calculate capital at risk (unhedged capital)
            self.capital_at_risk = max(0, self.total_tracked_capital - self.total_hedged_amount)
            
            # Calculate hedge effectiveness
            if self.total_hedged_amount > 0:
                # Simplified effectiveness calculation
                self.hedge_effectiveness = min(1.0, self.total_hedged_amount / (self.total_tracked_capital * self.hedge_percentage))
            else:
                self.hedge_effectiveness = 0.0
            
            # Update capital metrics in base class
            self.capital.hedged_amount = self.total_hedged_amount
            
            return {
                "capital_at_risk": self.capital_at_risk,
                "hedge_effectiveness": self.hedge_effectiveness,
                "hedge_ratio": self.total_hedged_amount / max(1.0, self.total_tracked_capital),
                "risk_score": self.capital_at_risk / max(1.0, self.total_tracked_capital)
            }
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
            return {"error": str(e)}
    
    async def _record_capital_snapshot(self):
        """Record a snapshot of current capital state"""
        try:
            snapshot = {
                "timestamp": time.time(),
                "total_capital": self.total_tracked_capital,
                "hedged_amount": self.total_hedged_amount,
                "capital_at_risk": self.capital_at_risk,
                "hedge_positions_count": len(self.hedge_positions),
                "hedge_effectiveness": self.hedge_effectiveness
            }
            
            self.capital_history.append(snapshot)
            
            # Keep only last 1000 snapshots to manage memory
            if len(self.capital_history) > 1000:
                self.capital_history = self.capital_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error recording capital snapshot: {e}")
    
    def update_tracked_capital(self, source: str, amount: float):
        """Update tracked capital for a specific source"""
        self.tracked_capital[source] = amount
        self.total_tracked_capital = sum(self.tracked_capital.values())
    
    def add_capital_source(self, source: str, initial_amount: float = 0.0):
        """Add a new capital source to track"""
        self.tracked_capital[source] = initial_amount
        self.total_tracked_capital = sum(self.tracked_capital.values())
    
    def remove_capital_source(self, source: str) -> float:
        """Remove a capital source and return its last tracked amount"""
        removed_amount = self.tracked_capital.pop(source, 0.0)
        self.total_tracked_capital = sum(self.tracked_capital.values())
        return removed_amount
    
    def _get_accounting_summary(self) -> Dict[str, Any]:
        """Get comprehensive accounting summary"""
        return {
            "total_tracked_capital": self.total_tracked_capital,
            "total_hedged_amount": self.total_hedged_amount,
            "capital_at_risk": self.capital_at_risk,
            "hedge_effectiveness": self.hedge_effectiveness,
            "hedge_positions_count": len(self.hedge_positions),
            "capital_sources": len(self.tracked_capital),
            "capital_history_length": len(self.capital_history),
            "risk_metrics": {
                "hedge_ratio": self.total_hedged_amount / max(1.0, self.total_tracked_capital),
                "risk_ratio": self.capital_at_risk / max(1.0, self.total_tracked_capital),
                "hedge_target_met": self.total_hedged_amount >= (self.total_tracked_capital * self.hedge_percentage)
            }
        }
    
    def get_capital_breakdown(self) -> Dict[str, Any]:
        """Get detailed capital breakdown"""
        return {
            "tracked_capital": self.tracked_capital.copy(),
            "hedge_positions": {
                pos_id: {
                    "amount": pos.amount,
                    "type": pos.hedge_type,
                    "current_value": pos.current_value,
                    "created_at": pos.created_at
                }
                for pos_id, pos in self.hedge_positions.items()
            },
            "risk_assessment": {
                "total_capital": self.total_tracked_capital,
                "hedged_amount": self.total_hedged_amount,
                "at_risk": self.capital_at_risk,
                "hedge_ratio": self.total_hedged_amount / max(1.0, self.total_tracked_capital)
            }
        }
    
    async def cleanup(self):
        """Cleanup accounting resources"""
        try:
            # Close all hedge positions (in production, this would execute actual trades)
            for position in self.hedge_positions.values():
                logger.debug(f"Closing hedge position {position.position_id}")
            
            self.hedge_positions.clear()
            self.total_hedged_amount = 0.0
            
            logger.info(f"AccountingAnt {self.ant_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during AccountingAnt cleanup: {e}") 