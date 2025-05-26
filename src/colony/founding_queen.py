"""
Founding Ant Queen - Top-level system coordinator

Manages multiple Ant Queens and handles system-wide coordination.
The origin wallet and highest level of the ant hierarchy.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any

from .base_ant import BaseAnt, AntRole, AntStatus
from .ant_queen import AntQueen

logger = logging.getLogger(__name__)

class FoundingAntQueen(BaseAnt):
    """Top-level coordinator managing multiple Queens and system-wide operations"""
    
    def __init__(self, ant_id: str = "founding_queen_0", initial_capital: float = 20.0):
        super().__init__(ant_id, AntRole.FOUNDING_QUEEN)
        
        # Founding Queen specific attributes
        self.queens: Dict[str, AntQueen] = {}
        self.retired_queens: List[str] = []
        
        # System-wide metrics
        self.system_metrics = {
            "total_ants": 1,  # Starting with founding queen
            "total_capital": initial_capital,
            "total_trades": 0,
            "system_profit": 0.0,
            "system_start_time": time.time(),
            "queens_created": 0,
            "queens_retired": 0
        }
        
        # Initialize capital
        self.capital.update_balance(initial_capital)
        
        logger.info(f"FoundingAntQueen {ant_id} created with {initial_capital} SOL capital")
    
    async def initialize(self) -> bool:
        """Initialize the Founding Queen and create initial Queen"""
        try:
            # Create initial Queen if we have enough capital
            if self.capital.current_balance >= 2.0:
                await self._create_initial_queen()
            
            logger.info(f"FoundingAntQueen {self.ant_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FoundingAntQueen {self.ant_id}: {e}")
            self.status = AntStatus.ERROR
            return False
    
    async def execute_cycle(self) -> Dict[str, Any]:
        """Execute Founding Queen coordination cycle"""
        if self.status != AntStatus.ACTIVE:
            return {"status": "inactive", "reason": f"Founding Queen status: {self.status.value}"}
        
        try:
            self.update_activity()
            
            # Coordinate all Queens
            queen_results = await self._coordinate_queens()
            
            # Check for Queen splitting or creation needs
            queen_management = await self._manage_queen_lifecycle()
            
            # Update system-wide metrics
            await self._update_system_metrics()
            
            # Check if we should create more Queens
            if self._should_create_queen():
                new_queen_result = await self._create_queen()
                queen_management["new_queen_created"] = new_queen_result
            
            return {
                "status": "active",
                "queen_coordination": queen_results,
                "queen_management": queen_management,
                "system_metrics": self.system_metrics,
                "queens_status": self._get_queens_status()
            }
            
        except Exception as e:
            logger.error(f"Error in FoundingAntQueen {self.ant_id} cycle: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _create_initial_queen(self) -> bool:
        """Create the first Queen under Founding Queen"""
        try:
            queen_id = f"queen_1"
            initial_capital = 2.0
            
            if not self.capital.allocate_capital(initial_capital):
                logger.warning("Insufficient capital to create initial Queen")
                return False
            
            queen = AntQueen(queen_id, self.ant_id, initial_capital)
            
            if await queen.initialize():
                self.queens[queen_id] = queen
                self.children.append(queen_id)
                self.system_metrics["queens_created"] += 1
                
                logger.info(f"FoundingAntQueen created initial Queen {queen_id}")
                return True
            else:
                # Return allocated capital if Queen creation failed
                self.capital.release_capital(initial_capital)
                return False
            
        except Exception as e:
            logger.error(f"Error creating initial Queen: {e}")
            return False
    
    async def _coordinate_queens(self) -> Dict[str, Any]:
        """Coordinate operations across all Queens"""
        try:
            coordination_results = {
                "queens_coordinated": 0,
                "total_workers": 0,
                "total_trades": 0,
                "total_profit": 0.0,
                "queen_results": {}
            }
            
            # Execute cycles for all active Queens
            for queen_id, queen in self.queens.items():
                if queen.status == AntStatus.ACTIVE:
                    queen_result = await queen.execute_cycle()
                    coordination_results["queen_results"][queen_id] = queen_result
                    coordination_results["queens_coordinated"] += 1
                    
                    # Aggregate metrics
                    if queen_result.get("status") == "active":
                        queen_metrics = queen_result.get("queen_metrics", {}).get("queen_specific", {})
                        workers_data = queen_metrics.get("workers", {})
                        
                        coordination_results["total_workers"] += workers_data.get("currently_active", 0)
                        coordination_results["total_profit"] += workers_data.get("total_worker_profit", 0.0)
                    
                    # Handle Queen splitting requests
                    if queen_result.get("status") == "splitting":
                        await self._handle_queen_split_request(queen_id, queen_result.get("split_result"))
            
            return coordination_results
            
        except Exception as e:
            logger.error(f"Error coordinating Queens: {e}")
            return {"error": str(e)}
    
    async def _handle_queen_split_request(self, queen_id: str, split_data: Dict[str, Any]) -> bool:
        """Handle Queen split request when $1500 threshold is reached"""
        try:
            if not split_data or split_data.get("type") != "queen_split":
                return False
            
            # Create new Queen from split
            new_queen_id = f"queen_{len(self.queens) + 1}"
            new_capital = split_data.get("new_queen_capital", 1.0)
            
            new_queen = AntQueen(new_queen_id, self.ant_id, new_capital)
            
            # Apply inheritance data
            inheritance_data = split_data.get("inheritance_data", {})
            if inheritance_data:
                await self._apply_inheritance(new_queen, inheritance_data)
            
            if await new_queen.initialize():
                self.queens[new_queen_id] = new_queen
                self.children.append(new_queen_id)
                self.system_metrics["queens_created"] += 1
                
                # Original Queen stops operating (as per architecture)
                original_queen = self.queens.get(queen_id)
                if original_queen:
                    await self._retire_queen(queen_id)
                
                logger.info(f"Queen split completed: {queen_id} -> {new_queen_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error handling Queen split: {e}")
            return False
    
    async def _apply_inheritance(self, new_queen: AntQueen, inheritance_data: Dict[str, Any]):
        """Apply inheritance data to new Queen"""
        try:
            # Store inheritance data for Queen to use
            new_queen.metadata["inherited_ai_data"] = inheritance_data.get("ai_learning_data", {})
            new_queen.metadata["inherited_strategies"] = inheritance_data.get("worker_strategies", {})
            new_queen.metadata["inherited_patterns"] = inheritance_data.get("performance_patterns", {})
            
        except Exception as e:
            logger.error(f"Error applying inheritance: {e}")
    
    async def _manage_queen_lifecycle(self) -> Dict[str, Any]:
        """Manage Queen lifecycle events"""
        try:
            management_results = {
                "queens_retired": 0,
                "underperforming_queens": [],
                "high_performing_queens": []
            }
            
            # Identify underperforming Queens for potential retirement
            underperforming = []
            high_performing = []
            
            for queen_id, queen in self.queens.items():
                if queen.should_merge():  # Using merge logic for retirement decisions
                    underperforming.append(queen_id)
                elif queen.performance.win_rate > 70 and queen.performance.total_profit > 1.0:
                    high_performing.append(queen_id)
            
            # Retire underperforming Queens (reclaim capital)
            for queen_id in underperforming:
                await self._retire_queen(queen_id)
                management_results["queens_retired"] += 1
            
            management_results["underperforming_queens"] = underperforming
            management_results["high_performing_queens"] = high_performing
            
            return management_results
            
        except Exception as e:
            logger.error(f"Error managing Queen lifecycle: {e}")
            return {"error": str(e)}
    
    async def _retire_queen(self, queen_id: str) -> bool:
        """Retire a Queen and reclaim capital"""
        try:
            if queen_id not in self.queens:
                return False
            
            queen = self.queens[queen_id]
            
            # Get final capital from Queen (including worker capital)
            final_capital = queen.capital.current_balance
            for worker in queen.workers.values():
                final_capital += worker.capital.current_balance
            
            # Cleanup Queen
            await queen.cleanup()
            
            # Remove from tracking
            del self.queens[queen_id]
            if queen_id in self.children:
                self.children.remove(queen_id)
            
            self.retired_queens.append(queen_id)
            self.system_metrics["queens_retired"] += 1
            
            # Reclaim capital
            self.capital.update_balance(self.capital.current_balance + final_capital)
            
            logger.info(f"FoundingAntQueen retired Queen {queen_id}, reclaimed {final_capital} SOL")
            return True
            
        except Exception as e:
            logger.error(f"Error retiring Queen {queen_id}: {e}")
            return False
    
    def _should_create_queen(self) -> bool:
        """Determine if we should create a new Queen"""
        # Create new Queen if we have enough capital and not too many active Queens
        return (self.capital.available_capital >= self.config["split_threshold"] and 
                len(self.queens) < self.config["max_children"])
    
    async def _create_queen(self) -> Dict[str, Any]:
        """Create a new Queen"""
        try:
            queen_id = f"queen_{len(self.queens) + len(self.retired_queens) + 1}"
            initial_capital = self.config["split_threshold"]  # 20 SOL
            
            if not self.capital.allocate_capital(initial_capital):
                return {"success": False, "reason": "Insufficient capital"}
            
            queen = AntQueen(queen_id, self.ant_id, initial_capital)
            
            if await queen.initialize():
                self.queens[queen_id] = queen
                self.children.append(queen_id)
                self.system_metrics["queens_created"] += 1
                
                logger.info(f"FoundingAntQueen created new Queen {queen_id}")
                return {"success": True, "queen_id": queen_id, "capital": initial_capital}
            else:
                self.capital.release_capital(initial_capital)
                return {"success": False, "reason": "Queen initialization failed"}
            
        except Exception as e:
            logger.error(f"Error creating Queen: {e}")
            return {"success": False, "error": str(e)}
    
    async def _update_system_metrics(self):
        """Update system-wide metrics"""
        try:
            total_capital = self.capital.current_balance
            total_trades = 0
            total_profit = 0.0
            total_ants = 1  # Founding Queen
            
            # Aggregate from all Queens
            for queen in self.queens.values():
                total_capital += queen.capital.current_balance
                total_trades += queen.performance.total_trades
                total_profit += queen.performance.total_profit
                total_ants += 1  # Queen
                
                # Add Worker metrics
                for worker in queen.workers.values():
                    total_capital += worker.capital.current_balance
                    total_trades += worker.performance.total_trades
                    total_profit += worker.performance.total_profit
                    total_ants += 1  # Worker
            
            self.system_metrics.update({
                "total_ants": total_ants,
                "total_capital": total_capital,
                "total_trades": total_trades,
                "system_profit": total_profit,
                "runtime_hours": (time.time() - self.system_metrics["system_start_time"]) / 3600,
                "active_queens": len(self.queens),
                "total_queens_created": self.system_metrics["queens_created"]
            })
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def _get_queens_status(self) -> Dict[str, Any]:
        """Get status of all Queens"""
        return {
            queen_id: {
                "status": queen.status.value,
                "capital": queen.capital.current_balance,
                "workers": len(queen.workers),
                "profit": queen.performance.total_profit
            }
            for queen_id, queen in self.queens.items()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        base_status = self.get_status_summary()
        
        base_status.update({
            "system_metrics": self.system_metrics,
            "queens_status": self._get_queens_status(),
            "system_health": {
                "active_queens": len(self.queens),
                "retired_queens": len(self.retired_queens),
                "total_system_capital": self.system_metrics["total_capital"],
                "system_profit": self.system_metrics["system_profit"]
            }
        })
        
        return base_status
    
    async def cleanup(self):
        """Cleanup all Queens and system resources"""
        try:
            # Cleanup all Queens
            for queen in self.queens.values():
                await queen.cleanup()
            
            logger.info(f"FoundingAntQueen {self.ant_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during FoundingAntQueen cleanup: {e}") 