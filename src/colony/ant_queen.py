"""
Ant Queen - Capital management and worker coordination

Manages Worker Ants and contains an Ant Drone for AI coordination.
Handles capital management, worker lifecycle, and $1500 splitting threshold.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple

from .base_ant import BaseAnt, AntRole, AntStatus
from .worker_ant import WorkerAnt  
from .ant_drone import AntDrone
from .accounting_ant import AccountingAnt
from ..lifecycle.splitting_logic import SplittingLogic
from ..lifecycle.merging_logic import MergingLogic
from ..lifecycle.inheritance_system import InheritanceSystem

logger = logging.getLogger(__name__)

class AntQueen(BaseAnt):
    """Capital management and worker coordination hub"""
    
    def __init__(self, ant_id: str, parent_id: Optional[str] = None, initial_capital: float = 2.0):
        super().__init__(ant_id, AntRole.QUEEN, parent_id)
        
        # Queen-specific components
        self.ant_drone: Optional[AntDrone] = None
        self.accounting_ant: Optional[AccountingAnt] = None
        
        # Worker management
        self.workers: Dict[str, WorkerAnt] = {}
        self.worker_queue: List[str] = []  # Workers waiting for tasks
        self.active_workers: List[str] = []  # Workers currently trading
        
        # Lifecycle management systems
        self.splitting_logic: Optional[SplittingLogic] = None
        self.merging_logic: Optional[MergingLogic] = None
        self.inheritance_system: Optional[InheritanceSystem] = None
        
        # Queen-specific metrics
        self.workers_created: int = 0
        self.workers_retired: int = 0
        self.workers_merged: int = 0
        self.capital_deployed: float = 0.0
        self.total_worker_profit: float = 0.0
        
        # Split threshold tracking
        self.target_split_value = self.config["target_split_amount"]  # $1500
        self.current_usd_value: float = 0.0
        
        # Initialize capital
        self.capital.update_balance(initial_capital)
        
        logger.info(f"AntQueen {ant_id} created with {initial_capital} SOL capital")
    
    async def initialize(self) -> bool:
        """Initialize the Queen with Drone and management systems"""
        try:
            # Create and initialize Ant Drone for AI coordination
            drone_id = f"{self.ant_id}_drone"
            self.ant_drone = AntDrone(drone_id, self.ant_id)
            if not await self.ant_drone.initialize():
                logger.error(f"Failed to initialize AntDrone for Queen {self.ant_id}")
                return False
            
            # Create and initialize Accounting Ant
            accounting_id = f"{self.ant_id}_accounting"
            self.accounting_ant = AccountingAnt(accounting_id, self.ant_id)
            if not await self.accounting_ant.initialize():
                logger.warning(f"Accounting Ant initialization failed for Queen {self.ant_id}")
            
            # Initialize lifecycle management systems
            self.splitting_logic = SplittingLogic()
            await self.splitting_logic.initialize()
            
            self.merging_logic = MergingLogic()
            await self.merging_logic.initialize()
            
            self.inheritance_system = InheritanceSystem()
            await self.inheritance_system.initialize()
            
            # Create initial Worker Ant if we have enough capital
            if self.capital.available_capital >= 0.4:
                await self._create_initial_worker()
            
            logger.info(f"AntQueen {self.ant_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AntQueen {self.ant_id}: {e}")
            self.status = AntStatus.ERROR
            return False
    
    async def execute_cycle(self) -> Dict[str, Any]:
        """Execute Queen management cycle"""
        if self.status != AntStatus.ACTIVE:
            return {"status": "inactive", "reason": f"Queen status: {self.status.value}"}
        
        try:
            self.update_activity()
            
            # Update current USD value for split threshold checking
            await self._update_usd_value()
            
            # Check if Queen should split based on $1500 threshold
            if await self._should_split_queen():
                split_result = await self._prepare_queen_split()
                return {"status": "splitting", "split_result": split_result}
            
            # Execute Drone AI coordination cycle
            drone_result = {}
            if self.ant_drone:
                drone_result = await self.ant_drone.execute_cycle()
            
            # Manage Worker Ants lifecycle
            worker_management_result = await self._manage_workers()
            
            # Handle Worker splitting and merging
            lifecycle_result = await self._handle_worker_lifecycle()
            
            # Update accounting and capital tracking
            accounting_result = {}
            if self.accounting_ant:
                accounting_result = await self.accounting_ant.execute_cycle()
            
            # Coordinate resource allocation
            allocation_result = await self._allocate_resources()
            
            return {
                "status": "active",
                "drone_intelligence": drone_result,
                "worker_management": worker_management_result,
                "lifecycle_management": lifecycle_result,
                "accounting": accounting_result,
                "resource_allocation": allocation_result,
                "queen_metrics": self._get_queen_metrics(),
                "split_status": {
                    "current_usd_value": self.current_usd_value,
                    "target_split_value": self.target_split_value,
                    "progress_to_split": self.current_usd_value / self.target_split_value
                }
            }
            
        except Exception as e:
            logger.error(f"Error in AntQueen {self.ant_id} cycle: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _create_initial_worker(self) -> bool:
        """Create the first Worker Ant"""
        try:
            worker_id = f"{self.ant_id}_worker_1"
            initial_capital = min(0.4, self.capital.available_capital * 0.8)
            
            worker = WorkerAnt(worker_id, self.ant_id, initial_capital)
            if await worker.initialize():
                self.workers[worker_id] = worker
                self.worker_queue.append(worker_id)
                self.children.append(worker_id)
                self.workers_created += 1
                
                # Allocate capital
                self.capital.allocate_capital(initial_capital)
                self.capital_deployed += initial_capital
                
                logger.info(f"AntQueen {self.ant_id} created initial worker {worker_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error creating initial worker for Queen {self.ant_id}: {e}")
            return False
    
    async def _update_usd_value(self):
        """Update current USD value of total capital for split threshold checking"""
        try:
            # Get SOL to USD price (simplified - would use real price feed)
            sol_price_usd = 100.0  # Placeholder - replace with real price feed
            
            total_capital = self.capital.current_balance + self.total_worker_profit
            self.current_usd_value = total_capital * sol_price_usd
            
        except Exception as e:
            logger.error(f"Error updating USD value for Queen {self.ant_id}: {e}")
    
    async def _should_split_queen(self) -> bool:
        """Check if Queen should split based on $1500 threshold"""
        return self.current_usd_value >= self.target_split_value
    
    async def _prepare_queen_split(self) -> Dict[str, Any]:
        """Prepare Queen split when $1500 threshold is reached"""
        try:
            self.status = AntStatus.SPLITTING
            
            # Calculate capital allocation for new Queen
            total_capital = self.capital.current_balance
            new_queen_capital = total_capital * 0.5  # Split 50/50
            remaining_capital = total_capital - new_queen_capital
            
            # Export learned behavior and data
            inheritance_data = await self._export_inheritance_data()
            
            # Prepare worker distribution
            worker_count = len(self.workers)
            workers_to_transfer = worker_count // 2
            
            split_request = {
                "type": "queen_split",
                "parent_queen_id": self.ant_id,
                "new_queen_capital": new_queen_capital,
                "remaining_capital": remaining_capital,
                "workers_to_transfer": workers_to_transfer,
                "inheritance_data": inheritance_data,
                "trigger_reason": f"Reached ${self.current_usd_value:.2f} USD value",
                "split_timestamp": time.time()
            }
            
            logger.info(f"AntQueen {self.ant_id} preparing to split at ${self.current_usd_value:.2f}")
            return split_request
            
        except Exception as e:
            logger.error(f"Error preparing Queen split for {self.ant_id}: {e}")
            return {"error": str(e)}
    
    async def _manage_workers(self) -> Dict[str, Any]:
        """Manage Worker Ant operations and coordination"""
        try:
            management_results = {
                "workers_managed": 0,
                "new_workers_created": 0,
                "workers_retired": 0,
                "total_worker_profit": 0.0,
                "worker_status": {}
            }
            
            # Execute cycles for all active workers
            for worker_id, worker in self.workers.items():
                if worker.status == AntStatus.ACTIVE:
                    worker_result = await worker.execute_cycle()
                    management_results["worker_status"][worker_id] = worker_result
                    management_results["workers_managed"] += 1
                    
                    # Track worker profit
                    if worker_result.get("trading_result", {}).get("trade_executed"):
                        profit = worker_result["trading_result"].get("profit", 0.0)
                        self.total_worker_profit += profit
                        management_results["total_worker_profit"] += profit
                    
                    # Handle worker splitting requests
                    if worker_result.get("status") == "splitting":
                        split_result = await self._handle_worker_split(worker_id, worker_result.get("split_request"))
                        management_results["new_workers_created"] += split_result.get("workers_created", 0)
                    
                    # Handle worker retirement
                    elif worker_result.get("status") == "retiring":
                        await self._retire_worker(worker_id)
                        management_results["workers_retired"] += 1
            
            return management_results
            
        except Exception as e:
            logger.error(f"Error managing workers for Queen {self.ant_id}: {e}")
            return {"error": str(e)}
    
    async def _handle_worker_split(self, parent_worker_id: str, split_request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Worker Ant splitting into 5 new workers"""
        try:
            if not split_request or split_request.get("type") != "worker_split":
                return {"error": "Invalid split request"}
            
            workers_created = 0
            new_worker_ids = []
            
            # Create 5 new workers from the split
            capital_per_worker = split_request.get("capital_per_worker", 0.0)
            inheritance_data = split_request.get("inheritance_data", {})
            
            for i in range(5):
                new_worker_id = f"{self.ant_id}_worker_{self.workers_created + i + 1}"
                
                # Create new worker with inherited data
                new_worker = WorkerAnt(new_worker_id, self.ant_id, capital_per_worker)
                
                # Apply inheritance
                if self.inheritance_system:
                    await self.inheritance_system.apply_inheritance(new_worker, inheritance_data)
                
                if await new_worker.initialize():
                    self.workers[new_worker_id] = new_worker
                    self.worker_queue.append(new_worker_id)
                    self.children.append(new_worker_id)
                    new_worker_ids.append(new_worker_id)
                    workers_created += 1
            
            # Retire the parent worker
            await self._retire_worker(parent_worker_id)
            
            self.workers_created += workers_created
            
            logger.info(f"AntQueen {self.ant_id} split worker {parent_worker_id} into {workers_created} new workers")
            
            return {
                "workers_created": workers_created,
                "new_worker_ids": new_worker_ids,
                "parent_worker_retired": parent_worker_id
            }
            
        except Exception as e:
            logger.error(f"Error handling worker split for Queen {self.ant_id}: {e}")
            return {"error": str(e)}
    
    async def _handle_worker_lifecycle(self) -> Dict[str, Any]:
        """Handle Worker Ant lifecycle events (merging, retirement)"""
        try:
            lifecycle_results = {
                "merges_performed": 0,
                "retirements_processed": 0,
                "underperforming_workers": [],
                "high_performing_workers": []
            }
            
            # Identify underperforming workers for merging
            underperforming = []
            high_performing = []
            
            for worker_id, worker in self.workers.items():
                if worker.should_merge():
                    underperforming.append(worker_id)
                elif worker.performance.win_rate > 70 and worker.performance.total_profit > 0.2:
                    high_performing.append(worker_id)
            
            # Merge underperforming workers with high-performing ones
            if underperforming and high_performing and self.merging_logic:
                merge_result = await self.merging_logic.merge_workers(
                    underperforming, high_performing, self.workers
                )
                lifecycle_results["merges_performed"] = merge_result.get("merges_completed", 0)
                self.workers_merged += merge_result.get("merges_completed", 0)
            
            lifecycle_results["underperforming_workers"] = underperforming
            lifecycle_results["high_performing_workers"] = high_performing
            
            return lifecycle_results
            
        except Exception as e:
            logger.error(f"Error handling worker lifecycle for Queen {self.ant_id}: {e}")
            return {"error": str(e)}
    
    async def _retire_worker(self, worker_id: str) -> bool:
        """Retire a Worker Ant and reclaim capital"""
        try:
            if worker_id not in self.workers:
                return False
            
            worker = self.workers[worker_id]
            
            # Get final capital from worker
            final_capital = worker.capital.current_balance
            
            # Transfer capital back to Queen
            self.capital.release_capital(final_capital)
            self.capital.update_balance(self.capital.current_balance + final_capital)
            
            # Cleanup worker
            await worker.cleanup()
            
            # Remove from tracking
            del self.workers[worker_id]
            if worker_id in self.worker_queue:
                self.worker_queue.remove(worker_id)
            if worker_id in self.active_workers:
                self.active_workers.remove(worker_id)
            if worker_id in self.children:
                self.children.remove(worker_id)
            
            self.workers_retired += 1
            
            logger.info(f"AntQueen {self.ant_id} retired worker {worker_id}, reclaimed {final_capital} SOL")
            return True
            
        except Exception as e:
            logger.error(f"Error retiring worker {worker_id} for Queen {self.ant_id}: {e}")
            return False
    
    async def _allocate_resources(self) -> Dict[str, Any]:
        """Allocate capital and resources to workers based on performance"""
        try:
            allocation_results = {
                "allocations_made": 0,
                "total_allocated": 0.0,
                "worker_allocations": {}
            }
            
            available_capital = self.capital.available_capital
            
            if available_capital > 0.5:  # Only allocate if we have significant capital
                # Prioritize high-performing workers
                worker_performance = []
                
                for worker_id, worker in self.workers.items():
                    if worker.status == AntStatus.ACTIVE:
                        performance_score = (
                            worker.performance.win_rate * 0.4 +
                            (worker.performance.total_profit * 100) * 0.3 +
                            (worker.performance.compound_factor - 1.0) * 100 * 0.3
                        )
                        worker_performance.append((worker_id, performance_score, worker))
                
                # Sort by performance (highest first)
                worker_performance.sort(key=lambda x: x[1], reverse=True)
                
                # Allocate to top performers
                allocation_per_worker = min(0.3, available_capital / max(1, len(worker_performance)))
                
                for worker_id, score, worker in worker_performance[:5]:  # Top 5 workers
                    if score > 50:  # Only allocate to decent performers
                        worker.capital.update_balance(worker.capital.current_balance + allocation_per_worker)
                        self.capital.allocate_capital(allocation_per_worker)
                        
                        allocation_results["worker_allocations"][worker_id] = allocation_per_worker
                        allocation_results["total_allocated"] += allocation_per_worker
                        allocation_results["allocations_made"] += 1
            
            return allocation_results
            
        except Exception as e:
            logger.error(f"Error allocating resources for Queen {self.ant_id}: {e}")
            return {"error": str(e)}
    
    async def _export_inheritance_data(self) -> Dict[str, Any]:
        """Export learned behavior and data for inheritance"""
        try:
            inheritance_data = {
                "queen_intelligence": {},
                "worker_strategies": {},
                "performance_patterns": {},
                "ai_learning_data": {}
            }
            
            # Export from Drone AI
            if self.ant_drone:
                inheritance_data["ai_learning_data"] = await self.ant_drone.export_learning_data()
            
            # Export successful worker strategies
            for worker_id, worker in self.workers.items():
                if worker.performance.win_rate > 60:  # Only successful workers
                    worker_data = await worker._export_learned_strategies()
                    inheritance_data["worker_strategies"][worker_id] = worker_data
            
            # Export Queen-level patterns
            inheritance_data["performance_patterns"] = {
                "total_workers_created": self.workers_created,
                "successful_worker_ratio": (self.workers_created - self.workers_retired) / max(1, self.workers_created),
                "average_worker_profit": self.total_worker_profit / max(1, self.workers_created),
                "capital_deployment_efficiency": self.total_worker_profit / max(1, self.capital_deployed)
            }
            
            return inheritance_data
            
        except Exception as e:
            logger.error(f"Error exporting inheritance data for Queen {self.ant_id}: {e}")
            return {}
    
    def _get_queen_metrics(self) -> Dict[str, Any]:
        """Get comprehensive Queen metrics"""
        base_metrics = self.get_status_summary()
        
        queen_specific = {
            "workers": {
                "total_created": self.workers_created,
                "currently_active": len([w for w in self.workers.values() if w.status == AntStatus.ACTIVE]),
                "retired": self.workers_retired,
                "merged": self.workers_merged,
                "in_queue": len(self.worker_queue),
                "total_worker_profit": self.total_worker_profit
            },
            "capital_management": {
                "capital_deployed": self.capital_deployed,
                "deployment_efficiency": self.total_worker_profit / max(1, self.capital_deployed),
                "current_usd_value": self.current_usd_value,
                "split_progress": self.current_usd_value / self.target_split_value
            },
            "ai_coordination": {
                "drone_active": self.ant_drone is not None and self.ant_drone.status == AntStatus.ACTIVE,
                "accounting_active": self.accounting_ant is not None and self.accounting_ant.status == AntStatus.ACTIVE
            }
        }
        
        base_metrics.update({"queen_specific": queen_specific})
        return base_metrics
    
    async def get_worker_recommendations(self, worker_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI recommendations for a specific worker"""
        if self.ant_drone:
            return await self.ant_drone.get_trading_recommendations(worker_context)
        return {"error": "No AI Drone available"}
    
    async def cleanup(self):
        """Cleanup Queen and all managed resources"""
        try:
            # Cleanup all workers
            for worker_id, worker in self.workers.items():
                await worker.cleanup()
            
            # Cleanup AI components
            if self.ant_drone:
                await self.ant_drone.cleanup()
            
            if self.accounting_ant:
                await self.accounting_ant.cleanup()
            
            logger.info(f"AntQueen {self.ant_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during AntQueen {self.ant_id} cleanup: {e}") 