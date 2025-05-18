from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime
from .ant_princess import AntPrincess
from .grok_engine import GrokEngine
from ..config import ANT_QUEEN_CONFIG

logger = logging.getLogger(__name__)

class AntQueen:
    def __init__(self):
        self.config = ANT_QUEEN_CONFIG
        self.grok_engine = GrokEngine()
        self._workers: List[AntPrincess] = []
        self._performance_history = []
        self._last_update = datetime.now()
        
    async def initialize(self):
        """Initialize the Ant Queen."""
        await self.grok_engine.initialize()
        
        # Create initial workers
        for _ in range(self.config["initial_workers"]):
            await self._create_worker()
            
    async def close(self):
        """Close the Ant Queen."""
        await self.grok_engine.close()
        
    async def _create_worker(self) -> AntPrincess:
        """Create a new worker (Ant Princess)."""
        try:
            worker = AntPrincess(self.grok_engine)
            self._workers.append(worker)
            return worker
            
        except Exception as e:
            logger.error(f"Error creating worker: {str(e)}")
            return None
            
    async def manage_workers(self):
        """Manage the worker pool."""
        try:
            # Check worker performance
            for worker in self._workers:
                metrics = worker.get_performance_metrics()
                if "error" not in metrics:
                    self._performance_history.append({
                        "worker_id": id(worker),
                        "metrics": metrics,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            # Remove underperforming workers
            self._remove_underperforming_workers()
            
            # Create new workers if needed
            if len(self._workers) < self.config["min_workers"]:
                await self._create_worker()
                
            # Check for worker multiplication
            for worker in self._workers:
                if worker.should_multiply():
                    await self._create_worker()
                    
        except Exception as e:
            logger.error(f"Error managing workers: {str(e)}")
            
    def _remove_underperforming_workers(self):
        """Remove workers that are performing below threshold."""
        try:
            # Calculate performance threshold
            threshold = self.config["performance_threshold"]
            
            # Filter out underperforming workers
            self._workers = [
                worker for worker in self._workers
                if worker.get_performance_metrics().get("performance_score", 0.0) >= threshold
            ]
            
        except Exception as e:
            logger.error(f"Error removing underperforming workers: {str(e)}")
            
    async def distribute_work(self, market_data: Dict, wallet_data: Dict) -> List[Dict]:
        """Distribute work among workers."""
        try:
            results = []
            
            # Distribute work to all workers
            tasks = [
                worker.analyze_opportunity(market_data, wallet_data)
                for worker in self._workers
            ]
            
            # Wait for all workers to complete
            worker_results = await asyncio.gather(*tasks)
            
            # Process results
            for result in worker_results:
                if "error" not in result:
                    results.append(result)
                    
            return results
            
        except Exception as e:
            logger.error(f"Error distributing work: {str(e)}")
            return []
            
    def get_queen_metrics(self) -> Dict:
        """Get metrics about the Ant Queen's performance."""
        try:
            return {
                "worker_count": len(self._workers),
                "performance_history": self._performance_history[-self.config["history_size"]:],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting queen metrics: {str(e)}")
            return {"error": str(e)}
            
    async def share_experience(self):
        """Share experience among workers."""
        try:
            # Get performance metrics from all workers
            worker_metrics = [
                worker.get_performance_metrics()
                for worker in self._workers
            ]
            
            # Calculate average performance
            valid_metrics = [m for m in worker_metrics if "error" not in m]
            if valid_metrics:
                avg_performance = sum(
                    m.get("performance_score", 0.0) for m in valid_metrics
                ) / len(valid_metrics)
                
                # Update worker thresholds based on average performance
                for worker in self._workers:
                    worker.update_performance({
                        "success": True,
                        "profit": avg_performance,
                        "risk": 1.0
                    })
                    
        except Exception as e:
            logger.error(f"Error sharing experience: {str(e)}") 