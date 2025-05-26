from typing import Dict, List, Optional
import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from .ant_princess import AntPrincess
from .grok_engine import GrokEngine
from config.ant_princess_config import QUEEN_CONFIG as ANT_QUEEN_CONFIG, SYSTEM_CONSTANTS as AI_CONFIG
from .ai_metrics import ai_metrics

logger = logging.getLogger(__name__)

class AntQueen:
    def __init__(self):
        self.config = ANT_QUEEN_CONFIG
        self.grok_engine = GrokEngine()
        self._workers: List[AntPrincess] = []
        self._performance_history = []
        self._last_update = datetime.now()
        self._session = None
        self._model = None
        self.training_data = []
        self.accuracy_history = []
        
    @ai_metrics.track_training()
    async def initialize(self):
        """Initialize the Ant Queen."""
        await self.grok_engine.initialize()
        
        # Create initial workers
        for _ in range(self.config["initial_workers"]):
            await self._create_worker()
            
        # Initialize the model
        await self._initialize_model()
        
        return {
            "success": True,
            "sample_count": len(self.training_data)
        }
        
    async def _initialize_model(self):
        """Initialize the model."""
        try:
            logger.info("Initializing Ant Queen model")
            
            # Load training data if available
            await self._load_training_data()
            
            # Track the number of samples
            ai_metrics.update_training_samples(len(self.training_data))
            
            return {
                "success": True,
                "sample_count": len(self.training_data)
            }
            
        except Exception as e:
            logger.error(f"Error initializing Ant Queen: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def _load_training_data(self):
        """Load training data if available."""
        try:
            # Simulated data loading
            self.training_data = [
                {"input": "feature_set_1", "output": "buy", "profit": 0.05},
                {"input": "feature_set_2", "output": "sell", "profit": 0.03},
                {"input": "feature_set_3", "output": "hold", "profit": 0.0}
            ]
            
            # Calculate initial accuracy (simulated)
            current_accuracy = 0.7  # Starting accuracy
            ai_metrics.update_accuracy(current_accuracy)
            
            # Initialize accuracy history
            self.accuracy_history = [current_accuracy]
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            raise
    
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
            
    @ai_metrics.track_prediction(prediction_type="market_decision")      
    async def analyze_market(self, market_data: Dict) -> Dict:
        """
        Analyze market data and make a trading decision.
        """
        try:
            # Simulated decision making
            features = self._extract_features(market_data)
            
            # Make prediction with confidence
            if features.get("volatility", 0) > 0.1:
                decision = "buy" if features.get("trend", 0) > 0 else "sell"
                confidence = min(0.7 + features.get("volatility", 0), 0.95)
            else:
                decision = "hold"
                confidence = 0.6
                
            # Record the confidence
            ai_metrics.record_confidence(confidence)
                
            return {
                "decision": decision,
                "confidence": confidence,
                "reasoning": f"Based on volatility ({features.get('volatility')}) and trend ({features.get('trend')})"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market: {str(e)}")
            return {"decision": "hold", "confidence": 0.5, "error": str(e)}
    
    def _extract_features(self, market_data: Dict) -> Dict:
        """Extract features from market data."""
        # Simple feature extraction
        features = {
            "volatility": market_data.get("price_change_1h", 0),
            "volume": market_data.get("volume_24h", 0) / 1000000,
            "trend": market_data.get("price_change_24h", 0),
            "liquidity": market_data.get("liquidity", 0) / 10000
        }
        
        return features
    
    @ai_metrics.track_training()
    async def learn_from_trade(self, trade_data: Dict) -> Dict:
        """
        Learn from trade outcomes to improve future predictions.
        """
        try:
            # Add trade to training data
            self.training_data.append(trade_data)
            
            # Update metrics
            ai_metrics.update_training_samples(len(self.training_data))
            
            # Simulate improvement in model accuracy
            # In a real implementation, this would come from actual model evaluation
            last_accuracy = self.accuracy_history[-1] if self.accuracy_history else 0.7
            
            # Small accuracy improvement
            new_accuracy = min(0.99, last_accuracy + 0.01)
            self.accuracy_history.append(new_accuracy)
            
            # Update accuracy metric
            ai_metrics.update_accuracy(new_accuracy)
            
            # Update accuracy by decision type
            decision_type = trade_data.get("decision", "default")
            type_accuracy = new_accuracy * (0.9 + 0.1 * (trade_data.get("profit", 0) > 0))
            ai_metrics.update_accuracy_by_type(decision_type, type_accuracy)
            
            return {
                "success": True,
                "accuracy": new_accuracy,
                "sample_count": len(self.training_data)
            }
            
        except Exception as e:
            logger.error(f"Error learning from trade: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_accuracy_trend(self) -> List[float]:
        """Get the historical accuracy trend."""
        return self.accuracy_history 