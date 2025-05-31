"""
Model Deployment Manager - Safe Deployment with Resource Management

This module implements practical model deployment with automated rollback
and resource-aware training for local environments.
"""

import asyncio
import time
import logging
import psutil
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

class DeploymentStage(Enum):
    DEVELOPMENT = "development"
    CANARY = "canary"
    PRODUCTION = "production"
    ROLLBACK = "rollback"

class ResourceStatus(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ModelPerformance:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    response_time_ms: float
    confidence_calibration: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class ResourceMetrics:
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    available_memory_gb: float
    temperature: Optional[float] = None  # If available
    timestamp: float = field(default_factory=time.time)

@dataclass
class RollbackCriteria:
    min_accuracy: float = 0.7
    max_response_time_ms: float = 2000
    min_confidence_calibration: float = 0.6
    max_error_rate: float = 0.3
    observation_window_minutes: int = 30
    min_samples_for_evaluation: int = 50

class ModelDeploymentManager:
    """
    Safe model deployment with automated rollback and resource management
    
    Features:
    - Canary deployment with automatic rollback
    - Resource-aware training scheduling
    - Performance monitoring and alerting
    - Safe rollback procedures
    """
    
    def __init__(self, rollback_criteria: Optional[RollbackCriteria] = None):
        self.rollback_criteria = rollback_criteria or RollbackCriteria()
        
        # Deployment tracking
        self.active_models: Dict[str, Dict[str, Any]] = {}
        self.model_performance: Dict[str, deque] = {}
        self.deployment_history: List[Dict[str, Any]] = []
        
        # Resource management
        self.resource_monitor = ResourceMonitor()
        self.training_scheduler = TrainingScheduler()
        
        # Performance tracking
        self.canary_traffic_percent = 5.0
        self.rollback_in_progress = False
        
        # Configuration
        self.models_directory = Path("models/")
        self.models_directory.mkdir(exist_ok=True)
        
        logger.info("🚀 Model Deployment Manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize deployment manager"""
        try:
            await self.resource_monitor.initialize()
            await self.training_scheduler.initialize()
            
            # Start monitoring tasks
            asyncio.create_task(self._performance_monitoring_loop())
            asyncio.create_task(self._resource_monitoring_loop())
            
            logger.info("✅ Model Deployment Manager initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Model Deployment Manager: {str(e)}")
            return False
    
    async def deploy_canary(self, model_id: str, model_artifact: Any) -> Dict[str, Any]:
        """Deploy model to canary stage with monitoring"""
        try:
            start_time = time.time()
            
            # Check resource availability
            resource_status = await self.resource_monitor.get_current_status()
            if resource_status.cpu_percent > 80:
                return {
                    "success": False,
                    "error": "System resources too high for deployment",
                    "cpu_usage": resource_status.cpu_percent
                }
            
            # Store model artifact
            model_path = await self._store_model_artifact(model_id, model_artifact)
            
            # Initialize performance tracking
            self.model_performance[model_id] = deque(maxlen=1000)
            
            # Register model as canary
            self.active_models[model_id] = {
                "stage": DeploymentStage.CANARY,
                "deployment_time": time.time(),
                "model_path": model_path,
                "traffic_percent": self.canary_traffic_percent,
                "performance_baseline": None
            }
            
            # Log deployment
            deployment_record = {
                "model_id": model_id,
                "action": "canary_deployment",
                "timestamp": time.time(),
                "resource_snapshot": resource_status
            }
            self.deployment_history.append(deployment_record)
            
            deployment_time = time.time() - start_time
            
            logger.info(f"🕯️ Canary deployment successful: {model_id} "
                       f"({self.canary_traffic_percent}% traffic, {deployment_time:.2f}s)")
            
            return {
                "success": True,
                "model_id": model_id,
                "stage": DeploymentStage.CANARY.value,
                "traffic_percent": self.canary_traffic_percent,
                "deployment_time_s": deployment_time
            }
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def record_prediction_result(self, model_id: str, prediction_time_ms: float, 
                                     was_correct: bool, confidence: float) -> bool:
        """Record prediction result for performance evaluation"""
        try:
            if model_id not in self.model_performance:
                return False
            
            # Calculate performance metrics
            performance = ModelPerformance(
                accuracy=1.0 if was_correct else 0.0,
                precision=confidence if was_correct else (1.0 - confidence),
                recall=1.0 if was_correct else 0.0,
                f1_score=1.0 if was_correct else 0.0,
                response_time_ms=prediction_time_ms,
                confidence_calibration=confidence if was_correct else (1.0 - confidence)
            )
            
            self.model_performance[model_id].append(performance)
            
            # Check rollback conditions
            await self._evaluate_rollback_conditions(model_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording prediction result: {str(e)}")
            return False
    
    async def _evaluate_rollback_conditions(self, model_id: str) -> bool:
        """Evaluate if model should be rolled back"""
        try:
            if self.rollback_in_progress or model_id not in self.model_performance:
                return False
            
            performance_data = list(self.model_performance[model_id])
            
            # Check minimum samples
            if len(performance_data) < self.rollback_criteria.min_samples_for_evaluation:
                return False
            
            # Get recent performance within observation window
            current_time = time.time()
            window_start = current_time - (self.rollback_criteria.observation_window_minutes * 60)
            recent_performance = [
                p for p in performance_data 
                if p.timestamp >= window_start
            ]
            
            if len(recent_performance) < 10:  # Minimum recent samples
                return False
            
            # Calculate aggregated metrics
            avg_accuracy = sum(p.accuracy for p in recent_performance) / len(recent_performance)
            avg_response_time = sum(p.response_time_ms for p in recent_performance) / len(recent_performance)
            avg_confidence_calibration = sum(p.confidence_calibration for p in recent_performance) / len(recent_performance)
            error_rate = 1.0 - avg_accuracy
            
            # Check rollback criteria
            rollback_reasons = []
            
            if avg_accuracy < self.rollback_criteria.min_accuracy:
                rollback_reasons.append(f"Accuracy too low: {avg_accuracy:.3f} < {self.rollback_criteria.min_accuracy}")
            
            if avg_response_time > self.rollback_criteria.max_response_time_ms:
                rollback_reasons.append(f"Response time too high: {avg_response_time:.1f}ms > {self.rollback_criteria.max_response_time_ms}ms")
            
            if avg_confidence_calibration < self.rollback_criteria.min_confidence_calibration:
                rollback_reasons.append(f"Confidence calibration poor: {avg_confidence_calibration:.3f} < {self.rollback_criteria.min_confidence_calibration}")
            
            if error_rate > self.rollback_criteria.max_error_rate:
                rollback_reasons.append(f"Error rate too high: {error_rate:.3f} > {self.rollback_criteria.max_error_rate}")
            
            # Trigger rollback if criteria met
            if rollback_reasons:
                logger.warning(f"🚨 ROLLBACK TRIGGERED for {model_id}: {'; '.join(rollback_reasons)}")
                await self._execute_rollback(model_id, rollback_reasons)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating rollback conditions: {str(e)}")
            return False
    
    async def _execute_rollback(self, model_id: str, reasons: List[str]) -> bool:
        """Execute safe rollback procedure"""
        try:
            self.rollback_in_progress = True
            start_time = time.time()
            
            logger.critical(f"🔄 EXECUTING ROLLBACK: {model_id}")
            
            # Find previous stable model
            stable_model_id = await self._find_stable_model()
            
            if stable_model_id:
                # Switch traffic to stable model
                if stable_model_id in self.active_models:
                    self.active_models[stable_model_id]["traffic_percent"] = 100.0
                    self.active_models[stable_model_id]["stage"] = DeploymentStage.PRODUCTION
                
                # Mark problematic model for rollback
                if model_id in self.active_models:
                    self.active_models[model_id]["stage"] = DeploymentStage.ROLLBACK
                    self.active_models[model_id]["traffic_percent"] = 0.0
                
                # Log rollback
                rollback_record = {
                    "model_id": model_id,
                    "action": "rollback",
                    "reasons": reasons,
                    "rolled_back_to": stable_model_id,
                    "timestamp": time.time(),
                    "execution_time_s": time.time() - start_time
                }
                self.deployment_history.append(rollback_record)
                
                logger.info(f"✅ Rollback completed: {model_id} -> {stable_model_id} "
                           f"({time.time() - start_time:.2f}s)")
                
                success = True
            else:
                logger.error("❌ No stable model found for rollback")
                success = False
            
            self.rollback_in_progress = False
            return success
            
        except Exception as e:
            logger.error(f"Rollback execution failed: {str(e)}")
            self.rollback_in_progress = False
            return False
    
    async def _find_stable_model(self) -> Optional[str]:
        """Find most recent stable model for rollback"""
        try:
            # Look for production models with good performance
            for model_id, model_info in self.active_models.items():
                if model_info["stage"] == DeploymentStage.PRODUCTION:
                    # Check recent performance
                    if model_id in self.model_performance:
                        recent_perf = list(self.model_performance[model_id])[-50:]  # Last 50 predictions
                        if recent_perf:
                            avg_accuracy = sum(p.accuracy for p in recent_perf) / len(recent_perf)
                            if avg_accuracy >= self.rollback_criteria.min_accuracy:
                                return model_id
            
            # Fallback: look in deployment history for last successful deployment
            for deployment in reversed(self.deployment_history):
                if deployment.get("action") == "promotion_to_production":
                    return deployment.get("model_id")
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding stable model: {str(e)}")
            return None
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        return {
            "active_models": {
                model_id: {
                    "stage": info["stage"].value,
                    "traffic_percent": info["traffic_percent"],
                    "deployment_age_hours": (time.time() - info["deployment_time"]) / 3600,
                    "recent_performance": self._get_recent_performance_summary(model_id)
                }
                for model_id, info in self.active_models.items()
            },
            "rollback_in_progress": self.rollback_in_progress,
            "total_deployments": len(self.deployment_history),
            "current_resource_status": self.resource_monitor.get_latest_status()
        }
    
    def _get_recent_performance_summary(self, model_id: str) -> Dict[str, float]:
        """Get recent performance summary for model"""
        if model_id not in self.model_performance:
            return {}
        
        recent_perf = list(self.model_performance[model_id])[-20:]  # Last 20 predictions
        if not recent_perf:
            return {}
        
        return {
            "accuracy": sum(p.accuracy for p in recent_perf) / len(recent_perf),
            "avg_response_time_ms": sum(p.response_time_ms for p in recent_perf) / len(recent_perf),
            "confidence_calibration": sum(p.confidence_calibration for p in recent_perf) / len(recent_perf),
            "sample_count": len(recent_perf)
        }

class ResourceMonitor:
    """Monitor system resources for training scheduling"""
    
    def __init__(self):
        self.resource_history = deque(maxlen=100)
        self.resource_thresholds = {
            "cpu_high": 75.0,
            "memory_high": 80.0,
            "disk_high": 90.0
        }
    
    async def initialize(self):
        """Initialize resource monitoring"""
        logger.info("📊 Resource Monitor initialized")
    
    async def get_current_status(self) -> ResourceMetrics:
        """Get current system resource status"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                available_memory_gb=memory.available / (1024**3)
            )
            
            self.resource_history.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting resource status: {str(e)}")
            return ResourceMetrics(0.0, 0.0, 0.0, 0.0)
    
    def get_latest_status(self) -> Dict[str, Any]:
        """Get latest resource status summary"""
        if not self.resource_history:
            return {"status": "no_data"}
        
        latest = self.resource_history[-1]
        return {
            "cpu_percent": latest.cpu_percent,
            "memory_percent": latest.memory_percent,
            "disk_percent": latest.disk_usage_percent,
            "available_memory_gb": latest.available_memory_gb,
            "status": self._determine_resource_status(latest)
        }
    
    def _determine_resource_status(self, metrics: ResourceMetrics) -> str:
        """Determine overall resource status"""
        if (metrics.cpu_percent > 90 or 
            metrics.memory_percent > 95 or 
            metrics.disk_usage_percent > 95):
            return "critical"
        elif (metrics.cpu_percent > self.resource_thresholds["cpu_high"] or
              metrics.memory_percent > self.resource_thresholds["memory_high"] or
              metrics.disk_usage_percent > self.resource_thresholds["disk_high"]):
            return "high"
        elif metrics.cpu_percent > 50 or metrics.memory_percent > 60:
            return "medium"
        else:
            return "low"

class TrainingScheduler:
    """Schedule training based on resource availability"""
    
    def __init__(self):
        self.training_queue = asyncio.Queue()
        self.active_training = False
    
    async def initialize(self):
        """Initialize training scheduler"""
        asyncio.create_task(self._training_scheduler_loop())
        logger.info("⏰ Training Scheduler initialized")
    
    async def _training_scheduler_loop(self):
        """Main training scheduling loop"""
        while True:
            try:
                # Check resource availability
                resource_status = psutil.cpu_percent(interval=1)
                
                # Schedule training during low resource usage
                if resource_status < 60 and not self.active_training:
                    # Check if training is queued
                    if not self.training_queue.empty():
                        training_task = await self.training_queue.get()
                        await self._execute_training(training_task)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Training scheduler error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _execute_training(self, training_task: Dict[str, Any]):
        """Execute training task"""
        try:
            self.active_training = True
            logger.info(f"🎯 Starting training: {training_task.get('model_id', 'unknown')}")
            
            # Simulate training execution
            await asyncio.sleep(5)  # Placeholder for actual training
            
            logger.info("✅ Training completed")
            
        except Exception as e:
            logger.error(f"Training execution error: {str(e)}")
        finally:
            self.active_training = False 