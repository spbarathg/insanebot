"""
Self-Replication System for Ant Bot

This module enables the Ant Bot system to autonomously replicate itself when certain
conditions are met, creating new independent instances with proper state isolation.

Features:
- Automatic system cloning when capital/performance thresholds are reached
- State isolation between instances
- Dynamic configuration management
- Resource allocation for new instances
- Monitoring and coordination between instances
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)

@dataclass
class ReplicationTrigger:
    """Conditions that trigger system replication"""
    capital_threshold: float = 50.0  # 50 SOL total system capital
    profit_threshold: float = 20.0   # 20 SOL profit
    performance_threshold: float = 0.8  # 80% win rate
    time_threshold: int = 86400      # 24 hours uptime
    max_instances: int = 5           # Maximum replicated instances

@dataclass
class InstanceConfig:
    """Configuration for a replicated instance"""
    instance_id: str
    parent_instance_id: Optional[str]
    allocated_capital: float
    port_offset: int
    workspace_path: str
    config_overrides: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    status: str = "initializing"
    pid: Optional[int] = None

class ResourceAllocator:
    """Manages resource allocation between instances"""
    
    def __init__(self):
        self.allocated_ports = set()
        self.allocated_memory = 0
        self.allocated_cpu = 0.0
        self.base_port = 8000
        
    def allocate_resources(self, instance_count: int) -> Dict[str, Any]:
        """Allocate system resources for new instance"""
        try:
            # Check available system resources
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            # Calculate resource allocation per instance
            available_memory_gb = (memory.available // (1024**3))
            memory_per_instance = min(2.0, available_memory_gb / (instance_count + 1))  # Max 2GB per instance
            cpu_per_instance = 1.0 / (instance_count + 1)  # Distribute CPU evenly
            
            # Find available port
            port = self._find_available_port()
            
            if not port:
                raise Exception("No available ports")
            
            self.allocated_ports.add(port)
            
            return {
                "port": port,
                "memory_limit_gb": memory_per_instance,
                "cpu_limit": cpu_per_instance,
                "available_memory_gb": available_memory_gb,
                "total_cpu_cores": cpu_count
            }
            
        except Exception as e:
            logger.error(f"Resource allocation error: {str(e)}")
            return {}
    
    def _find_available_port(self) -> Optional[int]:
        """Find an available port for the new instance"""
        import socket
        
        for offset in range(100):  # Try 100 ports
            port = self.base_port + offset
            if port not in self.allocated_ports:
                # Test if port is actually available
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('localhost', port))
                        return port
                except OSError:
                    continue
        return None
    
    def release_resources(self, port: int):
        """Release resources when instance is terminated"""
        self.allocated_ports.discard(port)

class ConfigurationManager:
    """Manages configuration for replicated instances"""
    
    def __init__(self, base_config_path: str = "config.json"):
        self.base_config_path = base_config_path
        self.base_config = self._load_base_config()
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration"""
        try:
            if os.path.exists(self.base_config_path):
                with open(self.base_config_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load base config: {str(e)}")
            return {}
    
    def generate_instance_config(self, instance_id: str, allocated_capital: float, 
                               resources: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration for new instance"""
        try:
            # Clone base configuration
            instance_config = self.base_config.copy()
            
            # Instance-specific overrides
            instance_config.update({
                "instance_id": instance_id,
                "initial_capital": allocated_capital,
                "port": resources.get("port"),
                "memory_limit": resources.get("memory_limit_gb"),
                "cpu_limit": resources.get("cpu_limit"),
                
                # Separate log directories
                "log_directory": f"logs/instance_{instance_id}",
                "data_directory": f"data/instance_{instance_id}",
                
                # Database separation
                "database_name": f"antbot_instance_{instance_id}",
                
                # API rate limiting adjustments
                "api_rate_limit": instance_config.get("api_rate_limit", 100) // 2,  # Reduce to avoid conflicts
                
                # Trading parameters
                "trading_parameters": {
                    **instance_config.get("trading_parameters", {}),
                    "instance_id": instance_id,
                    "max_concurrent_trades": 3,  # Reduced for resource sharing
                }
            })
            
            return instance_config
            
        except Exception as e:
            logger.error(f"Config generation error: {str(e)}")
            return {}
    
    def save_instance_config(self, instance_id: str, config: Dict[str, Any], 
                           workspace_path: str) -> str:
        """Save instance configuration to file"""
        try:
            config_path = os.path.join(workspace_path, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved configuration for instance {instance_id}")
            return config_path
            
        except Exception as e:
            logger.error(f"Config save error: {str(e)}")
            return ""

class WorkspaceManager:
    """Manages workspace isolation for instances"""
    
    def __init__(self, base_workspace: str = "."):
        self.base_workspace = os.path.abspath(base_workspace)
        self.instances_dir = os.path.join(self.base_workspace, "instances")
        os.makedirs(self.instances_dir, exist_ok=True)
    
    def create_instance_workspace(self, instance_id: str) -> str:
        """Create isolated workspace for new instance"""
        try:
            workspace_path = os.path.join(self.instances_dir, instance_id)
            
            if os.path.exists(workspace_path):
                shutil.rmtree(workspace_path)
            
            # Copy essential files
            shutil.copytree(self.base_workspace, workspace_path, 
                          ignore=shutil.ignore_patterns(
                              'instances', 'logs/*', 'data/*', '__pycache__', 
                              '*.pyc', '.git', 'htmlcov', '.pytest_cache'
                          ))
            
            # Create instance-specific directories
            os.makedirs(os.path.join(workspace_path, "logs"), exist_ok=True)
            os.makedirs(os.path.join(workspace_path, "data"), exist_ok=True)
            
            logger.info(f"Created workspace for instance {instance_id} at {workspace_path}")
            return workspace_path
            
        except Exception as e:
            logger.error(f"Workspace creation error: {str(e)}")
            return ""
    
    def cleanup_instance_workspace(self, instance_id: str) -> bool:
        """Clean up workspace when instance is terminated"""
        try:
            workspace_path = os.path.join(self.instances_dir, instance_id)
            if os.path.exists(workspace_path):
                shutil.rmtree(workspace_path)
                logger.info(f"Cleaned up workspace for instance {instance_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Workspace cleanup error: {str(e)}")
            return False

class ProcessManager:
    """Manages process lifecycle for replicated instances"""
    
    def __init__(self):
        self.running_instances: Dict[str, subprocess.Popen] = {}
    
    def start_instance(self, instance_config: InstanceConfig) -> bool:
        """Start a new instance process"""
        try:
            # Prepare environment variables
            env = os.environ.copy()
            env.update({
                "INSTANCE_ID": instance_config.instance_id,
                "INSTANCE_CAPITAL": str(instance_config.allocated_capital),
                "PYTHONPATH": instance_config.workspace_path,
            })
            
            # Start process in instance workspace
            cmd = ["python", "src/main.py"]
            process = subprocess.Popen(
                cmd,
                cwd=instance_config.workspace_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            self.running_instances[instance_config.instance_id] = process
            instance_config.pid = process.pid
            instance_config.status = "running"
            
            logger.info(f"Started instance {instance_config.instance_id} with PID {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start instance {instance_config.instance_id}: {str(e)}")
            instance_config.status = "failed"
            return False
    
    def stop_instance(self, instance_id: str) -> bool:
        """Stop a running instance"""
        try:
            if instance_id in self.running_instances:
                process = self.running_instances[instance_id]
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                
                del self.running_instances[instance_id]
                logger.info(f"Stopped instance {instance_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to stop instance {instance_id}: {str(e)}")
            return False
    
    def check_instance_health(self, instance_id: str) -> bool:
        """Check if instance is still running and healthy"""
        try:
            if instance_id in self.running_instances:
                process = self.running_instances[instance_id]
                return process.poll() is None
            return False
            
        except Exception as e:
            logger.error(f"Health check error for {instance_id}: {str(e)}")
            return False

class SystemReplicator:
    """Main system replication coordinator"""
    
    def __init__(self, founding_queen_instance):
        self.founding_queen = founding_queen_instance
        self.replication_trigger = ReplicationTrigger()
        self.resource_allocator = ResourceAllocator()
        self.config_manager = ConfigurationManager()
        self.workspace_manager = WorkspaceManager()
        self.process_manager = ProcessManager()
        
        self.instances: Dict[str, InstanceConfig] = {}
        self.replication_history: List[Dict[str, Any]] = []
        self.last_replication_check = time.time()
        
    async def monitor_replication_conditions(self) -> bool:
        """Monitor system conditions for replication triggers"""
        try:
            # Get current system metrics
            system_status = self.founding_queen.get_system_status()
            system_metrics = system_status.get("system_metrics", {})
            
            current_time = time.time()
            
            # Check each trigger condition
            triggers_met = []
            
            # Capital threshold
            total_capital = system_metrics.get("total_capital", 0)
            if total_capital >= self.replication_trigger.capital_threshold:
                triggers_met.append(f"Capital threshold: {total_capital:.2f} >= {self.replication_trigger.capital_threshold}")
            
            # Profit threshold
            system_profit = system_metrics.get("system_profit", 0)
            if system_profit >= self.replication_trigger.profit_threshold:
                triggers_met.append(f"Profit threshold: {system_profit:.2f} >= {self.replication_trigger.profit_threshold}")
            
            # Time threshold
            uptime = system_metrics.get("uptime", 0)
            if uptime >= self.replication_trigger.time_threshold:
                triggers_met.append(f"Uptime threshold: {uptime/3600:.1f}h >= {self.replication_trigger.time_threshold/3600:.1f}h")
            
            # Performance threshold (average win rate across all princesses)
            total_trades = system_metrics.get("total_trades", 0)
            if total_trades > 10:  # Need some trades to evaluate performance
                avg_win_rate = self._calculate_system_win_rate(system_status)
                if avg_win_rate >= self.replication_trigger.performance_threshold:
                    triggers_met.append(f"Performance threshold: {avg_win_rate:.2f} >= {self.replication_trigger.performance_threshold}")
            
            # Check instance limit
            active_instances = len([i for i in self.instances.values() if i.status == "running"])
            if active_instances >= self.replication_trigger.max_instances:
                logger.info(f"Max instances reached: {active_instances}/{self.replication_trigger.max_instances}")
                return False
            
            # Check replication cooldown (don't replicate too frequently)
            time_since_last = current_time - self.last_replication_check
            if time_since_last < 3600:  # 1 hour cooldown
                return False
            
            # Need at least 2 triggers to replicate
            if len(triggers_met) >= 2:
                logger.info(f"Replication conditions met: {triggers_met}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Replication monitoring error: {str(e)}")
            return False
    
    def _calculate_system_win_rate(self, system_status: Dict[str, Any]) -> float:
        """Calculate overall system win rate"""
        try:
            total_trades = 0
            total_wins = 0
            
            for queen_detail in system_status.get("queen_details", []):
                for princess_detail in queen_detail.get("princess_details", []):
                    performance = princess_detail.get("performance", {})
                    trades = performance.get("total_trades", 0)
                    win_rate = performance.get("win_rate", 0)
                    
                    total_trades += trades
                    total_wins += trades * (win_rate / 100.0)
            
            return (total_wins / total_trades) if total_trades > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Win rate calculation error: {str(e)}")
            return 0.0
    
    async def replicate_system(self) -> Optional[str]:
        """Create a new system instance"""
        try:
            # Generate instance ID
            instance_id = f"ant_instance_{int(time.time())}_{len(self.instances)}"
            
            # Calculate capital allocation (split current capital)
            system_metrics = self.founding_queen.get_system_status().get("system_metrics", {})
            total_capital = system_metrics.get("total_capital", 0)
            allocated_capital = min(total_capital * 0.3, 10.0)  # Allocate 30% or max 10 SOL
            
            if allocated_capital < 2.0:  # Need minimum capital to start
                logger.warning(f"Insufficient capital for replication: {allocated_capital}")
                return None
            
            # Allocate system resources
            resources = self.resource_allocator.allocate_resources(len(self.instances))
            if not resources:
                logger.error("Failed to allocate resources for new instance")
                return None
            
            # Create workspace
            workspace_path = self.workspace_manager.create_instance_workspace(instance_id)
            if not workspace_path:
                logger.error("Failed to create workspace for new instance")
                return None
            
            # Generate configuration
            instance_config_data = self.config_manager.generate_instance_config(
                instance_id, allocated_capital, resources
            )
            
            config_path = self.config_manager.save_instance_config(
                instance_id, instance_config_data, workspace_path
            )
            
            # Create instance configuration
            instance_config = InstanceConfig(
                instance_id=instance_id,
                parent_instance_id=self.founding_queen.ant_id,
                allocated_capital=allocated_capital,
                port_offset=resources["port"] - self.resource_allocator.base_port,
                workspace_path=workspace_path,
                config_overrides=instance_config_data
            )
            
            # Start the new instance
            if self.process_manager.start_instance(instance_config):
                self.instances[instance_id] = instance_config
                self.last_replication_check = time.time()
                
                # Record replication event
                replication_event = {
                    "instance_id": instance_id,
                    "timestamp": time.time(),
                    "allocated_capital": allocated_capital,
                    "parent_capital": total_capital,
                    "trigger_reason": "Automatic replication based on performance thresholds"
                }
                self.replication_history.append(replication_event)
                
                # Deduct capital from parent
                await self._transfer_capital_to_instance(allocated_capital)
                
                logger.info(f"Successfully replicated system as instance {instance_id}")
                return instance_id
            else:
                # Cleanup on failure
                self.workspace_manager.cleanup_instance_workspace(instance_id)
                self.resource_allocator.release_resources(resources["port"])
                logger.error(f"Failed to start replicated instance {instance_id}")
                return None
                
        except Exception as e:
            logger.error(f"System replication error: {str(e)}")
            return None
    
    async def _transfer_capital_to_instance(self, amount: float) -> bool:
        """Transfer capital from parent to new instance"""
        try:
            # This would involve actual wallet transfers in production
            # For now, we'll just update the founding queen's capital
            current_balance = self.founding_queen.capital.current_balance
            if current_balance >= amount:
                self.founding_queen.capital.update_balance(current_balance - amount)
                logger.info(f"Transferred {amount} SOL to new instance")
                return True
            else:
                logger.error(f"Insufficient capital for transfer: {current_balance} < {amount}")
                return False
                
        except Exception as e:
            logger.error(f"Capital transfer error: {str(e)}")
            return False
    
    async def manage_instances(self) -> Dict[str, Any]:
        """Manage lifecycle of all instances"""
        try:
            management_results = {
                "health_checks": {},
                "terminated_instances": [],
                "resource_usage": {}
            }
            
            # Health check all instances
            for instance_id, instance_config in list(self.instances.items()):
                is_healthy = self.process_manager.check_instance_health(instance_id)
                management_results["health_checks"][instance_id] = is_healthy
                
                if not is_healthy and instance_config.status == "running":
                    logger.warning(f"Instance {instance_id} appears unhealthy, terminating")
                    await self.terminate_instance(instance_id)
                    management_results["terminated_instances"].append(instance_id)
            
            # Monitor resource usage
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            management_results["resource_usage"] = {
                "memory_percent": memory.percent,
                "cpu_percent": cpu_percent,
                "active_instances": len([i for i in self.instances.values() if i.status == "running"]),
                "total_instances": len(self.instances)
            }
            
            return management_results
            
        except Exception as e:
            logger.error(f"Instance management error: {str(e)}")
            return {"error": str(e)}
    
    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a specific instance"""
        try:
            if instance_id not in self.instances:
                return False
            
            instance_config = self.instances[instance_id]
            
            # Stop the process
            self.process_manager.stop_instance(instance_id)
            
            # Release resources
            if hasattr(instance_config, 'port_offset'):
                port = self.resource_allocator.base_port + instance_config.port_offset
                self.resource_allocator.release_resources(port)
            
            # Cleanup workspace
            self.workspace_manager.cleanup_instance_workspace(instance_id)
            
            # Update status
            instance_config.status = "terminated"
            
            logger.info(f"Terminated instance {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Instance termination error: {str(e)}")
            return False
    
    def get_replication_status(self) -> Dict[str, Any]:
        """Get comprehensive replication system status"""
        try:
            return {
                "total_instances": len(self.instances),
                "active_instances": len([i for i in self.instances.values() if i.status == "running"]),
                "failed_instances": len([i for i in self.instances.values() if i.status == "failed"]),
                "replication_trigger": {
                    "capital_threshold": self.replication_trigger.capital_threshold,
                    "profit_threshold": self.replication_trigger.profit_threshold,
                    "performance_threshold": self.replication_trigger.performance_threshold,
                    "max_instances": self.replication_trigger.max_instances
                },
                "instances": {
                    instance_id: {
                        "status": config.status,
                        "allocated_capital": config.allocated_capital,
                        "created_at": config.created_at,
                        "uptime": time.time() - config.created_at if config.status == "running" else 0,
                        "pid": config.pid
                    }
                    for instance_id, config in self.instances.items()
                },
                "replication_history": self.replication_history,
                "next_check_in": max(0, 3600 - (time.time() - self.last_replication_check))
            }
            
        except Exception as e:
            logger.error(f"Status summary error: {str(e)}")
            return {"error": str(e)} 