"""
Self-Replication System for Ant Bot with Optimized Resource Management

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

# System Replication Constants
class ReplicationConstants:
    """Centralized constants for system replication"""
    # Default thresholds
    DEFAULT_CAPITAL_THRESHOLD = 50.0  # 50 SOL total system capital
    DEFAULT_PROFIT_THRESHOLD = 20.0   # 20 SOL profit
    DEFAULT_PERFORMANCE_THRESHOLD = 0.8  # 80% win rate
    DEFAULT_TIME_THRESHOLD = 86400      # 24 hours uptime
    
    # Resource limits
    MAX_INSTANCES = 5                # Maximum replicated instances
    MAX_MEMORY_GB_PER_INSTANCE = 2.0  # 2GB memory limit per instance
    BASE_PORT = 8000                 # Starting port for instances
    PORT_SCAN_RANGE = 100           # Number of ports to try
    
    # Resource allocation
    DEFAULT_API_RATE_LIMIT = 100     # Default API rate limit
    MAX_CONCURRENT_TRADES = 3        # Max concurrent trades per instance
    MEMORY_SAFETY_FACTOR = 0.8      # Use 80% of available memory
    
    # Process management
    STARTUP_TIMEOUT_SECONDS = 60    # Instance startup timeout
    SHUTDOWN_TIMEOUT_SECONDS = 30   # Instance shutdown timeout
    HEALTH_CHECK_INTERVAL = 30      # Health check interval
    
    # File management
    IGNORE_PATTERNS = [
        'instances', 'logs/*', 'data/*', '__pycache__', 
        '*.pyc', '.git', 'htmlcov', '.pytest_cache',
        '*.log', 'node_modules', '.env'
    ]

@dataclass
class ReplicationTrigger:
    """Conditions that trigger system replication with validation"""
    capital_threshold: float = ReplicationConstants.DEFAULT_CAPITAL_THRESHOLD
    profit_threshold: float = ReplicationConstants.DEFAULT_PROFIT_THRESHOLD
    performance_threshold: float = ReplicationConstants.DEFAULT_PERFORMANCE_THRESHOLD
    time_threshold: int = ReplicationConstants.DEFAULT_TIME_THRESHOLD
    max_instances: int = ReplicationConstants.MAX_INSTANCES
    
    def __post_init__(self):
        """Validate trigger parameters"""
        if self.capital_threshold <= 0:
            raise ValueError(f"Capital threshold must be positive, got: {self.capital_threshold}")
        if self.profit_threshold <= 0:
            raise ValueError(f"Profit threshold must be positive, got: {self.profit_threshold}")
        if not 0 < self.performance_threshold <= 1:
            raise ValueError(f"Performance threshold must be between 0 and 1, got: {self.performance_threshold}")
        if self.time_threshold <= 0:
            raise ValueError(f"Time threshold must be positive, got: {self.time_threshold}")
        if self.max_instances <= 0 or self.max_instances > 10:
            raise ValueError(f"Max instances must be between 1 and 10, got: {self.max_instances}")

@dataclass
class InstanceConfig:
    """Configuration for a replicated instance with validation"""
    instance_id: str
    parent_instance_id: Optional[str]
    allocated_capital: float
    port_offset: int
    workspace_path: str
    config_overrides: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    status: str = "initializing"
    pid: Optional[int] = None
    
    def __post_init__(self):
        """Validate instance configuration"""
        if not self.instance_id:
            raise ValueError("Instance ID cannot be empty")
        if self.allocated_capital <= 0:
            raise ValueError(f"Allocated capital must be positive, got: {self.allocated_capital}")
        if not os.path.exists(self.workspace_path):
            raise ValueError(f"Workspace path does not exist: {self.workspace_path}")

class ResourceAllocator:
    """Manages resource allocation between instances with safety checks"""
    
    def __init__(self):
        self.allocated_ports = set()
        self.allocated_memory = 0.0
        self.allocated_cpu = 0.0
        self.base_port = ReplicationConstants.BASE_PORT
        
        # Resource tracking
        self._total_system_memory = psutil.virtual_memory().total
        self._total_system_cpu = psutil.cpu_count()
        
        logger.debug(f"ResourceAllocator initialized: {self._total_system_memory/(1024**3):.1f}GB RAM, "
                    f"{self._total_system_cpu} CPU cores")
        
    def allocate_resources(self, instance_count: int) -> Dict[str, Any]:
        """Allocate system resources for new instance with safety checks"""
        try:
            if instance_count <= 0:
                raise ValueError(f"Instance count must be positive, got: {instance_count}")
            
            # Check available system resources
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Calculate safe resource allocation
            available_memory_gb = (memory.available / (1024**3)) * ReplicationConstants.MEMORY_SAFETY_FACTOR
            memory_per_instance = min(
                ReplicationConstants.MAX_MEMORY_GB_PER_INSTANCE, 
                available_memory_gb / (instance_count + 1)
            )
            
            # Ensure minimum viable memory allocation
            if memory_per_instance < 0.5:  # Less than 512MB
                raise Exception(f"Insufficient memory for new instance. Available: {available_memory_gb:.2f}GB")
            
            # CPU allocation with safety margin
            cpu_per_instance = min(0.8, 1.0 / (instance_count + 1))  # Max 80% CPU per instance
            
            # Find available port
            port = self._find_available_port()
            if not port:
                raise Exception("No available ports in range")
            
            self.allocated_ports.add(port)
            self.allocated_memory += memory_per_instance
            self.allocated_cpu += cpu_per_instance
            
            allocation = {
                "port": port,
                "memory_limit_gb": memory_per_instance,
                "cpu_limit": cpu_per_instance,
                "available_memory_gb": available_memory_gb,
                "total_cpu_cores": self._total_system_cpu,
                "system_cpu_usage": cpu_percent
            }
            
            logger.info(f"Resources allocated: Port {port}, Memory {memory_per_instance:.2f}GB, "
                       f"CPU {cpu_per_instance:.1%}")
            return allocation
            
        except Exception as e:
            logger.error(f"Resource allocation error: {str(e)}")
            return {}
    
    def _find_available_port(self) -> Optional[int]:
        """Find an available port for the new instance with improved scanning"""
        import socket
        
        for offset in range(ReplicationConstants.PORT_SCAN_RANGE):
            port = self.base_port + offset
            if port not in self.allocated_ports:
                # Test if port is actually available
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1)  # 1 second timeout
                        s.bind(('localhost', port))
                        return port
                except OSError as e:
                    logger.debug(f"Port {port} unavailable: {str(e)}")
                    continue
        
        logger.error(f"No available ports found in range {self.base_port}-{self.base_port + ReplicationConstants.PORT_SCAN_RANGE}")
        return None
    
    def release_resources(self, port: int, memory_gb: float = 0, cpu_percent: float = 0):
        """Release resources when instance is terminated with tracking"""
        self.allocated_ports.discard(port)
        self.allocated_memory = max(0, self.allocated_memory - memory_gb)
        self.allocated_cpu = max(0, self.allocated_cpu - cpu_percent)
        
        logger.debug(f"Resources released: Port {port}, Memory {memory_gb:.2f}GB, CPU {cpu_percent:.1%}")
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource allocation status"""
        return {
            "allocated_ports": list(self.allocated_ports),
            "allocated_memory_gb": self.allocated_memory,
            "allocated_cpu_percent": self.allocated_cpu,
            "system_memory_gb": self._total_system_memory / (1024**3),
            "system_cpu_cores": self._total_system_cpu,
            "memory_utilization": self.allocated_memory / (self._total_system_memory / (1024**3)),
            "cpu_utilization": self.allocated_cpu
        }

class ConfigurationManager:
    """Manages configuration for replicated instances with validation"""
    
    def __init__(self, base_config_path: str = "config.json"):
        self.base_config_path = base_config_path
        self.base_config = self._load_base_config()
        
        if not self.base_config:
            logger.warning("No base configuration loaded, using defaults")
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration with error handling"""
        try:
            if os.path.exists(self.base_config_path):
                with open(self.base_config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded base configuration from {self.base_config_path}")
                    return config
            else:
                logger.warning(f"Base config file not found: {self.base_config_path}")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in base config: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load base config: {str(e)}")
            return {}
    
    def generate_instance_config(self, instance_id: str, allocated_capital: float, 
                               resources: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration for new instance with validation"""
        try:
            if not instance_id:
                raise ValueError("Instance ID cannot be empty")
            if allocated_capital <= 0:
                raise ValueError(f"Allocated capital must be positive, got: {allocated_capital}")
            if not resources:
                raise ValueError("Resources dictionary cannot be empty")
            
            # Clone base configuration safely
            instance_config = {}
            if self.base_config:
                instance_config = json.loads(json.dumps(self.base_config))  # Deep copy
            
            # Instance-specific overrides with validation
            instance_config.update({
                "instance_id": instance_id,
                "initial_capital": allocated_capital,
                "port": resources.get("port"),
                "memory_limit": resources.get("memory_limit_gb"),
                "cpu_limit": resources.get("cpu_limit"),
                
                # Separate directories with instance prefix
                "log_directory": f"logs/instance_{instance_id}",
                "data_directory": f"data/instance_{instance_id}",
                
                # Database separation
                "database_name": f"antbot_instance_{instance_id}",
                
                # API rate limiting adjustments
                "api_rate_limit": max(10, instance_config.get("api_rate_limit", ReplicationConstants.DEFAULT_API_RATE_LIMIT) // 2),
                
                # Trading parameters with resource constraints
                "trading_parameters": {
                    **instance_config.get("trading_parameters", {}),
                    "instance_id": instance_id,
                    "max_concurrent_trades": ReplicationConstants.MAX_CONCURRENT_TRADES,
                    "resource_constraints": {
                        "memory_limit_gb": resources.get("memory_limit_gb"),
                        "cpu_limit": resources.get("cpu_limit")
                    }
                }
            })
            
            # Validate generated configuration
            required_fields = ["instance_id", "initial_capital", "port"]
            for field in required_fields:
                if field not in instance_config or instance_config[field] is None:
                    raise ValueError(f"Required field missing in generated config: {field}")
            
            logger.debug(f"Generated configuration for instance {instance_id}")
            return instance_config
            
        except Exception as e:
            logger.error(f"Config generation error: {str(e)}")
            return {}
    
    def save_instance_config(self, instance_id: str, config: Dict[str, Any], 
                           workspace_path: str) -> str:
        """Save instance configuration to file with validation"""
        try:
            if not instance_id:
                raise ValueError("Instance ID cannot be empty")
            if not config:
                raise ValueError("Configuration cannot be empty")
            if not os.path.exists(workspace_path):
                raise ValueError(f"Workspace path does not exist: {workspace_path}")
            
            config_path = os.path.join(workspace_path, "config.json")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Save with atomic write
            temp_path = config_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(config, f, indent=2, sort_keys=True)
            
            # Atomic move
            os.replace(temp_path, config_path)
            
            logger.info(f"Saved configuration for instance {instance_id} to {config_path}")
            return config_path
            
        except Exception as e:
            logger.error(f"Config save error: {str(e)}")
            # Cleanup temp file if it exists
            temp_path = os.path.join(workspace_path, "config.json.tmp")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return ""

class WorkspaceManager:
    """Manages workspace isolation for instances with improved error handling"""
    
    def __init__(self, base_workspace: str = "."):
        self.base_workspace = os.path.abspath(base_workspace)
        self.instances_dir = os.path.join(self.base_workspace, "instances")
        
        try:
            os.makedirs(self.instances_dir, exist_ok=True)
            logger.debug(f"WorkspaceManager initialized: base={self.base_workspace}, instances={self.instances_dir}")
        except Exception as e:
            logger.error(f"Failed to create instances directory: {str(e)}")
            raise
    
    def create_instance_workspace(self, instance_id: str) -> str:
        """Create isolated workspace for new instance with proper cleanup"""
        try:
            if not instance_id:
                raise ValueError("Instance ID cannot be empty")
            
            workspace_path = os.path.join(self.instances_dir, instance_id)
            
            # Clean up existing workspace if it exists
            if os.path.exists(workspace_path):
                logger.warning(f"Removing existing workspace: {workspace_path}")
                shutil.rmtree(workspace_path, ignore_errors=True)
            
            # Copy essential files with improved filtering
            try:
                shutil.copytree(
                    self.base_workspace, 
                    workspace_path, 
                    ignore=shutil.ignore_patterns(*ReplicationConstants.IGNORE_PATTERNS),
                    symlinks=False,
                    dirs_exist_ok=False
                )
            except Exception as e:
                logger.error(f"Failed to copy workspace: {str(e)}")
                # Cleanup failed copy
                if os.path.exists(workspace_path):
                    shutil.rmtree(workspace_path, ignore_errors=True)
                raise
            
            # Create instance-specific directories
            required_dirs = ["logs", "data", "temp"]
            for dir_name in required_dirs:
                dir_path = os.path.join(workspace_path, dir_name)
                os.makedirs(dir_path, exist_ok=True)
            
            # Set appropriate permissions (Unix-like systems)
            if hasattr(os, 'chmod'):
                # Set secure directory permissions (accessible by owner only)
                os.chmod(workspace_path, 0o700)
            
            logger.info(f"Created workspace for instance {instance_id} at {workspace_path}")
            return workspace_path
            
        except Exception as e:
            logger.error(f"Workspace creation error: {str(e)}")
            return ""
    
    def cleanup_instance_workspace(self, instance_id: str) -> bool:
        """Clean up workspace when instance is terminated with retry logic"""
        try:
            if not instance_id:
                logger.warning("Cannot cleanup workspace: empty instance ID")
                return False
                
            workspace_path = os.path.join(self.instances_dir, instance_id)
            
            if not os.path.exists(workspace_path):
                logger.warning(f"Workspace does not exist: {workspace_path}")
                return True  # Consider success if already cleaned
            
            # Retry cleanup with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(workspace_path)
                    logger.info(f"Cleaned up workspace for instance {instance_id}")
                    return True
                except OSError as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # 1, 2, 4 seconds
                        logger.warning(f"Cleanup attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to cleanup workspace after {max_retries} attempts: {str(e)}")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Workspace cleanup error: {str(e)}")
            return False
    
    def get_workspace_info(self, instance_id: str) -> Dict[str, Any]:
        """Get information about instance workspace"""
        try:
            workspace_path = os.path.join(self.instances_dir, instance_id)
            
            if not os.path.exists(workspace_path):
                return {"exists": False, "path": workspace_path}
            
            # Get directory size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(workspace_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except:
                        pass  # Skip files we can't read
            
            return {
                "exists": True,
                "path": workspace_path,
                "size_bytes": total_size,
                "size_mb": total_size / (1024 * 1024),
                "created": os.path.getctime(workspace_path),
                "modified": os.path.getmtime(workspace_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting workspace info: {str(e)}")
            return {"error": str(e)}

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