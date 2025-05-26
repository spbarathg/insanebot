"""
Ant Bot Ultimate Bot System - Main coordinator implementing complete ant colony architecture

This is the main entry point that coordinates all components of the Ant Bot system:
- Founding Ant Queen and Queen hierarchy
- Worker Ants with compounding behavior
- AI Intelligence coordination (Grok + Local LLM)
- 5-layer compounding system
- Flywheel effect implementation
- Lifecycle management (splitting, merging, retirement)
- Extension and modularity framework
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .colony.founding_queen import FoundingAntQueen
from .colony.ant_queen import AntQueen
from .colony.ant_princess import AntPrincess
from .flywheel.feedback_loops import FeedbackLoops
from .flywheel.architecture_iteration import ArchitectureIteration
from .flywheel.performance_amplification import PerformanceAmplification
from .compounding.monetary_layer import MonetaryLayer
from .compounding.worker_layer import WorkerLayer
from .compounding.carwash_layer import CarwashLayer
from .compounding.intelligence_layer import IntelligenceLayer
from .compounding.data_layer import DataLayer
from .extensions.plugin_system import PluginSystem
from .core.config_manager import ConfigManager
from .core.logger import SystemLogger
from .core.system_metrics import SystemMetrics
from .core.security_manager import SecurityManager

logger = logging.getLogger(__name__)

@dataclass
class SystemStatus:
    """Overall system status tracking"""
    initialized: bool = False
    running: bool = False
    total_queens: int = 0
    total_workers: int = 0
    total_capital_sol: float = 0.0
    total_capital_usd: float = 0.0
    system_uptime: float = 0.0
    last_update: float = 0.0
    flywheel_score: float = 0.0
    compounding_efficiency: float = 0.0

class AntBotSystem:
    """Main Ant Bot Ultimate Bot system coordinator"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Core system components
        self.config_manager = ConfigManager(config_path)
        self.system_logger = SystemLogger()
        self.security_manager = SecurityManager()
        self.system_metrics = SystemMetrics()
        
        # System status
        self.status = SystemStatus()
        self.start_time = time.time()
        
        # Ant hierarchy
        self.founding_queen: Optional[FoundingAntQueen] = None
        self.queens: Dict[str, AntQueen] = {}
        self.princess: Optional[AntPrincess] = None
        
        # Flywheel systems
        self.feedback_loops: Optional[FeedbackLoops] = None
        self.architecture_iteration: Optional[ArchitectureIteration] = None
        self.performance_amplification: Optional[PerformanceAmplification] = None
        
        # Compounding layers
        self.compounding_layers = {
            "monetary": None,
            "worker": None,
            "carwash": None,
            "intelligence": None,
            "data": None
        }
        
        # Extension system
        self.plugin_system: Optional[PluginSystem] = None
        
        # Operation control
        self.running = False
        self.cycle_interval = 30.0  # 30 seconds between cycles
        
        logger.info("AntBotSystem initialized")
    
    async def initialize(self) -> bool:
        """Initialize the complete Ant Bot system"""
        try:
            logger.info("Starting Ant Bot Ultimate Bot system initialization...")
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize compounding layers
            await self._initialize_compounding_layers()
            
            # Initialize flywheel systems
            await self._initialize_flywheel_systems()
            
            # Initialize ant hierarchy
            await self._initialize_ant_hierarchy()
            
            # Initialize extension system
            await self._initialize_extension_system()
            
            # Perform system validation
            if await self._validate_system():
                self.status.initialized = True
                logger.info("Ant Bot Ultimate Bot system initialized successfully")
                return True
            else:
                logger.error("System validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Ant Bot system: {e}")
            return False
    
    async def _initialize_core_components(self):
        """Initialize core system components"""
        # Initialize configuration
        self.config_manager.load_config()
        
        # Initialize security
        await self.security_manager.initialize()
        
        # Initialize logging system
        await self.system_logger.initialize()
        
        # Initialize metrics collection
        await self.system_metrics.initialize()
        
        logger.info("Core components initialized")
    
    async def _initialize_compounding_layers(self):
        """Initialize all 5 compounding layers"""
        try:
            # Layer 1: Monetary compounding
            self.compounding_layers["monetary"] = MonetaryLayer()
            await self.compounding_layers["monetary"].initialize()
            
            # Layer 2: Worker compounding 
            self.compounding_layers["worker"] = WorkerLayer()
            await self.compounding_layers["worker"].initialize()
            
            # Layer 3: Carwash (cleanup/reset) compounding
            self.compounding_layers["carwash"] = CarwashLayer()
            await self.compounding_layers["carwash"].initialize()
            
            # Layer 4: Intelligence compounding
            self.compounding_layers["intelligence"] = IntelligenceLayer()
            await self.compounding_layers["intelligence"].initialize()
            
            # Layer 5: Data compounding
            self.compounding_layers["data"] = DataLayer()
            await self.compounding_layers["data"].initialize()
            
            logger.info("All 5 compounding layers initialized")
            
        except Exception as e:
            logger.error(f"Error initializing compounding layers: {e}")
            raise
    
    async def _initialize_flywheel_systems(self):
        """Initialize flywheel effect systems"""
        try:
            # Feedback loops for AI improvement
            self.feedback_loops = FeedbackLoops()
            await self.feedback_loops.initialize()
            
            # Architecture iteration system
            self.architecture_iteration = ArchitectureIteration()
            await self.architecture_iteration.initialize()
            
            # Performance amplification mechanisms
            self.performance_amplification = PerformanceAmplification()
            await self.performance_amplification.initialize()
            
            # Connect flywheel systems
            await self._connect_flywheel_systems()
            
            logger.info("Flywheel systems initialized")
            
        except Exception as e:
            logger.error(f"Error initializing flywheel systems: {e}")
            raise
    
    async def _initialize_ant_hierarchy(self):
        """Initialize the ant colony hierarchy"""
        try:
            # Get initial capital from configuration
            initial_capital = self.config_manager.get("initial_capital", 20.0)
            
            # Create Founding Ant Queen
            founding_queen_id = "founding_queen_0"
            self.founding_queen = FoundingAntQueen(founding_queen_id, initial_capital)
            
            if not await self.founding_queen.initialize():
                raise Exception("Failed to initialize Founding Ant Queen")
            
            # Create Ant Princess for accumulation
            princess_id = "ant_princess_0"
            self.princess = AntPrincess(princess_id)
            await self.princess.initialize()
            
            # Create initial Ant Queen if we have enough capital
            if initial_capital >= 2.0:
                await self._create_initial_queen()
            
            logger.info("Ant hierarchy initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ant hierarchy: {e}")
            raise
    
    async def _create_initial_queen(self) -> bool:
        """Create the first Ant Queen under Founding Queen"""
        try:
            queen_id = f"queen_{len(self.queens) + 1}"
            initial_capital = 2.0
            
            # Create and initialize Queen
            queen = AntQueen(queen_id, self.founding_queen.ant_id, initial_capital)
            
            if await queen.initialize():
                self.queens[queen_id] = queen
                self.founding_queen.children.append(queen_id)
                self.status.total_queens += 1
                
                logger.info(f"Created initial Queen: {queen_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error creating initial queen: {e}")
            return False
    
    async def _initialize_extension_system(self):
        """Initialize the plugin and extension system"""
        try:
            self.plugin_system = PluginSystem()
            await self.plugin_system.initialize()
            
            # Load available plugins
            await self.plugin_system.discover_plugins()
            
            logger.info("Extension system initialized")
            
        except Exception as e:
            logger.error(f"Error initializing extension system: {e}")
            # Non-critical, continue without extensions
            pass
    
    async def _connect_flywheel_systems(self):
        """Connect flywheel systems for coordinated operation"""
        # Connect feedback loops to compounding layers
        for layer_name, layer in self.compounding_layers.items():
            if layer:
                await self.feedback_loops.add_feedback_source(layer_name, layer)
        
        # Connect architecture iteration to system metrics
        await self.architecture_iteration.connect_metrics(self.system_metrics)
        
        # Connect performance amplification to all systems
        await self.performance_amplification.connect_systems(
            feedback_loops=self.feedback_loops,
            compounding_layers=self.compounding_layers,
            architecture_iteration=self.architecture_iteration
        )
    
    async def start(self) -> bool:
        """Start the Ant Bot system operation"""
        try:
            if not self.status.initialized:
                logger.error("System not initialized. Call initialize() first.")
                return False
            
            if self.running:
                logger.warning("System already running")
                return True
            
            self.running = True
            self.status.running = True
            
            logger.info("Starting Ant Bot Ultimate Bot system operation...")
            
            # Start main operation loop
            asyncio.create_task(self._main_operation_loop())
            
            # Start monitoring tasks
            asyncio.create_task(self._monitoring_loop())
            asyncio.create_task(self._flywheel_loop())
            asyncio.create_task(self._compounding_loop())
            
            logger.info("Ant Bot system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Ant Bot system: {e}")
            self.running = False
            self.status.running = False
            return False
    
    async def _main_operation_loop(self):
        """Main system operation loop"""
        while self.running:
            try:
                cycle_start = time.time()
                
                # Execute Founding Queen cycle
                if self.founding_queen:
                    founding_result = await self.founding_queen.execute_cycle()
                    await self._process_founding_queen_result(founding_result)
                
                # Execute Queen cycles
                for queen_id, queen in self.queens.items():
                    queen_result = await queen.execute_cycle()
                    await self._process_queen_result(queen_id, queen_result)
                
                # Execute Princess cycle
                if self.princess:
                    princess_result = await self.princess.execute_cycle()
                    await self._process_princess_result(princess_result)
                
                # Update system status
                await self._update_system_status()
                
                # Calculate cycle duration and sleep
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, self.cycle_interval - cycle_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in main operation loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _process_founding_queen_result(self, result: Dict[str, Any]):
        """Process results from Founding Queen cycle"""
        if result.get("status") == "splitting":
            # Handle Founding Queen creating new Queens
            await self._handle_founding_queen_split(result)
        
        elif result.get("new_queens"):
            # Handle new Queen creation
            for queen_data in result["new_queens"]:
                await self._create_queen_from_data(queen_data)
    
    async def _process_queen_result(self, queen_id: str, result: Dict[str, Any]):
        """Process results from Queen cycle"""
        if result.get("status") == "splitting":
            # Handle Queen splitting
            await self._handle_queen_split(queen_id, result.get("split_result"))
        
        # Update worker count
        worker_count = result.get("queen_metrics", {}).get("queen_specific", {}).get("workers", {}).get("currently_active", 0)
        self.status.total_workers += worker_count
    
    async def _process_princess_result(self, result: Dict[str, Any]):
        """Process results from Princess cycle"""
        # Handle accumulation and withdrawal logic
        if result.get("withdrawal_ready"):
            await self._handle_princess_withdrawal(result)
    
    async def _handle_queen_split(self, parent_queen_id: str, split_data: Dict[str, Any]):
        """Handle Queen splitting when $1500 threshold is reached"""
        try:
            if not split_data or split_data.get("type") != "queen_split":
                return
            
            # Create new Queen
            new_queen_id = f"queen_{len(self.queens) + 1}"
            new_capital = split_data.get("new_queen_capital", 1.0)
            
            new_queen = AntQueen(new_queen_id, self.founding_queen.ant_id, new_capital)
            
            # Apply inheritance data
            inheritance_data = split_data.get("inheritance_data", {})
            if inheritance_data:
                # Apply learned behaviors and strategies
                await self._apply_queen_inheritance(new_queen, inheritance_data)
            
            if await new_queen.initialize():
                self.queens[new_queen_id] = new_queen
                self.status.total_queens += 1
                
                # Original Queen ceases operation (as per requirements)
                original_queen = self.queens.get(parent_queen_id)
                if original_queen:
                    await original_queen.cleanup()
                    del self.queens[parent_queen_id]
                
                logger.info(f"Queen split: {parent_queen_id} -> {new_queen_id}, original ceased")
            
        except Exception as e:
            logger.error(f"Error handling Queen split: {e}")
    
    async def _apply_queen_inheritance(self, new_queen: AntQueen, inheritance_data: Dict[str, Any]):
        """Apply inheritance data to new Queen"""
        try:
            # Apply AI learning data
            ai_data = inheritance_data.get("ai_learning_data", {})
            if ai_data and new_queen.ant_drone:
                await new_queen.ant_drone.learning_engine.import_knowledge_base(ai_data)
            
            # Apply worker strategies
            worker_strategies = inheritance_data.get("worker_strategies", {})
            if worker_strategies:
                # Store strategies for future worker creation
                new_queen.metadata["inherited_strategies"] = worker_strategies
            
            # Apply performance patterns
            performance_patterns = inheritance_data.get("performance_patterns", {})
            if performance_patterns:
                new_queen.metadata["inherited_patterns"] = performance_patterns
            
        except Exception as e:
            logger.error(f"Error applying Queen inheritance: {e}")
    
    async def _flywheel_loop(self):
        """Flywheel effect processing loop"""
        while self.running:
            try:
                # Update feedback loops
                if self.feedback_loops:
                    await self.feedback_loops.process_feedback_cycle()
                
                # Update architecture iteration
                if self.architecture_iteration:
                    iteration_result = await self.architecture_iteration.iterate_architecture()
                    if iteration_result.get("improvements"):
                        await self._apply_architecture_improvements(iteration_result["improvements"])
                
                # Update performance amplification
                if self.performance_amplification:
                    await self.performance_amplification.amplify_performance()
                
                # Calculate flywheel score
                self.status.flywheel_score = await self._calculate_flywheel_score()
                
                await asyncio.sleep(60)  # Flywheel processes every minute
                
            except Exception as e:
                logger.error(f"Error in flywheel loop: {e}")
                await asyncio.sleep(10)
    
    async def _compounding_loop(self):
        """Compounding layer processing loop"""
        while self.running:
            try:
                # Process each compounding layer
                for layer_name, layer in self.compounding_layers.items():
                    if layer:
                        await layer.process_compounding_cycle()
                
                # Calculate overall compounding efficiency
                self.status.compounding_efficiency = await self._calculate_compounding_efficiency()
                
                await asyncio.sleep(120)  # Compounding processes every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in compounding loop: {e}")
                await asyncio.sleep(15)
    
    async def _monitoring_loop(self):
        """System monitoring and health check loop"""
        while self.running:
            try:
                # Update system metrics
                await self.system_metrics.update_metrics({
                    "total_queens": self.status.total_queens,
                    "total_workers": self.status.total_workers,
                    "total_capital_sol": self.status.total_capital_sol,
                    "total_capital_usd": self.status.total_capital_usd,
                    "flywheel_score": self.status.flywheel_score,
                    "compounding_efficiency": self.status.compounding_efficiency,
                    "system_uptime": time.time() - self.start_time
                })
                
                # Perform health checks
                health_status = await self._perform_health_checks()
                
                if not health_status.get("healthy", True):
                    logger.warning(f"System health issues detected: {health_status}")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _update_system_status(self):
        """Update overall system status"""
        try:
            # Calculate total capital
            total_sol = 0.0
            
            if self.founding_queen:
                total_sol += self.founding_queen.capital.current_balance
            
            for queen in self.queens.values():
                total_sol += queen.capital.current_balance
                # Add worker capital
                for worker in queen.workers.values():
                    total_sol += worker.capital.current_balance
            
            if self.princess:
                total_sol += self.princess.capital.current_balance
            
            self.status.total_capital_sol = total_sol
            
            # Convert to USD (simplified)
            sol_price = 100.0  # Placeholder
            self.status.total_capital_usd = total_sol * sol_price
            
            # Update worker count
            total_workers = 0
            for queen in self.queens.values():
                total_workers += len(queen.workers)
            self.status.total_workers = total_workers
            
            # Update uptime
            self.status.system_uptime = time.time() - self.start_time
            self.status.last_update = time.time()
            
        except Exception as e:
            logger.error(f"Error updating system status: {e}")
    
    async def _calculate_flywheel_score(self) -> float:
        """Calculate overall flywheel effectiveness score"""
        try:
            scores = []
            
            if self.feedback_loops:
                scores.append(await self.feedback_loops.get_effectiveness_score())
            
            if self.architecture_iteration:
                scores.append(await self.architecture_iteration.get_improvement_score())
            
            if self.performance_amplification:
                scores.append(await self.performance_amplification.get_amplification_score())
            
            return sum(scores) / len(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating flywheel score: {e}")
            return 0.0
    
    async def _calculate_compounding_efficiency(self) -> float:
        """Calculate overall compounding efficiency"""
        try:
            efficiencies = []
            
            for layer in self.compounding_layers.values():
                if layer:
                    efficiency = await layer.get_efficiency_score()
                    efficiencies.append(efficiency)
            
            return sum(efficiencies) / len(efficiencies) if efficiencies else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating compounding efficiency: {e}")
            return 0.0
    
    async def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform comprehensive system health checks"""
        health_status = {"healthy": True, "issues": []}
        
        try:
            # Check Founding Queen health
            if not self.founding_queen or self.founding_queen.status != "active":
                health_status["issues"].append("Founding Queen not active")
                health_status["healthy"] = False
            
            # Check Queen health
            inactive_queens = [qid for qid, q in self.queens.items() if q.status != "active"]
            if inactive_queens:
                health_status["issues"].append(f"Inactive Queens: {inactive_queens}")
            
            # Check capital levels
            if self.status.total_capital_sol < 1.0:
                health_status["issues"].append("Low capital levels")
                health_status["healthy"] = False
            
            # Check flywheel performance
            if self.status.flywheel_score < 0.3:
                health_status["issues"].append("Poor flywheel performance")
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
            return {"healthy": False, "issues": [f"Health check error: {e}"]}
    
    async def _validate_system(self) -> bool:
        """Validate system initialization"""
        try:
            # Check core components
            if not self.config_manager or not self.system_logger:
                return False
            
            # Check compounding layers
            if not all(self.compounding_layers.values()):
                logger.warning("Some compounding layers not initialized")
            
            # Check ant hierarchy
            if not self.founding_queen:
                return False
            
            # Check flywheel systems
            if not self.feedback_loops:
                logger.warning("Feedback loops not initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"System validation error: {e}")
            return False
    
    async def stop(self):
        """Stop the Ant Bot system"""
        try:
            logger.info("Stopping Ant Bot Ultimate Bot system...")
            
            self.running = False
            self.status.running = False
            
            # Cleanup all components
            if self.founding_queen:
                await self.founding_queen.cleanup()
            
            for queen in self.queens.values():
                await queen.cleanup()
            
            if self.princess:
                await self.princess.cleanup()
            
            logger.info("Ant Bot system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system": {
                "initialized": self.status.initialized,
                "running": self.status.running,
                "uptime": self.status.system_uptime,
                "last_update": self.status.last_update
            },
            "hierarchy": {
                "founding_queen_active": self.founding_queen is not None,
                "total_queens": self.status.total_queens,
                "total_workers": self.status.total_workers,
                "princess_active": self.princess is not None
            },
            "capital": {
                "total_sol": self.status.total_capital_sol,
                "total_usd": self.status.total_capital_usd
            },
            "performance": {
                "flywheel_score": self.status.flywheel_score,
                "compounding_efficiency": self.status.compounding_efficiency
            },
            "compounding_layers": {
                layer_name: layer is not None 
                for layer_name, layer in self.compounding_layers.items()
            }
        }

# Utility function to create and run the system
async def create_and_run_ant_bot(config_path: Optional[str] = None) -> AntBotSystem:
    """Create, initialize, and start the Ant Bot system"""
    system = AntBotSystem(config_path)
    
    if await system.initialize():
        if await system.start():
            logger.info("Ant Bot Ultimate Bot system is running")
            return system
        else:
            logger.error("Failed to start Ant Bot system")
    else:
        logger.error("Failed to initialize Ant Bot system")
    
    return system 