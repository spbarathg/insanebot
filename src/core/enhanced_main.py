"""
Enhanced Main Coordinator for Ant Bot System

This is the new main entry point that integrates:
- Ant Hierarchy System (Founding Queen -> Queens -> Princesses)
- Enhanced AI Coordination with learning feedback loops
- Self-Replication System for autonomous scaling
- Complete Worker Ant lifecycle management

This replaces the old main.py with a true Ant Bot architecture.
"""

import asyncio
import json
import logging
import time
import signal
import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.ai.ant_hierarchy import FoundingAntQueen, AntRole
from src.core.ai.enhanced_ai_coordinator import AICoordinator
from src.core.system_replicator import SystemReplicator
from src.core.wallet_manager import WalletManager
from src.core.portfolio_risk_manager import PortfolioRiskManager
from src.core.data_ingestion import DataIngestion
from src.core.helius_service import HeliusService
from src.core.jupiter_service import JupiterService
from src.core.quicknode_service import QuickNodeService

logger = logging.getLogger(__name__)

class AntBotSystem:
    """
    Enhanced Ant Bot System implementing the complete architecture:
    
    - Founding Ant Queen: Top-level system coordinator
    - Ant Queens: Manage pools of Worker Ants (Princesses)
    - Ant Princesses: Individual trading agents with 5-10 trade lifecycle
    - AI Coordinator: Manages Grok + Local LLM collaboration with learning
    - System Replicator: Handles autonomous replication and scaling
    """
    
    def __init__(self, initial_capital: float = 20.0):
        self.initial_capital = initial_capital
        self.start_time = time.time()
        self.shutdown_requested = False
        
        # Core Components
        self.founding_queen: Optional[FoundingAntQueen] = None
        self.ai_coordinator: Optional[AICoordinator] = None
        self.system_replicator: Optional[SystemReplicator] = None
        self.wallet_manager: Optional[WalletManager] = None
        self.portfolio_risk_manager: Optional[PortfolioRiskManager] = None
        self.data_ingestion: Optional[DataIngestion] = None
        
        # External Services
        self.helius_service: Optional[HeliusService] = None
        self.jupiter_service: Optional[JupiterService] = None
        self.quicknode_service: Optional[QuickNodeService] = None
        
        # System State
        self.system_metrics = {
            "total_trades_executed": 0,
            "total_profit": 0.0,
            "system_uptime": 0.0,
            "active_ants": 0,
            "replication_count": 0
        }
        
        # Main operation loop
        self.main_loop_running = False
        
    async def initialize(self) -> bool:
        """Initialize the complete Ant Bot system"""
        try:
            logger.info("🐜 Initializing Enhanced Ant Bot System...")
            
            # Step 1: Initialize external services
            logger.info("📡 Initializing external services...")
            await self._initialize_external_services()
            
            # Step 2: Initialize core infrastructure
            logger.info("🏗️ Initializing core infrastructure...")
            await self._initialize_core_infrastructure()
            
            # Step 3: Initialize AI Coordinator
            logger.info("🧠 Initializing AI Coordinator...")
            self.ai_coordinator = AICoordinator()
            if not await self.ai_coordinator.initialize():
                raise Exception("AI Coordinator initialization failed")
            
            # Step 4: Initialize Founding Ant Queen
            logger.info("👑 Initializing Founding Ant Queen...")
            self.founding_queen = FoundingAntQueen("founding_queen_0", self.initial_capital)
            if not await self.founding_queen.initialize():
                raise Exception("Founding Queen initialization failed")
            
            # Step 5: Initialize System Replicator
            logger.info("🔄 Initializing System Replicator...")
            self.system_replicator = SystemReplicator(self.founding_queen)
            
            # Step 6: Initialize data ingestion
            logger.info("📊 Initializing Data Ingestion...")
            self.data_ingestion = DataIngestion(
                quicknode_service=self.quicknode_service,
                helius_service=self.helius_service,
                jupiter_service=self.jupiter_service
            )
            await self.data_ingestion.start()
            
            logger.info("✅ Ant Bot System initialization complete!")
            
            # Print initialization summary
            self._print_initialization_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {str(e)}")
            return False
    
    async def _initialize_external_services(self) -> bool:
        """Initialize external services (Helius, Jupiter, QuickNode)"""
        try:
            # Initialize QuickNode service (PRIORITY - most reliable)
            self.quicknode_service = QuickNodeService()
            
            # Initialize Helius service (backup)
            self.helius_service = HeliusService()
            
            # Initialize Jupiter service
            self.jupiter_service = JupiterService()
            
            logger.info("✅ External services initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ External services initialization failed: {str(e)}")
            return False
    
    async def _initialize_core_infrastructure(self) -> bool:
        """Initialize core infrastructure components"""
        try:
            # Initialize wallet manager
            self.wallet_manager = WalletManager()
            if not await self.wallet_manager.initialize():
                raise Exception("Wallet manager initialization failed")
            
            # Initialize portfolio risk manager
            self.portfolio_risk_manager = PortfolioRiskManager(None)  # Will be set later
            if not await self.portfolio_risk_manager.initialize():
                raise Exception("Portfolio risk manager initialization failed")
            
            logger.info("✅ Core infrastructure initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Core infrastructure initialization failed: {str(e)}")
            return False
    
    def _print_initialization_summary(self):
        """Print system initialization summary"""
        summary = f"""
🐜 ========================================
   ANT BOT SYSTEM - INITIALIZATION COMPLETE
========================================

👑 Founding Queen ID: {self.founding_queen.ant_id}
💰 Initial Capital: {self.initial_capital} SOL
🧠 AI Models: Grok AI + Local LLM
🔄 Self-Replication: Enabled
📊 Data Ingestion: Active

🏗️ Architecture:
   └── Founding Ant Queen
       ├── AI Coordinator (Grok + Local LLM)
       ├── System Replicator (Auto-scaling)
       └── Ant Queens
           └── Worker Ants (Princesses)

⚙️ System Ready for Operations!
========================================
        """
        logger.info(summary)
    
    async def start(self):
        """Start the main Ant Bot system operations"""
        try:
            logger.info("🚀 Starting Ant Bot System operations...")
            self.main_loop_running = True
            
            # Start the main operation loop
            await self._main_operation_loop()
            
        except KeyboardInterrupt:
            logger.info("👋 Shutdown requested by user")
            self.shutdown_requested = True
        except Exception as e:
            logger.error(f"💥 Critical error in main operations: {str(e)}")
        finally:
            await self.shutdown()
    
    async def _main_operation_loop(self):
        """Main system operation loop"""
        logger.info("🔄 Starting main operation loop...")
        
        loop_count = 0
        
        while self.main_loop_running and not self.shutdown_requested:
            try:
                loop_start_time = time.time()
                loop_count += 1
                
                logger.info(f"🔄 Operation Loop #{loop_count} starting...")
                
                # Step 1: Get market opportunities
                market_opportunities = await self._scan_market_opportunities()
                
                # Step 2: Coordinate Ant system for opportunities
                coordination_results = await self.founding_queen.coordinate_system(market_opportunities)
                
                # Step 3: Execute trading decisions
                execution_results = await self._execute_trading_decisions(coordination_results["decisions"])
                
                # Step 4: Learn from outcomes
                await self._process_learning_feedback(execution_results)
                
                # Step 5: Check replication conditions
                await self._check_replication_conditions()
                
                # Step 6: Update system metrics
                await self._update_system_metrics()
                
                # Step 7: Log system status
                await self._log_system_status(loop_count)
                
                # Wait before next loop (30 seconds)
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, 30 - loop_duration)
                
                if sleep_time > 0:
                    logger.debug(f"💤 Sleeping for {sleep_time:.1f} seconds until next loop...")
                    await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"💥 Error in operation loop #{loop_count}: {str(e)}")
                await asyncio.sleep(10)  # Short sleep on error
    
    async def _scan_market_opportunities(self) -> List[Dict[str, Any]]:
        """Scan for market opportunities using data ingestion"""
        try:
            # Get market data from data ingestion
            market_data = await self.data_ingestion.get_market_data()
            
            # Filter and enhance opportunities
            opportunities = []
            for token_data in market_data[:10]:  # Limit to top 10 opportunities
                # Enhance with additional metrics
                enhanced_data = {
                    **token_data,
                    "timestamp": time.time(),
                    "source": "market_scanner"
                }
                opportunities.append(enhanced_data)
            
            logger.info(f"📊 Found {len(opportunities)} market opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"❌ Market scanning error: {str(e)}")
            return []
    
    async def _execute_trading_decisions(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute trading decisions from Ant system"""
        execution_results = []
        
        try:
            for decision_data in decisions:
                princess_id = decision_data["princess_id"]
                decision = decision_data["decision"]
                
                if decision["action"] in ["buy", "sell"]:
                    # Find the princess and execute trade
                    princess = self._find_princess_by_id(princess_id)
                    if princess:
                        result = await princess.execute_trade(decision, self.wallet_manager)
                        if result.get("success"):
                            execution_results.append({
                                "princess_id": princess_id,
                                "trade_id": result["trade_record"]["trade_id"],
                                "decision": decision,
                                "result": result,
                                "timestamp": time.time()
                            })
                            
                            # Update system metrics
                            self.system_metrics["total_trades_executed"] += 1
                            self.system_metrics["total_profit"] += result["trade_record"]["profit"]
            
            logger.info(f"⚡ Executed {len(execution_results)} trades")
            return execution_results
            
        except Exception as e:
            logger.error(f"❌ Trading execution error: {str(e)}")
            return execution_results
    
    def _find_princess_by_id(self, princess_id: str):
        """Find a princess by ID across all queens"""
        try:
            for queen in self.founding_queen.queens.values():
                if princess_id in queen.princesses:
                    return queen.princesses[princess_id]
            return None
        except Exception as e:
            logger.error(f"Princess lookup error: {str(e)}")
            return None
    
    async def _process_learning_feedback(self, execution_results: List[Dict[str, Any]]):
        """Process learning feedback from trade outcomes"""
        try:
            for result in execution_results:
                trade_outcome = {
                    "profit": result["result"]["trade_record"]["profit"],
                    "success": result["result"]["trade_record"]["success"],
                    "market_conditions": {
                        "timestamp": result["timestamp"]
                    }
                }
                
                # Send feedback to AI coordinator
                await self.ai_coordinator.learn_from_outcome(
                    result["trade_id"],
                    result["decision"],  # This should be an AIDecision object
                    trade_outcome
                )
            
            if execution_results:
                logger.info(f"🧠 Processed learning feedback for {len(execution_results)} trades")
                
        except Exception as e:
            logger.error(f"❌ Learning feedback error: {str(e)}")
    
    async def _check_replication_conditions(self):
        """Check and handle system replication"""
        try:
            # Check if replication conditions are met
            should_replicate = await self.system_replicator.monitor_replication_conditions()
            
            if should_replicate:
                logger.info("🔄 Replication conditions met, attempting to replicate system...")
                new_instance_id = await self.system_replicator.replicate_system()
                
                if new_instance_id:
                    self.system_metrics["replication_count"] += 1
                    logger.info(f"✅ Successfully replicated system as instance: {new_instance_id}")
                else:
                    logger.warning("⚠️ System replication attempt failed")
            
            # Manage existing instances
            await self.system_replicator.manage_instances()
            
        except Exception as e:
            logger.error(f"❌ Replication check error: {str(e)}")
    
    async def _update_system_metrics(self):
        """Update system-wide metrics"""
        try:
            # Update uptime
            self.system_metrics["system_uptime"] = time.time() - self.start_time
            
            # Count active ants
            total_ants = 1  # Founding Queen
            total_ants += len(self.founding_queen.queens)  # Queens
            
            for queen in self.founding_queen.queens.values():
                total_ants += len(queen.princesses)  # Princesses
            
            self.system_metrics["active_ants"] = total_ants
            
        except Exception as e:
            logger.error(f"❌ Metrics update error: {str(e)}")
    
    async def _log_system_status(self, loop_count: int):
        """Log comprehensive system status"""
        try:
            # Get system status from all components
            founding_queen_status = self.founding_queen.get_system_status()
            ai_performance = self.ai_coordinator.get_performance_summary()
            replication_status = self.system_replicator.get_replication_status()
            
            # Create status summary
            status_summary = {
                "loop_count": loop_count,
                "system_metrics": self.system_metrics,
                "founding_queen": {
                    "active_queens": founding_queen_status.get("active_queens", 0),
                    "total_capital": founding_queen_status.get("system_metrics", {}).get("total_capital", 0),
                    "total_trades": founding_queen_status.get("system_metrics", {}).get("total_trades", 0),
                    "system_profit": founding_queen_status.get("system_metrics", {}).get("system_profit", 0)
                },
                "ai_performance": ai_performance,
                "replication": {
                    "active_instances": replication_status.get("active_instances", 0),
                    "total_instances": replication_status.get("total_instances", 0)
                }
            }
            
            # Log detailed status every 10 loops (5 minutes)
            if loop_count % 10 == 0:
                logger.info(f"📊 SYSTEM STATUS REPORT (Loop #{loop_count}):")
                logger.info(f"   🐜 Total Ants: {self.system_metrics['active_ants']}")
                logger.info(f"   💰 Total Capital: {status_summary['founding_queen']['total_capital']:.4f} SOL")
                logger.info(f"   📈 Total Profit: {status_summary['founding_queen']['system_profit']:.4f} SOL")
                logger.info(f"   ⚡ Total Trades: {status_summary['founding_queen']['total_trades']}")
                logger.info(f"   🔄 Replicated Instances: {status_summary['replication']['active_instances']}")
                logger.info(f"   ⏱️ Uptime: {self.system_metrics['system_uptime']/3600:.1f} hours")
            else:
                # Brief status for other loops
                logger.info(f"📊 Loop #{loop_count}: {self.system_metrics['active_ants']} ants, {status_summary['founding_queen']['system_profit']:.4f} SOL profit")
            
        except Exception as e:
            logger.error(f"❌ Status logging error: {str(e)}")
    
    async def shutdown(self):
        """Graceful system shutdown"""
        try:
            logger.info("🛑 Initiating system shutdown...")
            self.main_loop_running = False
            
            # Stop data ingestion
            if self.data_ingestion:
                await self.data_ingestion.close()
                logger.info("✅ Data ingestion stopped")
            
            # Terminate replicated instances
            if self.system_replicator:
                for instance_id in list(self.system_replicator.instances.keys()):
                    await self.system_replicator.terminate_instance(instance_id)
                logger.info("✅ Replicated instances terminated")
            
            # Close wallet manager
            if self.wallet_manager:
                await self.wallet_manager.close()
                logger.info("✅ Wallet manager closed")
            
            # Close portfolio risk manager
            if self.portfolio_risk_manager:
                await self.portfolio_risk_manager.close()
                logger.info("✅ Portfolio risk manager closed")
            
            # Close external services
            if self.helius_service:
                await self.helius_service.close()
            if self.jupiter_service:
                await self.jupiter_service.close()
            if self.quicknode_service:
                await self.quicknode_service.close()
            logger.info("✅ External services closed")
            
            # Save final system state
            await self._save_system_state()
            
            logger.info("🏁 Ant Bot System shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {str(e)}")
    
    async def _save_system_state(self):
        """Save final system state for recovery"""
        try:
            final_state = {
                "shutdown_time": time.time(),
                "system_metrics": self.system_metrics,
                "founding_queen_status": self.founding_queen.get_system_status() if self.founding_queen else {},
                "ai_performance": self.ai_coordinator.get_performance_summary() if self.ai_coordinator else {},
                "replication_history": self.system_replicator.replication_history if self.system_replicator else []
            }
            
            os.makedirs("data/system_state", exist_ok=True)
            with open("data/system_state/final_state.json", "w") as f:
                json.dump(final_state, f, indent=2)
            
            logger.info("💾 System state saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Save state error: {str(e)}")
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        try:
            overview = {
                "system_info": {
                    "start_time": self.start_time,
                    "uptime": time.time() - self.start_time,
                    "status": "running" if self.main_loop_running else "stopped"
                },
                "metrics": self.system_metrics,
                "components": {
                    "founding_queen": self.founding_queen.get_system_status() if self.founding_queen else None,
                    "ai_coordinator": self.ai_coordinator.get_performance_summary() if self.ai_coordinator else None,
                    "replication": self.system_replicator.get_replication_status() if self.system_replicator else None
                }
            }
            return overview
            
        except Exception as e:
            logger.error(f"System overview error: {str(e)}")
            return {"error": str(e)}

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
    # Set global shutdown flag (this would be handled by the main function)
    sys.exit(0)

async def main():
    """Main entry point for the Enhanced Ant Bot System"""
    logger.info("🐜 Starting Enhanced Ant Bot System...")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get initial capital from environment or use default
    initial_capital = float(os.getenv("INITIAL_CAPITAL", "20.0"))
    
    # Create and start the system
    system = AntBotSystem(initial_capital)
    
    if await system.initialize():
        await system.start()
    else:
        logger.error("💀 Failed to initialize Ant Bot System")
        sys.exit(1)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the system
    asyncio.run(main()) 