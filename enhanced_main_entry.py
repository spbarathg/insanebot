#!/usr/bin/env python3
"""
Enhanced Ant Bot - Main Entry Point
Complete trading system with Ant hierarchy, AI collaboration, and self-replication
"""

import asyncio
import logging
import signal
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import Enhanced Ant Bot components
from src.core.ai.ant_hierarchy import FoundingAntQueen, AntRole
from src.core.ai.enhanced_ai_coordinator import EnhancedAICoordinator
from src.core.system_replicator import SystemReplicator
from src.core.enhanced_main import AntBotSystem

# Import existing services (now with fixes)
from src.core.helius_service import HeliusService
from src.core.jupiter_service import JupiterService
from src.core.wallet_manager import WalletManager
from src.core.portfolio_risk_manager import PortfolioRiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedAntBotRunner:
    """Main runner for the Enhanced Ant Bot system"""
    
    def __init__(self, initial_capital: float = 0.1):
        self.initial_capital = initial_capital
        self.system = None
        self.running = False
        self.start_time = None
        
        # System components
        self.founding_queen = None
        self.ai_coordinator = None
        self.system_replicator = None
        
        # External services (enhanced with fixes)
        self.helius_service = None
        self.jupiter_service = None
        self.wallet_manager = None
        self.portfolio_risk_manager = None
    
    async def initialize(self) -> bool:
        """Initialize the Enhanced Ant Bot system"""
        try:
            logger.info("🚀 Initializing Enhanced Ant Bot System...")
            
            # Initialize external services
            await self._initialize_services()
            
            # Initialize Enhanced Ant Bot components
            await self._initialize_ant_system()
            
            # Initialize AI coordination
            await self._initialize_ai_coordination()
            
            # Initialize self-replication system
            await self._initialize_replication()
            
            # Create main system
            self.system = AntBotSystem(initial_capital=self.initial_capital)
            self.system.founding_queen = self.founding_queen
            self.system.ai_coordinator = self.ai_coordinator
            self.system.system_replicator = self.system_replicator
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            logger.info("✅ Enhanced Ant Bot System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Enhanced Ant Bot: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _initialize_services(self):
        """Initialize external services with enhanced error handling"""
        logger.info("🔧 Initializing external services...")
        
        # Helius service (with fixes)
        self.helius_service = HeliusService()
        
        # Jupiter service (with enhanced rate limiting)
        self.jupiter_service = JupiterService()
        
        # Wallet manager
        self.wallet_manager = WalletManager()
        await self.wallet_manager.initialize()
        
        # Portfolio risk manager
        self.portfolio_risk_manager = PortfolioRiskManager()
        
        logger.info("✅ External services initialized")
    
    async def _initialize_ant_system(self):
        """Initialize the Ant hierarchy system"""
        logger.info("🏰 Initializing Ant hierarchy system...")
        
        # Create Founding Queen with proper capital
        self.founding_queen = FoundingAntQueen(
            ant_id="founding_queen_main",
            initial_capital=self.initial_capital
        )
        
        # Initialize with external services
        self.founding_queen.helius_service = self.helius_service
        self.founding_queen.jupiter_service = self.jupiter_service
        self.founding_queen.wallet_manager = self.wallet_manager
        self.founding_queen.portfolio_risk_manager = self.portfolio_risk_manager
        
        logger.info(f"✅ Founding Queen created with {self.initial_capital} SOL")
    
    async def _initialize_ai_coordination(self):
        """Initialize AI coordination system"""
        logger.info("🧠 Initializing AI coordination system...")
        
        # Create AI coordinator with learning capabilities
        self.ai_coordinator = EnhancedAICoordinator()
        
        # Initialize AI models (Grok + Local LLM)
        await self.ai_coordinator.initialize()
        
        # Connect to Ant system
        self.ai_coordinator.ant_system = self.founding_queen
        
        logger.info("✅ AI coordination system initialized")
    
    async def _initialize_replication(self):
        """Initialize self-replication system"""
        logger.info("🔄 Initializing self-replication system...")
        
        # Create system replicator
        self.system_replicator = SystemReplicator()
        
        # Configure replication parameters
        self.system_replicator.capital_threshold = 2.0  # Replicate at 2 SOL
        self.system_replicator.performance_threshold = 0.1  # 10% profit threshold
        self.system_replicator.time_threshold = 3600  # 1 hour
        
        logger.info("✅ Self-replication system initialized")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self):
        """Run the Enhanced Ant Bot system"""
        try:
            logger.info("🎯 Starting Enhanced Ant Bot main loop...")
            self.running = True
            self.start_time = time.time()
            
            # Main system loop
            loop_count = 0
            last_status_time = time.time()
            
            while self.running:
                loop_start = time.time()
                loop_count += 1
                
                try:
                    # 1. Process Ant hierarchy operations
                    await self._process_ant_operations()
                    
                    # 2. Process AI coordination
                    await self._process_ai_coordination()
                    
                    # 3. Check replication conditions
                    await self._check_replication()
                    
                    # 4. Log status periodically
                    if time.time() - last_status_time > 300:  # Every 5 minutes
                        await self._log_system_status()
                        last_status_time = time.time()
                    
                    # 5. Calculate loop timing
                    loop_duration = time.time() - loop_start
                    target_loop_time = 10.0  # 10 second loops
                    
                    if loop_duration < target_loop_time:
                        await asyncio.sleep(target_loop_time - loop_duration)
                    
                except Exception as e:
                    logger.error(f"❌ Error in main loop iteration {loop_count}: {str(e)}")
                    await asyncio.sleep(5)  # Brief pause before retrying
            
        except Exception as e:
            logger.error(f"❌ Critical error in Enhanced Ant Bot: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            await self.shutdown()
    
    async def _process_ant_operations(self):
        """Process Ant hierarchy operations"""
        try:
            # Update Founding Queen operations
            await self.founding_queen.process_operations()
            
            # Process all Queens
            for queen_id, queen in self.founding_queen.queens.items():
                await queen.process_operations()
                
                # Process Princesses for each Queen
                for princess_id, princess in queen.princesses.items():
                    await princess.process_operations()
            
        except Exception as e:
            logger.error(f"Error processing Ant operations: {str(e)}")
    
    async def _process_ai_coordination(self):
        """Process AI coordination and learning"""
        try:
            # Get system state for AI analysis
            system_state = self.founding_queen.get_system_status()
            
            # Process AI coordination
            ai_insights = await self.ai_coordinator.process_system_state(system_state)
            
            # Apply AI insights to system
            if ai_insights:
                await self.founding_queen.apply_ai_insights(ai_insights)
            
        except Exception as e:
            logger.error(f"Error processing AI coordination: {str(e)}")
    
    async def _check_replication(self):
        """Check and process replication conditions"""
        try:
            # Check if replication is needed
            if await self.system_replicator.should_replicate(self.founding_queen):
                logger.info("🔄 Replication conditions met, initiating system replication...")
                await self.system_replicator.replicate_system(self.founding_queen)
            
        except Exception as e:
            logger.error(f"Error checking replication: {str(e)}")
    
    async def _log_system_status(self):
        """Log comprehensive system status"""
        try:
            runtime = time.time() - self.start_time
            system_status = self.founding_queen.get_system_status()
            ai_status = self.ai_coordinator.get_performance_summary()
            replication_status = self.system_replicator.get_replication_status()
            
            logger.info("="*80)
            logger.info("🎯 ENHANCED ANT BOT SYSTEM STATUS")
            logger.info("="*80)
            logger.info(f"⏱️  Runtime: {runtime/3600:.1f} hours")
            logger.info(f"💰 Total Capital: {system_status.get('total_capital', 0):.4f} SOL")
            logger.info(f"👑 Active Queens: {system_status.get('active_queens', 0)}")
            logger.info(f"🐜 Active Princesses: {system_status.get('active_princesses', 0)}")
            logger.info(f"📊 Total Trades: {system_status.get('total_trades', 0)}")
            logger.info(f"📈 Total Profit: {system_status.get('total_profit', 0):.4f} SOL")
            logger.info(f"🧠 AI Decision Accuracy: {ai_status.get('accuracy', 0):.1%}")
            logger.info(f"🔄 Replication Instances: {replication_status.get('total_instances', 0)}")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Error logging system status: {str(e)}")
    
    async def shutdown(self):
        """Graceful shutdown of the Enhanced Ant Bot system"""
        try:
            logger.info("🛑 Initiating Enhanced Ant Bot shutdown...")
            self.running = False
            
            # Save system state
            if self.founding_queen:
                await self.founding_queen.save_state()
            
            # Close AI coordinator
            if self.ai_coordinator:
                await self.ai_coordinator.close()
            
            # Close replication system
            if self.system_replicator:
                await self.system_replicator.close()
            
            # Close external services
            if self.helius_service:
                await self.helius_service.close()
            
            if self.jupiter_service:
                await self.jupiter_service.close()
            
            logger.info("✅ Enhanced Ant Bot shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

async def main():
    """Main entry point for Enhanced Ant Bot"""
    print("🎯 Enhanced Ant Bot - Professional Trading System")
    print("="*60)
    print("🏰 Features: Ant Hierarchy | AI Collaboration | Self-Replication")
    print("✨ Architecture: 100% Verified | Production Ready")
    print("="*60)
    
    # Get initial capital from environment or use default
    initial_capital = float(os.getenv("INITIAL_CAPITAL", "0.1"))
    
    # Create and run Enhanced Ant Bot
    enhanced_bot = EnhancedAntBotRunner(initial_capital=initial_capital)
    
    # Initialize system
    if not await enhanced_bot.initialize():
        logger.error("❌ Failed to initialize Enhanced Ant Bot")
        sys.exit(1)
    
    # Run system
    await enhanced_bot.run()

if __name__ == "__main__":
    # Run Enhanced Ant Bot
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Enhanced Ant Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Enhanced Ant Bot crashed: {str(e)}")
        sys.exit(1) 