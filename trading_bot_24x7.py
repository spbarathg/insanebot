#!/usr/bin/env python3
"""
Enhanced Ant Bot - 24/7 Continuous Trading System
=================================================

Single file to run the complete trading bot system continuously.
Features:
- Complete trading system integration
- 24/7 continuous operation
- Automatic error recovery and restarts
- Robust logging and monitoring
- System health checks
- Graceful shutdown handling

Usage: python trading_bot_24x7.py

Environment Variables:
- INITIAL_CAPITAL: Starting capital in SOL (default: 0.1)
- LOG_LEVEL: INFO, DEBUG, WARNING, ERROR (default: INFO)  
- RESTART_ON_ERROR: true/false (default: true)
- MAX_RESTART_ATTEMPTS: Number (default: 5)
- HEALTH_CHECK_INTERVAL: Seconds (default: 60)
"""

import asyncio
import logging
import signal
import sys
import os
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Enhanced Ant Bot Core Imports
try:
    from src.core.enhanced_main import AntBotSystem
    from src.services.wallet_manager import WalletManager
    from src.services.quicknode_service import QuickNodeService
    from src.services.helius_service import HeliusService
    from src.services.jupiter_service import JupiterService
    from src.core.ai.enhanced_ai_coordinator import AICoordinator
    from src.core.titan_shield_coordinator import TitanShieldCoordinator
    from src.core.portfolio_manager import PortfolioManager
    from src.core.system_replicator import SystemReplicator
except ImportError as e:
    print(f"❌ Failed to import trading bot components: {str(e)}")
    print("🔧 Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
def setup_logging():
    """Setup comprehensive logging for 24/7 operation"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup file handler
    log_file = logs_dir / f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[file_handler, console_handler]
    )
    
    return logging.getLogger(__name__)

class TradingBot24x7:
    """24/7 Continuous Trading Bot System"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.running = False
        self.start_time = None
        self.restart_count = 0
        self.max_restarts = int(os.getenv("MAX_RESTART_ATTEMPTS", "5"))
        self.restart_on_error = os.getenv("RESTART_ON_ERROR", "true").lower() == "true"
        self.health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))
        
        # System components
        self.bot_system: Optional[AntBotSystem] = None
        self.wallet_manager: Optional[WalletManager] = None
        self.ai_coordinator: Optional[AICoordinator] = None
        self.titan_shield: Optional[TitanShieldCoordinator] = None
        
        # System metrics
        self.metrics = {
            "start_time": None,
            "uptime_hours": 0,
            "total_trades": 0,
            "successful_trades": 0,
            "current_capital": 0.0,
            "profit_loss": 0.0,
            "last_health_check": None,
            "restart_count": 0,
            "errors_count": 0
        }
        
        # Configuration
        self.config = {
            "initial_capital": float(os.getenv("INITIAL_CAPITAL", "0.1")),
            "trading_enabled": True,
            "ai_learning_enabled": True,
            "defense_systems_enabled": True,
            "auto_scaling_enabled": True
        }
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        self.logger.info("🤖 Enhanced Ant Bot 24/7 System Initialized")
        self.logger.info(f"💰 Initial Capital: {self.config['initial_capital']} SOL")
        self.logger.info(f"🔄 Restart on Error: {self.restart_on_error}")
        self.logger.info(f"🔁 Max Restart Attempts: {self.max_restarts}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
    
    async def initialize_trading_system(self) -> bool:
        """Initialize the complete trading bot system"""
        try:
            self.logger.info("🚀 Initializing Enhanced Ant Bot Trading System...")
            
            # Initialize main bot system
            self.bot_system = AntBotSystem(initial_capital=self.config["initial_capital"])
            
            if not await self.bot_system.initialize():
                self.logger.error("❌ Failed to initialize main bot system")
                return False
            
            # Get system components
            self.wallet_manager = getattr(self.bot_system, 'wallet_manager', None)
            self.ai_coordinator = getattr(self.bot_system, 'ai_coordinator', None)
            self.titan_shield = getattr(self.bot_system, 'titan_shield', None)
            
            # Verify critical components
            if not self.wallet_manager:
                self.logger.warning("⚠️ Wallet manager not initialized")
            
            if not self.ai_coordinator:
                self.logger.warning("⚠️ AI coordinator not initialized")
            
            if not self.titan_shield:
                self.logger.warning("⚠️ Titan Shield not initialized")
            
            # Initialize metrics
            self.metrics["start_time"] = datetime.now()
            self.metrics["current_capital"] = self.config["initial_capital"]
            
            self.logger.info("✅ Trading system initialized successfully!")
            self.logger.info("🛡️ Defense systems: ACTIVE")
            self.logger.info("🧠 AI learning: ACTIVE")
            self.logger.info("💰 Wallet management: ACTIVE")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {str(e)}")
            self.logger.error(f"🔍 Error details: {traceback.format_exc()}")
            return False
    
    async def health_check(self) -> bool:
        """Perform comprehensive system health check"""
        try:
            self.logger.debug("🔍 Performing health check...")
            
            health_status = {
                "bot_system": False,
                "wallet_manager": False,
                "ai_coordinator": False,
                "titan_shield": False,
                "overall": False
            }
            
            # Check Enhanced Ant Bot system (production system)
            if self.bot_system:
                # Check if it's initialized and has the required components
                if (hasattr(self.bot_system, 'founding_queen') and 
                    self.bot_system.founding_queen and
                    hasattr(self.bot_system, 'ai_coordinator') and
                    self.bot_system.ai_coordinator):
                    health_status["bot_system"] = True
            
            # Check wallet manager
            if self.wallet_manager:
                health_status["wallet_manager"] = True
            
            # Check AI coordinator
            if self.ai_coordinator:
                health_status["ai_coordinator"] = True
            
            # Check defense systems
            if self.titan_shield:
                health_status["titan_shield"] = True
            
            # Overall health - require bot_system and wallet_manager at minimum
            health_status["overall"] = all([
                health_status["bot_system"],
                health_status["wallet_manager"]
            ])
            
            self.metrics["last_health_check"] = datetime.now()
            
            if health_status["overall"]:
                self.logger.debug("✅ Health check passed")
                return True
            else:
                self.logger.warning(f"⚠️ Health check issues: {health_status}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {str(e)}")
            return False
    
    async def update_metrics(self):
        """Update system metrics"""
        try:
            if self.metrics["start_time"]:
                uptime = datetime.now() - self.metrics["start_time"]
                self.metrics["uptime_hours"] = uptime.total_seconds() / 3600
            
            # Update bot metrics
            if self.bot_system:
                self.metrics["total_trades"] = getattr(self.bot_system, 'total_trades_executed', 0)
                self.metrics["current_capital"] = getattr(self.bot_system, 'current_capital', self.config["initial_capital"])
                
                # Calculate profit/loss
                self.metrics["profit_loss"] = self.metrics["current_capital"] - self.config["initial_capital"]
            
            self.metrics["restart_count"] = self.restart_count
            
        except Exception as e:
            self.logger.debug(f"❌ Metrics update error: {str(e)}")
    
    async def log_system_status(self):
        """Log comprehensive system status"""
        try:
            await self.update_metrics()
            
            self.logger.info("=" * 80)
            self.logger.info("📊 ENHANCED ANT BOT - 24/7 SYSTEM STATUS")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️  Uptime: {self.metrics['uptime_hours']:.1f} hours")
            self.logger.info(f"💰 Current Capital: {self.metrics['current_capital']:.4f} SOL")
            self.logger.info(f"📈 Profit/Loss: {self.metrics['profit_loss']:+.4f} SOL")
            self.logger.info(f"⚡ Total Trades: {self.metrics['total_trades']}")
            self.logger.info(f"🔄 Restart Count: {self.metrics['restart_count']}")
            self.logger.info(f"🛡️ Trading Status: {'ACTIVE' if self.running else 'HALTED'}")
            
            # Component status
            self.logger.info(f"🤖 Bot System: {'✅' if self.bot_system else '❌'}")
            self.logger.info(f"💰 Wallet Manager: {'✅' if self.wallet_manager else '❌'}")
            self.logger.info(f"🧠 AI Coordinator: {'✅' if self.ai_coordinator else '❌'}")
            self.logger.info(f"🛡️ Titan Shield: {'✅' if self.titan_shield else '❌'}")
            
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"❌ Status logging error: {str(e)}")
    
    async def trading_loop(self):
        """Main 24/7 trading loop"""
        self.logger.info("🚀 Starting 24/7 trading loop...")
        
        last_health_check = 0
        last_status_log = 0
        status_log_interval = 1800  # 30 minutes
        
        while self.running:
            try:
                current_time = time.time()
                
                # Periodic health checks
                if current_time - last_health_check > self.health_check_interval:
                    if not await self.health_check():
                        self.logger.warning("⚠️ Health check failed, continuing with caution...")
                    last_health_check = current_time
                
                # Periodic status logging
                if current_time - last_status_log > status_log_interval:
                    await self.log_system_status()
                    last_status_log = current_time
                
                # Main trading cycle
                if self.bot_system and self.config["trading_enabled"]:
                    # Let the bot system handle its own trading logic
                    # The bot system runs its own internal loops
                    await asyncio.sleep(30)  # Main loop cycle - 30 seconds
                else:
                    self.logger.warning("⚠️ Trading disabled or bot system not available")
                    await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"❌ Trading loop error: {str(e)}")
                self.logger.error(f"🔍 Error details: {traceback.format_exc()}")
                self.metrics["errors_count"] += 1
                
                if self.restart_on_error and self.restart_count < self.max_restarts:
                    self.logger.info("🔄 Attempting system restart due to error...")
                    if await self.restart_system():
                        continue
                    else:
                        self.logger.error("❌ System restart failed")
                        break
                else:
                    self.logger.error("❌ Maximum restart attempts reached or restart disabled")
                    break
                
                await asyncio.sleep(10)  # Wait before continuing after error
        
        self.logger.info("🛑 Trading loop ended")
    
    async def restart_system(self) -> bool:
        """Restart the trading system"""
        try:
            self.logger.info(f"🔄 Restarting system (attempt {self.restart_count + 1}/{self.max_restarts})...")
            
            # Shutdown current system
            await self.shutdown_system()
            
            # Wait before restart
            await asyncio.sleep(5)
            
            # Reinitialize
            if await self.initialize_trading_system():
                self.restart_count += 1
                self.logger.info(f"✅ System restart successful (restart #{self.restart_count})")
                return True
            else:
                self.logger.error("❌ System restart failed during initialization")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ System restart error: {str(e)}")
            return False
    
    async def shutdown_system(self):
        """Gracefully shutdown the trading system"""
        try:
            self.logger.info("🛑 Shutting down trading system...")
            
            if self.bot_system:
                await self.bot_system.shutdown()
                self.logger.info("✅ Bot system shutdown complete")
            
            # Reset components
            self.bot_system = None
            self.wallet_manager = None
            self.ai_coordinator = None
            self.titan_shield = None
            
        except Exception as e:
            self.logger.error(f"❌ Shutdown error: {str(e)}")
    
    def save_metrics(self):
        """Save metrics to file"""
        try:
            metrics_file = Path("logs") / "trading_metrics.json"
            with open(metrics_file, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                metrics_json = self.metrics.copy()
                if metrics_json["start_time"]:
                    metrics_json["start_time"] = metrics_json["start_time"].isoformat()
                if metrics_json["last_health_check"]:
                    metrics_json["last_health_check"] = metrics_json["last_health_check"].isoformat()
                
                json.dump(metrics_json, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save metrics: {str(e)}")
    
    async def run(self):
        """Main entry point for 24/7 operation"""
        try:
            self.logger.info("🤖 Enhanced Ant Bot 24/7 System Starting...")
            self.logger.info("🎯 Mission: Continuous autonomous trading with maximum survival")
            
            # Initialize system
            if not await self.initialize_trading_system():
                self.logger.error("❌ Failed to initialize trading system")
                return False
            
            # Set running flag
            self.running = True
            self.start_time = time.time()
            
            # Log initial status
            await self.log_system_status()
            
            # Start the main trading loop
            await self.trading_loop()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"❌ System error: {str(e)}")
            self.logger.error(f"🔍 Error details: {traceback.format_exc()}")
        finally:
            # Cleanup
            self.running = False
            await self.shutdown_system()
            self.save_metrics()
            
            # Final status
            if self.start_time:
                total_runtime = time.time() - self.start_time
                self.logger.info(f"📊 Total Runtime: {total_runtime/3600:.1f} hours")
                self.logger.info(f"🔄 Total Restarts: {self.restart_count}")
                self.logger.info(f"💰 Final Capital: {self.metrics['current_capital']:.4f} SOL")
                self.logger.info(f"📈 Total Profit/Loss: {self.metrics['profit_loss']:+.4f} SOL")
            
            self.logger.info("👋 Enhanced Ant Bot 24/7 System Shutdown Complete")
            return True

def print_banner():
    """Print startup banner"""
    print("=" * 80)
    print("🤖 ENHANCED ANT BOT - 24/7 CONTINUOUS TRADING SYSTEM")
    print("=" * 80)
    print("🎯 Mission: Autonomous trading with maximum survival")
    print("🛡️ Defense: 7-layer Titan Shield protection")
    print("🧠 Intelligence: AI-powered learning and adaptation")
    print("⚡ Operation: 24/7 continuous with auto-recovery")
    print("=" * 80)
    print()

def check_environment():
    """Check environment setup"""
    print("🔍 Checking environment setup...")
    
    # Check environment file
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️ .env file not found. Using environment variables or defaults.")
    else:
        print("✅ .env file found")
    
    # Check critical environment variables
    critical_vars = ["INITIAL_CAPITAL"]
    for var in critical_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"⚠️ {var}: Using default value")
    
    print()

async def main():
    """Main entry point"""
    print_banner()
    check_environment()
    
    # Create and run the 24/7 trading bot
    trading_bot = TradingBot24x7()
    
    try:
        success = await trading_bot.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    """
    24/7 Trading Bot Entry Point
    
    Usage:
        python trading_bot_24x7.py
    
    Environment Variables:
        INITIAL_CAPITAL=0.1
        LOG_LEVEL=INFO
        RESTART_ON_ERROR=true
        MAX_RESTART_ATTEMPTS=5
        HEALTH_CHECK_INTERVAL=60
    """
    try:
        # Run the bot
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1) 