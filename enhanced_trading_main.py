#!/usr/bin/env python3
"""
Enhanced Ant Bot Trading Main - Audit Improved Version

PRODUCTION-READY ENTRY POINT with INSTITUTIONAL-GRADE IMPROVEMENTS:
✅ Enhanced Signal Processing: Technical (30%), Pump.fun (25%), Smart Money (25%), Sentiment (15%), AI (5%)
✅ Advanced Exit Management: Trailing stops, partial profit taking, volume exhaustion detection
✅ MEV Protection: Jito bundles, sandwich detection, timing randomization
✅ Improved Risk Management: 5% max slippage, 2% max position size, 5% daily loss limit
✅ Titan Shield Defense: 7-layer protection system
✅ Ant Colony Architecture: Micro-capital scaling with defense integration

AUDIT IMPROVEMENTS IMPLEMENTED:
- Reduced slippage tolerance from 15% to 5% for better profit margins
- Reduced position sizing from 10% to 2% for better risk management
- Enhanced signal weights: Technical analysis now primary signal (30%)
- MEV protection with Jito bundle support and sandwich attack detection
- Advanced exit strategies with trailing stops and partial profit taking
- Improved API timeouts and retry logic for better reliability
- Enhanced risk management with daily loss limits and portfolio exposure controls

Run this script to start the enhanced trading bot with all audit improvements applied.
"""

import asyncio
import logging
import os
import sys
import signal
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import enhanced system components
from src.core.enhanced_main import AntBotSystem
from config.core_config import (
    ENHANCED_TRADING_CONSTANTS,
    load_validated_config,
    TradingConfig,
    SecurityConfig,
    MEVProtectionConfig,
    ExitStrategyConfig,
    AdvancedRiskConfig
)

# Configure enhanced logging
def setup_enhanced_logging():
    """Setup enhanced logging with detailed formatting"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with detailed format
    file_handler = logging.FileHandler(
        log_dir / f"enhanced_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Performance log handler
    performance_handler = logging.FileHandler(log_dir / "performance.log")
    performance_handler.setLevel(logging.INFO)
    performance_handler.setFormatter(simple_formatter)
    
    performance_logger = logging.getLogger("performance")
    performance_logger.addHandler(performance_handler)
    performance_logger.setLevel(logging.INFO)
    
    return root_logger

def validate_environment():
    """Validate environment variables and configuration"""
    logger = logging.getLogger(__name__)
    
    required_env_vars = [
        "PRIVATE_KEY",
        "WALLET_ADDRESS", 
        "HELIUS_API_KEY",
        "QUICKNODE_ENDPOINT"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        logger.error("Please check your .env file or environment configuration")
        return False
    
    # Validate configuration
    try:
        config = load_validated_config()
        logger.info("✅ Configuration validation passed")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False

def print_enhanced_startup_banner():
    """Print enhanced startup banner with audit improvements"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 ENHANCED ANT BOT - AUDIT IMPROVED VERSION 🚀              ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  🎯 INSTITUTIONAL-GRADE IMPROVEMENTS IMPLEMENTED:                              ║
║                                                                               ║
║  ✅ Enhanced Signal Processing                                                 ║
║     • Technical Analysis: 30% (PRIMARY SIGNAL)                               ║
║     • Pump.fun Monitoring: 25% (REDUCED FROM 35%)                            ║
║     • Smart Money Tracking: 25%                                              ║
║     • Social Sentiment: 15% (REDUCED FROM 25%)                               ║
║     • AI Analysis: 5% (MINIMAL WEIGHT)                                       ║
║                                                                               ║
║  ✅ Advanced Exit Management                                                   ║
║     • Trailing Stop Losses with Volatility Adjustment                        ║
║     • Partial Profit Taking at Multiple Levels                               ║
║     • Volume Exhaustion Detection                                            ║
║     • RSI Divergence Exit Triggers                                           ║
║     • Time-based Exit Strategies                                             ║
║                                                                               ║
║  ✅ MEV Protection (Jito Bundle Support)                                       ║
║     • Sandwich Attack Detection                                              ║
║     • Front-running Protection                                               ║
║     • Timing Randomization                                                   ║
║     • Dynamic Priority Fees                                                  ║
║     • Private Mempool Routing                                                ║
║                                                                               ║
║  ✅ Improved Risk Management                                                   ║
║     • Max Slippage: 5% (REDUCED FROM 15%)                                    ║
║     • Max Position Size: 2% (REDUCED FROM 10%)                               ║
║     • Max Daily Loss: 5% (STRICT LIMIT)                                      ║
║     • Max Meme Exposure: 15% (PORTFOLIO PROTECTION)                          ║
║     • Kelly Criterion Position Sizing                                        ║
║                                                                               ║
║  ✅ Enhanced API Performance                                                   ║
║     • API Timeout: 5s (REDUCED FROM 10s)                                     ║
║     • Max Retries: 2 (REDUCED FROM 3)                                        ║
║     • Parallel RPC Endpoints: 3                                              ║
║     • Sub-100ms Execution Targeting                                          ║
║                                                                               ║
║  🛡️ Titan Shield Defense System (7-Layer Protection)                          ║
║     • Token Vetting Fortress                                                 ║
║     • Volatility Adaptive Armor                                              ║
║     • AI Deception Shield                                                    ║
║     • Transaction Warfare System                                             ║
║     • Capital Forcefields                                                    ║
║     • Adversarial Learning Core                                              ║
║     • Counter-Attack Profit Engines                                          ║
║                                                                               ║
║  🐜 Ant Colony Architecture                                                    ║
║     • Founding Queen: Supreme Coordinator                                    ║
║     • Ant Queens: Capital Managers                                           ║
║     • Ant Princesses: Protected Trading Agents                              ║
║     • Self-Replication with Defense State Propagation                       ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

    🎯 TARGET: Maximum profitability with institutional-grade risk management
    ⚡ EXECUTION: Sub-100ms targeting with MEV protection
    🛡️ PROTECTION: 7-layer defense system for 95%+ survival rate
    💰 CAPITAL: Micro-scaling from 0.1 SOL with 2% position sizing
    📊 ANALYTICS: Real-time performance tracking with enhanced metrics
    
    Ready for Production Trading! 🚀💰
    """
    print(banner)

def print_configuration_summary(config: Dict[str, Any]):
    """Print configuration summary with audit improvements"""
    logger = logging.getLogger(__name__)
    
    trading_config = config.get('trading', {})
    security_config = config.get('security', {})
    mev_config = config.get('mev_protection', {})
    exit_config = config.get('exit_strategy', {})
    
    logger.info("=" * 80)
    logger.info("📋 ENHANCED CONFIGURATION SUMMARY (AUDIT IMPROVED)")
    logger.info("=" * 80)
    
    # Trading parameters
    logger.info("🎯 TRADING PARAMETERS:")
    logger.info(f"   Max Slippage: {trading_config.get('max_slippage', 0.05)*100:.1f}% (IMPROVED ⭐)")
    logger.info(f"   Max Position Size: {trading_config.get('max_position_size', 0.02)*100:.1f}% (IMPROVED ⭐)")
    logger.info(f"   Min Confidence: {trading_config.get('min_confidence_threshold', 0.7)*100:.0f}% (NEW ⭐)")
    logger.info(f"   Min Liquidity: {trading_config.get('min_liquidity', 10000):,} SOL")
    logger.info(f"   Cooldown Period: {trading_config.get('cooldown_period', 300)} seconds")
    
    # Signal weights
    logger.info("🧠 ENHANCED SIGNAL WEIGHTS (REBALANCED ⭐):")
    signal_weights = trading_config.get('signal_weights', ENHANCED_TRADING_CONSTANTS['SIGNAL_WEIGHTS'])
    for signal_type, weight in signal_weights.items():
        logger.info(f"   {signal_type.replace('_', ' ').title()}: {weight*100:.0f}%")
    
    # Risk management
    logger.info("🛡️ RISK MANAGEMENT (ENHANCED ⭐):")
    logger.info(f"   Max Daily Loss: {security_config.get('max_daily_loss', 0.05)*100:.0f}% (IMPROVED ⭐)")
    logger.info(f"   Emergency Stop: {security_config.get('emergency_stop_loss', 0.15)*100:.0f}% (IMPROVED ⭐)")
    logger.info(f"   Max Meme Exposure: {security_config.get('max_portfolio_exposure', 0.15)*100:.0f}% (NEW ⭐)")
    logger.info(f"   Max Concurrent Trades: {security_config.get('max_concurrent_trades', 3)} (REDUCED ⭐)")
    
    # MEV protection
    logger.info("⚡ MEV PROTECTION (NEW ⭐):")
    logger.info(f"   Protection Level: {mev_config.get('protection_level', 'advanced').upper()}")
    logger.info(f"   Jito Bundles: {'✅ ENABLED' if mev_config.get('enable_jito_bundles', True) else '❌ DISABLED'}")
    logger.info(f"   Sandwich Detection: {'✅ ENABLED' if mev_config.get('sandwich_detection_enabled', True) else '❌ DISABLED'}")
    logger.info(f"   Timing Randomization: {'✅ ENABLED' if mev_config.get('randomize_timing', True) else '❌ DISABLED'}")
    
    # Exit strategies
    logger.info("🚪 EXIT STRATEGIES (NEW ⭐):")
    logger.info(f"   Trailing Stops: {'✅ ENABLED' if exit_config.get('enable_trailing_stops', True) else '❌ DISABLED'}")
    logger.info(f"   Partial Exits: {'✅ ENABLED' if exit_config.get('enable_partial_exits', True) else '❌ DISABLED'}")
    logger.info(f"   Volume Exhaustion: {'✅ ENABLED' if exit_config.get('enable_volume_exhaustion', True) else '❌ DISABLED'}")
    logger.info(f"   Max Hold Time: {exit_config.get('max_hold_time_hours', 24)} hours")
    
    logger.info("=" * 80)

async def monitor_system_health(system: AntBotSystem):
    """Monitor system health and performance"""
    logger = logging.getLogger("performance")
    
    try:
        while system.main_loop_running:
            # Get system metrics
            metrics = system.system_metrics
            profit_metrics = system.profit_metrics
            
            # Log performance summary
            logger.info(f"📊 SYSTEM HEALTH: "
                       f"Trades: {metrics.get('total_trades_executed', 0)} | "
                       f"Success Rate: {profit_metrics.get('signal_accuracy_rate', 0):.1f}% | "
                       f"MEV Threats: {metrics.get('mev_threats_detected', 0)} | "
                       f"Uptime: {(time.time() - system.start_time)/3600:.1f}h")
            
            # Check for critical issues
            if metrics.get('survival_score', 100) < 70:
                logger.warning(f"⚠️ LOW SURVIVAL SCORE: {metrics.get('survival_score', 0):.1f}%")
            
            if metrics.get('mev_threats_detected', 0) > 10:
                logger.warning(f"⚠️ HIGH MEV ACTIVITY: {metrics.get('mev_threats_detected', 0)} threats detected")
            
            # Sleep for 5 minutes
            await asyncio.sleep(300)
            
    except Exception as e:
        logger.error(f"Error in system health monitoring: {e}")

async def enhanced_shutdown_handler(system: AntBotSystem):
    """Enhanced graceful shutdown handler"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🛑 Initiating enhanced graceful shutdown...")
        
        # Stop main trading loop
        system.main_loop_running = False
        system.shutdown_requested = True
        
        # Save final performance metrics
        final_metrics = {
            'shutdown_timestamp': datetime.now().isoformat(),
            'total_runtime_hours': (time.time() - system.start_time) / 3600,
            'system_metrics': system.system_metrics,
            'profit_metrics': system.profit_metrics,
            'mev_protection_summary': system.mev_protection.get_protection_summary(),
            'exit_management_summary': system.advanced_exit_manager.get_performance_summary()
        }
        
        # Save to file
        performance_dir = Path("data/performance")
        performance_dir.mkdir(parents=True, exist_ok=True)
        
        with open(performance_dir / f"final_metrics_{int(time.time())}.json", "w") as f:
            json.dump(final_metrics, f, indent=2)
        
        # Cleanup system components
        await system.shutdown() if hasattr(system, 'shutdown') else None
        
        logger.info("✅ Enhanced graceful shutdown completed")
        logger.info(f"📊 Final Performance Summary:")
        logger.info(f"   Total Trades: {final_metrics['system_metrics'].get('total_trades_executed', 0)}")
        logger.info(f"   Success Rate: {final_metrics['profit_metrics'].get('signal_accuracy_rate', 0):.1f}%")
        logger.info(f"   MEV Threats Mitigated: {final_metrics['system_metrics'].get('mev_threats_mitigated', 0)}")
        logger.info(f"   Runtime: {final_metrics['total_runtime_hours']:.1f} hours")
        
    except Exception as e:
        logger.error(f"Error during enhanced shutdown: {e}")

def setup_signal_handlers(system: AntBotSystem):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger = logging.getLogger(__name__)
        logger.info(f"🔔 Received signal {signum}, initiating shutdown...")
        
        # Create shutdown task
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(enhanced_shutdown_handler(system))
        else:
            asyncio.run(enhanced_shutdown_handler(system))
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Enhanced main entry point with all audit improvements"""
    
    # Setup enhanced logging
    logger = setup_enhanced_logging()
    
    try:
        # Print startup banner
        print_enhanced_startup_banner()
        
        # Validate environment
        logger.info("🔍 Validating environment and configuration...")
        if not validate_environment():
            logger.error("❌ Environment validation failed")
            sys.exit(1)
        
        # Load enhanced configuration
        logger.info("📋 Loading enhanced configuration...")
        try:
            config = load_validated_config()
            print_configuration_summary(config)
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            logger.info("🔄 Falling back to enhanced default configuration...")
            config = {
                'trading': ENHANCED_TRADING_CONSTANTS,
                'security': {'max_daily_loss': 0.05, 'max_position_size': 0.02},
                'mev_protection': {'protection_level': 'advanced', 'enable_jito_bundles': True},
                'exit_strategy': {'enable_trailing_stops': True, 'enable_partial_exits': True}
            }
        
        # Get initial capital
        initial_capital = float(os.getenv("INITIAL_CAPITAL", "0.1"))  # Start with 0.1 SOL as recommended
        logger.info(f"💰 Initial capital: {initial_capital} SOL")
        
        # Create enhanced system
        logger.info("🏗️ Creating Enhanced Ant Bot System...")
        system = AntBotSystem(initial_capital=initial_capital)
        
        # Setup signal handlers
        setup_signal_handlers(system)
        
        # Initialize system
        logger.info("⚡ Initializing Enhanced Ant Bot System with audit improvements...")
        initialization_start = time.time()
        
        if await system.initialize():
            initialization_time = time.time() - initialization_start
            logger.info(f"✅ System initialized successfully in {initialization_time:.2f} seconds")
            
            # Start health monitoring
            health_monitor_task = asyncio.create_task(monitor_system_health(system))
            
            # Start enhanced trading
            logger.info("🚀 Starting enhanced trading operations...")
            logger.info("🎯 System ready for maximum profitability with institutional-grade risk management!")
            
            # Create main trading task
            trading_task = asyncio.create_task(system.start())
            
            # Wait for either task to complete
            try:
                await asyncio.gather(trading_task, health_monitor_task)
            except KeyboardInterrupt:
                logger.info("👋 Shutdown requested by user")
                await enhanced_shutdown_handler(system)
            except Exception as e:
                logger.error(f"💥 Critical error in main operations: {e}")
                await enhanced_shutdown_handler(system)
                raise
                
        else:
            logger.error("💀 Failed to initialize Enhanced Ant Bot System")
            logger.error("📋 Please check your configuration and environment variables")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"💥 Fatal error in main: {e}")
        sys.exit(1)
    
    finally:
        logger.info("🏁 Enhanced Ant Bot System terminated")

if __name__ == "__main__":
    # Verify Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    # Set event loop policy for Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run enhanced system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Enhanced Ant Bot System shutting down...")
    except Exception as e:
        print(f"💥 Fatal startup error: {e}")
        sys.exit(1) 