"""
Lightweight Solana trading bot with robust error handling and type safety.
This version avoids heavy ML dependencies that can cause import issues.
"""
import asyncio
import signal
import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add the app directory to Python path for Docker compatibility
app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if app_root not in sys.path:
    sys.path.insert(0, app_root)

# Set environment variables to disable heavy components
os.environ["USE_LOCAL_LLM"] = "false"
os.environ["ENABLE_ML_PREDICTIONS"] = "false"
os.environ["ENABLE_PATTERN_RECOGNITION"] = "false"
os.environ["ENABLE_SENTIMENT_ANALYSIS"] = "false"

# Configure detailed logging
from loguru import logger

# Create log directories
os.makedirs("logs", exist_ok=True)

# Configure very detailed logging
logger.remove()  # Remove default handler

# Console logging with colors and detailed format
logger.add(
    sys.stderr, 
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

# Main log file with everything
logger.add(
    "logs/bot_detailed.log", 
    rotation="10 MB", 
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    retention="7 days"
)

# Activity-specific log files
logger.add(
    "logs/trades.log", 
    rotation="5 MB", 
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    filter=lambda record: "TRADE" in record["extra"] or "trade" in record["message"].lower()
)

logger.add(
    "logs/monitoring.log", 
    rotation="5 MB", 
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    filter=lambda record: "MONITOR" in record["extra"] or any(word in record["message"].lower() for word in ["scanning", "checking", "monitoring", "watching"])
)

logger.add(
    "logs/errors.log", 
    rotation="5 MB", 
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    retention="30 days"
)

# Initialize shutdown flag
shutdown_requested = False

class LightweightBot:
    """Lightweight trading bot without heavy ML dependencies."""
    
    def __init__(self):
        """Initialize the lightweight bot."""
        self.helius_service = None
        self.jupiter_service = None
        self.running = False
        
    async def initialize(self) -> bool:
        """Initialize bot components."""
        try:
            logger.info("🚀 Initializing Lightweight Trading Bot...")
            
            # Import and initialize core services
            from src.core.helius_service import HeliusService
            from src.core.jupiter_service import JupiterService
            
            self.helius_service = HeliusService()
            self.jupiter_service = JupiterService()
            
            logger.info("✅ Core services initialized successfully")
            
            # Test connectivity
            logger.info("🧪 Testing API connectivity...")
            
            # Test Jupiter API
            tokens = await self.jupiter_service.get_supported_tokens()
            if tokens:
                logger.info(f"✅ Jupiter API working - {len(tokens)} tokens available")
            else:
                logger.warning("⚠️ Jupiter API returned no data")
            
            # Test Helius API (will show warning if no API key)
            metadata = await self.helius_service.get_token_metadata("So11111111111111111111111111111111111111112")
            if metadata:
                logger.info(f"✅ Helius API working - Got metadata for {metadata.get('symbol', 'SOL')}")
            else:
                logger.info("ℹ️ Helius API in limited mode (no API key)")
            
            logger.info("✅ Lightweight bot initialization completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize lightweight bot: {str(e)}")
            logger.exception("Full initialization error:")
            return False
    
    async def start(self):
        """Start the lightweight bot."""
        if not self.helius_service or not self.jupiter_service:
            logger.error("❌ Bot not initialized. Call initialize() first.")
            return
        
        self.running = True
        logger.info("🎯 Lightweight Trading Bot started!")
        
        try:
            loop_count = 0
            while self.running and not shutdown_requested:
                loop_count += 1
                
                logger.bind(ACTIVITY="MONITOR").info(f"🔄 Bot monitoring loop #{loop_count}")
                
                # Basic monitoring without heavy ML
                try:
                    # Get some random tokens to monitor
                    tokens = await self.jupiter_service.get_random_tokens(count=3)
                    
                    if tokens:
                        logger.bind(SCANNER=True).info(f"📊 Monitoring {len(tokens)} tokens:")
                        
                        for token in tokens:
                            symbol = token.get('symbol', 'UNKNOWN')
                            address = token.get('address', '')
                            
                            logger.bind(SCANNER=True).info(f"  • {symbol} ({address[:8]}...)")
                            
                            # Get basic metadata
                            metadata = await self.helius_service.get_token_metadata(address)
                            if metadata:
                                logger.bind(SCANNER=True).debug(f"    Metadata: {metadata.get('name', 'Unknown')}")
                    
                    # Portfolio summary (simulated)
                    logger.bind(PORTFOLIO=True).info(
                        f"💰 Portfolio Status: 1.0 SOL (Simulation Mode) | "
                        f"Loops: {loop_count} | Status: ✅ Operational"
                    )
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                
                # Sleep between loops
                await asyncio.sleep(30)  # 30 second intervals
                
        except KeyboardInterrupt:
            logger.info("⏹️ Bot stopped by user")
        except Exception as e:
            logger.error(f"💥 Error in bot main loop: {str(e)}")
            logger.exception("Full error traceback:")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the bot and cleanup."""
        logger.info("🛑 Stopping Lightweight Trading Bot...")
        self.running = False
        
        try:
            if self.helius_service:
                await self.helius_service.close()
            if self.jupiter_service:
                await self.jupiter_service.close()
            
            logger.info("✅ Bot stopped successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

async def main():
    """Main entry point for the lightweight trading bot."""
    logger.bind(ACTIVITY="STARTUP").info("🚀 Starting Lightweight Solana Trading Bot...")
    
    try:
        # Create and initialize bot
        bot = LightweightBot()
        
        logger.bind(ACTIVITY="INIT").info("⚙️ Starting bot initialization...")
        if await bot.initialize():
            logger.bind(ACTIVITY="INIT").success("✅ Bot initialization completed successfully!")
            
            logger.bind(ACTIVITY="STARTUP").info("🏃 Starting main bot execution loop...")
            
            # Start the bot
            await bot.start()
        else:
            logger.bind(ACTIVITY="INIT").error("❌ Bot initialization failed!")
            
    except KeyboardInterrupt:
        logger.bind(ACTIVITY="SHUTDOWN").info("⏹️ Bot execution interrupted by user")
    except asyncio.CancelledError:
        logger.bind(ACTIVITY="SHUTDOWN").info("⏹️ Bot execution cancelled")
    except Exception as e:
        logger.bind(ACTIVITY="ERROR").error(f"💥 Critical error in bot: {str(e)}")
        logger.bind(ACTIVITY="ERROR").exception("Full error traceback:")
    finally:
        logger.bind(ACTIVITY="SHUTDOWN").info("🔄 Bot shutdown sequence initiated...")
        logger.bind(ACTIVITY="SHUTDOWN").info("🛑 Bot shutdown complete")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global shutdown_requested
    logger.bind(ACTIVITY="SIGNAL").warning(f"📡 Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True
    sys.exit(0)

def setup():
    """Setup environment and dependencies."""
    try:
        logger.bind(ACTIVITY="SETUP").info("🔧 Starting environment setup...")
        
        # Create necessary directories
        directories = ["data", "logs", "config", "models"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.bind(ACTIVITY="SETUP").debug(f"📁 Created/verified directory: {directory}")
            
        # Ensure directories are writable (skip on Windows if permission denied)
        for directory in directories:
            path = Path(directory)
            try:
                # Try to make directories writable (only on Unix-like systems)
                import platform
                if platform.system() != "Windows":
                    path.chmod(0o777)
                    logger.bind(ACTIVITY="SETUP").debug(f"🔐 Set permissions for {directory}")
                else:
                    # On Windows, just verify the directory is accessible
                    if path.exists() and path.is_dir():
                        logger.bind(ACTIVITY="SETUP").debug(f"🔐 Verified directory access for {directory}")
                    else:
                        logger.bind(ACTIVITY="SETUP").warning(f"⚠️ Directory not accessible: {directory}")
            except Exception as e:
                logger.bind(ACTIVITY="SETUP").warning(f"⚠️ Could not set permissions for {directory}: {str(e)}")
                
        # Set up signal handlers (only on Unix-like systems)
        try:
            import platform
            if platform.system() != "Windows":
                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)
                logger.bind(ACTIVITY="SETUP").debug("📡 Signal handlers configured")
            else:
                # On Windows, only set up SIGINT (Ctrl+C)
                signal.signal(signal.SIGINT, signal_handler)
                logger.bind(ACTIVITY="SETUP").debug("📡 Windows signal handlers configured")
        except Exception as e:
            logger.bind(ACTIVITY="SETUP").warning(f"⚠️ Could not set up signal handlers: {str(e)}")
    
        logger.bind(ACTIVITY="SETUP").success("✅ Environment setup complete")
        return True
    except Exception as e:
        logger.bind(ACTIVITY="SETUP").error(f"❌ Setup error: {str(e)}")
        logger.bind(ACTIVITY="SETUP").exception("Full setup error traceback:")
        return False

if __name__ == "__main__":
    logger.bind(ACTIVITY="MAIN").info("🎯 Lightweight Solana Trading Bot Starting...")
    
    # Perform setup
    if setup():
        logger.bind(ACTIVITY="MAIN").info("▶️ Launching lightweight bot main loop...")
        # Run the bot
        asyncio.run(main())
    else:
        logger.bind(ACTIVITY="MAIN").error("💀 Failed to set up the bot environment")
        sys.exit(1) 