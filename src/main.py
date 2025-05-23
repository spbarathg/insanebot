"""
Minimalistic Solana trading bot with robust error handling and type safety.
"""
import asyncio
import signal
import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

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

async def main():
    """Main entry point for the trading bot."""
    logger.bind(ACTIVITY="STARTUP").info("🚀 Starting Solana trading bot in detailed monitoring mode...")
    
    try:
        # Import and initialize the actual bot
        logger.debug("📦 Importing core bot modules...")
        from src.core.main import MemeCoinBot
        
        logger.bind(ACTIVITY="INIT").info("🔧 Initializing MemeCoinBot...")
        bot = MemeCoinBot()
        
        logger.bind(ACTIVITY="INIT").info("⚙️ Starting bot initialization process...")
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
        try:
            if 'bot' in locals():
                await bot.close()
                logger.bind(ACTIVITY="SHUTDOWN").success("✅ Bot cleanup completed")
        except Exception as e:
            logger.bind(ACTIVITY="SHUTDOWN").error(f"❌ Error during cleanup: {str(e)}")
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
            
        # Ensure directories are writable
        for directory in directories:
            path = Path(directory)
            try:
                # Try to make directories writable
                path.chmod(0o777)
                logger.bind(ACTIVITY="SETUP").debug(f"🔐 Set permissions for {directory}")
            except Exception as e:
                logger.bind(ACTIVITY="SETUP").warning(f"⚠️ Could not set permissions for {directory}: {str(e)}")
                
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.bind(ACTIVITY="SETUP").debug("📡 Signal handlers configured")
    
        logger.bind(ACTIVITY="SETUP").success("✅ Environment setup complete")
        return True
    except Exception as e:
        logger.bind(ACTIVITY="SETUP").error(f"❌ Setup error: {str(e)}")
        logger.bind(ACTIVITY="SETUP").exception("Full setup error traceback:")
        return False

if __name__ == "__main__":
    logger.bind(ACTIVITY="MAIN").info("🎯 Solana Trading Bot Starting...")
    
    # Perform setup
    if setup():
        logger.bind(ACTIVITY="MAIN").info("▶️ Launching bot main loop...")
        # Run the bot
        asyncio.run(main())
    else:
        logger.bind(ACTIVITY="MAIN").error("💀 Failed to set up the bot environment")
        sys.exit(1) 