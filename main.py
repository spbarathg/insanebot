#!/usr/bin/env python3
"""
Solana Trading Bot Entry Point
"""
import asyncio
import logging
import os
import signal
import sys
from dotenv import load_dotenv
from pathlib import Path
from src.core.main import MemeCoinBot
from src.utils.logging_config import initialize_logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("main")

# Initialize data directories
data_dir = Path("data")
logs_dir = Path("logs")
models_dir = Path("models")

for directory in [data_dir, logs_dir, models_dir]:
    if not directory.exists():
        directory.mkdir(parents=True)
        logger.info(f"Created directory: {directory}")

# Global reference to the bot
bot = None

async def shutdown(signal, loop):
    """Graceful shutdown on signal"""
    logger.info(f"Received exit signal {signal.name}...")
    
    # Close the bot
    if bot:
        logger.info("Shutting down bot...")
        await bot.close()
        
    # Cancel tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Stop the event loop
    loop.stop()
    logger.info("Shutdown complete")

async def main():
    """Main entry point"""
    global bot
    
    try:
        # Initialize bot
        logger.info("Initializing trading bot...")
        bot = MemeCoinBot()
        
        # Initialize components
        if not await bot.initialize():
            logger.error("Failed to initialize trading bot")
            return
            
        # Register signal handlers for graceful shutdown
        for signal_name in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(
                signal_name,
                lambda s=signal_name: asyncio.create_task(shutdown(s, loop))
            )
            
        # Start the bot
        logger.info("Starting trading bot...")
        await bot.start()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        if bot:
            await bot.close()
        
if __name__ == "__main__":
    # Set up the asyncio event loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        logger.info("Closing event loop")
        loop.close() 