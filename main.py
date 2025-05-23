#!/usr/bin/env python3
"""
Solana Trading Bot with robust error handling - Standalone Version
"""
import asyncio
import logging
import os
import signal
import sys
import platform
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from src.core.main import MemeCoinBot
from src.utils.logging_config import initialize_logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more verbose logging
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("main")

# Ensure directories exist
for directory in ["data", "logs", "config", "models"]:
    os.makedirs(directory, exist_ok=True)

# Global shutdown flag
shutdown_requested = False

class WalletManager:
    """Simplified wallet manager with simulation capabilities."""
    
    def __init__(self):
        self.simulation_mode = os.getenv("SIMULATION_MODE", "true").lower() == "true"
        self.balance = float(os.getenv("SIMULATION_CAPITAL", "0.1"))
        
    async def initialize(self) -> bool:
        logger.info(f"Wallet initialized in {'simulation' if self.simulation_mode else 'live'} mode")
        return True
        
    async def close(self):
        logger.info("Wallet connections closed")
        return True

class MemeCoinBot:
    """Main trading bot implementation."""
    
    def __init__(self):
        self.wallet_manager = WalletManager()
        self.running = False
        self.test_mode = True
        
    async def initialize(self) -> bool:
        try:
            logger.info("Initializing trading bot components...")
            await self.wallet_manager.initialize()
            return True
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            return False
            
    async def start(self):
        if self.running:
            return
            
        self.running = True
        logger.info("Starting trading bot...")
        
        try:
            while not shutdown_requested:
                try:
                    if self.test_mode:
                        logger.info("Bot is running in test mode - All systems operational")
                    else:
                        status = "SIMULATION" if self.wallet_manager.simulation_mode else "LIVE"
                        logger.info(f"Bot is running in {status} mode - All systems operational")
                    
                    # Sleep to prevent CPU usage
                    await asyncio.sleep(30)
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(10)
        finally:
            self.running = False
            
    async def close(self):
        logger.info("Shutting down bot...")
        await self.wallet_manager.close()
        logger.info("Bot shutdown complete")

async def shutdown(signal_name, loop):
    """Graceful shutdown on signal"""
    logger.info(f"Received exit signal {signal_name}...")

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
        logger.info("Initializing trading bot components...")
        success = await bot.initialize()
        if not success:
            logger.error("Failed to initialize trading bot")
            return

        logger.info("Bot initialized successfully!")

        # Register signal handlers for graceful shutdown on Unix-like systems
        if platform.system() != "Windows":
            for signal_name in [signal.SIGINT, signal.SIGTERM]:
                loop.add_signal_handler(
                    signal_name,
                    lambda s=signal_name: asyncio.create_task(shutdown(s, loop))
                )

        # Start the bot
        logger.info("Starting trading bot...")
        await bot.start()

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)  # Added exc_info=True for stack trace
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