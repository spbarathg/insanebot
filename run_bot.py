#!/usr/bin/env python3
"""
Solana Trading Bot - Runner Script
"""
import os
import sys
import argparse
import asyncio
import logging
import json
import time
import signal
from pathlib import Path
import colorama
from colorama import Fore, Style
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the trading bot
from src.core.main import MemeCoinBot

# Initialize colorama
colorama.init()

# Configure logging
logger = logging.getLogger()

def setup_logging(log_level: str, log_file: str = None):
    """Set up logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

def print_header():
    """Print a nice header for the bot"""
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}                      SOLANA MEMECOIN TRADING BOT{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}• Advanced technical analysis")
    print(f"• Portfolio tracking and management")
    print(f"• Risk assessment and position sizing")
    print(f"• Market scanning for new opportunities{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Solana Memecoin Trading Bot")
    
    parser.add_argument("--simulation", action="store_true", default=True,
                        help="Run in simulation mode (default: True)")
    parser.add_argument("--balance", type=float, default=1.0,
                        help="Initial balance for simulation (in SOL)")
    parser.add_argument("--log-level", type=str, choices=["debug", "info", "warning", "error", "critical"],
                        default="info", help="Logging level")
    parser.add_argument("--log-file", type=str, default="logs/trading_bot.log",
                        help="Log file path")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Configuration file path")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")
    
    return parser.parse_args()

async def run_bot(args):
    """Run the trading bot"""
    # Load configuration
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    # Create and initialize the bot
    bot = MemeCoinBot()
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    for s in [signal.SIGINT, signal.SIGTERM]:
        try:
            loop.add_signal_handler(
                s, lambda s=s: asyncio.create_task(shutdown(s, loop, bot))
            )
        except NotImplementedError:
            # Windows doesn't support SIGINT handler
            pass
    
    # Initialize and start the bot
    logger.info("Initializing trading bot...")
    success = await bot.initialize()
    
    if success:
        logger.info("Starting trading bot...")
        try:
            # Start the bot
            await bot.start()
        except asyncio.CancelledError:
            logger.info("Bot execution was cancelled")
        finally:
            # Ensure bot is properly shut down
            await bot.close()
    else:
        logger.error("Failed to initialize bot")

async def shutdown(signal_name, loop, bot):
    """Handle graceful shutdown"""
    logger.info(f"Received {signal_name}, shutting down...")
    await bot.close()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Disable colors if requested
    if args.no_color:
        colorama.deinit()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    
    # Print header
    print_header()
    
    # Create data directories if they don't exist
    for directory in ["data", "logs", "config", "models"]:
        os.makedirs(directory, exist_ok=True)
    
    # Set environment variables for simulation
    if args.simulation:
        os.environ["SIMULATION_MODE"] = "true"
        os.environ["SIMULATION_CAPITAL"] = str(args.balance)
        logger.info(f"Running in simulation mode with {args.balance} SOL")
    
    # Run the bot
    try:
        asyncio.run(run_bot(args))
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    main() 