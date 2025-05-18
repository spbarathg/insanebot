"""
Minimalistic Solana trading bot with robust error handling.
"""
import asyncio
import signal
import sys
import time
import json
import aiohttp
import os
from typing import List, Dict, Optional, Set
from loguru import logger
from solana.rpc.async_api import AsyncClient
from solana.transaction import Transaction
from solana.keypair import Keypair
from solana.rpc.commitment import Confirmed
from core.config import CORE_CONFIG, MARKET_CONFIG, TRADING_CONFIG
from core.error_handler import ErrorHandler
from core.dex import RaydiumDEX
from core.wallet import WalletManager
from core.simulator import TradingSimulator

class SimpleTradingBot:
    def __init__(self, wallet_address: str, simulation_mode: bool = False):
        self.wallet_address = wallet_address
        self.error_handler = ErrorHandler()
        self.is_running = False
        self.last_trade_time = 0
        self.active_trades: Dict[str, Dict] = {}
        self.known_tokens: Set[str] = set()
        self.simulation_mode = simulation_mode
        
        # Initialize core components
        self.rpc_client = AsyncClient("https://api.mainnet-beta.solana.com", commitment=Confirmed)
        self.dex = RaydiumDEX(self.rpc_client)
        self.wallet = WalletManager(self.rpc_client)
        self.simulator = TradingSimulator(self.rpc_client) if simulation_mode else None

    async def initialize(self) -> None:
        """Initialize the trading bot with proper error handling."""
        try:
            # Load known tokens
            self._load_known_tokens()
            
            # Initialize wallet
            await self._initialize_wallet()
            
            logger.info(f"Initialized trading bot for wallet: {self.wallet_address}")
            logger.info(f"Loaded {len(self.known_tokens)} known tokens")
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {str(e)}")
            self.error_handler.add_error(e)
            raise

    def _load_known_tokens(self) -> None:
        """Load known tokens from file."""
        try:
            with open("known_tokens.json", "r") as f:
                self.known_tokens = set(json.load(f))
        except FileNotFoundError:
            self.known_tokens = set()

    async def _initialize_wallet(self) -> None:
        """Initialize wallet with balance check."""
        if self.simulation_mode:
            self.simulator.load_state()
            logger.info(f"Simulation mode active. Balance: {self.simulator.get_balance()} SOL")
        else:
            balance = await self.wallet.get_balance()
            logger.info(f"Wallet balance: {balance} SOL")
            
            if balance < TRADING_CONFIG["min_position_size"]:
                raise ValueError("Insufficient wallet balance")

    async def _get_tokens_to_check(self) -> List[str]:
        """Get list of new tokens to check."""
        try:
            response = await self.rpc_client.get_program_accounts(
                "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                encoding="jsonParsed",
                limit=100
            )
            
            new_tokens = [
                account["pubkey"]
                for account in response["result"]
                if account["pubkey"] not in self.known_tokens
            ]
            
            # Update known tokens
            self.known_tokens.update(new_tokens)
            self._save_known_tokens()
            
            return new_tokens
            
        except Exception as e:
            logger.error(f"Error getting tokens: {str(e)}")
            self.error_handler.add_error(e)
            return []

    def _save_known_tokens(self) -> None:
        """Save known tokens to file."""
        with open("known_tokens.json", "w") as f:
            json.dump(list(self.known_tokens), f)

    async def _get_token_data(self, token_address: str) -> Optional[Dict]:
        """Get token data with validation."""
        try:
            # Get token info
            token_info = await self.rpc_client.get_account_info(token_address)
            if not token_info["result"]["value"]:
                return None
            
            # Get price and liquidity
            price_data = await self.dex.get_token_price(token_address)
            liquidity = await self.dex.get_liquidity(token_address)
            
            if not price_data or not liquidity:
                return None
            
            # Calculate trading signals
            should_buy = (
                price_data["price_change_1h"] > MARKET_CONFIG["volatility_threshold"] and
                price_data["volume_change_1h"] > 0 and
                liquidity >= MARKET_CONFIG["min_liquidity"]
            )
            
            should_sell = (
                price_data["price_change_1h"] < -MARKET_CONFIG["volatility_threshold"] or
                price_data["volume_change_1h"] < 0
            )
            
            return {
                "address": token_address,
                "price": price_data["price"],
                "liquidity": liquidity,
                "should_buy": should_buy,
                "should_sell": should_sell
            }
            
        except Exception as e:
            logger.error(f"Error getting token data: {str(e)}")
            self.error_handler.add_error(e)
            return None

    async def execute_trade(self, token_address: str, is_buy: bool) -> bool:
        """Execute a trade with comprehensive error handling."""
        try:
            # Check cooldown
            if time.time() - self.last_trade_time < CORE_CONFIG["trading"]["cooldown_period"]:
                return False
            
            # Get token data
            token_data = await self._get_token_data(token_address)
            if not token_data:
                return False
            
            # Validate trade conditions
            if is_buy and not token_data["should_buy"]:
                return False
            if not is_buy and not token_data["should_sell"]:
                return False
            
            # Calculate trade size
            trade_size = min(
                TRADING_CONFIG["max_position_size"],
                token_data["liquidity"] * 0.01
            )
            
            # Execute trade
            if self.simulation_mode:
                success = await self.simulator.simulate_trade(token_address, trade_size, is_buy)
            else:
                transaction = await self.dex.create_swap_transaction(
                    token_address,
                    trade_size,
                    is_buy,
                    self.wallet.get_keypair()
                )
                if not transaction:
                    return False
                success = await self.wallet.send_transaction(transaction)
            
            if success:
                self.last_trade_time = time.time()
                self.active_trades[token_address] = {
                    "size": trade_size,
                    "is_buy": is_buy,
                    "timestamp": time.time()
                }
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            self.error_handler.add_error(e)
            return False

    async def run(self) -> None:
        """Main trading loop."""
        self.is_running = True
        logger.info("Starting trading bot...")
        
        while self.is_running:
            try:
                # Check for critical errors
                if self.error_handler.should_stop_trading():
                    logger.error("Critical error threshold reached. Stopping bot.")
                    break
                
                # Get new tokens
                new_tokens = await self._get_tokens_to_check()
                
                # Process each token
                for token in new_tokens:
                    if not self.is_running:
                        break
                    
                    # Check for buy opportunity
                    if await self.execute_trade(token, is_buy=True):
                        logger.info(f"Executed buy trade for {token}")
                    
                    # Check for sell opportunity
                    if token in self.active_trades:
                        if await self.execute_trade(token, is_buy=False):
                            logger.info(f"Executed sell trade for {token}")
                
                # Sleep between iterations
                await asyncio.sleep(CORE_CONFIG["monitoring"]["check_interval"])
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                self.error_handler.add_error(e)
                await asyncio.sleep(CORE_CONFIG["monitoring"]["retry_delay"])

    async def close(self) -> None:
        """Cleanup and shutdown."""
        self.is_running = False
        await self.rpc_client.close()
        self._save_known_tokens()
        logger.info("Trading bot stopped.")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Received shutdown signal")
    sys.exit(0)

async def main():
    """Main entry point."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Get wallet address from environment
        wallet_address = os.getenv("WALLET_ADDRESS")
        if not wallet_address:
            raise ValueError("WALLET_ADDRESS not found in environment")
            
        # Initialize bot
        bot = SimpleTradingBot(wallet_address)
        await bot.initialize()
        
        # Run bot
        await bot.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
        
    finally:
        # Cleanup
        if 'bot' in locals():
            await bot.close()

if __name__ == "__main__":
    asyncio.run(main()) 