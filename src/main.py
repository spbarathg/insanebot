"""
Minimalistic Solana trading bot with robust error handling and type safety.
"""
import asyncio
import signal
import sys
import time
import json
import aiohttp
import os
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass
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

@dataclass
class TradeInfo:
    """Data class for storing trade information."""
    size: float
    is_buy: bool
    timestamp: float
    price: float
    token_address: str

class SimpleTradingBot:
    """
    A Solana trading bot that monitors and trades tokens based on configurable parameters.
    
    Attributes:
        wallet_address (str): The Solana wallet address to use for trading
        simulation_mode (bool): Whether to run in simulation mode
        error_handler (ErrorHandler): Handles error tracking and recovery
        is_running (bool): Whether the bot is currently running
        last_trade_time (float): Timestamp of the last trade
        active_trades (Dict[str, TradeInfo]): Currently active trades
        known_tokens (Set[str]): Set of known token addresses
    """
    
    def __init__(self, wallet_address: str, simulation_mode: bool = False):
        self.wallet_address = wallet_address
        self.error_handler = ErrorHandler()
        self.is_running = False
        self.last_trade_time = 0
        self.active_trades: Dict[str, TradeInfo] = {}
        self.known_tokens: Set[str] = set()
        self.simulation_mode = simulation_mode
        
        # Initialize core components
        self.rpc_client = AsyncClient("https://api.mainnet-beta.solana.com", commitment=Confirmed)
        self.dex = RaydiumDEX(self.rpc_client)
        self.wallet = WalletManager(self.rpc_client)
        self.simulator = TradingSimulator(self.rpc_client) if simulation_mode else None

    async def initialize(self) -> None:
        """
        Initialize the trading bot with proper error handling.
        
        Raises:
            ValueError: If wallet initialization fails
            Exception: For other initialization errors
        """
        try:
            await self._load_known_tokens()
            await self._initialize_wallet()
            
            logger.info(f"Initialized trading bot for wallet: {self.wallet_address}")
            logger.info(f"Loaded {len(self.known_tokens)} known tokens")
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {str(e)}")
            self.error_handler.add_error(e)
            raise

    async def _load_known_tokens(self) -> None:
        """Load known tokens from file asynchronously."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("known_tokens.json") as response:
                    if response.status == 200:
                        self.known_tokens = set(await response.json())
                    else:
                        self.known_tokens = set()
        except Exception as e:
            logger.warning(f"Failed to load known tokens: {e}")
            self.known_tokens = set()

    async def _initialize_wallet(self) -> None:
        """
        Initialize wallet with balance check.
        
        Raises:
            ValueError: If wallet balance is insufficient
        """
        if self.simulation_mode:
            self.simulator.load_state()
            logger.info(f"Simulation mode active. Balance: {self.simulator.get_balance()} SOL")
        else:
            balance = await self.wallet.get_balance()
            logger.info(f"Wallet balance: {balance} SOL")
            
            if balance < TRADING_CONFIG["min_position_size"]:
                raise ValueError("Insufficient wallet balance")

    async def _get_tokens_to_check(self) -> List[str]:
        """
        Get list of new tokens to check.
        
        Returns:
            List[str]: List of new token addresses to check
        """
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
            await self._save_known_tokens()
            
            return new_tokens
            
        except Exception as e:
            logger.error(f"Error getting tokens: {str(e)}")
            self.error_handler.add_error(e)
            return []

    async def _save_known_tokens(self) -> None:
        """Save known tokens to file asynchronously."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "known_tokens.json",
                    json=list(self.known_tokens)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to save known tokens: {response.status}")
        except Exception as e:
            logger.error(f"Error saving known tokens: {e}")

    async def _get_token_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Get token data with validation.
        
        Args:
            token_address (str): The token address to check
            
        Returns:
            Optional[Dict[str, Any]]: Token data if valid, None otherwise
        """
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
            should_buy = self._should_buy_token(price_data, liquidity)
            should_sell = self._should_sell_token(price_data)
            
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

    def _should_buy_token(self, price_data: Dict[str, Any], liquidity: float) -> bool:
        """
        Determine if a token should be bought based on price data and liquidity.
        
        Args:
            price_data (Dict[str, Any]): Token price data
            liquidity (float): Token liquidity
            
        Returns:
            bool: True if token should be bought
        """
        return (
            price_data["price_change_1h"] > MARKET_CONFIG["volatility_threshold"] and
            price_data["volume_change_1h"] > 0 and
            liquidity >= MARKET_CONFIG["min_liquidity"]
        )

    def _should_sell_token(self, price_data: Dict[str, Any]) -> bool:
        """
        Determine if a token should be sold based on price data.
        
        Args:
            price_data (Dict[str, Any]): Token price data
            
        Returns:
            bool: True if token should be sold
        """
        return (
            price_data["price_change_1h"] < -MARKET_CONFIG["volatility_threshold"] or
            price_data["volume_change_1h"] < 0
        )

    async def execute_trade(self, token_address: str, is_buy: bool) -> bool:
        """
        Execute a trade with comprehensive error handling.
        
        Args:
            token_address (str): The token address to trade
            is_buy (bool): Whether this is a buy or sell trade
            
        Returns:
            bool: True if trade was successful
        """
        try:
            if not self._can_execute_trade():
                return False
            
            token_data = await self._get_token_data(token_address)
            if not token_data or not self._validate_trade_conditions(token_data, is_buy):
                return False
            
            trade_size = self._calculate_trade_size(token_data)
            success = await self._execute_trade_transaction(token_address, trade_size, is_buy)
            
            if success:
                self._update_trade_state(token_address, trade_size, is_buy, token_data["price"])
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            self.error_handler.add_error(e)
            return False

    def _can_execute_trade(self) -> bool:
        """Check if a trade can be executed based on cooldown period."""
        return time.time() - self.last_trade_time >= CORE_CONFIG["trading"]["cooldown_period"]

    def _validate_trade_conditions(self, token_data: Dict[str, Any], is_buy: bool) -> bool:
        """Validate trade conditions based on token data."""
        return token_data["should_buy"] if is_buy else token_data["should_sell"]

    def _calculate_trade_size(self, token_data: Dict[str, Any]) -> float:
        """Calculate appropriate trade size based on liquidity."""
        return min(
            TRADING_CONFIG["max_position_size"],
            token_data["liquidity"] * 0.01
        )

    async def _execute_trade_transaction(
        self,
        token_address: str,
        trade_size: float,
        is_buy: bool
    ) -> bool:
        """Execute the actual trade transaction."""
        if self.simulation_mode:
            return await self.simulator.simulate_trade(token_address, trade_size, is_buy)
        
        transaction = await self.dex.create_swap_transaction(
            token_address,
            trade_size,
            is_buy,
            self.wallet.get_keypair()
        )
        
        if not transaction:
            return False
            
        return await self.wallet.send_transaction(transaction)

    def _update_trade_state(
        self,
        token_address: str,
        trade_size: float,
        is_buy: bool,
        price: float
    ) -> None:
        """Update trade state after successful execution."""
        self.last_trade_time = time.time()
        self.active_trades[token_address] = TradeInfo(
            size=trade_size,
            is_buy=is_buy,
            timestamp=time.time(),
            price=price,
            token_address=token_address
        )

    async def run(self) -> None:
        """Main trading loop with error handling and monitoring."""
        self.is_running = True
        logger.info("Starting trading bot...")
        
        while self.is_running:
            try:
                if self.error_handler.should_stop_trading():
                    logger.error("Critical error threshold reached. Stopping bot.")
                    break
                
                await self._process_trading_cycle()
                await asyncio.sleep(CORE_CONFIG["monitoring"]["check_interval"])
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                self.error_handler.add_error(e)
                await asyncio.sleep(CORE_CONFIG["monitoring"]["retry_delay"])

    async def _process_trading_cycle(self) -> None:
        """Process one complete trading cycle."""
        new_tokens = await self._get_tokens_to_check()
        
        for token in new_tokens:
            if not self.is_running:
                break
            
            if await self.execute_trade(token, is_buy=True):
                logger.info(f"Executed buy trade for {token}")
            
            if token in self.active_trades:
                if await self.execute_trade(token, is_buy=False):
                    logger.info(f"Executed sell trade for {token}")

    async def close(self) -> None:
        """Cleanup and shutdown the trading bot."""
        self.is_running = False
        await self.rpc_client.close()
        await self._save_known_tokens()
        logger.info("Trading bot stopped.")

def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals."""
    logger.info("Received shutdown signal")
    sys.exit(0)

async def main() -> None:
    """Main entry point for the trading bot."""
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