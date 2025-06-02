"""
Trade execution module for Solana trading bot.
"""
import asyncio
import logging
import time
import os
import json
import base58
from typing import Dict, List, Optional, Any
from decimal import Decimal
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction
from solana.rpc.async_api import AsyncClient
from solders.commitment_config import CommitmentLevel
from solana.rpc.types import TxOpts
from config.core_config import TRADING_CONFIG

# Import settings
from src.services.jupiter_service import JupiterService
from src.services.helius_service import HeliusService
from src.services.wallet_manager import WalletManager

logger = logging.getLogger(__name__)

class TradeExecution:
    """Trade execution service for Solana trading bot."""
    
    def __init__(self):
        self.wallet_manager = WalletManager()
        self.jupiter = JupiterService()
        self.helius = HeliusService()
        self.session = None
        self.solana_client = None
        self.wallet = None
        self.wsol_mint = "So11111111111111111111111111111111111111112"
        self.usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC on Solana
        self.last_trade_time = 0
        self.trades_in_progress = {}
        self.trade_history = []
        
    async def initialize(self) -> bool:
        """Initialize trade execution service."""
        try:
            logger.info("Initializing trade execution service...")
            
            # Initialize wallet manager
            await self.wallet_manager.initialize()
            self.wallet = self.wallet_manager.get_keypair()
            
            # Initialize Solana client
            self.solana_client = AsyncClient("https://api.mainnet-beta.solana.com", commitment=CommitmentLevel.confirmed)
            
            # Initialize Jupiter
            await self.jupiter.initialize()
            
            # Initialize Helius
            await self.helius.initialize()
            
            # Load trade history if available
            self._load_trade_history()
            
            logger.info("Trade execution service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize trade execution service: {str(e)}")
            return False
            
    def _load_trade_history(self):
        """Load trade history from file."""
        try:
            trade_history_file = "trade_history.json"
            if os.path.exists(trade_history_file):
                with open(trade_history_file, 'r') as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded {len(self.trade_history)} trades from history")
            else:
                self.trade_history = []
                logger.info("No trade history file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
            self.trade_history = []
            
    def _save_trade_history(self):
        """Save trade history to file."""
        try:
            trade_history_file = "trade_history.json"
            os.makedirs(os.path.dirname(trade_history_file), exist_ok=True)
            with open(trade_history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=str)
            logger.debug(f"Saved {len(self.trade_history)} trades to history")
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
            
    async def close(self):
        """Close trade execution service."""
        try:
            await self.jupiter.close()
            await self.helius.close()
            await self.solana_client.close()
            logger.info("Trade execution service closed")
        except Exception as e:
            logger.error(f"Error closing trade execution service: {e}")
            
    async def execute_buy(self, token_address: str, amount_sol: float) -> Optional[Dict]:
        """Execute buy trade for a token using SOL."""
        try:
            # Check if we should execute trade
            if not self._can_execute_trade(token_address):
                return None
                
            # Get current SOL price for token
            price_data = await self.helius.get_token_price(token_address)
            if not price_data:
                logger.warning(f"No price data for token {token_address}")
                return None
                
            # Check if trade meets criteria
            if not self._validate_trade(token_address, True, amount_sol, price_data):
                return None
                
            # Get swap quote
            quote = await self.jupiter.get_swap_quote(
                self.wsol_mint,  # Input is SOL
                token_address,   # Output is token
                amount_sol       # Amount in SOL
            )
            
            if not quote:
                logger.warning(f"Could not get swap quote for {token_address}")
                return None
                
            # Create swap transaction
            swap_tx_data = await self.jupiter.create_swap_transaction(
                quote, 
                str(self.wallet.public_key)
            )
            
            if not swap_tx_data:
                logger.warning(f"Could not create swap transaction for {token_address}")
                return None
                
            # Process and send transaction
            tx_result = await self._process_and_send_transaction(swap_tx_data.get('swapTransaction'))
            
            if not tx_result:
                logger.warning(f"Transaction failed for {token_address}")
                return None
                
            # Record trade
            trade_record = {
                'type': 'buy',
                'token': token_address,
                'amount_sol': amount_sol,
                'price': swap_tx_data.get('price'),
                'token_amount': float(swap_tx_data.get('expectedOutputAmount', 0)) / 1e9,
                'timestamp': time.time(),
                'transaction': tx_result
            }
            
            # Update trade history
            self.trade_history.append(trade_record)
            self._save_trade_history()
            
            logger.info(f"Buy executed: {amount_sol} SOL → {trade_record['token_amount']} tokens at {trade_record['price']}")
            return trade_record
            
        except Exception as e:
            logger.error(f"Error executing buy: {str(e)}")
            return None
            
    async def execute_sell(self, token_address: str, amount: float) -> Optional[Dict]:
        """Execute sell trade for a token to SOL."""
        try:
            # Check if we should execute trade
            if not self._can_execute_trade(token_address):
                return None
                
            # Get current token data
            price_data = await self.helius.get_token_price(token_address)
            if not price_data:
                logger.warning(f"No price data for token {token_address}")
                return None
                
            # Check if trade meets criteria
            if not self._validate_trade(token_address, False, amount, price_data):
                return None
                
            # Get swap quote
            quote = await self.jupiter.get_swap_quote(
                token_address,  # Input is token
                self.wsol_mint,  # Output is SOL
                amount          # Amount in tokens
            )
            
            if not quote:
                logger.warning(f"Could not get swap quote for {token_address}")
                return None
                
            # Create swap transaction
            swap_tx_data = await self.jupiter.create_swap_transaction(
                quote, 
                str(self.wallet.public_key)
            )
            
            if not swap_tx_data:
                logger.warning(f"Could not create swap transaction for {token_address}")
                return None
                
            # Process and send transaction
            tx_result = await self._process_and_send_transaction(swap_tx_data.get('swapTransaction'))
            
            if not tx_result:
                logger.warning(f"Transaction failed for {token_address}")
                return None
                
            # Record trade
            trade_record = {
                'type': 'sell',
                'token': token_address,
                'amount_tokens': amount,
                'price': swap_tx_data.get('price'),
                'sol_amount': float(swap_tx_data.get('expectedOutputAmount', 0)) / 1e9,
                'timestamp': time.time(),
                'transaction': tx_result
            }
            
            # Update trade history
            self.trade_history.append(trade_record)
            self._save_trade_history()
            
            logger.info(f"Sell executed: {amount} tokens → {trade_record['sol_amount']} SOL at {trade_record['price']}")
            return trade_record
            
        except Exception as e:
            logger.error(f"Error executing sell: {str(e)}")
            return None
    
    async def _process_and_send_transaction(self, transaction_base64: str) -> Optional[str]:
        """Process and send a transaction."""
        try:
            # Decode the transaction
            decoded_tx = base58.b58decode(transaction_base64)
            
            # Create transaction object
            tx = Transaction.deserialize(decoded_tx)
            
            # Send the transaction
            result = await self.solana_client.send_raw_transaction(
                decoded_tx,
                opts={
                    'skip_preflight': True,
                    'max_retries': 3,  # Default max retries
                    'preflight_commitment': CommitmentLevel.confirmed
                }
            )
            
            if not result or 'result' not in result:
                logger.error(f"Failed to send transaction: {result}")
                return None
                
            tx_signature = result['result']
            
            # Monitor for confirmation
            confirmed = await self._wait_for_confirmation(tx_signature)
            if not confirmed:
                logger.warning(f"Transaction not confirmed: {tx_signature}")
                return None
                
            return tx_signature
            
        except Exception as e:
            logger.error(f"Error processing transaction: {str(e)}")
            return None
            
    async def _wait_for_confirmation(self, signature: str, max_retries: int = 30) -> bool:
        """Wait for transaction confirmation."""
        retries = 0
        while retries < max_retries:
            try:
                result = await self.solana_client.get_signature_statuses([signature])
                if result and 'result' in result and result['result']['value'][0]:
                    status = result['result']['value'][0]
                    if status.get('confirmationStatus') == 'confirmed' or status.get('confirmationStatus') == 'finalized':
                        return True
            except Exception as e:
                logger.error(f"Error checking confirmation: {str(e)}")
                
            await asyncio.sleep(1)
            retries += 1
            
        return False
        
    def _can_execute_trade(self, token_address: str) -> bool:
        """Check if we can execute a trade for this token."""
        # Check if we have an active trade for this token
        if token_address in self.trades_in_progress:
            logger.warning(f"Trade already in progress for {token_address}")
            return False
            
        # Check cooldown period
        current_time = time.time()
        trade_cooldown = 30  # Default 30 seconds cooldown
        if current_time - self.last_trade_time < trade_cooldown:
            logger.warning(f"Trade cooldown in effect, wait {trade_cooldown - (current_time - self.last_trade_time)} seconds")
            return False
            
        return True
        
    async def _validate_trade(self, token_address: str, is_buy: bool, amount: float, price_data: Dict) -> bool:
        """Validate if trade meets criteria."""
        try:
            # Get token info from Helius
            token_info = await self.helius.get_token_metadata(token_address)
            if not token_info:
                logger.warning(f"No token info for {token_address}")
                return False
                
            # Get liquidity info
            liquidity_data = await self.helius.get_token_liquidity(token_address)
            if not liquidity_data:
                logger.warning(f"No liquidity data for {token_address}")
                return False
                
            # Check minimum liquidity
            liquidity = liquidity_data.get('liquidity', 0)
            min_liquidity = 10000  # Default minimum liquidity
            if liquidity < min_liquidity:
                logger.warning(f"Insufficient liquidity for {token_address}: {liquidity} < {min_liquidity}")
                return False
                
            # Check token age for buys
            if is_buy:
                token_age = time.time() - token_info.get('created_at', time.time())
                max_token_age = 86400  # Default 24 hours
                if token_age < max_token_age:
                    logger.warning(f"Token too new for {token_address}: {token_age} < {max_token_age}")
                    return False
                
            # Check price impact
            price_impact = await self._calculate_price_impact(token_address, amount, is_buy)
            max_price_impact = 0.05  # Default 5% max price impact
            if price_impact > max_price_impact:
                logger.warning(f"Price impact too high for {token_address}: {price_impact} > {max_price_impact}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade: {str(e)}")
            return False
            
    async def _calculate_price_impact(self, token_address: str, amount: float, is_buy: bool) -> float:
        """Calculate price impact of a trade."""
        try:
            # For buy, input is SOL, output is token
            # For sell, input is token, output is SOL
            input_token = self.wsol_mint if is_buy else token_address
            output_token = token_address if is_buy else self.wsol_mint
            
            # Get quote
            quote = await self.jupiter.get_swap_quote(input_token, output_token, amount)
            if not quote:
                return 1.0  # Default to high impact if no quote
                
            # Extract price impact
            price_impact = float(quote.get('priceImpactPct', 0))
            
            return price_impact
            
        except Exception as e:
            logger.error(f"Error calculating price impact: {str(e)}")
            return 1.0  # Default to high impact on error
            
    async def get_wallet_balance(self) -> Optional[Dict]:
        """Get wallet balances."""
        try:
            balances = await self.helius.get_token_balances(str(self.wallet.public_key))
            return balances
        except Exception as e:
            logger.error(f"Error getting wallet balance: {str(e)}")
            return None 