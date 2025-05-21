"""
Wallet manager for Solana trading bot.
"""
import logging
import os
import base58
import json
import time
import asyncio
from typing import Dict, List, Optional, Any
from solders.keypair import Keypair
from solana.publickey import PublicKey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.transaction import Transaction
from solana.system_program import transfer

# Try to import SPL token functions, but continue even if they're not available
# These will only be needed for token-related operations
try:
    from spl.token.constants import TOKEN_PROGRAM_ID
    from spl.token.instructions import create_associated_token_account, transfer_checked
    SPL_TOKEN_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("SPL token module not available. Token operations will be limited.")
    SPL_TOKEN_AVAILABLE = False

from ..utils.config import settings
from ..utils.logging_config import (
    wallet_logger, error_logger, handle_errors, log_performance,
    WalletError, NetworkError
)

logger = logging.getLogger(__name__)

class WalletManager:
    """
    Manages wallet operations including key storage, balance checking,
    and transaction signing.
    """
    
    def __init__(self):
        """Initialize wallet manager with Solana client."""
        self.solana_client = None
        self.keypair = None
        self.public_key = None
        self.balance = 0.0
        self.balance_cache = None
        self.last_balance_update = 0
        self.initialize()
        
    def initialize(self):
        """Initialize wallet manager with RPC connection and keypair."""
        try:
            # Initialize Solana client
            rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
            self.solana_client = AsyncClient(rpc_url)
            
            # Load keypair
            self._load_keypair()
            
            # Check balance
            asyncio.run(self.check_balance())
            
            # Check if balance is sufficient
            min_balance = float(os.getenv("MIN_BALANCE", "0.05"))
            
            # Handle null balance by defaulting to 0
            if self.balance is None:
                logger.warning("Balance is None, defaulting to 0")
                self.balance = 0.0
                
            # Now we can safely compare with a float
            if self.balance < min_balance:
                logger.warning(f"Wallet balance {self.balance} SOL is below minimum {min_balance} SOL")
                
            logger.info("Wallet manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize bot: {str(e)}")
            raise
            
    def _load_keypair(self):
        """Load keypair from private key or create a new one in simulation mode."""
        try:
            if settings.SIMULATION_MODE:
                # In simulation mode, create a random keypair
                self.keypair = Keypair()
                logger.info("Created simulation keypair")
            else:
                # In real mode, load from environment variable
                if not settings.WALLET_PRIVATE_KEY and not settings.SOLANA_PRIVATE_KEY:
                    raise ValueError("WALLET_PRIVATE_KEY or SOLANA_PRIVATE_KEY not set in environment")
                    
                # Get private key from either environment variable
                private_key = settings.WALLET_PRIVATE_KEY or settings.SOLANA_PRIVATE_KEY
                
                try:
                    # Try base58 first
                    private_key_bytes = base58.b58decode(private_key)
                except Exception:
                    try:
                        # Then try hex
                        if private_key.startswith('0x'):
                            private_key = private_key[2:]
                        private_key_bytes = bytes.fromhex(private_key)
                    except Exception as e:
                        raise ValueError(f"Invalid private key format: {str(e)}")
                
                # Handle different Solana SDK versions
                try:
                    # Newer versions
                    self.keypair = Keypair.from_bytes(private_key_bytes)
                except AttributeError:
                    try:
                        # Alternative API
                        self.keypair = Keypair.from_secret_key(private_key_bytes)
                    except AttributeError:
                        # Last resort
                        raise ValueError("Could not create keypair - incompatible Solana SDK version")
                
            # Set public key
            self.public_key = self.keypair.pubkey()
            
        except Exception as e:
            logger.error(f"Error loading keypair: {str(e)}")
            raise
            
    async def check_balance(self) -> float:
        """Check wallet balance."""
        try:
            if settings.SIMULATION_MODE:
                # In simulation mode, use configured simulation capital
                self.balance = settings.SIMULATION_CAPITAL
                logger.info(f"Simulation balance: {self.balance} SOL")
                return self.balance
                
            # Get balance from RPC
            response = await self.solana_client.get_balance(self.public_key)
            if response and "result" in response and "value" in response["result"]:
                # Convert lamports to SOL
                self.balance = response["result"]["value"] / 1e9
                self.balance_cache = self.balance
                self.last_balance_update = time.time()
                logger.info(f"Wallet balance: {self.balance} SOL")
                return self.balance
            else:
                logger.warning("Could not retrieve wallet balance")
                return 0.0
        except Exception as e:
            logger.error(f"Error checking balance: {str(e)}")
            return 0.0
            
    def get_keypair(self) -> Keypair:
        """Get wallet keypair."""
        return self.keypair
        
    def get_public_key(self) -> PublicKey:
        """Get wallet public key."""
        return self.public_key
    
    async def close(self):
        """Close connections."""
        if self.solana_client:
            await self.solana_client.close()
            
    def __del__(self):
        """Ensure connections are closed on deletion."""
        try:
            if hasattr(self, 'solana_client') and self.solana_client:
                asyncio.create_task(self.solana_client.close())
        except Exception:
            pass

    @handle_errors(wallet_logger)
    @log_performance(wallet_logger)
    async def get_balance(self) -> Optional[float]:
        """Get wallet balance in SOL"""
        try:
            if not self.solana_client or not self.keypair:
                error_msg = "Wallet not initialized"
                wallet_logger.error(error_msg)
                raise WalletError(error_msg)

            response = await self.solana_client.get_balance(
                self.public_key,
                commitment=Confirmed
            )
            
            if response.value is not None:
                balance = response.value / 1e9  # Convert lamports to SOL
                self.balance_cache = balance
                self.last_balance_update = time.time()
                wallet_logger.info(f"Wallet balance: {balance} SOL")
                return balance
            
            error_msg = "Failed to get balance"
            wallet_logger.error(error_msg)
            raise WalletError(error_msg)

        except Exception as e:
            error_msg = f"Error getting balance: {str(e)}"
            wallet_logger.error(error_msg)
            raise WalletError(error_msg)

    @handle_errors(wallet_logger)
    @log_performance(wallet_logger)
    async def create_token_account(self, token_address: str) -> Optional[str]:
        """Create a token account for a specific token"""
        try:
            if not SPL_TOKEN_AVAILABLE:
                error_msg = "SPL token module not available. Cannot create token account."
                wallet_logger.error(error_msg)
                raise WalletError(error_msg)
                
            if not self.solana_client or not self.keypair:
                error_msg = "Wallet not initialized"
                wallet_logger.error(error_msg)
                raise WalletError(error_msg)

            # Check if account already exists
            if token_address in self.token_accounts:
                wallet_logger.debug(f"Token account already exists for {token_address}")
                return self.token_accounts[token_address]

            # Create token account
            transaction = Transaction()
            transaction.add(
                create_associated_token_account(
                    payer=self.public_key,
                    owner=self.public_key,
                    mint=token_address
                )
            )

            # Sign and send transaction
            result = await self.solana_client.send_transaction(
                transaction,
                self.keypair,
                opts={"skip_confirmation": False}
            )

            if result.value:
                account_address = result.value
                self.token_accounts[token_address] = account_address
                wallet_logger.info(f"Created token account for {token_address}: {account_address}")
                return account_address

            error_msg = f"Failed to create token account for {token_address}"
            wallet_logger.error(error_msg)
            raise WalletError(error_msg)

        except Exception as e:
            error_msg = f"Error creating token account for {token_address}: {str(e)}"
            wallet_logger.error(error_msg)
            raise WalletError(error_msg)

    @handle_errors(wallet_logger)
    @log_performance(wallet_logger)
    async def get_token_balance(self, token_address: str) -> Optional[float]:
        """Get token balance for a specific token"""
        try:
            if not self.solana_client or not self.keypair:
                error_msg = "Wallet not initialized"
                wallet_logger.error(error_msg)
                raise WalletError(error_msg)

            # Get token account
            account_address = await self.create_token_account(token_address)
            if not account_address:
                error_msg = f"Failed to get token account for {token_address}"
                wallet_logger.error(error_msg)
                raise WalletError(error_msg)

            # Get balance
            response = await self.solana_client.get_token_account_balance(
                account_address,
                commitment=Confirmed
            )

            if response.value:
                balance = float(response.value.amount) / (10 ** response.value.decimals)
                wallet_logger.info(f"Token balance for {token_address}: {balance}")
                return balance

            error_msg = f"Failed to get token balance for {token_address}"
            wallet_logger.error(error_msg)
            raise WalletError(error_msg)

        except Exception as e:
            error_msg = f"Error getting token balance for {token_address}: {str(e)}"
            wallet_logger.error(error_msg)
            raise WalletError(error_msg)

    @handle_errors(wallet_logger)
    @log_performance(wallet_logger)
    async def transfer_sol(self, recipient: str, amount: float) -> Optional[str]:
        """Transfer SOL to another wallet"""
        try:
            if not self.solana_client or not self.keypair:
                error_msg = "Wallet not initialized"
                wallet_logger.error(error_msg)
                raise WalletError(error_msg)

            # Create transfer instruction
            transaction = Transaction()
            transaction.add(
                transfer(
                    from_pubkey=self.public_key,
                    to_pubkey=recipient,
                    lamports=int(amount * 1e9)  # Convert SOL to lamports
                )
            )

            # Sign and send transaction
            result = await self.solana_client.send_transaction(
                transaction,
                self.keypair,
                opts={"skip_confirmation": False}
            )

            if result.value:
                wallet_logger.info(f"Transferred {amount} SOL to {recipient}")
                return result.value

            error_msg = f"Failed to transfer {amount} SOL to {recipient}"
            wallet_logger.error(error_msg)
            raise WalletError(error_msg)

        except Exception as e:
            error_msg = f"Error transferring SOL: {str(e)}"
            wallet_logger.error(error_msg)
            raise WalletError(error_msg)

    @handle_errors(wallet_logger)
    @log_performance(wallet_logger)
    async def transfer_token(self, token_address: str, recipient: str, amount: float) -> Optional[str]:
        """Transfer tokens to another wallet"""
        try:
            if not SPL_TOKEN_AVAILABLE:
                error_msg = "SPL token module not available. Cannot transfer tokens."
                wallet_logger.error(error_msg)
                raise WalletError(error_msg)
                
            if not self.solana_client or not self.keypair:
                error_msg = "Wallet not initialized"
                wallet_logger.error(error_msg)
                raise WalletError(error_msg)

            # Get token accounts
            source_account = await self.create_token_account(token_address)
            if not source_account:
                error_msg = f"Failed to get source token account for {token_address}"
                wallet_logger.error(error_msg)
                raise WalletError(error_msg)

            # Create recipient's token account if needed
            recipient_account = await self.create_token_account(token_address)
            if not recipient_account:
                error_msg = f"Failed to get recipient token account for {token_address}"
                wallet_logger.error(error_msg)
                raise WalletError(error_msg)

            # Create transfer instruction
            transaction = Transaction()
            transaction.add(
                transfer_checked(
                    program_id=TOKEN_PROGRAM_ID,
                    source=source_account,
                    mint=token_address,
                    dest=recipient_account,
                    owner=self.public_key,
                    amount=int(amount * 1e9),  # Convert to token decimals
                    decimals=9
                )
            )

            # Sign and send transaction
            result = await self.solana_client.send_transaction(
                transaction,
                self.keypair,
                opts={"skip_confirmation": False}
            )

            if result.value:
                wallet_logger.info(f"Transferred {amount} tokens to {recipient}")
                return result.value

            error_msg = f"Failed to transfer {amount} tokens to {recipient}"
            wallet_logger.error(error_msg)
            raise WalletError(error_msg)

        except Exception as e:
            error_msg = f"Error transferring token: {str(e)}"
            wallet_logger.error(error_msg)
            raise WalletError(error_msg) 