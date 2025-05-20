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
from solana.keypair import Keypair
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
    Manages a wallet for Solana trading bot.
    
    Handles key management, balance checks, and transaction monitoring.
    """
    
    def __init__(self):
        self.keypair = None
        self.public_key = None
        self.solana_client = None
        self.balance = 0.0
        self.transactions = []
        self.pending_transactions = {}
        self.token_accounts = {}
        self.balance_cache = 0.0
        self.last_balance_update = 0
        wallet_logger.info("WalletManager instance initialized")

    async def initialize(self) -> bool:
        """Initialize wallet manager."""
        try:
            logger.info("Initializing wallet manager...")
            
            # Load keypair from private key
            self._load_keypair()
            
            # Initialize Solana client
            self.solana_client = AsyncClient(settings.RPC_ENDPOINTS[0], commitment=Confirmed)
            
            # Check wallet balance
            await self.check_balance()
            
            logger.info(f"Wallet manager initialized for {self.public_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize wallet manager: {str(e)}")
            return False
            
    def _load_keypair(self):
        """Load keypair from private key or create a new one in simulation mode."""
        try:
            if settings.SIMULATION_MODE:
                # In simulation mode, create a random keypair
                self.keypair = Keypair()
                logger.info("Created simulation keypair")
            else:
                # In real mode, load from environment variable
                if not settings.SOLANA_PRIVATE_KEY:
                    raise ValueError("SOLANA_PRIVATE_KEY not set in environment")
                    
                # Convert private key to bytes and create keypair
                private_key_bytes = bytes.fromhex(settings.SOLANA_PRIVATE_KEY)
                self.keypair = Keypair.from_secret_key(private_key_bytes)
                
            # Set public key
            self.public_key = self.keypair.public_key
            
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
                return 0
        except Exception as e:
            logger.error(f"Error checking balance: {str(e)}")
            return 0
            
    def get_keypair(self) -> Keypair:
        """Get wallet keypair."""
        return self.keypair
        
    def get_public_key(self) -> PublicKey:
        """Get wallet public key."""
        return self.public_key
        
    async def get_token_accounts(self) -> List[Dict]:
        """Get token accounts for wallet."""
        try:
            response = await self.solana_client.get_token_accounts_by_owner(
                self.public_key,
                {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"}
            )
            
            token_accounts = []
            if response and "result" in response and "value" in response["result"]:
                for account in response["result"]["value"]:
                    token_accounts.append({
                        "pubkey": account["pubkey"],
                        "mint": account["account"]["data"]["parsed"]["info"]["mint"],
                        "amount": int(account["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"]) / 10**account["account"]["data"]["parsed"]["info"]["tokenAmount"]["decimals"],
                        "decimals": account["account"]["data"]["parsed"]["info"]["tokenAmount"]["decimals"]
                    })
            
            self.token_accounts = {account["mint"]: account["pubkey"] for account in token_accounts}
            wallet_logger.info(f"Retrieved {len(token_accounts)} token accounts")
            return token_accounts
        except Exception as e:
            logger.error(f"Error getting token accounts: {str(e)}")
            return []
            
    async def monitor_transaction(self, signature: str, max_retries: int = 30) -> bool:
        """Monitor transaction until confirmed or failed."""
        try:
            self.pending_transactions[signature] = {
                "status": "pending",
                "start_time": time.time()
            }
            
            retry_count = 0
            while retry_count < max_retries:
                # Check signature status
                response = await self.solana_client.get_signature_statuses([signature])
                
                if response and "result" in response and "value" in response["result"] and response["result"]["value"][0]:
                    status = response["result"]["value"][0]
                    
                    if status.get("confirmationStatus") == "confirmed" or status.get("confirmationStatus") == "finalized":
                        # Transaction confirmed
                        self.pending_transactions[signature]["status"] = "confirmed"
                        self.pending_transactions[signature]["confirmed_time"] = time.time()
                        
                        # Add to transactions history
                        self.transactions.append({
                            "signature": signature,
                            "status": "confirmed",
                            "timestamp": time.time()
                        })
                        
                        logger.info(f"Transaction confirmed: {signature}")
                        return True
                    elif status.get("err"):
                        # Transaction failed
                        self.pending_transactions[signature]["status"] = "failed"
                        self.pending_transactions[signature]["error"] = status.get("err")
                        
                        # Add to transactions history
                        self.transactions.append({
                            "signature": signature,
                            "status": "failed",
                            "error": status.get("err"),
                            "timestamp": time.time()
                        })
                        
                        logger.warning(f"Transaction failed: {signature} - {status.get('err')}")
                        return False
                
                # Sleep before retrying
                await asyncio.sleep(1)
                retry_count += 1
                
            # Max retries reached, consider transaction failed
            self.pending_transactions[signature]["status"] = "timeout"
            logger.warning(f"Transaction timeout: {signature}")
            return False
            
        except Exception as e:
            logger.error(f"Error monitoring transaction: {str(e)}")
            return False
            
    async def close(self):
        """Close wallet manager."""
        try:
            if self.solana_client:
                await self.solana_client.close()
                logger.info("Wallet manager closed")
        except Exception as e:
            logger.error(f"Error closing wallet manager: {str(e)}")
            raise WalletError(f"Error closing wallet manager: {str(e)}")

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