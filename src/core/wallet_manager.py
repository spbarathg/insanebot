from solana.rpc.async_api import AsyncClient
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.transaction import Transaction
from solana.system_program import SYS_PROGRAM_ID
from spl.token.instructions import get_associated_token_address
from spl.token.constants import TOKEN_PROGRAM_ID
import base58
from typing import Dict, Optional, Tuple
from solana.rpc.commitment import Confirmed
from ..utils.config import settings
from ..utils.logging_config import (
    wallet_logger, error_logger, handle_errors, log_performance,
    WalletError, NetworkError
)

class WalletManager:
    def __init__(self):
        self.client: Optional[AsyncClient] = None
        self.keypair: Optional[Keypair] = None
        self.token_accounts: Dict[str, str] = {}
        self.balance_cache = 0.0
        self.last_balance_update = 0
        wallet_logger.info("WalletManager instance initialized")

    @handle_errors(wallet_logger)
    @log_performance(wallet_logger)
    async def initialize(self) -> bool:
        """Initialize the wallet manager"""
        try:
            # Initialize Solana client
            self.client = AsyncClient(settings.SOLANA_RPC_URL)
            
            # Load private key
            private_key = bytes.fromhex(settings.SOLANA_PRIVATE_KEY)
            self.keypair = Keypair.from_secret_key(private_key)
            
            wallet_logger.info(f"Wallet initialized: {self.keypair.public_key}")
            return True

        except Exception as e:
            error_msg = f"Failed to initialize wallet: {str(e)}"
            wallet_logger.error(error_msg)
            raise WalletError(error_msg)

    @handle_errors(wallet_logger)
    async def close(self) -> None:
        """Close the wallet manager"""
        try:
            if self.client:
                await self.client.close()
            wallet_logger.info("Wallet manager closed successfully")
        except Exception as e:
            error_msg = f"Error closing wallet manager: {str(e)}"
            wallet_logger.error(error_msg)
            raise WalletError(error_msg)

    @handle_errors(wallet_logger)
    @log_performance(wallet_logger)
    async def get_balance(self) -> Optional[float]:
        """Get wallet balance in SOL"""
        try:
            if not self.client or not self.keypair:
                error_msg = "Wallet not initialized"
                wallet_logger.error(error_msg)
                raise WalletError(error_msg)

            response = await self.client.get_balance(
                self.keypair.public_key,
                commitment=Confirmed
            )
            
            if response.value is not None:
                balance = response.value / 1e9  # Convert lamports to SOL
                self.balance_cache = balance
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
            if not self.client or not self.keypair:
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
                    payer=self.keypair.public_key,
                    owner=self.keypair.public_key,
                    mint=token_address
                )
            )

            # Sign and send transaction
            result = await self.client.send_transaction(
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
            if not self.client or not self.keypair:
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
            response = await self.client.get_token_account_balance(
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
            if not self.client or not self.keypair:
                error_msg = "Wallet not initialized"
                wallet_logger.error(error_msg)
                raise WalletError(error_msg)

            # Create transfer instruction
            transaction = Transaction()
            transaction.add(
                transfer(
                    from_pubkey=self.keypair.public_key,
                    to_pubkey=recipient,
                    lamports=int(amount * 1e9)  # Convert SOL to lamports
                )
            )

            # Sign and send transaction
            result = await self.client.send_transaction(
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
            if not self.client or not self.keypair:
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
                    owner=self.keypair.public_key,
                    amount=int(amount * 1e9),  # Convert to token decimals
                    decimals=9
                )
            )

            # Sign and send transaction
            result = await self.client.send_transaction(
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

    @handle_errors(wallet_logger)
    @log_performance(wallet_logger)
    async def get_token_accounts(self) -> Dict[str, str]:
        """Get all token accounts for the wallet"""
        try:
            if not self.client or not self.keypair:
                error_msg = "Wallet not initialized"
                wallet_logger.error(error_msg)
                raise WalletError(error_msg)

            response = await self.client.get_token_accounts_by_owner(
                self.keypair.public_key,
                {"programId": TOKEN_PROGRAM_ID}
            )

            accounts = {}
            for account in response.value:
                mint = account.account.data.parsed["info"]["mint"]
                address = account.pubkey
                accounts[mint] = address

            self.token_accounts = accounts
            wallet_logger.info(f"Retrieved {len(accounts)} token accounts")
            return accounts

        except Exception as e:
            error_msg = f"Error getting token accounts: {str(e)}"
            wallet_logger.error(error_msg)
            raise WalletError(error_msg) 