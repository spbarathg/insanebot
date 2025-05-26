"""
Wallet manager for Solana trading bot with real blockchain integration.
"""
import logging
import os
import json
import time
import asyncio
import base58
from typing import Dict, List, Optional, Any, Tuple
from solana.rpc.async_api import AsyncClient
# Updated import for newer solana library version
from solders.commitment_config import CommitmentLevel
from solana.rpc.types import TxOpts
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction
from solders.system_program import transfer, TransferParams
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
from loguru import logger
# from src.core.validation import AddressValidator  # Removed to avoid circular import

class WalletSecurityError(Exception):
    """Raised when wallet security is compromised"""
    pass

class InsufficientFundsError(Exception):
    """Raised when wallet has insufficient funds"""
    pass

class WalletManager:
    """
    Advanced wallet manager with real Solana blockchain integration.
    Supports secure key management, balance checking, and transaction execution.
    """
    
    def __init__(self):
        """Initialize wallet manager with security and validation."""
        self.simulation_mode = os.getenv("SIMULATION_MODE", "true").lower() == "true"
        self.rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
        self.client: Optional[AsyncClient] = None
        self.keypair: Optional[Keypair] = None
        self.public_key: Optional[Pubkey] = None
        
        # Security settings
        self.master_password = os.getenv("WALLET_PASSWORD", "")
        self.encryption_salt = os.getenv("WALLET_SALT", "")
        
        # Transaction settings
        self.max_retries = 3
        self.retry_delay = 1.0
        self.transaction_timeout = 30.0
        
        # Simulation settings (fallback)
        self.simulation_balance = float(os.getenv("SIMULATION_CAPITAL", "0.1"))
        
        logger.info(f"Wallet manager initialized in {'simulation' if self.simulation_mode else 'live'} mode")
        
    async def initialize(self):
        """Initialize the wallet manager"""
        try:
            if self.simulation_mode:
                # In simulation mode, don't create real client to avoid compatibility issues
                logger.info("Wallet manager running in simulation mode - skipping real client initialization")
                return True
                
            # Only create real client in production mode
            from solana.rpc.async_api import AsyncClient
            self.client = AsyncClient("https://api.mainnet-beta.solana.com")
            logger.info("Wallet manager initialized with real Solana client")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize wallet: {str(e)}")
            # Still return True in simulation mode even if real client fails
            if self.simulation_mode:
                logger.warning("Continuing in simulation mode despite client initialization failure")
                return True
            return False
    
    def _validate_credentials(self) -> bool:
        """Validate that all required credentials are present and valid."""
        required_env_vars = [
            "SOLANA_PRIVATE_KEY",
            "WALLET_PASSWORD", 
            "WALLET_SALT"
        ]
        
        missing_vars = []
        for var in required_env_vars:
            value = os.getenv(var)
            if not value or value in ["", "demo_key_for_testing", "0000000000000000000000000000000000000000000000000000000000000000"]:
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"Missing or invalid environment variables: {missing_vars}")
            return False
        
        return True
    
    async def _initialize_real_wallet(self) -> None:
        """Initialize real Solana wallet with secure key management."""
        try:
            # Get private key from environment
            private_key_str = os.getenv("SOLANA_PRIVATE_KEY")
            if not private_key_str:
                raise WalletSecurityError("SOLANA_PRIVATE_KEY environment variable not set")
            
            # Decrypt private key if it's encrypted
            if self.master_password and self.encryption_salt:
                private_key_str = self._decrypt_private_key(private_key_str)
            
            # Parse private key
            try:
                if private_key_str.startswith('[') and private_key_str.endswith(']'):
                    # Array format [1,2,3,...]
                    key_bytes = bytes(json.loads(private_key_str))
                elif len(private_key_str) == 88:  # Base58 format
                    key_bytes = base58.b58decode(private_key_str)
                elif private_key_str.startswith('0x'):
                    # Hex format
                    key_bytes = bytes.fromhex(private_key_str[2:])
                else:
                    # Assume hex without prefix
                    key_bytes = bytes.fromhex(private_key_str)
                
                self.keypair = Keypair.from_secret_key_bytes(key_bytes)
                self.public_key = self.keypair.pubkey
                
            except Exception as e:
                raise WalletSecurityError(f"Invalid private key format: {str(e)}")
            
            logger.info(f"Wallet loaded successfully. Public key: {str(self.public_key)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize real wallet: {str(e)}")
            raise
    
    def _decrypt_private_key(self, encrypted_key: str) -> str:
        """Decrypt private key using master password."""
        try:
            salt = self.encryption_salt.encode()
            password = self.master_password.encode()
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            f = Fernet(key)
            
            return f.decrypt(encrypted_key.encode()).decode()
            
        except Exception as e:
            raise WalletSecurityError(f"Failed to decrypt private key: {str(e)}")
    
    async def check_balance(self) -> float:
        """Check current wallet balance."""
        try:
            if self.simulation_mode:
                logger.debug(f"Simulation balance: {self.simulation_balance} SOL")
                return self.simulation_balance
            
            if not self.client or not self.public_key:
                raise WalletSecurityError("Wallet not properly initialized")
            
            response = await self.client.get_balance(
                self.public_key,
                commitment=CommitmentLevel.confirmed()
            )
            
            if response.value is None:
                raise Exception("Failed to get balance from RPC")
            
            # Convert lamports to SOL
            balance_sol = response.value / 1_000_000_000
            logger.debug(f"Current balance: {balance_sol} SOL")
            
            return balance_sol
            
        except Exception as e:
            logger.error(f"Failed to check balance: {str(e)}")
            raise
    
    async def send_transaction(self, transaction: Transaction, max_retries: Optional[int] = None) -> str:
        """Send a transaction with retry logic and proper error handling."""
        if self.simulation_mode:
            # Simulate transaction
            tx_id = f"SIM_{int(time.time() * 1000000)}"
            logger.info(f"Simulated transaction: {tx_id}")
            await asyncio.sleep(0.1)  # Simulate network delay
            return tx_id
        
        if not self.client or not self.keypair:
            raise WalletSecurityError("Wallet not properly initialized")
        
        max_retries = max_retries or self.max_retries
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Get recent blockhash
                recent_blockhash = await self.client.get_recent_blockhash()
                transaction.recent_blockhash = recent_blockhash.value.blockhash
                
                # Sign transaction
                transaction.sign(self.keypair)
                
                # Send transaction
                response = await self.client.send_transaction(
                    transaction,
                    opts=TxOpts(skip_preflight=False, preflight_commitment=CommitmentLevel.confirmed())
                )
                
                if response.value:
                    logger.info(f"Transaction sent successfully: {response.value}")
                    return response.value
                else:
                    raise Exception("Transaction failed to send")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Transaction attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        raise Exception(f"Transaction failed after {max_retries} attempts. Last error: {str(last_error)}")
    
    async def transfer_sol(self, to_address: str, amount_sol: float) -> str:
        """Transfer SOL to another address."""
        try:
            if amount_sol <= 0:
                raise ValueError("Transfer amount must be positive")
            
            # Check sufficient balance
            current_balance = await self.check_balance()
            if current_balance < amount_sol:
                raise InsufficientFundsError(f"Insufficient balance. Have: {current_balance} SOL, Need: {amount_sol} SOL")
            
            if self.simulation_mode:
                # Update simulation balance
                self.simulation_balance -= amount_sol
                tx_id = f"SIM_TRANSFER_{int(time.time() * 1000000)}"
                logger.info(f"Simulated transfer: {amount_sol} SOL to {to_address}, tx: {tx_id}")
                return tx_id
            
            # Create transfer instruction
            to_pubkey = Pubkey(to_address)
            amount_lamports = int(amount_sol * 1_000_000_000)
            
            transfer_instruction = transfer(
                TransferParams(
                    from_pubkey=self.public_key,
                    to_pubkey=to_pubkey,
                    lamports=amount_lamports
                )
            )
            
            # Create and send transaction
            transaction = Transaction()
            transaction.add(transfer_instruction)
            
            tx_id = await self.send_transaction(transaction)
            logger.info(f"Transferred {amount_sol} SOL to {to_address}, tx: {tx_id}")
            
            return tx_id
            
        except Exception as e:
            logger.error(f"Failed to transfer SOL: {str(e)}")
            raise
    
    def get_keypair(self) -> Keypair:
        """Get wallet keypair (only for simulation or authorized operations)."""
        if self.simulation_mode:
            return "SIMULATION_KEYPAIR"
        
        if not self.keypair:
            raise WalletSecurityError("Keypair not available - wallet not initialized")
        
        return self.keypair
    
    def get_public_key(self) -> str:
        """Get wallet public key."""
        if self.simulation_mode:
            return "SIMULATION_PUBLIC_KEY"
        
        if not self.public_key:
            raise WalletSecurityError("Public key not available - wallet not initialized")
        
        return str(self.public_key)
    
    async def validate_transaction_params(self, amount: float, token_address: str = None) -> bool:
        """Validate transaction parameters before execution."""
        try:
            # Basic amount validation
            if amount <= 0:
                raise ValueError("Amount must be positive")
            
            if amount > 1000:  # Safety limit
                raise ValueError("Amount exceeds safety limit of 1000 SOL")
            
            # Check balance
            balance = await self.check_balance()
            if amount > balance * 0.95:  # Leave 5% buffer
                raise InsufficientFundsError(f"Amount too large. Balance: {balance} SOL, Requested: {amount} SOL")
            
            # Additional validation for token addresses
            if token_address:
                # Simple validation - check if it's a valid base58 string of correct length
                try:
                    if not isinstance(token_address, str) or not (32 <= len(token_address) <= 44):
                        raise ValueError(f"Invalid token address format: {token_address}")
                    
                    # Try to decode as base58 to validate format
                    decoded = base58.b58decode(token_address)
                    if len(decoded) != 32:
                        raise ValueError(f"Invalid token address length: {token_address}")
                        
                except Exception:
                    raise ValueError(f"Invalid token address: {token_address}")
            
            return True
            
        except Exception as e:
            logger.error(f"Transaction validation failed: {str(e)}")
            raise
    
    async def get_transaction_status(self, tx_id: str) -> Dict[str, Any]:
        """Get the status of a transaction."""
        if self.simulation_mode:
            return {
                "status": "confirmed",
                "slot": 12345,
                "confirmations": 32,
                "block_time": int(time.time())
            }
        
        if not self.client:
            raise WalletSecurityError("Client not initialized")
        
        try:
            response = await self.client.get_transaction(tx_id)
            if response.value:
                return {
                    "status": "confirmed" if response.value.meta.err is None else "failed",
                    "slot": response.value.slot,
                    "block_time": response.value.block_time,
                    "fee": response.value.meta.fee
                }
            else:
                return {"status": "not_found"}
                
        except Exception as e:
            logger.error(f"Failed to get transaction status: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def close(self) -> None:
        """Close wallet connections and cleanup."""
        try:
            if self.client:
                await self.client.close()
            
            # Clear sensitive data
            self.keypair = None
            self.master_password = ""
            
            logger.info("Wallet manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing wallet manager: {str(e)}") 