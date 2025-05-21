"""
Secure wallet management for Solana trading bot.
"""
import os
import json
import base64
import time
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from solders.keypair import Keypair
from solders.transaction import Transaction
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from loguru import logger
from dotenv import load_dotenv
import asyncio
import logging
import base58

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Changed to INFO to reduce logging overhead

async def await_confirmation(rpc_client, signature, timeout=15, poll_interval=0.5):  # Reduced timeout and poll interval
    """Poll Solana RPC for transaction confirmation."""
    start = time.time()
    while time.time() - start < timeout:
        resp = await rpc_client.get_signature_statuses([signature])
        status = resp['result']['value'][0]
        if status and status.get('confirmationStatus') in ('confirmed', 'finalized'):
            return True
        await asyncio.sleep(poll_interval)
    return False

class WalletManager:
    """
    Secure wallet management with encrypted key storage.
    
    Attributes:
        rpc_client: Solana RPC client
        keypair: Encrypted wallet keypair
        _fernet: Fernet instance for encryption
    """
    
    def __init__(self, rpc_client: AsyncClient):
        self.rpc_client = rpc_client
        self.keypair: Optional[Keypair] = None
        self._fernet = None
        self._balance_cache = None
        self._last_balance_update = 0
        self._balance_cache_ttl = 30  # Cache balance for 30 seconds
        self._load_wallet()
        
    def _initialize_encryption(self) -> None:
        """Initialize encryption with environment-based key derivation."""
        try:
            # Get encryption key from environment
            salt = os.getenv("WALLET_SALT", "").encode()
            password = os.getenv("WALLET_PASSWORD", "").encode()
            
            if not salt or not password:
                raise ValueError("Missing WALLET_SALT or WALLET_PASSWORD environment variables")
            
            # Derive encryption key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self._fernet = Fernet(key)
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise
    
    def _load_wallet(self) -> None:
        """Load wallet from private key."""
        try:
            # Get private key from environment
            private_key = os.getenv("WALLET_PRIVATE_KEY") or os.getenv("SOLANA_PRIVATE_KEY")
            
            if not private_key:
                if os.getenv("SIMULATION_MODE", "False").lower() == "true":
                    # Use a random keypair for simulation
                    self.keypair = Keypair()
                    return
                raise ValueError("No wallet private key found and not in simulation mode")
            
            # Try base58 first, then hex
            try:
                # For base58 encoded keys
                private_key_bytes = base58.b58decode(private_key)
            except Exception:
                try:
                    # For hex encoded keys, ensure we have valid hex
                    if private_key.startswith('0x'):
                        private_key = private_key[2:]
                    private_key_bytes = bytes.fromhex(private_key)
                except Exception as e:
                    logger.error(f"Invalid private key format: {str(e)}")
                    # Use a random keypair as fallback in any case
                    if os.getenv("SIMULATION_MODE", "False").lower() == "true":
                        self.keypair = Keypair()
                        return
                    raise ValueError(f"Invalid private key format. Must be base58 or hex: {str(e)}")
            
            # Create keypair from secret key bytes
            # Note: in newer solders versions, use create_from_bytes instead
            try:
                self.keypair = Keypair.from_bytes(private_key_bytes)
            except AttributeError:
                # Try alternative method if from_secret_key doesn't exist
                try:
                    self.keypair = Keypair.from_bytes(private_key_bytes)
                except Exception:
                    # Last resort, create a random keypair for simulation
                    if os.getenv("SIMULATION_MODE", "False").lower() == "true":
                        self.keypair = Keypair()
                    else:
                        raise ValueError("Could not create keypair from provided private key")
            
            logger.info(f"Loaded wallet: {self.keypair.pubkey()}")
            
        except Exception as e:
            if os.getenv("SIMULATION_MODE", "False").lower() == "true":
                self.keypair = Keypair()
            else:
                raise ValueError(f"Failed to load wallet: {str(e)}")
    
    def load_wallet(self, encrypted_key_path: str) -> None:
        """
        Load encrypted wallet from file.
        
        Args:
            encrypted_key_path: Path to encrypted wallet file
            
        Raises:
            ValueError: If wallet file is invalid or decryption fails
        """
        try:
            with open(encrypted_key_path, "rb") as f:
                encrypted_data = f.read()
            
            # Decrypt wallet data
            decrypted_data = self._fernet.decrypt(encrypted_data)
            wallet_data = json.loads(decrypted_data)
            
            # Create keypair from decrypted data
            self.keypair = Keypair.from_secret_key(bytes(wallet_data["secret_key"]))
            logger.info("Wallet loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load wallet: {e}")
            raise ValueError("Invalid wallet file or decryption failed")
    
    def save_wallet(self, keypair: Keypair, encrypted_key_path: str) -> None:
        """
        Save wallet with encryption.
        
        Args:
            keypair: Keypair to save
            encrypted_key_path: Path to save encrypted wallet
            
        Raises:
            ValueError: If encryption fails
        """
        try:
            # Prepare wallet data
            wallet_data = {
                "public_key": str(keypair.public_key),
                "secret_key": list(keypair.secret_key)
            }
            
            # Encrypt wallet data
            encrypted_data = self._fernet.encrypt(json.dumps(wallet_data).encode())
            
            # Save encrypted data
            with open(encrypted_key_path, "wb") as f:
                f.write(encrypted_data)
            
            logger.info("Wallet saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save wallet: {e}")
            raise ValueError("Failed to encrypt wallet")
    
    def get_keypair(self) -> Keypair:
        """
        Get the current keypair.
        
        Returns:
            Keypair: The current wallet keypair
            
        Raises:
            ValueError: If wallet is not loaded
        """
        if not self.keypair:
            raise ValueError("Wallet not loaded")
        return self.keypair
    
    async def get_balance(self) -> Optional[float]:
        """Get wallet balance in SOL with caching."""
        if not self.keypair:
            raise ValueError("Wallet not loaded")
            
        current_time = time.time()
        if (self._balance_cache is not None and 
            current_time - self._last_balance_update < self._balance_cache_ttl):
            return self._balance_cache
            
        try:
            pubkey = self.keypair.pubkey()
            response = await self.rpc_client.get_balance(pubkey)
            if response and "result" in response and "value" in response["result"]:
                self._balance_cache = response["result"]["value"] / 1e9
                self._last_balance_update = current_time
                return self._balance_cache
            return None
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None
    
    async def get_token_balance(self, token_address: str) -> Optional[float]:
        """Get token balance for a specific token."""
        try:
            if not self.keypair:
                raise ValueError("Wallet not loaded")
                
            if not token_address:
                raise ValueError("Invalid token address")
                
            # Ensure owner is a Pubkey object
            owner_pubkey = self.keypair.pubkey()
            
            # Ensure token_address is a Pubkey object
            try:
                mint_pubkey = Pubkey.from_string(token_address)
            except Exception as e:
                logger.error(f"Could not convert token address to Pubkey: {str(e)}")
                return 0.0
                
            # Get token account
            token_account = await self.rpc_client.get_token_accounts_by_owner(
                owner_pubkey,
                {"mint": mint_pubkey}
            )
            
            if not token_account or not token_account.get("result", {}).get("value"):
                return 0.0
                
            # Get balance
            balance = await self.rpc_client.get_token_account_balance(
                token_account["result"]["value"][0]["pubkey"]
            )
            
            if not balance or not balance.get("result", {}).get("value"):
                return 0.0
                
            return float(balance["result"]["value"]["uiAmount"])
            
        except ValueError as e:
            # Re-raise ValueError for validation
            raise
        except Exception as e:
            logger.error(f"Error getting token balance: {str(e)}")
            return None
            
    def get_public_key(self) -> Pubkey:
        """Get wallet public key."""
        if not self.keypair:
            raise ValueError("Wallet not loaded")
        return self.keypair.pubkey()
        
    async def send_transaction(self, transaction: Transaction) -> bool:
        """Send a transaction with optimized retry logic."""
        if not self.keypair:
            raise ValueError("Wallet not loaded")
            
        max_retries = 2  # Reduced retries
        retry_delay = 0.5  # Reduced delay
        
        for attempt in range(max_retries):
            try:
                transaction.sign([self.keypair])
                response = await self.rpc_client.send_transaction(transaction)
                if "result" in response:
                    signature = response["result"]
                    confirmed = await await_confirmation(self.rpc_client, signature)
                    if confirmed:
                        return True
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                continue
        return False

    async def initialize(self) -> bool:
        """Initialize the wallet manager."""
        try:
            # Initialize wallet here
            return True
        except Exception as e:
            logger.error(f"Failed to initialize wallet: {str(e)}")
            return False

    async def close(self) -> None:
        """Close the wallet manager."""
        try:
            # Cleanup here
            pass
        except Exception as e:
            logger.error(f"Error closing wallet: {str(e)}") 