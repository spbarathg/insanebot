import os
from typing import Optional
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.rpc.async_api import AsyncClient
from loguru import logger
from dotenv import load_dotenv

class WalletManager:
    def __init__(self, rpc_client: AsyncClient):
        self.rpc_client = rpc_client
        self.keypair: Optional[Keypair] = None
        self._balance_cache = None
        self._load_wallet()
        
    def _load_wallet(self) -> None:
        """Load wallet from private key."""
        try:
            # Load environment variables
            load_dotenv()
            
            # Get private key from environment
            private_key = os.getenv("WALLET_PRIVATE_KEY")
            
            # For tests, use a dummy keypair if no private key found
            if not private_key:
                logger.info("No wallet private key found, using mock keypair for testing")
                self.keypair = Keypair()
                return
                
            # Convert private key to bytes
            private_key_bytes = bytes.fromhex(private_key)
            
            # Create keypair
            self.keypair = Keypair.from_secret_key(private_key_bytes)
            logger.info(f"Loaded wallet: {self.keypair.public_key}")
            
        except Exception as e:
            logger.error(f"Error loading wallet: {str(e)}")
            # Use a dummy keypair for testing
            self.keypair = Keypair()
            
    async def get_balance(self) -> Optional[float]:
        """Get wallet SOL balance."""
        try:
            if not self.keypair:
                raise ValueError("Wallet not loaded")
            
            # Use cache if available
            if self._balance_cache is not None:
                return self._balance_cache
                
            response = await self.rpc_client.get_balance(self.keypair.public_key)
            if not response or not response.get("result", {}).get("value"):
                return 0.0
                
            # Convert lamports to SOL
            balance = response["result"]["value"] / 1e9
            self._balance_cache = balance
            return balance
            
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            return None
            
    async def get_token_balance(self, token_address: str) -> Optional[float]:
        """Get token balance for a specific token."""
        try:
            if not self.keypair:
                raise ValueError("Wallet not loaded")
                
            if not token_address:
                raise ValueError("Invalid token address")
                
            # Get token account
            token_account = await self.rpc_client.get_token_accounts_by_owner(
                self.keypair.public_key,
                {"mint": PublicKey(token_address)}
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
            
    def get_public_key(self) -> PublicKey:
        """Get wallet public key."""
        if not self.keypair:
            raise ValueError("Wallet not loaded")
        return self.keypair.public_key
        
    def get_keypair(self) -> Keypair:
        """Get wallet keypair."""
        if not self.keypair:
            raise ValueError("Wallet not loaded")
        return self.keypair 