"""
Secure wallet vault for managing wallet keys and transactions.
"""
import os
import json
import base64
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import ed25519
from solana.keypair import Keypair
import logging

logger = logging.getLogger(__name__)

class WalletVault:
    def __init__(self, master_key: Optional[str] = None):
        """Initialize the wallet vault with optional master key."""
        self._master_key = master_key or os.getenv("MASTER_KEY")
        if not self._master_key:
            raise ValueError("Master key must be provided or set in environment")
        
        self._vault_path = os.getenv("VAULT_PATH", "data/wallet_vault.json")
        self._encryption_key = self._derive_encryption_key()
        self._wallets: Dict[str, Dict] = {}
        self._load_vault()

    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from master key."""
        salt = b'wallet_vault_salt'  # In production, use a secure random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(self._master_key.encode()))

    def _load_vault(self) -> None:
        """Load encrypted wallet data from vault."""
        try:
            if os.path.exists(self._vault_path):
                with open(self._vault_path, 'rb') as f:
                    encrypted_data = f.read()
                f = Fernet(self._encryption_key)
                decrypted_data = f.decrypt(encrypted_data)
                self._wallets = json.loads(decrypted_data)
        except Exception as e:
            logger.error(f"Failed to load wallet vault: {str(e)}")
            self._wallets = {}

    def _save_vault(self) -> None:
        """Save encrypted wallet data to vault."""
        try:
            os.makedirs(os.path.dirname(self._vault_path), exist_ok=True)
            f = Fernet(self._encryption_key)
            encrypted_data = f.encrypt(json.dumps(self._wallets).encode())
            with open(self._vault_path, 'wb') as f:
                f.write(encrypted_data)
        except Exception as e:
            logger.error(f"Failed to save wallet vault: {str(e)}")
            raise

    def add_wallet(self, wallet_id: str, keypair: Keypair, metadata: Optional[Dict] = None) -> None:
        """Add a new wallet to the vault."""
        if wallet_id in self._wallets:
            raise ValueError(f"Wallet {wallet_id} already exists")

        # Encrypt private key
        f = Fernet(self._encryption_key)
        encrypted_private_key = f.encrypt(keypair.secret_key)

        self._wallets[wallet_id] = {
            "public_key": str(keypair.public_key),
            "private_key": base64.b64encode(encrypted_private_key).decode(),
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self._save_vault()
        logger.info(f"Added wallet {wallet_id} to vault")

    def get_wallet(self, wallet_id: str) -> Keypair:
        """Retrieve a wallet from the vault."""
        if wallet_id not in self._wallets:
            raise ValueError(f"Wallet {wallet_id} not found")

        try:
            # Decrypt private key
            f = Fernet(self._encryption_key)
            encrypted_private_key = base64.b64decode(self._wallets[wallet_id]["private_key"])
            private_key = f.decrypt(encrypted_private_key)

            # Update last used timestamp
            self._wallets[wallet_id]["last_used"] = datetime.now().isoformat()
            self._save_vault()

            return Keypair.from_secret_key(private_key)
        except Exception as e:
            logger.error(f"Failed to retrieve wallet {wallet_id}: {str(e)}")
            raise

    def remove_wallet(self, wallet_id: str) -> None:
        """Remove a wallet from the vault."""
        if wallet_id not in self._wallets:
            raise ValueError(f"Wallet {wallet_id} not found")

        del self._wallets[wallet_id]
        self._save_vault()
        logger.info(f"Removed wallet {wallet_id} from vault")

    def list_wallets(self) -> List[Dict]:
        """List all wallets in the vault (without private keys)."""
        return [
            {
                "id": wallet_id,
                "public_key": data["public_key"],
                "created_at": data["created_at"],
                "last_used": data["last_used"],
                "metadata": data["metadata"]
            }
            for wallet_id, data in self._wallets.items()
        ]

    def update_wallet_metadata(self, wallet_id: str, metadata: Dict) -> None:
        """Update wallet metadata."""
        if wallet_id not in self._wallets:
            raise ValueError(f"Wallet {wallet_id} not found")

        self._wallets[wallet_id]["metadata"].update(metadata)
        self._save_vault()
        logger.info(f"Updated metadata for wallet {wallet_id}")

    def rotate_wallet_key(self, wallet_id: str) -> None:
        """Rotate a wallet's private key."""
        if wallet_id not in self._wallets:
            raise ValueError(f"Wallet {wallet_id} not found")

        # Generate new keypair
        new_keypair = Keypair()
        
        # Update wallet with new key
        f = Fernet(self._encryption_key)
        encrypted_private_key = f.encrypt(new_keypair.secret_key)
        
        self._wallets[wallet_id].update({
            "public_key": str(new_keypair.public_key),
            "private_key": base64.b64encode(encrypted_private_key).decode(),
            "last_rotated": datetime.now().isoformat()
        })
        
        self._save_vault()
        logger.info(f"Rotated key for wallet {wallet_id}")

# Initialize global instance
wallet_vault = WalletVault() 