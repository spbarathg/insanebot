"""
Security module for managing API keys, request signing, and security features.
"""
import os
import time
import hmac
import hashlib
import json
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import base64
import logging

logger = logging.getLogger(__name__)

# Optional cryptography imports for production environments
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.warning("Cryptography library not available - using basic security")

class KeyManager:
    def __init__(self, master_key: Optional[str] = None):
        """Initialize the key manager with optional master key."""
        self._master_key = master_key or os.getenv("MASTER_KEY")
        
        # Use development mode if no master key provided
        self._development_mode = self._master_key is None
        if self._development_mode:
            logger.warning("⚠️ Security: Running in development mode - using mock security")
            self._master_key = "development_master_key_not_for_production"
        
        self._key_store: Dict[str, Dict] = {}
        self._rotation_period = int(os.getenv("KEY_ROTATION_PERIOD", "86400"))  # 24 hours
        self._initialize_key_store()

    def _initialize_key_store(self):
        """Initialize the key store with encrypted keys."""
        self._key_store = {
            "rpc": {
                "current": self._encrypt_key(os.getenv("RPC_KEY", "mock_rpc_key")),
                "previous": None,
                "rotation_time": time.time()
            },
            "dex": {
                "current": self._encrypt_key(os.getenv("DEX_KEY", "mock_dex_key")),
                "previous": None,
                "rotation_time": time.time()
            }
        }

    def _encrypt_key(self, key: str) -> bytes:
        """Encrypt a key using the master key."""
        if not CRYPTOGRAPHY_AVAILABLE or self._development_mode:
            # Basic encoding for development/testing
            return base64.b64encode(key.encode())
        
        # Production encryption
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(self._master_key.encode()))
        f = Fernet(derived_key)
        return f.encrypt(key.encode())

    def _decrypt_key(self, encrypted_key: bytes) -> str:
        """Decrypt a key using the master key."""
        if not CRYPTOGRAPHY_AVAILABLE or self._development_mode:
            # Basic decoding for development/testing
            return base64.b64decode(encrypted_key).decode()
        
        # Production decryption
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(self._master_key.encode()))
        f = Fernet(derived_key)
        return f.decrypt(encrypted_key).decode()

    def rotate_key(self, key_type: str) -> None:
        """Rotate a specific key type."""
        if key_type not in self._key_store:
            raise ValueError(f"Unknown key type: {key_type}")

        current_time = time.time()
        if current_time - self._key_store[key_type]["rotation_time"] < self._rotation_period:
            return

        # Store current key as previous
        self._key_store[key_type]["previous"] = self._key_store[key_type]["current"]
        
        # Generate and store new key
        new_key = os.urandom(32).hex() if not self._development_mode else f"mock_key_{int(current_time)}"
        self._key_store[key_type]["current"] = self._encrypt_key(new_key)
        self._key_store[key_type]["rotation_time"] = current_time

        logger.info(f"Rotated {key_type} key at {datetime.now()}")

    def get_key(self, key_type: str) -> str:
        """Get the current key for a specific type."""
        if key_type not in self._key_store:
            raise ValueError(f"Unknown key type: {key_type}")
        
        return self._decrypt_key(self._key_store[key_type]["current"])

    def sign_request(self, method: str, path: str, body: Optional[Dict] = None) -> str:
        """Sign a request using HMAC-SHA256."""
        timestamp = str(int(time.time()))
        message = f"{method}{path}{timestamp}"
        if body:
            message += json.dumps(body, sort_keys=True)
        
        key = self.get_key("rpc")
        signature = hmac.new(
            key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{timestamp}:{signature}"

class IPWhitelist:
    def __init__(self):
        """Initialize IP whitelist manager."""
        whitelist_env = os.getenv("IP_WHITELIST", "127.0.0.1,localhost")
        self._whitelist = set(whitelist_env.split(","))
        self._whitelist.discard("")  # Remove empty strings
        
        if not self._whitelist:
            logger.warning("⚠️ Security: No IP whitelist configured - allowing localhost only")
            self._whitelist = {"127.0.0.1", "localhost"}

    def is_allowed(self, ip: str) -> bool:
        """Check if an IP is whitelisted."""
        return ip in self._whitelist

    def add_ip(self, ip: str) -> None:
        """Add an IP to the whitelist."""
        self._whitelist.add(ip)

    def remove_ip(self, ip: str) -> None:
        """Remove an IP from the whitelist."""
        self._whitelist.discard(ip)

# Initialize global instances with safe defaults
try:
    key_manager = KeyManager()
    ip_whitelist = IPWhitelist()
    logger.info("✅ Security module initialized successfully")
except Exception as e:
    logger.error(f"❌ Security module initialization failed: {e}")
    # Create mock instances for development
    key_manager = None
    ip_whitelist = None 