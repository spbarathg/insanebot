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
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)

class KeyManager:
    def __init__(self, master_key: Optional[str] = None):
        """Initialize the key manager with optional master key."""
        self._master_key = master_key or os.getenv("MASTER_KEY")
        if not self._master_key:
            raise ValueError("Master key must be provided or set in environment")
        
        self._key_store: Dict[str, Dict] = {}
        self._rotation_period = int(os.getenv("KEY_ROTATION_PERIOD", "86400"))  # 24 hours
        self._initialize_key_store()

    def _initialize_key_store(self):
        """Initialize the key store with encrypted keys."""
        self._key_store = {
            "rpc": {
                "current": self._encrypt_key(os.getenv("RPC_KEY")),
                "previous": None,
                "rotation_time": time.time()
            },
            "dex": {
                "current": self._encrypt_key(os.getenv("DEX_KEY")),
                "previous": None,
                "rotation_time": time.time()
            }
        }

    def _encrypt_key(self, key: str) -> bytes:
        """Encrypt a key using the master key."""
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self._master_key.encode()))
        f = Fernet(key)
        return f.encrypt(key.encode())

    def _decrypt_key(self, encrypted_key: bytes) -> str:
        """Decrypt a key using the master key."""
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self._master_key.encode()))
        f = Fernet(key)
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
        new_key = os.urandom(32).hex()
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
        self._whitelist = set(os.getenv("IP_WHITELIST", "").split(","))
        self._whitelist.discard("")  # Remove empty strings

    def is_allowed(self, ip: str) -> bool:
        """Check if an IP is whitelisted."""
        return ip in self._whitelist

    def add_ip(self, ip: str) -> None:
        """Add an IP to the whitelist."""
        self._whitelist.add(ip)

    def remove_ip(self, ip: str) -> None:
        """Remove an IP from the whitelist."""
        self._whitelist.discard(ip)

# Initialize global instances
key_manager = KeyManager()
ip_whitelist = IPWhitelist() 