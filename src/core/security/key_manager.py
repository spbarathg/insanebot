"""
Key Manager - Cryptographic Key Management System

Handles secure key generation, storage, rotation, and lifecycle management
for all cryptographic operations in the trading system.
"""

import os
import secrets
import hashlib
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import time
import json

logger = logging.getLogger(__name__)

@dataclass
class KeyInfo:
    """Information about a managed key"""
    key_id: str
    key_type: str  # 'symmetric', 'asymmetric_private', 'asymmetric_public'
    algorithm: str  # 'AES-256', 'RSA-2048', etc.
    created_at: float
    expires_at: Optional[float] = None
    is_active: bool = True
    usage_count: int = 0
    max_usage: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class KeyManager:
    """
    Secure cryptographic key management system
    
    Features:
    - Automatic key generation and rotation
    - Secure key storage with encryption at rest
    - Key lifecycle management
    - Usage tracking and limits
    - Emergency key revocation
    """
    
    def __init__(self, master_password: Optional[str] = None):
        self.master_password = master_password or os.getenv('MASTER_KEY_PASSWORD', '')
        
        # Key storage (encrypted in memory)
        self._keys: Dict[str, bytes] = {}
        self._key_info: Dict[str, KeyInfo] = {}
        
        # Master encryption key for key storage
        self._master_key = self._derive_master_key()
        self._fernet = Fernet(base64.urlsafe_b64encode(self._master_key[:32]))
        
        # Key rotation settings
        self.default_key_lifetime = 86400 * 7  # 7 days
        self.auto_rotation_enabled = True
        
        # Usage statistics
        self.total_keys_generated = 0
        self.total_operations = 0
        self.failed_operations = 0
        
        logger.info("🔐 KeyManager initialized - Secure key management active")
    
    def _derive_master_key(self) -> bytes:
        """Derive master key from password"""
        if not self.master_password:
            # Generate a secure random master key if no password provided
            self.master_password = secrets.token_urlsafe(32)
            logger.warning("🔐 No master password provided, generated random key")
        
        # Use PBKDF2 to derive key from password
        salt = b'ant_bot_salt_2024'  # In production, use random salt per installation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(self.master_password.encode())
    
    def generate_symmetric_key(self, key_id: str, algorithm: str = "AES-256") -> str:
        """Generate a new symmetric encryption key"""
        try:
            if key_id in self._keys:
                raise ValueError(f"Key {key_id} already exists")
            
            # Generate random key
            if algorithm == "AES-256":
                key = Fernet.generate_key()
            else:
                # Fallback to 32-byte random key
                key = secrets.token_bytes(32)
            
            # Store encrypted key
            encrypted_key = self._fernet.encrypt(key)
            self._keys[key_id] = encrypted_key
            
            # Store key metadata
            self._key_info[key_id] = KeyInfo(
                key_id=key_id,
                key_type="symmetric",
                algorithm=algorithm,
                created_at=time.time(),
                expires_at=time.time() + self.default_key_lifetime if self.auto_rotation_enabled else None
            )
            
            self.total_keys_generated += 1
            logger.info(f"🔐 Generated symmetric key: {key_id} ({algorithm})")
            return key_id
            
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"Failed to generate symmetric key {key_id}: {e}")
            raise
    
    def generate_asymmetric_keypair(self, key_id: str, algorithm: str = "RSA-2048") -> str:
        """Generate a new asymmetric key pair"""
        try:
            if key_id in self._keys:
                raise ValueError(f"Key {key_id} already exists")
            
            if algorithm == "RSA-2048":
                # Generate RSA key pair
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                )
                
                # Serialize private key
                private_pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                
                # Serialize public key
                public_key = private_key.public_key()
                public_pem = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                
                # Store encrypted keys
                private_key_id = f"{key_id}_private"
                public_key_id = f"{key_id}_public"
                
                self._keys[private_key_id] = self._fernet.encrypt(private_pem)
                self._keys[public_key_id] = self._fernet.encrypt(public_pem)
                
                # Store metadata
                current_time = time.time()
                expires_at = current_time + self.default_key_lifetime if self.auto_rotation_enabled else None
                
                self._key_info[private_key_id] = KeyInfo(
                    key_id=private_key_id,
                    key_type="asymmetric_private",
                    algorithm=algorithm,
                    created_at=current_time,
                    expires_at=expires_at
                )
                
                self._key_info[public_key_id] = KeyInfo(
                    key_id=public_key_id,
                    key_type="asymmetric_public",
                    algorithm=algorithm,
                    created_at=current_time,
                    expires_at=expires_at
                )
                
                self.total_keys_generated += 2
                logger.info(f"🔐 Generated asymmetric keypair: {key_id} ({algorithm})")
                return key_id
            
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"Failed to generate asymmetric keypair {key_id}: {e}")
            raise
    
    def get_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve and decrypt a key"""
        try:
            if key_id not in self._keys:
                return None
            
            # Check if key is active and not expired
            if key_id in self._key_info:
                key_info = self._key_info[key_id]
                
                if not key_info.is_active:
                    logger.warning(f"Attempted to use inactive key: {key_id}")
                    return None
                
                if key_info.expires_at and time.time() > key_info.expires_at:
                    logger.warning(f"Attempted to use expired key: {key_id}")
                    return None
                
                # Update usage count
                key_info.usage_count += 1
                
                # Check usage limits
                if key_info.max_usage and key_info.usage_count > key_info.max_usage:
                    logger.warning(f"Key {key_id} exceeded usage limit")
                    return None
            
            # Decrypt and return key
            encrypted_key = self._keys[key_id]
            decrypted_key = self._fernet.decrypt(encrypted_key)
            
            self.total_operations += 1
            return decrypted_key
            
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"Failed to retrieve key {key_id}: {e}")
            return None
    
    def rotate_key(self, old_key_id: str, new_key_id: Optional[str] = None) -> str:
        """Rotate a key to a new version"""
        try:
            if old_key_id not in self._key_info:
                raise ValueError(f"Key {old_key_id} not found")
            
            old_key_info = self._key_info[old_key_id]
            new_key_id = new_key_id or f"{old_key_id}_rotated_{int(time.time())}"
            
            # Generate new key of same type
            if old_key_info.key_type == "symmetric":
                self.generate_symmetric_key(new_key_id, old_key_info.algorithm)
            elif old_key_info.key_type in ["asymmetric_private", "asymmetric_public"]:
                # For asymmetric keys, rotate the pair
                base_key_id = old_key_id.replace("_private", "").replace("_public", "")
                new_base_id = new_key_id.replace("_private", "").replace("_public", "")
                self.generate_asymmetric_keypair(new_base_id, old_key_info.algorithm)
                new_key_id = f"{new_base_id}_{old_key_info.key_type.split('_')[1]}"
            
            # Mark old key as inactive
            old_key_info.is_active = False
            
            logger.info(f"🔄 Rotated key: {old_key_id} -> {new_key_id}")
            return new_key_id
            
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"Failed to rotate key {old_key_id}: {e}")
            raise
    
    def revoke_key(self, key_id: str, reason: str = "manual_revocation"):
        """Revoke a key immediately"""
        try:
            if key_id in self._key_info:
                self._key_info[key_id].is_active = False
                self._key_info[key_id].metadata['revocation_reason'] = reason
                self._key_info[key_id].metadata['revoked_at'] = time.time()
                
                logger.warning(f"🚫 Revoked key: {key_id} - Reason: {reason}")
            
        except Exception as e:
            logger.error(f"Failed to revoke key {key_id}: {e}")
    
    def list_keys(self, include_inactive: bool = False) -> List[KeyInfo]:
        """List all managed keys"""
        keys = []
        for key_info in self._key_info.values():
            if include_inactive or key_info.is_active:
                keys.append(key_info)
        return keys
    
    def get_key_info(self, key_id: str) -> Optional[KeyInfo]:
        """Get information about a specific key"""
        return self._key_info.get(key_id)
    
    def cleanup_expired_keys(self) -> int:
        """Remove expired keys from memory"""
        current_time = time.time()
        removed_count = 0
        
        for key_id, key_info in list(self._key_info.items()):
            if (key_info.expires_at and current_time > key_info.expires_at and 
                not key_info.is_active):
                
                # Remove from storage
                if key_id in self._keys:
                    del self._keys[key_id]
                del self._key_info[key_id]
                removed_count += 1
                
                logger.debug(f"🗑️ Cleaned up expired key: {key_id}")
        
        if removed_count > 0:
            logger.info(f"🗑️ Cleaned up {removed_count} expired keys")
        
        return removed_count
    
    def export_key_metadata(self) -> Dict[str, Any]:
        """Export key metadata for auditing (no actual keys)"""
        metadata = {
            "total_keys": len(self._key_info),
            "active_keys": sum(1 for k in self._key_info.values() if k.is_active),
            "expired_keys": sum(1 for k in self._key_info.values() 
                              if k.expires_at and time.time() > k.expires_at),
            "total_operations": self.total_operations,
            "failed_operations": self.failed_operations,
            "keys": []
        }
        
        for key_info in self._key_info.values():
            metadata["keys"].append({
                "key_id": key_info.key_id,
                "key_type": key_info.key_type,
                "algorithm": key_info.algorithm,
                "created_at": key_info.created_at,
                "expires_at": key_info.expires_at,
                "is_active": key_info.is_active,
                "usage_count": key_info.usage_count
            })
        
        return metadata
    
    def get_status(self) -> Dict[str, Any]:
        """Get key manager status"""
        active_keys = sum(1 for k in self._key_info.values() if k.is_active)
        expired_keys = sum(1 for k in self._key_info.values() 
                          if k.expires_at and time.time() > k.expires_at)
        
        return {
            "total_keys_managed": len(self._key_info),
            "active_keys": active_keys,
            "expired_keys": expired_keys,
            "total_operations": self.total_operations,
            "failed_operations": self.failed_operations,
            "success_rate": ((self.total_operations - self.failed_operations) / 
                           max(1, self.total_operations)) * 100,
            "auto_rotation_enabled": self.auto_rotation_enabled,
            "default_lifetime_hours": self.default_key_lifetime / 3600
        } 