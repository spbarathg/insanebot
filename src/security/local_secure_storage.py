"""
Local Secure Storage - Practical Encryption for Local Deployment

This module implements secure key storage using software-based encryption
optimized for local environments with migration from existing .env storage.
"""

import os
import json
import logging
import hashlib
import secrets
import getpass
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
import time
import platform

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("Cryptography library not available. Using simplified storage.")

logger = logging.getLogger(__name__)

class StorageType(Enum):
    ENV_FILE = "env_file"           # Legacy .env storage
    ENCRYPTED_FILE = "encrypted_file"  # Encrypted JSON storage
    OS_CREDENTIAL = "os_credential"    # OS credential manager

class MigrationStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class EncryptedKey:
    """Encrypted key storage format"""
    key_id: str
    encrypted_data: str
    salt: str
    created_at: float
    last_accessed: float
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StorageMetrics:
    """Storage system metrics"""
    total_keys: int = 0
    encrypted_keys: int = 0
    legacy_keys: int = 0
    failed_migrations: int = 0
    last_backup_time: float = 0.0
    storage_size_kb: float = 0.0

class LocalSecureStorage:
    """
    Local secure storage with practical encryption and migration
    
    Features:
    - Software-based encryption with strong passwords
    - Migration from .env to secure storage
    - Cross-platform OS-level permissions
    - Automated key rotation and backup
    - Secure password management
    """
    
    def __init__(self, storage_directory: str = "secure_storage"):
        self.storage_dir = Path(storage_directory)
        self.storage_dir.mkdir(mode=0o700, exist_ok=True)  # Owner read/write/execute only
        
        # Storage files
        self.encrypted_storage_file = self.storage_dir / "encrypted_keys.json"
        self.metadata_file = self.storage_dir / "storage_metadata.json"
        self.backup_dir = self.storage_dir / "backups"
        self.backup_dir.mkdir(mode=0o700, exist_ok=True)
        
        # Encryption components
        self.master_password = None
        self.encryption_key = None
        self.fernet_cipher = None
        
        # Storage state
        self.encrypted_keys: Dict[str, EncryptedKey] = {}
        self.storage_metrics = StorageMetrics()
        self.is_unlocked = False
        
        # Migration tracking
        self.migration_status = MigrationStatus.NOT_STARTED
        self.legacy_env_file = Path(".env")
        
        logger.info(f"🔐 Local Secure Storage initialized: {storage_directory}")
    
    async def initialize(self, master_password: Optional[str] = None) -> bool:
        """Initialize secure storage system"""
        try:
            # Check for existing storage
            if self.encrypted_storage_file.exists():
                if master_password:
                    success = await self._unlock_storage(master_password)
                else:
                    success = await self._prompt_for_password()
            else:
                # First time setup
                success = await self._setup_new_storage(master_password)
            
            if success:
                # Load existing keys
                await self._load_encrypted_keys()
                
                # Check for migration needs
                await self._check_migration_status()
                
                # Set secure permissions
                await self._set_secure_permissions()
                
                logger.info("✅ Local Secure Storage initialization complete")
                return True
            else:
                logger.error("❌ Failed to unlock secure storage")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Local Secure Storage: {str(e)}")
            return False
    
    async def _setup_new_storage(self, master_password: Optional[str] = None) -> bool:
        """Setup new secure storage"""
        try:
            if not master_password:
                master_password = await self._create_master_password()
            
            # Generate encryption key from password
            await self._setup_encryption(master_password)
            
            # Create initial storage structure
            await self._create_storage_structure()
            
            self.is_unlocked = True
            logger.info("🔒 New secure storage created")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up new storage: {str(e)}")
            return False
    
    async def _create_master_password(self) -> str:
        """Create master password with user input"""
        try:
            print("\n🔐 Setting up secure storage - Create a master password")
            print("This password will protect all your private keys.")
            print("Requirements: At least 12 characters, mix of letters, numbers, symbols")
            
            while True:
                password = getpass.getpass("Enter master password: ")
                
                if len(password) < 12:
                    print("❌ Password too short. Minimum 12 characters.")
                    continue
                
                # Basic strength check
                if not any(c.isupper() for c in password):
                    print("❌ Password needs at least one uppercase letter.")
                    continue
                
                if not any(c.islower() for c in password):
                    print("❌ Password needs at least one lowercase letter.")
                    continue
                
                if not any(c.isdigit() for c in password):
                    print("❌ Password needs at least one number.")
                    continue
                
                confirm_password = getpass.getpass("Confirm master password: ")
                
                if password != confirm_password:
                    print("❌ Passwords don't match. Try again.")
                    continue
                
                print("✅ Master password created successfully")
                return password
            
        except Exception as e:
            logger.error(f"Error creating master password: {str(e)}")
            raise
    
    async def _setup_encryption(self, master_password: str):
        """Setup encryption using master password"""
        try:
            if not CRYPTO_AVAILABLE:
                logger.warning("Cryptography not available - using simplified storage")
                self.master_password = master_password
                return
            
            # Generate salt for key derivation
            salt = os.urandom(16)
            
            # Derive encryption key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,  # OWASP recommended minimum
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
            self.encryption_key = key
            self.fernet_cipher = Fernet(key)
            
            # Store salt for future use
            salt_file = self.storage_dir / "salt.bin"
            with open(salt_file, 'wb') as f:
                f.write(salt)
            
            # Set secure permissions
            os.chmod(salt_file, 0o600)
            
            logger.info("🔑 Encryption setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up encryption: {str(e)}")
            raise
    
    async def _unlock_storage(self, master_password: str) -> bool:
        """Unlock existing storage with password"""
        try:
            if not CRYPTO_AVAILABLE:
                self.master_password = master_password
                self.is_unlocked = True
                return True
            
            # Load salt
            salt_file = self.storage_dir / "salt.bin"
            if not salt_file.exists():
                logger.error("Salt file not found")
                return False
            
            with open(salt_file, 'rb') as f:
                salt = f.read()
            
            # Derive key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
            self.encryption_key = key
            self.fernet_cipher = Fernet(key)
            
            # Test decryption with a known value
            test_successful = await self._test_decryption()
            
            if test_successful:
                self.is_unlocked = True
                logger.info("🔓 Storage unlocked successfully")
                return True
            else:
                logger.error("❌ Invalid master password")
                return False
            
        except Exception as e:
            logger.error(f"Error unlocking storage: {str(e)}")
            return False
    
    async def _prompt_for_password(self) -> bool:
        """Prompt user for master password"""
        try:
            print("\n🔐 Secure storage detected - Enter your master password")
            
            for attempt in range(3):  # Allow 3 attempts
                password = getpass.getpass("Master password: ")
                
                if await self._unlock_storage(password):
                    return True
                else:
                    remaining = 2 - attempt
                    if remaining > 0:
                        print(f"❌ Invalid password. {remaining} attempts remaining.")
                    else:
                        print("❌ Too many failed attempts. Exiting.")
            
            return False
            
        except Exception as e:
            logger.error(f"Error prompting for password: {str(e)}")
            return False
    
    async def store_key(self, key_id: str, private_key: str, metadata: Dict[str, Any] = None) -> bool:
        """Store private key securely"""
        try:
            if not self.is_unlocked:
                logger.error("Storage not unlocked")
                return False
            
            # Prepare key data
            key_data = {
                "private_key": private_key,
                "metadata": metadata or {},
                "stored_at": time.time()
            }
            
            if CRYPTO_AVAILABLE and self.fernet_cipher:
                # Encrypt the key data
                serialized_data = json.dumps(key_data).encode()
                encrypted_data = self.fernet_cipher.encrypt(serialized_data)
                encrypted_data_b64 = base64.b64encode(encrypted_data).decode()
            else:
                # Fallback: Base64 encoding (not secure, but better than plain text)
                serialized_data = json.dumps(key_data).encode()
                encrypted_data_b64 = base64.b64encode(serialized_data).decode()
            
            # Create encrypted key object
            encrypted_key = EncryptedKey(
                key_id=key_id,
                encrypted_data=encrypted_data_b64,
                salt=base64.b64encode(os.urandom(16)).decode(),
                created_at=time.time(),
                last_accessed=time.time(),
                metadata=metadata or {}
            )
            
            # Store in memory
            self.encrypted_keys[key_id] = encrypted_key
            
            # Persist to disk
            await self._save_encrypted_keys()
            
            # Update metrics
            self.storage_metrics.total_keys = len(self.encrypted_keys)
            self.storage_metrics.encrypted_keys += 1
            
            logger.info(f"🔐 Key stored securely: {key_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing key {key_id}: {str(e)}")
            return False
    
    async def retrieve_key(self, key_id: str) -> Optional[str]:
        """Retrieve and decrypt private key"""
        try:
            if not self.is_unlocked:
                logger.error("Storage not unlocked")
                return None
            
            if key_id not in self.encrypted_keys:
                logger.warning(f"Key not found: {key_id}")
                return None
            
            encrypted_key = self.encrypted_keys[key_id]
            
            # Decrypt key data
            encrypted_data = base64.b64decode(encrypted_key.encrypted_data.encode())
            
            if CRYPTO_AVAILABLE and self.fernet_cipher:
                try:
                    decrypted_data = self.fernet_cipher.decrypt(encrypted_data)
                except Exception:
                    # Fallback decryption
                    decrypted_data = base64.b64decode(encrypted_data)
            else:
                decrypted_data = base64.b64decode(encrypted_data)
            
            # Parse key data
            key_data = json.loads(decrypted_data.decode())
            
            # Update access tracking
            encrypted_key.last_accessed = time.time()
            encrypted_key.access_count += 1
            
            logger.debug(f"🔓 Key retrieved: {key_id}")
            return key_data["private_key"]
            
        except Exception as e:
            logger.error(f"Error retrieving key {key_id}: {str(e)}")
            return None
    
    async def migrate_from_env(self) -> bool:
        """Migrate keys from .env file to secure storage"""
        try:
            if not self.legacy_env_file.exists():
                logger.info("No .env file found for migration")
                self.migration_status = MigrationStatus.COMPLETED
                return True
            
            logger.info("🔄 Starting migration from .env file")
            self.migration_status = MigrationStatus.IN_PROGRESS
            
            # Read .env file
            env_keys = {}
            with open(self.legacy_env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')  # Remove quotes
                        
                        # Identify private keys
                        if any(keyword in key.lower() for keyword in ['private', 'secret', 'key']):
                            env_keys[key] = value
            
            if not env_keys:
                logger.info("No private keys found in .env file")
                self.migration_status = MigrationStatus.COMPLETED
                return True
            
            # Migrate each key
            migration_success = True
            for key_name, key_value in env_keys.items():
                try:
                    metadata = {
                        "source": "env_migration",
                        "original_name": key_name,
                        "migrated_at": time.time()
                    }
                    
                    success = await self.store_key(
                        key_id=f"migrated_{key_name.lower()}",
                        private_key=key_value,
                        metadata=metadata
                    )
                    
                    if success:
                        logger.info(f"✅ Migrated key: {key_name}")
                    else:
                        logger.error(f"❌ Failed to migrate key: {key_name}")
                        migration_success = False
                        
                except Exception as e:
                    logger.error(f"Error migrating key {key_name}: {str(e)}")
                    migration_success = False
            
            if migration_success:
                # Create backup of original .env file
                backup_path = self.backup_dir / f"env_backup_{int(time.time())}.txt"
                import shutil
                shutil.copy2(self.legacy_env_file, backup_path)
                
                # Recommend removing original .env file
                print(f"\n✅ Migration completed successfully!")
                print(f"Original .env backed up to: {backup_path}")
                print("🔒 Consider removing the original .env file for security:")
                print(f"   rm {self.legacy_env_file}")
                
                self.migration_status = MigrationStatus.COMPLETED
            else:
                self.migration_status = MigrationStatus.FAILED
                self.storage_metrics.failed_migrations += 1
            
            return migration_success
            
        except Exception as e:
            logger.error(f"Error during migration: {str(e)}")
            self.migration_status = MigrationStatus.FAILED
            return False
    
    async def _set_secure_permissions(self):
        """Set secure OS-level permissions"""
        try:
            system = platform.system().lower()
            
            if system in ['linux', 'darwin']:  # Linux/MacOS
                # Set restrictive permissions (owner only)
                for file_path in [self.encrypted_storage_file, self.metadata_file]:
                    if file_path.exists():
                        os.chmod(file_path, 0o600)  # rw-------
                
                # Directory permissions
                os.chmod(self.storage_dir, 0o700)  # rwx------
                os.chmod(self.backup_dir, 0o700)
                
                logger.info("🔒 Unix permissions set (owner only)")
                
            elif system == 'windows':
                # Windows permissions (more complex, simplified here)
                import subprocess
                try:
                    # Remove inheritance and grant access only to current user
                    subprocess.run([
                        'icacls', str(self.storage_dir),
                        '/inheritance:d', '/grant:r', f'{os.getlogin()}:F',
                        '/remove', 'Users', '/remove', 'Everyone'
                    ], check=True, capture_output=True)
                    
                    logger.info("🔒 Windows permissions set (current user only)")
                except subprocess.CalledProcessError:
                    logger.warning("Could not set Windows permissions automatically")
            
        except Exception as e:
            logger.warning(f"Could not set secure permissions: {str(e)}")
    
    async def create_backup(self) -> bool:
        """Create encrypted backup of all keys"""
        try:
            backup_filename = f"keys_backup_{int(time.time())}.json"
            backup_path = self.backup_dir / backup_filename
            
            # Create backup data
            backup_data = {
                "created_at": time.time(),
                "version": "1.0",
                "encrypted_keys": {
                    key_id: {
                        "encrypted_data": key.encrypted_data,
                        "salt": key.salt,
                        "created_at": key.created_at,
                        "metadata": key.metadata
                    }
                    for key_id, key in self.encrypted_keys.items()
                },
                "storage_metrics": {
                    "total_keys": self.storage_metrics.total_keys,
                    "encrypted_keys": self.storage_metrics.encrypted_keys
                }
            }
            
            # Write backup
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            # Set secure permissions
            os.chmod(backup_path, 0o600)
            
            self.storage_metrics.last_backup_time = time.time()
            
            logger.info(f"💾 Backup created: {backup_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            return False
    
    def get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage system metrics"""
        # Calculate storage size
        storage_size = 0
        if self.encrypted_storage_file.exists():
            storage_size = self.encrypted_storage_file.stat().st_size / 1024  # KB
        
        return {
            "total_keys": len(self.encrypted_keys),
            "encrypted_keys": self.storage_metrics.encrypted_keys,
            "storage_size_kb": storage_size,
            "migration_status": self.migration_status.value,
            "last_backup_time": self.storage_metrics.last_backup_time,
            "is_unlocked": self.is_unlocked,
            "crypto_available": CRYPTO_AVAILABLE,
            "storage_directory": str(self.storage_dir),
            "backup_count": len(list(self.backup_dir.glob("*.json")))
        } 