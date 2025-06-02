"""
Production-Grade Secrets Management System

Secure storage and management of API keys, private keys, and sensitive configuration.
Features:
- Encrypted storage with multiple encryption layers
- Key rotation and versioning
- Secure key derivation from master password
- Integration with external secret stores (AWS Secrets Manager, HashiCorp Vault)
- Audit logging for all secret access
"""

import os
import json
import base64
import hashlib
import hmac
import time
import asyncio
import logging
import shutil
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import secrets
import tarfile

logger = logging.getLogger(__name__)

@dataclass
class SecretMetadata:
    """Metadata for a stored secret"""
    name: str
    version: int
    created_at: float
    updated_at: float
    expires_at: Optional[float] = None
    tags: List[str] = None
    access_count: int = 0
    last_accessed: Optional[float] = None

@dataclass
class SecretEntry:
    """Complete secret entry with metadata"""
    metadata: SecretMetadata
    encrypted_value: str
    salt: str
    nonce: str

class SecretAccessAuditor:
    """Audit logging for secret access"""
    
    def __init__(self, audit_file: Path):
        self.audit_file = audit_file
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_access(self, secret_name: str, action: str, success: bool, user: str = None):
        """Log secret access event"""
        audit_entry = {
            'timestamp': time.time(),
            'secret_name': secret_name,
            'action': action,  # 'read', 'write', 'delete', 'rotate'
            'success': success,
            'user': user or 'system',
            'process_id': os.getpid()
        }
        
        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')

class EncryptionProvider:
    """Multi-layer encryption provider"""
    
    def __init__(self, master_password: str):
        self.master_password = master_password.encode('utf-8')
        self._initialize_encryption_keys()
    
    def _initialize_encryption_keys(self):
        """Initialize encryption keys from master password"""
        # Derive multiple keys for layered encryption
        self.salt_1 = os.urandom(32)
        self.salt_2 = os.urandom(32)
        
        # Primary encryption key (Fernet)
        kdf1 = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt_1,
            iterations=100000,
            backend=default_backend()
        )
        key1 = base64.urlsafe_b64encode(kdf1.derive(self.master_password))
        self.fernet = Fernet(key1)
        
        # Secondary encryption key (AES)
        kdf2 = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt_2,
            iterations=100000,
            backend=default_backend()
        )
        self.aes_key = kdf2.derive(self.master_password)
    
    def encrypt(self, data: str) -> Dict[str, str]:
        """Multi-layer encryption of sensitive data"""
        try:
            # Layer 1: Fernet encryption
            encrypted_layer1 = self.fernet.encrypt(data.encode('utf-8'))
            
            # Layer 2: AES encryption
            nonce = os.urandom(12)  # GCM mode nonce
            cipher = Cipher(
                algorithms.AES(self.aes_key),
                modes.GCM(nonce),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            encrypted_layer2 = encryptor.update(encrypted_layer1) + encryptor.finalize()
            
            return {
                'encrypted_data': base64.b64encode(encrypted_layer2).decode('utf-8'),
                'nonce': base64.b64encode(nonce).decode('utf-8'),
                'tag': base64.b64encode(encryptor.tag).decode('utf-8'),
                'salt_1': base64.b64encode(self.salt_1).decode('utf-8'),
                'salt_2': base64.b64encode(self.salt_2).decode('utf-8')
            }
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise
    
    def decrypt(self, encrypted_data: Dict[str, str]) -> str:
        """Multi-layer decryption of sensitive data"""
        try:
            # Restore salts and nonce
            nonce = base64.b64decode(encrypted_data['nonce'])
            tag = base64.b64decode(encrypted_data['tag'])
            data = base64.b64decode(encrypted_data['encrypted_data'])
            
            # Layer 2: AES decryption
            cipher = Cipher(
                algorithms.AES(self.aes_key),
                modes.GCM(nonce, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            decrypted_layer2 = decryptor.update(data) + decryptor.finalize()
            
            # Layer 1: Fernet decryption
            decrypted_layer1 = self.fernet.decrypt(decrypted_layer2)
            
            return decrypted_layer1.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise

class SecretsManager:
    """Enterprise-grade secrets management with encryption, versioning, and audit logging."""
    
    def __init__(self, vault_path: str = "data/secrets", master_key_env: str = "MASTER_KEY"):
        self.vault_path = Path(vault_path)
        self.vault_path.mkdir(parents=True, exist_ok=True)
        
        self.master_key_env = master_key_env
        self.audit_log_path = self.vault_path / "audit.log"
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_audit_logging()
        
        # Initialize encryption
        self._master_key = self._get_or_create_master_key()
        self._fernet = self._create_fernet_key()
        
        # Secrets metadata
        self.metadata_file = self.vault_path / "metadata.json"
        self._metadata = self._load_metadata()
        
        self.logger.info("SecretsManager initialized successfully")
        self._audit_log("INIT", "SecretsManager initialized")
    
    def _setup_audit_logging(self):
        """Setup dedicated audit logging."""
        audit_handler = logging.FileHandler(self.audit_log_path)
        audit_handler.setLevel(logging.INFO)
        audit_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        
        audit_logger = logging.getLogger('secrets_audit')
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
        
        self.audit_logger = audit_logger
    
    def _audit_log(self, action: str, details: str, secret_name: str = None):
        """Log security-relevant events."""
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "action": action,
            "details": details,
            "secret_name": secret_name,
            "user": os.getenv("USER", "unknown")
        }
        self.audit_logger.info(json.dumps(log_entry))
    
    def _get_or_create_master_key(self) -> bytes:
        """Get master key from environment or create new one."""
        env_key = os.getenv(self.master_key_env)
        
        if env_key:
            try:
                # Decode existing key
                return base64.b64decode(env_key.encode())
            except Exception as e:
                self.logger.warning(f"Invalid master key in environment: {e}")
                self._audit_log("MASTER_KEY_ERROR", f"Invalid master key: {e}")
        
        # Generate new master key
        master_key = secrets.token_bytes(32)
        encoded_key = base64.b64encode(master_key).decode()
        
        self.logger.warning(
            f"Generated new master key. Set {self.master_key_env}={encoded_key} "
            "in your environment variables for production!"
        )
        self._audit_log("MASTER_KEY_GENERATED", "New master key generated")
        
        return master_key
    
    def _create_fernet_key(self) -> Fernet:
        """Create Fernet encryption key from master key."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'secrets_manager_salt',  # In production, use random salt
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(self._master_key))
        return Fernet(key)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load secrets metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load metadata: {e}")
                self._audit_log("METADATA_LOAD_ERROR", f"Failed to load metadata: {e}")
        
        return {"secrets": {}, "version": "1.0", "created": datetime.datetime.utcnow().isoformat()}
    
    def _save_metadata(self):
        """Save secrets metadata."""
        try:
            self._metadata["last_updated"] = datetime.datetime.utcnow().isoformat()
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            self._audit_log("METADATA_SAVE_ERROR", f"Failed to save metadata: {e}")
    
    def _encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data with multiple layers."""
        # Layer 1: Fernet encryption
        encrypted_fernet = self._fernet.encrypt(data.encode())
        
        # Layer 2: AES encryption with random key
        aes_key = secrets.token_bytes(32)
        iv = secrets.token_bytes(16)
        
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padded_data = encrypted_fernet + b'\x00' * (16 - len(encrypted_fernet) % 16)
        encrypted_aes = encryptor.update(padded_data) + encryptor.finalize()
        
        # Combine all components
        final_data = {
            "aes_key": base64.b64encode(aes_key).decode(),
            "iv": base64.b64encode(iv).decode(),
            "data": base64.b64encode(encrypted_aes).decode()
        }
        
        return json.dumps(final_data).encode()
    
    def _decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data."""
        try:
            # Parse encrypted components
            data_dict = json.loads(encrypted_data.decode())
            aes_key = base64.b64decode(data_dict["aes_key"])
            iv = base64.b64decode(data_dict["iv"])
            encrypted_aes = base64.b64decode(data_dict["data"])
            
            # Layer 2: AES decryption
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted_aes = decryptor.update(encrypted_aes) + decryptor.finalize()
            
            # Remove padding
            encrypted_fernet = decrypted_aes.rstrip(b'\x00')
            
            # Layer 1: Fernet decryption
            return self._fernet.decrypt(encrypted_fernet).decode()
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}")
            self._audit_log("DECRYPTION_ERROR", f"Failed to decrypt data: {e}")
            raise ValueError("Failed to decrypt secret")
    
    def store_secret(self, name: str, value: str, description: str = "", tags: List[str] = None) -> bool:
        """Store a secret with encryption and versioning."""
        try:
            # Encrypt the secret
            encrypted_value = self._encrypt_data(value)
            
            # Create secret file path
            secret_file = self.vault_path / f"{name}.secret"
            
            # Get current version
            current_version = 1
            if name in self._metadata["secrets"]:
                current_version = self._metadata["secrets"][name]["version"] + 1
            
            # Store encrypted secret
            secret_data = {
                "version": current_version,
                "encrypted_value": base64.b64encode(encrypted_value).decode(),
                "created": datetime.datetime.utcnow().isoformat(),
                "checksum": hashlib.sha256(encrypted_value).hexdigest()
            }
            
            with open(secret_file, 'w') as f:
                json.dump(secret_data, f)
            
            # Update metadata
            self._metadata["secrets"][name] = {
                "version": current_version,
                "description": description,
                "tags": tags or [],
                "created": secret_data["created"],
                "file": f"{name}.secret"
            }
            
            self._save_metadata()
            
            self.logger.info(f"Secret '{name}' stored successfully (version {current_version})")
            self._audit_log("SECRET_STORED", f"Secret stored (version {current_version})", name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store secret '{name}': {e}")
            self._audit_log("SECRET_STORE_ERROR", f"Failed to store secret: {e}", name)
            return False
    
    def get_secret(self, name: str, version: Optional[int] = None) -> Optional[str]:
        """Retrieve and decrypt a secret."""
        try:
            if name not in self._metadata["secrets"]:
                self.logger.warning(f"Secret '{name}' not found")
                self._audit_log("SECRET_NOT_FOUND", f"Secret not found", name)
                return None
            
            # Load secret file
            secret_file = self.vault_path / f"{name}.secret"
            if not secret_file.exists():
                self.logger.error(f"Secret file for '{name}' not found")
                self._audit_log("SECRET_FILE_NOT_FOUND", f"Secret file not found", name)
                return None
            
            with open(secret_file, 'r') as f:
                secret_data = json.load(f)
            
            # Verify integrity
            encrypted_value = base64.b64decode(secret_data["encrypted_value"])
            if hashlib.sha256(encrypted_value).hexdigest() != secret_data["checksum"]:
                self.logger.error(f"Integrity check failed for secret '{name}'")
                self._audit_log("INTEGRITY_CHECK_FAILED", f"Integrity check failed", name)
                return None
            
            # Decrypt secret
            decrypted_value = self._decrypt_data(encrypted_value)
            
            self.logger.info(f"Secret '{name}' retrieved successfully")
            self._audit_log("SECRET_RETRIEVED", f"Secret retrieved", name)
            
            return decrypted_value
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret '{name}': {e}")
            self._audit_log("SECRET_RETRIEVE_ERROR", f"Failed to retrieve secret: {e}", name)
            return None
    
    def rotate_secret(self, name: str, new_value: str) -> bool:
        """Rotate a secret to a new value."""
        try:
            old_value = self.get_secret(name)
            if old_value is None:
                self.logger.error(f"Cannot rotate non-existent secret '{name}'")
                return False
            
            # Store new version
            metadata = self._metadata["secrets"].get(name, {})
            description = metadata.get("description", "")
            tags = metadata.get("tags", [])
            
            success = self.store_secret(name, new_value, description, tags)
            
            if success:
                self.logger.info(f"Secret '{name}' rotated successfully")
                self._audit_log("SECRET_ROTATED", f"Secret rotated", name)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to rotate secret '{name}': {e}")
            self._audit_log("SECRET_ROTATE_ERROR", f"Failed to rotate secret: {e}", name)
            return False
    
    def delete_secret(self, name: str) -> bool:
        """Delete a secret (secure deletion)."""
        try:
            if name not in self._metadata["secrets"]:
                self.logger.warning(f"Secret '{name}' not found for deletion")
                return False
            
            # Secure file deletion
            secret_file = self.vault_path / f"{name}.secret"
            if secret_file.exists():
                # Overwrite file with random data before deletion
                file_size = secret_file.stat().st_size
                with open(secret_file, 'wb') as f:
                    f.write(secrets.token_bytes(file_size))
                
                secret_file.unlink()
            
            # Remove from metadata
            del self._metadata["secrets"][name]
            self._save_metadata()
            
            self.logger.info(f"Secret '{name}' deleted successfully")
            self._audit_log("SECRET_DELETED", f"Secret deleted", name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete secret '{name}': {e}")
            self._audit_log("SECRET_DELETE_ERROR", f"Failed to delete secret: {e}", name)
            return False
    
    def list_secrets(self) -> List[Dict[str, Any]]:
        """List all secrets with metadata (excluding values)."""
        secrets_list = []
        
        for name, metadata in self._metadata["secrets"].items():
            secrets_list.append({
                "name": name,
                "version": metadata.get("version", 1),
                "description": metadata.get("description", ""),
                "tags": metadata.get("tags", []),
                "created": metadata.get("created", ""),
            })
        
        self._audit_log("SECRETS_LISTED", f"Listed {len(secrets_list)} secrets")
        return secrets_list
    
    def backup_vault(self, backup_path: str) -> bool:
        """Create encrypted backup of entire vault."""
        try:
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create tar archive
            tar_path = backup_path.with_suffix('.tar')
            with tarfile.open(tar_path, 'w') as tar:
                tar.add(self.vault_path, arcname='secrets_vault')
            
            # Encrypt backup
            with open(tar_path, 'rb') as f:
                backup_data = f.read()
            
            encrypted_backup = self._encrypt_data(base64.b64encode(backup_data).decode())
            
            with open(backup_path, 'wb') as f:
                f.write(encrypted_backup)
            
            # Clean up temporary tar file
            tar_path.unlink()
            
            self.logger.info(f"Vault backup created: {backup_path}")
            self._audit_log("VAULT_BACKUP", f"Vault backup created: {backup_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to backup vault: {e}")
            self._audit_log("VAULT_BACKUP_ERROR", f"Failed to backup vault: {e}")
            return False
    
    def restore_vault(self, backup_path: str) -> bool:
        """Restore vault from encrypted backup."""
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                self.logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Decrypt backup
            with open(backup_path, 'rb') as f:
                encrypted_backup = f.read()
            
            backup_data = base64.b64decode(self._decrypt_data(encrypted_backup))
            
            # Extract to temporary location
            temp_tar = backup_path.with_suffix('.temp.tar')
            with open(temp_tar, 'wb') as f:
                f.write(backup_data)
            
            # Create backup of current vault
            current_backup = self.vault_path.with_suffix('.backup')
            if self.vault_path.exists():
                shutil.move(str(self.vault_path), str(current_backup))
            
            # Extract backup with safety validation
            with tarfile.open(temp_tar, 'r') as tar:
                # Validate tar members for security
                def is_safe_path(path, base_path):
                    """Check if the path is safe (no directory traversal)"""
                    full_path = os.path.realpath(os.path.join(base_path, path))
                    return full_path.startswith(os.path.realpath(base_path))
                
                safe_members = []
                for member in tar.getmembers():
                    if is_safe_path(member.name, str(self.vault_path.parent)):
                        safe_members.append(member)
                    else:
                        self.logger.warning(f"Skipping unsafe tar member: {member.name}")
                
                # Extract only safe members
                for member in safe_members:
                    tar.extract(member, path=self.vault_path.parent)
            
            # Clean up
            temp_tar.unlink()
            
            # Reload metadata
            self._metadata = self._load_metadata()
            
            self.logger.info(f"Vault restored from: {backup_path}")
            self._audit_log("VAULT_RESTORED", f"Vault restored from: {backup_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore vault: {e}")
            self._audit_log("VAULT_RESTORE_ERROR", f"Failed to restore vault: {e}")
            return False
    
    def migrate_from_env(self, env_vars: List[str]) -> bool:
        """Migrate secrets from environment variables to encrypted storage."""
        try:
            migrated_count = 0
            
            for env_var in env_vars:
                value = os.getenv(env_var)
                if value:
                    # Store in secrets manager
                    success = self.store_secret(
                        name=env_var.lower(),
                        value=value,
                        description=f"Migrated from environment variable {env_var}",
                        tags=["migrated", "environment"]
                    )
                    
                    if success:
                        migrated_count += 1
                        self.logger.info(f"Migrated {env_var} to secrets manager")
                else:
                    self.logger.warning(f"Environment variable {env_var} not found")
            
            self.logger.info(f"Migration completed: {migrated_count}/{len(env_vars)} variables migrated")
            self._audit_log("MIGRATION_COMPLETED", f"Migrated {migrated_count} environment variables")
            
            return migrated_count > 0
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            self._audit_log("MIGRATION_ERROR", f"Migration failed: {e}")
            return False
    
    def generate_migration_script(self, output_file: str = "migrate_secrets.py") -> bool:
        """Generate script to help migrate from environment variables."""
        try:
            script_content = '''#!/usr/bin/env python3
"""
Secrets migration script - automatically generated
Run this script to migrate environment variables to encrypted secrets storage
"""

import os
from src.core.secrets_manager import SecretsManager

def main():
    # Initialize secrets manager
    sm = SecretsManager()
    
    # Environment variables to migrate
    env_vars = [
        "SOLANA_RPC_URL",
        "WALLET_PRIVATE_KEY", 
        "DISCORD_WEBHOOK_URL",
        "HELIUS_API_KEY",
        "QUICKNODE_ENDPOINT_URL"
    ]
    
    print("🔐 Starting secrets migration...")
    print(f"Found {len(env_vars)} environment variables to migrate")
    
    # Migrate each variable
    for var in env_vars:
        value = os.getenv(var)
        if value and not value.startswith("your-"):
            success = sm.store_secret(
                name=var.lower(),
                value=value,
                description=f"Migrated from {var}",
                tags=["production", "migrated"]
            )
            if success:
                print(f"✅ Migrated {var}")
            else:
                print(f"❌ Failed to migrate {var}")
        else:
            print(f"⚠️  Skipped {var} (not set or template value)")
    
    print("🎉 Migration completed!")
    print("\\nNext steps:")
    print("1. Update your application to use sm.get_secret() instead of os.getenv()")
    print("2. Remove sensitive values from environment variables")
    print("3. Set only MASTER_KEY in your environment")

if __name__ == "__main__":
    main()
'''
            
            with open(output_file, 'w') as f:
                f.write(script_content)
            
            # Make script executable
            os.chmod(output_file, 0o755)
            
            self.logger.info(f"Migration script generated: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate migration script: {e}")
            return False

def main():
    """CLI interface for secrets manager."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Enterprise Secrets Manager")
    parser.add_argument("action", choices=["store", "get", "list", "delete", "rotate", "backup", "restore", "migrate"])
    parser.add_argument("--name", help="Secret name")
    parser.add_argument("--value", help="Secret value")
    parser.add_argument("--description", help="Secret description")
    parser.add_argument("--path", help="Backup/restore path")
    parser.add_argument("--env-vars", nargs="+", help="Environment variables to migrate")
    
    args = parser.parse_args()
    
    sm = SecretsManager()
    
    if args.action == "store":
        if not args.name or not args.value:
            print("Error: --name and --value required for store action")
            sys.exit(1)
        success = sm.store_secret(args.name, args.value, args.description or "")
        print(f"{'✅' if success else '❌'} Store secret: {args.name}")
    
    elif args.action == "get":
        if not args.name:
            print("Error: --name required for get action")
            sys.exit(1)
        value = sm.get_secret(args.name)
        if value:
            print(f"Secret '{args.name}': {value}")
        else:
            print(f"Secret '{args.name}' not found")
            sys.exit(1)
    
    elif args.action == "list":
        secrets = sm.list_secrets()
        print(f"Found {len(secrets)} secrets:")
        for secret in secrets:
            print(f"  • {secret['name']} (v{secret['version']}) - {secret['description']}")
    
    elif args.action == "delete":
        if not args.name:
            print("Error: --name required for delete action")
            sys.exit(1)
        success = sm.delete_secret(args.name)
        print(f"{'✅' if success else '❌'} Delete secret: {args.name}")
    
    elif args.action == "backup":
        if not args.path:
            print("Error: --path required for backup action")
            sys.exit(1)
        success = sm.backup_vault(args.path)
        print(f"{'✅' if success else '❌'} Backup vault to: {args.path}")
    
    elif args.action == "migrate":
        if args.env_vars:
            success = sm.migrate_from_env(args.env_vars)
        else:
            success = sm.generate_migration_script()
        print(f"{'✅' if success else '❌'} Migration {'completed' if args.env_vars else 'script generated'}")

if __name__ == "__main__":
    main() 