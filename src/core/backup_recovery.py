"""
Production-Grade Backup and Recovery System

Automated backup and disaster recovery for trading bot data:
- Automated scheduled backups
- Data integrity verification
- Point-in-time recovery
- Cross-platform compatibility
- Encrypted backup storage
- Monitoring and alerting
"""

import os
import json
import asyncio
import logging
import hashlib
import time
import shutil
import tarfile
import gzip
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
import schedule
import threading
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class BackupMetadata:
    """Metadata for backup files"""
    backup_id: str
    timestamp: str
    backup_type: str  # 'full', 'incremental', 'differential'
    size_bytes: int
    file_count: int
    checksum: str
    compression_ratio: float
    encryption_enabled: bool
    source_paths: List[str]
    backup_file: str

@dataclass
class RecoveryPoint:
    """Recovery point information."""
    backup_id: str
    timestamp: str
    backup_type: str
    description: str
    integrity_verified: bool
    size_mb: float

class BackupConfig:
    """Backup configuration"""
    backup_dir: Path
    retention_days: int = 30
    max_backups: int = 100
    compression_enabled: bool = True
    encryption_enabled: bool = True
    encryption_key: Optional[str] = None
    schedule_full: str = "daily"  # daily, weekly, monthly
    schedule_incremental: str = "hourly"  # hourly, every_6h, every_12h
    remote_backup_enabled: bool = False
    remote_backup_config: Dict[str, Any] = None

class DataIntegrityChecker:
    """Verify data integrity for backups"""
    
    def __init__(self):
        self.hash_algorithm = hashlib.sha256
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash for a single file"""
        hasher = self.hash_algorithm()
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            return ""
    
    def calculate_directory_hash(self, directory: Path) -> Dict[str, str]:
        """Calculate hashes for all files in directory"""
        file_hashes = {}
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(directory)
                file_hashes[str(relative_path)] = self.calculate_file_hash(file_path)
        return file_hashes
    
    def verify_backup_integrity(self, backup_path: Path, expected_checksum: str) -> bool:
        """Verify backup file integrity"""
        if not backup_path.exists():
            return False
        
        actual_checksum = self.calculate_file_hash(backup_path)
        return actual_checksum == expected_checksum

class BackupCompressor:
    """Handle backup compression and decompression"""
    
    def compress_directory(self, source_dir: Path, target_file: Path) -> bool:
        """Compress directory to tar.gz file"""
        try:
            with tarfile.open(target_file, 'w:gz') as tar:
                tar.add(source_dir, arcname=source_dir.name)
            return True
        except Exception as e:
            logger.error(f"Compression failed: {str(e)}")
            return False
    
    def decompress_archive(self, archive_file: Path, target_dir: Path) -> bool:
        """Decompress tar.gz file to directory with security validation"""
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(archive_file, 'r:gz') as tar:
                # Validate tar members for security
                def is_safe_path(path, base_path):
                    """Check if the path is safe (no directory traversal)"""
                    full_path = os.path.realpath(os.path.join(base_path, path))
                    return full_path.startswith(os.path.realpath(base_path))
                
                safe_members = []
                for member in tar.getmembers():
                    if is_safe_path(member.name, str(target_dir)):
                        safe_members.append(member)
                    else:
                        logger.warning(f"Skipping unsafe tar member: {member.name}")
                
                # Extract only safe members
                for member in safe_members:
                    tar.extract(member, path=target_dir)
                    
            return True
        except Exception as e:
            logger.error(f"Decompression failed: {str(e)}")
            return False

class BackupEncryption:
    """Handle backup encryption and decryption"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        if encryption_key:
            self.fernet = Fernet(encryption_key.encode())
        else:
            # Generate new key
            key = Fernet.generate_key()
            self.fernet = Fernet(key)
            self.encryption_key = key.decode()
    
    def encrypt_file(self, source_file: Path, target_file: Path) -> bool:
        """Encrypt file"""
        try:
            with open(source_file, 'rb') as f:
                plaintext = f.read()
            
            encrypted_data = self.fernet.encrypt(plaintext)
            
            with open(target_file, 'wb') as f:
                f.write(encrypted_data)
            
            return True
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            return False
    
    def decrypt_file(self, source_file: Path, target_file: Path) -> bool:
        """Decrypt file"""
        try:
            with open(source_file, 'rb') as f:
                encrypted_data = f.read()
            
            plaintext = self.fernet.decrypt(encrypted_data)
            
            with open(target_file, 'wb') as f:
                f.write(plaintext)
            
            return True
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            return False

class BackupRecoverySystem:
    """Main backup and recovery system"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.config.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize backup-specific logger
        self.backup_logger = logging.getLogger(f"{__name__}.backup")
        
        # Set backup root directory
        self.backup_root = self.config.backup_dir
        
        # Initialize components
        self.integrity_checker = DataIntegrityChecker()
        self.compressor = BackupCompressor()
        self.encryptor = BackupEncryption(config.encryption_key) if config.encryption_enabled else None
        
        # Backup metadata storage
        self.metadata_file = self.config.backup_dir / "backup_metadata.json"
        self.backup_metadata: List[BackupMetadata] = []
        self._load_metadata()
        
        # State tracking
        self.last_full_backup = None
        self.last_incremental_backup = None
        self.scheduler_running = False
        self.scheduler_thread = None
        
        # Critical paths to backup
        self.backup_paths = [
            Path("data"),
            Path("config"),
            Path("logs"),
            Path(".env"),
        ]
        
        # Backup directories
        self.full_backups_dir = self.config.backup_dir / "full"
        self.incremental_backups_dir = self.config.backup_dir / "incremental"
        self.metadata_dir = self.config.backup_dir / "metadata"
        
        for dir_path in [self.full_backups_dir, self.incremental_backups_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Metadata database
        self.metadata_db = self.metadata_dir / "backup_metadata.db"
        self._init_metadata_db()
        
        # Backup settings
        self.compression_level = 9
        self.retention_days = 30
        self.max_backup_size_gb = 50
        
        self.logger.info("BackupRecoverySystem initialized successfully")
    
    def _load_metadata(self):
        """Load backup metadata from storage"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata_dicts = json.load(f)
                    self.backup_metadata = [
                        BackupMetadata(**meta) for meta in metadata_dicts
                    ]
                logger.info(f"Loaded {len(self.backup_metadata)} backup records")
        except Exception as e:
            logger.error(f"Failed to load backup metadata: {str(e)}")
            self.backup_metadata = []
    
    def _save_metadata(self):
        """Save backup metadata to storage"""
        try:
            metadata_dicts = [asdict(meta) for meta in self.backup_metadata]
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_dicts, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {str(e)}")
    
    def create_backup(self, backup_type: str = "full", description: str = "") -> Optional[str]:
        """Create a new backup"""
        try:
            timestamp = time.time()
            backup_id = f"{backup_type}_{datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Creating {backup_type} backup: {backup_id}")
            
            # Create temporary staging directory
            staging_dir = self.config.backup_dir / f"staging_{backup_id}"
            staging_dir.mkdir(exist_ok=True)
            
            try:
                # Copy files to staging
                total_size = 0
                file_count = 0
                
                for path in self.backup_paths:
                    if path.exists():
                        if path.is_file():
                            target_file = staging_dir / path.name
                            shutil.copy2(path, target_file)
                            total_size += target_file.stat().st_size
                            file_count += 1
                        elif path.is_dir():
                            target_dir = staging_dir / path.name
                            shutil.copytree(path, target_dir, dirs_exist_ok=True)
                            for file_path in target_dir.rglob('*'):
                                if file_path.is_file():
                                    total_size += file_path.stat().st_size
                                    file_count += 1
                
                # Create compressed archive
                archive_file = self.config.backup_dir / f"{backup_id}.tar.gz"
                if not self.compressor.compress_directory(staging_dir, archive_file):
                    raise Exception("Compression failed")
                
                # Encrypt if enabled
                final_backup_file = archive_file
                if self.config.encryption_enabled and self.encryptor:
                    encrypted_file = self.config.backup_dir / f"{backup_id}.encrypted"
                    if self.encryptor.encrypt_file(archive_file, encrypted_file):
                        archive_file.unlink()  # Remove unencrypted file
                        final_backup_file = encrypted_file
                    else:
                        raise Exception("Encryption failed")
                
                # Calculate checksum
                checksum = self.integrity_checker.calculate_file_hash(final_backup_file)
                
                # Create metadata
                metadata = BackupMetadata(
                    backup_id=backup_id,
                    timestamp=timestamp,
                    backup_type=backup_type,
                    size_bytes=final_backup_file.stat().st_size,
                    file_count=file_count,
                    checksum=checksum,
                    compression_ratio=final_backup_file.stat().st_size / final_backup_file.stat().st_size if final_backup_file.stat().st_size > 0 else 1.0,
                    encryption_enabled=self.config.encryption_enabled,
                    source_paths=[str(p) for p in self.backup_paths],
                    backup_file=str(final_backup_file)
                )
                
                # Save metadata
                self.backup_metadata.append(metadata)
                self._save_metadata()
                
                # Update tracking
                if backup_type == "full":
                    self.last_full_backup = timestamp
                elif backup_type == "incremental":
                    self.last_incremental_backup = timestamp
                
                logger.info(f"Backup created successfully: {backup_id}")
                logger.info(f"Files: {file_count}, Size: {total_size / 1024 / 1024:.2f} MB")
                
                # Cleanup old backups
                self._cleanup_old_backups()
                
                return backup_id
                
            finally:
                # Cleanup staging directory
                if staging_dir.exists():
                    shutil.rmtree(staging_dir)
                
        except Exception as e:
            logger.error(f"Backup creation failed: {str(e)}")
            return None
    
    def restore_backup(self, backup_id: str, target_dir: Optional[Path] = None) -> bool:
        """Restore from backup"""
        try:
            # Find backup metadata
            metadata = next((m for m in self.backup_metadata if m.backup_id == backup_id), None)
            if not metadata:
                logger.error(f"Backup not found: {backup_id}")
                return False
            
            # Find backup file
            backup_file = None
            for ext in ['.encrypted', '.tar.gz']:
                potential_file = self.config.backup_dir / f"{backup_id}{ext}"
                if potential_file.exists():
                    backup_file = potential_file
                    break
            
            if not backup_file:
                logger.error(f"Backup file not found: {backup_id}")
                return False
            
            # Verify integrity
            if not self.integrity_checker.verify_backup_integrity(backup_file, metadata.checksum):
                logger.error(f"Backup integrity check failed: {backup_id}")
                return False
            
            logger.info(f"Restoring backup: {backup_id}")
            
            # Create restore directory
            if target_dir is None:
                target_dir = Path(f"restore_{backup_id}_{int(time.time())}")
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Decrypt if necessary
            archive_file = backup_file
            if metadata.encryption_enabled and self.encryptor:
                decrypted_file = target_dir / f"{backup_id}_decrypted.tar.gz"
                if not self.encryptor.decrypt_file(backup_file, decrypted_file):
                    logger.error("Decryption failed")
                    return False
                archive_file = decrypted_file
            
            # Decompress
            if not self.compressor.decompress_archive(archive_file, target_dir):
                logger.error("Decompression failed")
                return False
            
            logger.info(f"Backup restored successfully to: {target_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            return False
    
    def list_backups(self) -> List[BackupMetadata]:
        """List all available backups"""
        # Sort by timestamp, newest first
        return sorted(self.backup_metadata, key=lambda x: x.timestamp, reverse=True)
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a specific backup"""
        try:
            # Find metadata
            metadata = next((m for m in self.backup_metadata if m.backup_id == backup_id), None)
            if not metadata:
                logger.error(f"Backup not found: {backup_id}")
                return False
            
            # Delete backup file
            for ext in ['.encrypted', '.tar.gz']:
                backup_file = self.config.backup_dir / f"{backup_id}{ext}"
                if backup_file.exists():
                    backup_file.unlink()
                    break
            
            # Remove from metadata
            self.backup_metadata = [m for m in self.backup_metadata if m.backup_id != backup_id]
            self._save_metadata()
            
            logger.info(f"Backup deleted: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup: {str(e)}")
            return False
    
    def _cleanup_old_backups(self):
        """Remove old backups based on retention policy"""
        try:
            now = time.time()
            cutoff_time = now - (self.config.retention_days * 24 * 3600)
            
            # Find old backups
            old_backups = [
                m for m in self.backup_metadata 
                if m.timestamp < cutoff_time
            ]
            
            # Keep at least one backup if retention would delete all
            if len(old_backups) >= len(self.backup_metadata):
                old_backups = old_backups[:-1]  # Keep the newest old backup
            
            # Delete old backups
            for metadata in old_backups:
                self.delete_backup(metadata.backup_id)
            
            # Enforce max backup count
            if len(self.backup_metadata) > self.config.max_backups:
                excess_count = len(self.backup_metadata) - self.config.max_backups
                # Sort by timestamp, delete oldest
                sorted_backups = sorted(self.backup_metadata, key=lambda x: x.timestamp)
                for metadata in sorted_backups[:excess_count]:
                    self.delete_backup(metadata.backup_id)
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
    
    def start_scheduled_backups(self):
        """Start automated backup scheduler"""
        if self.scheduler_running:
            logger.warning("Scheduler already running")
            return
        
        # Clear any existing jobs
        schedule.clear()
        
        # Schedule full backups
        if self.config.schedule_full == "daily":
            schedule.every().day.at("02:00").do(self._scheduled_full_backup)
        elif self.config.schedule_full == "weekly":
            schedule.every().week.do(self._scheduled_full_backup)
        elif self.config.schedule_full == "monthly":
            schedule.every(30).days.do(self._scheduled_full_backup)
        
        # Schedule incremental backups
        if self.config.schedule_incremental == "hourly":
            schedule.every().hour.do(self._scheduled_incremental_backup)
        elif self.config.schedule_incremental == "every_6h":
            schedule.every(6).hours.do(self._scheduled_incremental_backup)
        elif self.config.schedule_incremental == "every_12h":
            schedule.every(12).hours.do(self._scheduled_incremental_backup)
        
        # Start scheduler thread
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Backup scheduler started")
    
    def stop_scheduled_backups(self):
        """Stop automated backup scheduler"""
        self.scheduler_running = False
        schedule.clear()
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Backup scheduler stopped")
    
    def _run_scheduler(self):
        """Run the backup scheduler"""
        while self.scheduler_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _scheduled_full_backup(self):
        """Execute scheduled full backup"""
        logger.info("Executing scheduled full backup")
        backup_id = self.create_backup("full", "Scheduled full backup")
        if backup_id:
            logger.info(f"Scheduled full backup completed: {backup_id}")
        else:
            logger.error("Scheduled full backup failed")
    
    def _scheduled_incremental_backup(self):
        """Execute scheduled incremental backup"""
        logger.info("Executing scheduled incremental backup")
        backup_id = self.create_backup("incremental", "Scheduled incremental backup")
        if backup_id:
            logger.info(f"Scheduled incremental backup completed: {backup_id}")
        else:
            logger.error("Scheduled incremental backup failed")
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup system statistics"""
        if not self.backup_metadata:
            return {
                "total_backups": 0,
                "total_size": 0,
                "last_backup": None,
                "oldest_backup": None
            }
        
        total_size = sum(m.size_bytes for m in self.backup_metadata)
        latest_backup = max(self.backup_metadata, key=lambda x: x.timestamp)
        oldest_backup = min(self.backup_metadata, key=lambda x: x.timestamp)
        
        return {
            "total_backups": len(self.backup_metadata),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / 1024 / 1024,
            "last_backup": {
                "id": latest_backup.backup_id,
                "timestamp": latest_backup.timestamp,
                "type": latest_backup.backup_type
            },
            "oldest_backup": {
                "id": oldest_backup.backup_id,
                "timestamp": oldest_backup.timestamp,
                "type": oldest_backup.backup_type
            },
            "backup_types": {
                backup_type: len([m for m in self.backup_metadata if m.backup_type == backup_type])
                for backup_type in set(m.backup_type for m in self.backup_metadata)
            }
        }
    
    def create_disaster_recovery_kit(self) -> bool:
        """Create complete disaster recovery package"""
        try:
            logger.info("Creating disaster recovery kit")
            
            # Create full backup
            backup_id = self.create_backup("full", "Disaster recovery backup")
            if not backup_id:
                return False
            
            # Create recovery kit directory
            kit_dir = self.config.backup_dir / f"disaster_recovery_{int(time.time())}"
            kit_dir.mkdir()
            
            # Copy backup file
            backup_metadata = next(m for m in self.backup_metadata if m.backup_id == backup_id)
            backup_file = self.config.backup_dir / f"{backup_id}.{'encrypted' if backup_metadata.encryption_enabled else 'tar.gz'}"
            shutil.copy2(backup_file, kit_dir / backup_file.name)
            
            # Create recovery instructions
            instructions = {
                "recovery_instructions": {
                    "1": "Install Python 3.8+ and required dependencies",
                    "2": "Restore backup using: python -m src.core.backup_recovery restore <backup_id>",
                    "3": "Configure environment variables from restored .env file",
                    "4": "Verify configuration using: python -m src.core.environment_validator",
                    "5": "Start system using: python trading_bot_24x7.py"
                },
                "backup_metadata": asdict(backup_metadata),
                "system_requirements": {
                    "python_version": "3.8+",
                    "disk_space_mb": backup_metadata.size_bytes / 1024 / 1024 * 2,  # 2x for extraction
                    "memory_mb": 1024
                }
            }
            
            with open(kit_dir / "recovery_instructions.json", 'w') as f:
                json.dump(instructions, f, indent=2)
            
            # Create recovery script
            recovery_script = '''#!/usr/bin/env python3
"""
Disaster Recovery Script for Trading Bot
"""
import sys
import json
from pathlib import Path

def main():
    print("Trading Bot Disaster Recovery")
    print("=" * 40)
    
    # Load instructions
    instructions_file = Path(__file__).parent / "recovery_instructions.json"
    if instructions_file.exists():
        with open(instructions_file) as f:
            instructions = json.load(f)
        
        print("Recovery Instructions:")
        for step, instruction in instructions["recovery_instructions"].items():
            print(f"{step}. {instruction}")
        
        print(f"\nBackup ID: {instructions['backup_metadata']['backup_id']}")
        print(f"Backup Size: {instructions['backup_metadata']['size_bytes'] / 1024 / 1024:.2f} MB")
    
    print("\nFor detailed recovery procedures, see the documentation.")

if __name__ == "__main__":
    main()
'''
            
            with open(kit_dir / "recover.py", 'w') as f:
                f.write(recovery_script)
            
            logger.info(f"Disaster recovery kit created: {kit_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create disaster recovery kit: {str(e)}")
            return False

    def _init_metadata_db(self):
        """Initialize metadata database."""
        try:
            conn = sqlite3.connect(self.metadata_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backups (
                    backup_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    backup_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    file_count INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    compression_ratio REAL NOT NULL,
                    encryption_enabled BOOLEAN NOT NULL,
                    source_paths TEXT NOT NULL,
                    backup_file TEXT NOT NULL,
                    integrity_verified BOOLEAN DEFAULT FALSE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recovery_operations (
                    recovery_id TEXT PRIMARY KEY,
                    backup_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    recovery_path TEXT NOT NULL,
                    files_recovered INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (backup_id) REFERENCES backups (backup_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize metadata database: {e}")
            raise

    def _save_backup_metadata(self, metadata: BackupMetadata):
        """Save backup metadata to database."""
        try:
            conn = sqlite3.connect(self.metadata_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO backups (
                    backup_id, timestamp, backup_type, size_bytes, file_count,
                    checksum, compression_ratio, encryption_enabled, source_paths, backup_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.backup_id,
                metadata.timestamp,
                metadata.backup_type,
                metadata.size_bytes,
                metadata.file_count,
                metadata.checksum,
                metadata.compression_ratio,
                metadata.encryption_enabled,
                json.dumps(metadata.source_paths),
                metadata.backup_file
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save backup metadata: {e}")
            raise

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()

    def _compress_and_encrypt(self, source_file: Path, target_file: Path) -> Tuple[int, int]:
        """Compress and optionally encrypt a file."""
        original_size = source_file.stat().st_size
        
        with open(source_file, 'rb') as f_in:
            # Compress
            compressed_data = gzip.compress(f_in.read(), compresslevel=self.compression_level)
            
            # Encrypt if key is available
            if self.encryptor:
                encrypted_data = self.encryptor.encrypt(compressed_data)
                final_data = encrypted_data
            else:
                final_data = compressed_data
            
            # Write final data
            with open(target_file, 'wb') as f_out:
                f_out.write(final_data)
        
        compressed_size = target_file.stat().st_size
        return original_size, compressed_size

    def _decompress_and_decrypt(self, source_file: Path, target_file: Path) -> bool:
        """Decompress and decrypt a file."""
        try:
            with open(source_file, 'rb') as f_in:
                data = f_in.read()
                
                # Decrypt if key is available
                if self.encryptor:
                    decrypted_data = self.encryptor.decrypt(data)
                    final_data = decrypted_data
                else:
                    final_data = data
                
                # Decompress
                decompressed_data = gzip.decompress(final_data)
                
                # Write decompressed data
                with open(target_file, 'wb') as f_out:
                    f_out.write(decompressed_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to decompress/decrypt {source_file}: {e}")
            return False

    def create_full_backup(self, source_paths: List[str] = None, description: str = "") -> Optional[str]:
        """Create a full backup of specified paths."""
        try:
            if source_paths is None:
                source_paths = self.backup_paths
            
            # Generate backup ID
            timestamp = datetime.datetime.utcnow()
            backup_id = f"full_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info(f"Starting full backup: {backup_id}")
            self.backup_logger.info(f"FULL_BACKUP_START: {backup_id} - {description}")
            
            # Create temporary tar file
            temp_tar = self.backup_root / f"{backup_id}_temp.tar"
            backup_file = self.full_backups_dir / f"{backup_id}.backup"
            
            file_count = 0
            
            with tarfile.open(temp_tar, 'w') as tar:
                for source_path in source_paths:
                    if os.path.exists(source_path):
                        if os.path.isfile(source_path):
                            tar.add(source_path, arcname=source_path)
                            file_count += 1
                        elif os.path.isdir(source_path):
                            for root, dirs, files in os.walk(source_path):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    tar.add(file_path, arcname=file_path)
                                    file_count += 1
                        
                        self.logger.debug(f"Added to backup: {source_path}")
                    else:
                        self.logger.warning(f"Source path not found: {source_path}")
            
            # Compress and encrypt
            original_size, compressed_size = self._compress_and_encrypt(temp_tar, backup_file)
            
            # Calculate checksum
            checksum = self._calculate_file_checksum(backup_file)
            
            # Clean up temporary file
            temp_tar.unlink()
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp.isoformat(),
                backup_type="full",
                size_bytes=compressed_size,
                file_count=file_count,
                checksum=checksum,
                compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
                encryption_enabled=self.encryptor is not None,
                source_paths=[str(p) for p in self.backup_paths],
                backup_file=str(backup_file)
            )
            
            # Save metadata
            self._save_backup_metadata(metadata)
            
            self.logger.info(f"Full backup completed: {backup_id}")
            self.logger.info(f"  Files: {file_count}")
            self.logger.info(f"  Size: {compressed_size / 1024 / 1024:.2f} MB")
            self.logger.info(f"  Compression: {metadata.compression_ratio:.2f}x")
            
            self.backup_logger.info(f"FULL_BACKUP_COMPLETE: {backup_id} - {file_count} files, {compressed_size} bytes")
            
            return backup_id
            
        except Exception as e:
            self.logger.error(f"Failed to create full backup: {e}")
            self.backup_logger.error(f"FULL_BACKUP_FAILED: {e}")
            return None

    def create_incremental_backup(self, source_paths: List[str] = None, since_backup_id: str = None) -> Optional[str]:
        """Create an incremental backup (only changed files since last backup)."""
        try:
            if source_paths is None:
                source_paths = self.backup_paths
            
            # Find the last backup if not specified
            if since_backup_id is None:
                since_backup_id = self.get_latest_backup_id()
                if since_backup_id is None:
                    self.logger.warning("No previous backup found, creating full backup instead")
                    return self.create_full_backup(source_paths)
            
            # Get timestamp of last backup
            last_backup_timestamp = self._get_backup_timestamp(since_backup_id)
            if last_backup_timestamp is None:
                self.logger.error(f"Could not find backup timestamp for {since_backup_id}")
                return None
            
            # Generate backup ID
            timestamp = datetime.datetime.utcnow()
            backup_id = f"incr_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info(f"Starting incremental backup: {backup_id}")
            self.backup_logger.info(f"INCREMENTAL_BACKUP_START: {backup_id} since {since_backup_id}")
            
            # Find changed files
            changed_files = []
            last_backup_time = datetime.datetime.fromisoformat(last_backup_timestamp).timestamp()
            
            for source_path in source_paths:
                if os.path.exists(source_path):
                    if os.path.isfile(source_path):
                        if os.path.getmtime(source_path) > last_backup_time:
                            changed_files.append(source_path)
                    elif os.path.isdir(source_path):
                        for root, dirs, files in os.walk(source_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                if os.path.getmtime(file_path) > last_backup_time:
                                    changed_files.append(file_path)
            
            if not changed_files:
                self.logger.info("No changed files found, skipping incremental backup")
                return None
            
            # Create backup
            temp_tar = self.backup_root / f"{backup_id}_temp.tar"
            backup_file = self.incremental_backups_dir / f"{backup_id}.backup"
            
            with tarfile.open(temp_tar, 'w') as tar:
                for file_path in changed_files:
                    tar.add(file_path, arcname=file_path)
            
            # Compress and encrypt
            original_size, compressed_size = self._compress_and_encrypt(temp_tar, backup_file)
            
            # Calculate checksum
            checksum = self._calculate_file_checksum(backup_file)
            
            # Clean up temporary file
            temp_tar.unlink()
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp.isoformat(),
                backup_type="incremental",
                size_bytes=compressed_size,
                file_count=len(changed_files),
                checksum=checksum,
                compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
                encryption_enabled=self.encryptor is not None,
                source_paths=[str(p) for p in self.backup_paths],
                backup_file=str(backup_file)
            )
            
            # Save metadata
            self._save_backup_metadata(metadata)
            
            self.logger.info(f"Incremental backup completed: {backup_id}")
            self.logger.info(f"  Changed files: {len(changed_files)}")
            self.logger.info(f"  Size: {compressed_size / 1024 / 1024:.2f} MB")
            
            self.backup_logger.info(f"INCREMENTAL_BACKUP_COMPLETE: {backup_id} - {len(changed_files)} files")
            
            return backup_id
            
        except Exception as e:
            self.logger.error(f"Failed to create incremental backup: {e}")
            self.backup_logger.error(f"INCREMENTAL_BACKUP_FAILED: {e}")
            return None

    def verify_backup_integrity(self, backup_id: str) -> bool:
        """Verify integrity of a backup file."""
        try:
            # Get backup metadata
            metadata = self._get_backup_metadata(backup_id)
            if metadata is None:
                self.logger.error(f"Backup metadata not found: {backup_id}")
                return False
            
            backup_file = Path(metadata['backup_file'])
            if not backup_file.exists():
                self.logger.error(f"Backup file not found: {backup_file}")
                return False
            
            # Verify checksum
            current_checksum = self._calculate_file_checksum(backup_file)
            if current_checksum != metadata['checksum']:
                self.logger.error(f"Backup integrity check failed: {backup_id}")
                self.logger.error(f"Expected: {metadata['checksum']}")
                self.logger.error(f"Actual: {current_checksum}")
                return False
            
            # Update integrity status in database
            conn = sqlite3.connect(self.metadata_db)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE backups SET integrity_verified = ? WHERE backup_id = ?",
                (True, backup_id)
            )
            conn.commit()
            conn.close()
            
            self.logger.info(f"Backup integrity verified: {backup_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to verify backup integrity: {e}")
            return False

    def restore_backup(self, backup_id: str, restore_path: str = "restored", selective_paths: List[str] = None) -> bool:
        """Restore files from a backup."""
        try:
            # Get backup metadata
            metadata = self._get_backup_metadata(backup_id)
            if metadata is None:
                self.logger.error(f"Backup metadata not found: {backup_id}")
                return False
            
            backup_file = Path(metadata['backup_file'])
            if not backup_file.exists():
                self.logger.error(f"Backup file not found: {backup_file}")
                return False
            
            # Verify integrity first
            if not self.verify_backup_integrity(backup_id):
                self.logger.error(f"Backup integrity check failed, aborting restore: {backup_id}")
                return False
            
            self.logger.info(f"Starting restore from backup: {backup_id}")
            self.backup_logger.info(f"RESTORE_START: {backup_id} to {restore_path}")
            
            # Create restore directory
            restore_dir = Path(restore_path)
            restore_dir.mkdir(parents=True, exist_ok=True)
            
            # Decompress and decrypt to temporary file
            temp_tar = self.backup_root / f"restore_{backup_id}_temp.tar"
            
            if not self._decompress_and_decrypt(backup_file, temp_tar):
                self.logger.error(f"Failed to decompress/decrypt backup: {backup_id}")
                return False
            
            # Extract files
            files_restored = 0
            
            with tarfile.open(temp_tar, 'r') as tar:
                for member in tar.getmembers():
                    if selective_paths is None or any(member.name.startswith(path) for path in selective_paths):
                        tar.extract(member, restore_dir)
                        files_restored += 1
            
            # Clean up temporary file
            temp_tar.unlink()
            
            # Log recovery operation
            recovery_id = f"recovery_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self._log_recovery_operation(recovery_id, backup_id, restore_path, files_restored)
            
            self.logger.info(f"Restore completed: {backup_id}")
            self.logger.info(f"  Files restored: {files_restored}")
            self.logger.info(f"  Restore path: {restore_dir}")
            
            self.backup_logger.info(f"RESTORE_COMPLETE: {backup_id} - {files_restored} files to {restore_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            self.backup_logger.error(f"RESTORE_FAILED: {backup_id} - {e}")
            return False

    def _get_backup_metadata(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Get backup metadata from database."""
        try:
            conn = sqlite3.connect(self.metadata_db)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM backups WHERE backup_id = ?",
                (backup_id,)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get backup metadata: {e}")
            return None

    def _get_backup_timestamp(self, backup_id: str) -> Optional[str]:
        """Get backup timestamp."""
        metadata = self._get_backup_metadata(backup_id)
        return metadata['timestamp'] if metadata else None

    def _log_recovery_operation(self, recovery_id: str, backup_id: str, restore_path: str, files_recovered: int):
        """Log recovery operation to database."""
        try:
            conn = sqlite3.connect(self.metadata_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO recovery_operations (
                    recovery_id, backup_id, timestamp, status, recovery_path, files_recovered
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                recovery_id,
                backup_id,
                datetime.datetime.utcnow().isoformat(),
                'completed',
                restore_path,
                files_recovered
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log recovery operation: {e}")

    def get_latest_backup_id(self) -> Optional[str]:
        """Get the ID of the most recent backup."""
        try:
            conn = sqlite3.connect(self.metadata_db)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT backup_id FROM backups ORDER BY timestamp DESC LIMIT 1"
            )
            
            row = cursor.fetchone()
            conn.close()
            
            return row[0] if row else None
            
        except Exception as e:
            self.logger.error(f"Failed to get latest backup ID: {e}")
            return None

    def list_backups(self) -> List[RecoveryPoint]:
        """List all available recovery points."""
        try:
            conn = sqlite3.connect(self.metadata_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT backup_id, timestamp, backup_type, size_bytes, integrity_verified 
                FROM backups ORDER BY timestamp DESC
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            recovery_points = []
            for row in rows:
                recovery_points.append(RecoveryPoint(
                    backup_id=row[0],
                    timestamp=row[1],
                    backup_type=row[2],
                    description=f"{row[2].title()} backup",
                    integrity_verified=bool(row[4]),
                    size_mb=row[3] / 1024 / 1024
                ))
            
            return recovery_points
            
        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")
            return []

    def cleanup_old_backups(self, retention_days: int = None) -> int:
        """Remove old backups based on retention policy."""
        try:
            if retention_days is None:
                retention_days = self.retention_days
            
            cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=retention_days)
            
            conn = sqlite3.connect(self.metadata_db)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT backup_id, backup_file FROM backups WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            
            old_backups = cursor.fetchall()
            
            cleaned_count = 0
            for backup_id, backup_file in old_backups:
                try:
                    # Remove backup file
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                    
                    # Remove from database
                    cursor.execute("DELETE FROM backups WHERE backup_id = ?", (backup_id,))
                    cleaned_count += 1
                    
                    self.logger.info(f"Cleaned up old backup: {backup_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to clean up backup {backup_id}: {e}")
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleanup completed: {cleaned_count} old backups removed")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {e}")
            return 0

    def start_automated_backups(self, full_backup_schedule: str = "daily", incremental_schedule: str = "hourly"):
        """Start automated backup scheduling."""
        try:
            if self.scheduler_running:
                self.logger.warning("Automated backups already running")
                return
            
            # Schedule full backups
            if full_backup_schedule == "daily":
                schedule.every().day.at("02:00").do(self._scheduled_full_backup)
            elif full_backup_schedule == "weekly":
                schedule.every().sunday.at("02:00").do(self._scheduled_full_backup)
            
            # Schedule incremental backups
            if incremental_schedule == "hourly":
                schedule.every().hour.do(self._scheduled_incremental_backup)
            elif incremental_schedule == "4hours":
                schedule.every(4).hours.do(self._scheduled_incremental_backup)
            
            # Schedule cleanup
            schedule.every().day.at("03:00").do(self._scheduled_cleanup)
            
            # Start scheduler thread
            self.scheduler_running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            self.logger.info("Automated backup scheduling started")
            self.logger.info(f"  Full backups: {full_backup_schedule}")
            self.logger.info(f"  Incremental backups: {incremental_schedule}")
            
        except Exception as e:
            self.logger.error(f"Failed to start automated backups: {e}")

    def stop_automated_backups(self):
        """Stop automated backup scheduling."""
        self.scheduler_running = False
        schedule.clear()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        self.logger.info("Automated backup scheduling stopped")

    def _run_scheduler(self):
        """Run the backup scheduler."""
        while self.scheduler_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def _scheduled_full_backup(self):
        """Perform scheduled full backup."""
        try:
            backup_id = self.create_full_backup(description="Scheduled full backup")
            if backup_id:
                self.logger.info(f"Scheduled full backup completed: {backup_id}")
            else:
                self.logger.error("Scheduled full backup failed")
        except Exception as e:
            self.logger.error(f"Scheduled full backup error: {e}")

    def _scheduled_incremental_backup(self):
        """Perform scheduled incremental backup."""
        try:
            backup_id = self.create_incremental_backup()
            if backup_id:
                self.logger.info(f"Scheduled incremental backup completed: {backup_id}")
            else:
                self.logger.debug("Scheduled incremental backup skipped (no changes)")
        except Exception as e:
            self.logger.error(f"Scheduled incremental backup error: {e}")

    def _scheduled_cleanup(self):
        """Perform scheduled cleanup."""
        try:
            cleaned = self.cleanup_old_backups()
            self.logger.info(f"Scheduled cleanup completed: {cleaned} backups removed")
        except Exception as e:
            self.logger.error(f"Scheduled cleanup error: {e}")

    def create_disaster_recovery_kit(self, output_path: str) -> bool:
        """Create a disaster recovery kit with scripts and documentation."""
        try:
            kit_dir = Path(output_path)
            kit_dir.mkdir(parents=True, exist_ok=True)
            
            # Create recovery script
            recovery_script = kit_dir / "disaster_recovery.py"
            script_content = f'''#!/usr/bin/env python3
"""
Disaster Recovery Script
Generated: {datetime.datetime.utcnow().isoformat()}

This script helps recover your trading bot from backups.
"""

import os
import sys
import json
from pathlib import Path

def main():
    print("🚨 DISASTER RECOVERY TOOL")
    print("=" * 50)
    
    # Check if backup directory exists
    backup_dir = Path("{self.backup_root}")
    if not backup_dir.exists():
        print(f"❌ Backup directory not found: {{backup_dir}}")
        sys.exit(1)
    
    print(f"✅ Backup directory found: {{backup_dir}}")
    
    # List available backups
    print("\nAvailable backups:")
    from src.core.backup_recovery import BackupRecoverySystem
    
    brs = BackupRecoverySystem()
    backups = brs.list_backups()
    
    if not backups:
        print("❌ No backups found!")
        sys.exit(1)
    
    for i, backup in enumerate(backups):
        status = "✅ Verified" if backup.integrity_verified else "⚠️  Not verified"
        print(f"  {i+1}. {backup.backup_id} - {backup.backup_type} ({backup.size_mb:.1f} MB) {status}")
    
    # Get user selection
    while True:
        try:
            choice = int(input("\nSelect backup to restore (number): ")) - 1
            if 0 <= choice < len(backups):
                selected_backup = backups[choice]
                break
            else:
                print("Invalid selection!")
        except ValueError:
            print("Please enter a number!")
    
    # Confirm recovery
    print(f"\nSelected backup: {{selected_backup.backup_id}}")
    confirm = input("⚠️  This will restore files. Continue? (yes/no): ")
    
    if confirm.lower() == 'yes':
        restore_path = input("Enter restore path (default: ./restored): ") or "./restored"
        
        print("\n🔄 Starting recovery...")
        success = brs.restore_backup(selected_backup.backup_id, restore_path)
        
        if success:
            print(f"✅ Recovery completed! Files restored to: {{restore_path}}")
            print("\nNext steps:")
            print("1. Check restored files")
            print("2. Update configuration if needed")
            print("3. Restart your trading bot")
        else:
            print("❌ Recovery failed! Check logs for details.")
    else:
        print("Recovery cancelled.")

if __name__ == "__main__":
    main()
'''
            
            with open(recovery_script, 'w') as f:
                f.write(script_content)
            
            # Make script executable
            os.chmod(recovery_script, 0o755)
            
            # Create documentation
            readme = kit_dir / "DISASTER_RECOVERY_README.md"
            readme_content = f'''# Disaster Recovery Kit

Generated: {datetime.datetime.utcnow().isoformat()}

## Quick Recovery Steps

1. **Immediate Actions**
   - Stop the trading bot if it's running
   - Assess the damage/data loss
   - Check if backup directory is accessible

2. **Run Recovery Tool**
   ```bash
   python disaster_recovery.py
   ```

3. **Manual Recovery** (if tool fails)
   - Navigate to backup directory: `{self.backup_root}`
   - Find latest backup in `full/` or `incremental/`
   - Extract backup manually

## Backup Information

- **Backup Root**: `{self.backup_root}`
- **Full Backups**: `{self.full_backups_dir}`
- **Incremental Backups**: `{self.incremental_backups_dir}`
- **Metadata**: `{self.metadata_dir}`

## Important Files to Recover

1. **Configuration Files**
   - `config/config.yaml`
   - `.env` (environment variables)
   - `config/trading_config.json`

2. **Data Files**
   - `data/trading_data.db`
   - `data/secrets/` (if using secrets manager)

3. **Logs** (for analysis)
   - `logs/trading_bot.log`
   - `logs/errors.log`

## Emergency Contacts

- Review your monitoring alerts
- Check Discord/Telegram notifications
- Analyze trading performance impact

## Prevention for Future

1. **Test Recovery Process** regularly
2. **Monitor Backup Health** daily
3. **Keep Multiple Backup Locations**
4. **Document Critical Procedures**

---
*This kit was automatically generated by the Backup Recovery System*
'''
            
            with open(readme, 'w') as f:
                f.write(readme_content)
            
            # Create backup verification script
            verify_script = kit_dir / "verify_backups.py"
            verify_content = '''#!/usr/bin/env python3
"""
Backup Verification Script
"""

from src.core.backup_recovery import BackupRecoverySystem

def main():
    print("🔍 BACKUP VERIFICATION")
    print("=" * 30)
    
    brs = BackupRecoverySystem()
    backups = brs.list_backups()
    
    if not backups:
        print("❌ No backups found!")
        return
    
    verified_count = 0
    failed_count = 0
    
    for backup in backups:
        print(f"Verifying {backup.backup_id}...", end=" ")
        
        if brs.verify_backup_integrity(backup.backup_id):
            print("✅ OK")
            verified_count += 1
        else:
            print("❌ FAILED")
            failed_count += 1
    
    print("\nResults: {verified_count} verified, {failed_count} failed")
    
    if failed_count > 0:
        print("⚠️  Some backups failed verification!")
        print("Consider creating new backups immediately.")

if __name__ == "__main__":
    main()
'''
            
            with open(verify_script, 'w') as f:
                f.write(verify_content)
            
            os.chmod(verify_script, 0o755)
            
            self.logger.info(f"Disaster recovery kit created: {kit_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create disaster recovery kit: {e}")
            return False

def main():
    """CLI interface for backup and recovery operations."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Backup and Recovery System")
    parser.add_argument("action", choices=["backup", "restore", "list", "verify", "cleanup", "schedule", "disaster-kit"])
    parser.add_argument("--type", choices=["full", "incremental"], default="full", help="Backup type")
    parser.add_argument("--backup-id", help="Backup ID for restore/verify operations")
    parser.add_argument("--restore-path", default="restored", help="Path to restore files")
    parser.add_argument("--sources", nargs="+", help="Source paths to backup")
    parser.add_argument("--retention-days", type=int, default=30, help="Retention period for cleanup")
    parser.add_argument("--output-path", default="disaster_recovery_kit", help="Output path for disaster kit")
    
    args = parser.parse_args()
    
    # Initialize system
    brs = BackupRecoverySystem()
    
    if args.action == "backup":
        if args.type == "full":
            backup_id = brs.create_full_backup(args.sources)
        else:
            backup_id = brs.create_incremental_backup(args.sources)
        
        if backup_id:
            print(f"✅ Backup created: {backup_id}")
        else:
            print("❌ Backup failed")
            sys.exit(1)
    
    elif args.action == "restore":
        if not args.backup_id:
            print("Error: --backup-id required for restore")
            sys.exit(1)
        
        success = brs.restore_backup(args.backup_id, args.restore_path)
        print(f"{'✅' if success else '❌'} Restore {'completed' if success else 'failed'}")
    
    elif args.action == "list":
        backups = brs.list_backups()
        print(f"Found {len(backups)} backups:")
        
        for i, backup in enumerate(backups):
            status = "✅ Verified" if backup.integrity_verified else "⚠️  Not verified"
            print(f"  • {i+1}. {backup.backup_id} - {backup.backup_type} ({backup.size_mb:.1f} MB) {status}")
    
    elif args.action == "verify":
        if args.backup_id:
            success = brs.verify_backup_integrity(args.backup_id)
            print(f"{'✅' if success else '❌'} Verification {'passed' if success else 'failed'}")
        else:
            # Verify all backups
            backups = brs.list_backups()
            verified = sum(1 for backup in backups if brs.verify_backup_integrity(backup.backup_id))
            print(f"Verified {verified}/{len(backups)} backups")
    
    elif args.action == "cleanup":
        cleaned = brs.cleanup_old_backups(args.retention_days)
        print(f"✅ Cleaned up {cleaned} old backups")
    
    elif args.action == "schedule":
        brs.start_automated_backups()
        print("✅ Automated backups started")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            brs.stop_automated_backups()
            print("\n✅ Automated backups stopped")
    
    elif args.action == "disaster-kit":
        success = brs.create_disaster_recovery_kit(args.output_path)
        print(f"{'✅' if success else '❌'} Disaster recovery kit {'created' if success else 'failed'}")

if __name__ == "__main__":
    main() 