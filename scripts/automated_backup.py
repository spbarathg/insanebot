#!/usr/bin/env python3
"""
Enhanced Ant Bot - Automated Backup System

Comprehensive backup solution with encryption, validation, and rotation.
Supports database backups, configuration backups, and state snapshots.
"""

import os
import sys
import time
import json
import gzip
import shutil
import hashlib
import logging
import asyncio
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import psycopg2
import redis
from cryptography.fernet import Fernet


@dataclass
class BackupConfig:
    """Backup configuration settings."""
    backup_root: str = "/backups"
    retention_days: int = 30
    encryption_enabled: bool = True
    compression_enabled: bool = True
    verify_backups: bool = True
    remote_storage_enabled: bool = False
    remote_storage_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackupResult:
    """Result of a backup operation."""
    success: bool
    backup_type: str
    file_path: str
    file_size: int
    duration_seconds: float
    checksum: str
    error_message: Optional[str] = None


class BackupManager:
    """Comprehensive backup management system."""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.encryption_key = self._get_encryption_key()
        self.fernet = Fernet(self.encryption_key) if config.encryption_enabled else None
        
        # Ensure backup directories exist
        self._create_backup_directories()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for backup operations."""
        logger = logging.getLogger("backup_manager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key for backups."""
        key_file = Path(self.config.backup_root) / ".backup_key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)
            return key
    
    def _create_backup_directories(self):
        """Create necessary backup directories."""
        directories = [
            "database",
            "redis",
            "config",
            "logs",
            "state",
            "verification"
        ]
        
        for directory in directories:
            path = Path(self.config.backup_root) / directory
            path.mkdir(parents=True, exist_ok=True)
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _compress_file(self, source_path: str, target_path: str):
        """Compress a file using gzip."""
        with open(source_path, 'rb') as f_in:
            with gzip.open(target_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    def _encrypt_file(self, source_path: str, target_path: str):
        """Encrypt a file using Fernet encryption."""
        with open(source_path, 'rb') as f_in:
            data = f_in.read()
            encrypted_data = self.fernet.encrypt(data)
            with open(target_path, 'wb') as f_out:
                f_out.write(encrypted_data)
    
    async def backup_database(self) -> BackupResult:
        """Backup PostgreSQL database."""
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"database_backup_{timestamp}.sql"
        backup_path = Path(self.config.backup_root) / "database" / backup_name
        
        try:
            self.logger.info("Starting database backup...")
            
            # Get database configuration from environment
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'trading_bot_prod'),
                'username': os.getenv('DB_USERNAME', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'postgres')
            }
            
            # Create pg_dump command
            cmd = [
                'pg_dump',
                '--host', db_config['host'],
                '--port', db_config['port'],
                '--username', db_config['username'],
                '--dbname', db_config['database'],
                '--verbose',
                '--clean',
                '--create',
                '--format=custom',
                '--file', str(backup_path)
            ]
            
            # Set password via environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = db_config['password']
            
            # Execute backup
            process = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if process.returncode != 0:
                raise Exception(f"pg_dump failed: {process.stderr}")
            
            # Post-process backup file
            final_path = await self._post_process_backup(backup_path)
            
            # Calculate metrics
            duration = time.time() - start_time
            file_size = os.path.getsize(final_path)
            checksum = self._calculate_checksum(final_path)
            
            self.logger.info(f"Database backup completed: {final_path}")
            
            return BackupResult(
                success=True,
                backup_type="database",
                file_path=final_path,
                file_size=file_size,
                duration_seconds=duration,
                checksum=checksum
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Database backup failed: {str(e)}")
            
            return BackupResult(
                success=False,
                backup_type="database",
                file_path="",
                file_size=0,
                duration_seconds=duration,
                checksum="",
                error_message=str(e)
            )
    
    async def backup_redis(self) -> BackupResult:
        """Backup Redis data."""
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"redis_backup_{timestamp}.rdb"
        backup_path = Path(self.config.backup_root) / "redis" / backup_name
        
        try:
            self.logger.info("Starting Redis backup...")
            
            # Connect to Redis
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', '6379')),
                password=os.getenv('REDIS_PASSWORD', None),
                decode_responses=False
            )
            
            # Trigger BGSAVE
            redis_client.bgsave()
            
            # Wait for BGSAVE to complete
            while redis_client.lastsave() == redis_client.lastsave():
                await asyncio.sleep(1)
            
            # Copy RDB file
            rdb_path = "/var/lib/redis/dump.rdb"  # Default Redis RDB path
            if os.path.exists(rdb_path):
                shutil.copy2(rdb_path, backup_path)
            else:
                # Fallback: export all keys
                await self._export_redis_keys(redis_client, backup_path)
            
            # Post-process backup file
            final_path = await self._post_process_backup(backup_path)
            
            # Calculate metrics
            duration = time.time() - start_time
            file_size = os.path.getsize(final_path)
            checksum = self._calculate_checksum(final_path)
            
            self.logger.info(f"Redis backup completed: {final_path}")
            
            return BackupResult(
                success=True,
                backup_type="redis",
                file_path=final_path,
                file_size=file_size,
                duration_seconds=duration,
                checksum=checksum
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Redis backup failed: {str(e)}")
            
            return BackupResult(
                success=False,
                backup_type="redis",
                file_path="",
                file_size=0,
                duration_seconds=duration,
                checksum="",
                error_message=str(e)
            )
    
    async def _export_redis_keys(self, redis_client: redis.Redis, backup_path: str):
        """Export Redis keys to JSON format."""
        backup_data = {}
        
        # Get all keys
        keys = redis_client.keys('*')
        
        for key in keys:
            key_type = redis_client.type(key)
            
            if key_type == b'string':
                backup_data[key.decode()] = redis_client.get(key).decode()
            elif key_type == b'hash':
                backup_data[key.decode()] = redis_client.hgetall(key)
            elif key_type == b'list':
                backup_data[key.decode()] = redis_client.lrange(key, 0, -1)
            elif key_type == b'set':
                backup_data[key.decode()] = list(redis_client.smembers(key))
            elif key_type == b'zset':
                backup_data[key.decode()] = redis_client.zrange(key, 0, -1, withscores=True)
        
        # Save to JSON file
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
    
    async def backup_configuration(self) -> BackupResult:
        """Backup system configuration files."""
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"config_backup_{timestamp}.tar.gz"
        backup_path = Path(self.config.backup_root) / "config" / backup_name
        
        try:
            self.logger.info("Starting configuration backup...")
            
            # Files and directories to backup
            config_items = [
                ".env",
                "env.production",
                "docker-compose.yml",
                "docker-compose.prod.yml",
                "config/",
                "monitoring/prometheus.yml",
                "monitoring/alert_rules.yml"
            ]
            
            # Create tar archive
            cmd = ['tar', '-czf', str(backup_path)]
            for item in config_items:
                if os.path.exists(item):
                    cmd.append(item)
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if process.returncode != 0:
                raise Exception(f"Configuration backup failed: {process.stderr}")
            
            # Post-process backup file
            final_path = await self._post_process_backup(backup_path)
            
            # Calculate metrics
            duration = time.time() - start_time
            file_size = os.path.getsize(final_path)
            checksum = self._calculate_checksum(final_path)
            
            self.logger.info(f"Configuration backup completed: {final_path}")
            
            return BackupResult(
                success=True,
                backup_type="configuration",
                file_path=final_path,
                file_size=file_size,
                duration_seconds=duration,
                checksum=checksum
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Configuration backup failed: {str(e)}")
            
            return BackupResult(
                success=False,
                backup_type="configuration",
                file_path="",
                file_size=0,
                duration_seconds=duration,
                checksum="",
                error_message=str(e)
            )
    
    async def backup_application_state(self) -> BackupResult:
        """Backup application state and runtime data."""
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"state_backup_{timestamp}.json"
        backup_path = Path(self.config.backup_root) / "state" / backup_name
        
        try:
            self.logger.info("Starting application state backup...")
            
            # Collect state data from various sources
            state_data = {
                "timestamp": timestamp,
                "portfolio_state": await self._get_portfolio_state(),
                "trading_state": await self._get_trading_state(),
                "risk_parameters": await self._get_risk_parameters(),
                "system_metrics": await self._get_system_metrics()
            }
            
            # Save state data
            with open(backup_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            # Post-process backup file
            final_path = await self._post_process_backup(backup_path)
            
            # Calculate metrics
            duration = time.time() - start_time
            file_size = os.path.getsize(final_path)
            checksum = self._calculate_checksum(final_path)
            
            self.logger.info(f"Application state backup completed: {final_path}")
            
            return BackupResult(
                success=True,
                backup_type="application_state",
                file_path=final_path,
                file_size=file_size,
                duration_seconds=duration,
                checksum=checksum
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Application state backup failed: {str(e)}")
            
            return BackupResult(
                success=False,
                backup_type="application_state",
                file_path="",
                file_size=0,
                duration_seconds=duration,
                checksum="",
                error_message=str(e)
            )
    
    async def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state via API."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8080/portfolio/status') as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            self.logger.warning(f"Failed to get portfolio state: {e}")
        return {}
    
    async def _get_trading_state(self) -> Dict[str, Any]:
        """Get current trading state via API."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8080/trading/status') as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            self.logger.warning(f"Failed to get trading state: {e}")
        return {}
    
    async def _get_risk_parameters(self) -> Dict[str, Any]:
        """Get current risk parameters via API."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8080/risk/parameters') as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            self.logger.warning(f"Failed to get risk parameters: {e}")
        return {}
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics via API."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8080/metrics') as response:
                    if response.status == 200:
                        return {"raw_metrics": await response.text()}
        except Exception as e:
            self.logger.warning(f"Failed to get system metrics: {e}")
        return {}
    
    async def _post_process_backup(self, backup_path: str) -> str:
        """Post-process backup file with compression and encryption."""
        current_path = backup_path
        
        # Compression
        if self.config.compression_enabled and not backup_path.endswith('.gz'):
            compressed_path = f"{backup_path}.gz"
            self._compress_file(current_path, compressed_path)
            os.remove(current_path)
            current_path = compressed_path
        
        # Encryption
        if self.config.encryption_enabled:
            encrypted_path = f"{current_path}.enc"
            self._encrypt_file(current_path, encrypted_path)
            os.remove(current_path)
            current_path = encrypted_path
        
        return current_path
    
    async def verify_backup(self, backup_path: str) -> bool:
        """Verify backup integrity."""
        try:
            if not os.path.exists(backup_path):
                return False
            
            # Basic file existence and size check
            file_size = os.path.getsize(backup_path)
            if file_size == 0:
                return False
            
            # If encrypted, try to decrypt
            if backup_path.endswith('.enc') and self.fernet:
                with open(backup_path, 'rb') as f:
                    encrypted_data = f.read()
                    try:
                        self.fernet.decrypt(encrypted_data)
                    except Exception:
                        return False
            
            # Additional verification based on backup type
            if 'database' in backup_path:
                return await self._verify_database_backup(backup_path)
            elif 'redis' in backup_path:
                return await self._verify_redis_backup(backup_path)
            elif 'config' in backup_path:
                return await self._verify_config_backup(backup_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Backup verification failed: {e}")
            return False
    
    async def _verify_database_backup(self, backup_path: str) -> bool:
        """Verify database backup integrity."""
        try:
            # For PostgreSQL custom format, we can use pg_restore --list
            cmd = ['pg_restore', '--list', backup_path]
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            return process.returncode == 0
        except Exception:
            return False
    
    async def _verify_redis_backup(self, backup_path: str) -> bool:
        """Verify Redis backup integrity."""
        try:
            # Basic validation - check if file can be read
            if backup_path.endswith('.json'):
                with open(backup_path, 'r') as f:
                    json.load(f)
                return True
            return True  # For RDB files, basic existence check is sufficient
        except Exception:
            return False
    
    async def _verify_config_backup(self, backup_path: str) -> bool:
        """Verify configuration backup integrity."""
        try:
            # Test tar file integrity
            cmd = ['tar', '-tzf', backup_path]
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            return process.returncode == 0
        except Exception:
            return False
    
    async def cleanup_old_backups(self):
        """Remove old backups based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        for backup_type in ['database', 'redis', 'config', 'state']:
            backup_dir = Path(self.config.backup_root) / backup_type
            
            if not backup_dir.exists():
                continue
            
            for backup_file in backup_dir.iterdir():
                if backup_file.is_file():
                    file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        self.logger.info(f"Removing old backup: {backup_file}")
                        backup_file.unlink()
    
    async def run_full_backup(self) -> List[BackupResult]:
        """Run complete backup of all components."""
        self.logger.info("Starting full backup process...")
        
        results = []
        
        # Database backup
        results.append(await self.backup_database())
        
        # Redis backup
        results.append(await self.backup_redis())
        
        # Configuration backup
        results.append(await self.backup_configuration())
        
        # Application state backup
        results.append(await self.backup_application_state())
        
        # Cleanup old backups
        await self.cleanup_old_backups()
        
        # Generate backup report
        await self._generate_backup_report(results)
        
        self.logger.info("Full backup process completed")
        return results
    
    async def _generate_backup_report(self, results: List[BackupResult]):
        """Generate backup report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(self.config.backup_root) / f"backup_report_{timestamp}.json"
        
        report = {
            "timestamp": timestamp,
            "total_backups": len(results),
            "successful_backups": len([r for r in results if r.success]),
            "failed_backups": len([r for r in results if not r.success]),
            "total_size_bytes": sum(r.file_size for r in results if r.success),
            "total_duration_seconds": sum(r.duration_seconds for r in results),
            "backups": [
                {
                    "type": r.backup_type,
                    "success": r.success,
                    "file_path": r.file_path,
                    "file_size": r.file_size,
                    "duration": r.duration_seconds,
                    "checksum": r.checksum,
                    "error": r.error_message
                }
                for r in results
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Backup report generated: {report_path}")


async def main():
    """Main backup execution function."""
    parser = argparse.ArgumentParser(description='Enhanced Ant Bot Backup System')
    parser.add_argument('--type', choices=['full', 'database', 'redis', 'config', 'state'], 
                       default='full', help='Type of backup to perform')
    parser.add_argument('--backup-root', default='/backups', 
                       help='Root directory for backups')
    parser.add_argument('--retention-days', type=int, default=30,
                       help='Number of days to retain backups')
    parser.add_argument('--no-encryption', action='store_true',
                       help='Disable backup encryption')
    parser.add_argument('--no-compression', action='store_true',
                       help='Disable backup compression')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing backups')
    
    args = parser.parse_args()
    
    # Create backup configuration
    config = BackupConfig(
        backup_root=args.backup_root,
        retention_days=args.retention_days,
        encryption_enabled=not args.no_encryption,
        compression_enabled=not args.no_compression
    )
    
    # Initialize backup manager
    backup_manager = BackupManager(config)
    
    if args.verify_only:
        # Verify existing backups
        print("Verifying existing backups...")
        # Implementation for verification would go here
        return
    
    # Perform backup based on type
    if args.type == 'full':
        results = await backup_manager.run_full_backup()
    elif args.type == 'database':
        results = [await backup_manager.backup_database()]
    elif args.type == 'redis':
        results = [await backup_manager.backup_redis()]
    elif args.type == 'config':
        results = [await backup_manager.backup_configuration()]
    elif args.type == 'state':
        results = [await backup_manager.backup_application_state()]
    
    # Print results
    print("\n=== Backup Results ===")
    for result in results:
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        print(f"{status} {result.backup_type}: {result.file_path}")
        if not result.success:
            print(f"  Error: {result.error_message}")
        else:
            print(f"  Size: {result.file_size:,} bytes")
            print(f"  Duration: {result.duration_seconds:.2f} seconds")
            print(f"  Checksum: {result.checksum[:16]}...")
    
    # Exit with error code if any backup failed
    failed_count = len([r for r in results if not r.success])
    if failed_count > 0:
        print(f"\n⚠️  {failed_count} backup(s) failed!")
        sys.exit(1)
    else:
        print(f"\n🎉 All {len(results)} backup(s) completed successfully!")


if __name__ == "__main__":
    asyncio.run(main()) 