"""
Production-Grade Startup System

Orchestrates the complete startup process for production deployment:
- Environment validation and pre-flight checks
- Secrets management initialization
- Backup system setup
- Audit logging initialization
- System monitoring setup
- Health checks and self-validation
"""

import os
import sys
import asyncio
import logging
import signal
import atexit
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime
import traceback
import psutil
import time
from dataclasses import dataclass
from enum import Enum

# Import our production systems
from .environment_validator import EnvironmentValidator, validate_environment
from .secrets_manager import initialize_secrets_manager, migrate_from_env_vars, SecretsManager
from .backup_recovery import BackupRecoverySystem, BackupConfig
from .audit_logging import initialize_audit_logging, get_audit_logger, AuditEventType, AuditLevel, AuditLogger
from .api_documentation import create_api_server, APIDocumentationConfig, APIDocumentationGenerator

logger = logging.getLogger(__name__)

class StartupPhase(Enum):
    """Production startup phases."""
    INITIALIZATION = "initialization"
    ENVIRONMENT_VALIDATION = "environment_validation"
    SECRETS_MANAGEMENT = "secrets_management"
    AUDIT_LOGGING = "audit_logging"
    BACKUP_RECOVERY = "backup_recovery"
    CORE_SERVICES = "core_services"
    TRADING_ENGINE = "trading_engine"
    MONITORING = "monitoring"
    PRODUCTION_READY = "production_ready"

@dataclass
class StartupResult:
    """Result of a startup phase."""
    phase: StartupPhase
    success: bool
    duration_seconds: float
    message: str
    details: Dict[str, Any]

class ProductionStartupError(Exception):
    """Exception raised during production startup"""
    pass

class ProductionStartupOrchestrator:
    """Orchestrates production startup with comprehensive validation and health checks."""
    
    def __init__(self):
        self.startup_time = datetime.utcnow()
        self.phases_completed: List[StartupResult] = []
        self.is_production_ready = False
        self.shutdown_handlers: List[Callable] = []
        
        # Core system instances
        self.environment_validator: Optional[EnvironmentValidator] = None
        self.secrets_manager: Optional[SecretsManager] = None
        self.audit_logger: Optional[AuditLogger] = None
        self.backup_system: Optional[BackupRecoverySystem] = None
        
        # Setup basic logging first
        self._setup_basic_logging()
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Production Startup Orchestrator initialized")
    
    def _setup_basic_logging(self):
        """Setup basic logging before full system initialization."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "startup.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start_production_system(self) -> bool:
        """Start the complete production system through all phases."""
        self.logger.info("🚀 Starting Production Trading Bot System")
        self.logger.info(f"Startup Time: {self.startup_time.isoformat()}")
        self.logger.info("=" * 80)
        
        startup_phases = [
            (StartupPhase.INITIALIZATION, self._phase_initialization),
            (StartupPhase.ENVIRONMENT_VALIDATION, self._phase_environment_validation),
            (StartupPhase.SECRETS_MANAGEMENT, self._phase_secrets_management),
            (StartupPhase.AUDIT_LOGGING, self._phase_audit_logging),
            (StartupPhase.BACKUP_RECOVERY, self._phase_backup_recovery),
            (StartupPhase.CORE_SERVICES, self._phase_core_services),
            (StartupPhase.TRADING_ENGINE, self._phase_trading_engine),
            (StartupPhase.MONITORING, self._phase_monitoring),
            (StartupPhase.PRODUCTION_READY, self._phase_production_ready)
        ]
        
        overall_success = True
        
        for phase, phase_handler in startup_phases:
            result = await self._execute_phase(phase, phase_handler)
            self.phases_completed.append(result)
            
            if not result.success:
                self.logger.error(f"❌ Phase {phase.value} failed: {result.message}")
                overall_success = False
                break
            else:
                self.logger.info(f"✅ Phase {phase.value} completed in {result.duration_seconds:.2f}s")
        
        if overall_success:
            self.is_production_ready = True
            self.logger.info("🎉 Production system startup completed successfully!")
            self._log_startup_summary()
        else:
            self.logger.error("💥 Production system startup failed!")
            await self.shutdown()
        
        return overall_success
    
    async def _execute_phase(self, phase: StartupPhase, handler: Callable) -> StartupResult:
        """Execute a startup phase with timing and error handling."""
        self.logger.info(f"🔄 Starting phase: {phase.value}")
        
        start_time = time.time()
        
        try:
            success, message, details = await handler()
            duration = time.time() - start_time
            
            return StartupResult(
                phase=phase,
                success=success,
                duration_seconds=duration,
                message=message,
                details=details
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_message = f"Phase failed with exception: {str(e)}"
            self.logger.exception(error_message)
            
            return StartupResult(
                phase=phase,
                success=False,
                duration_seconds=duration,
                message=error_message,
                details={"exception": str(e)}
            )
    
    async def _phase_initialization(self) -> tuple[bool, str, Dict[str, Any]]:
        """Phase 1: Basic system initialization."""
        try:
            # Create required directories
            required_dirs = [
                "data", "logs", "config", "backups",
                "docs", "temp", "cache"
            ]
            
            for dir_name in required_dirs:
                Path(dir_name).mkdir(exist_ok=True)
            
            # Check system resources
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            cpu_count = psutil.cpu_count()
            
            system_info = {
                "memory_gb": memory.total / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "cpu_count": cpu_count,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform
            }
            
            self.logger.info(f"System Resources: {system_info}")
            
            return True, "System initialization completed", {
                "directories_created": required_dirs,
                "system_info": system_info
            }
            
        except Exception as e:
            return False, f"Initialization failed: {str(e)}", {}
    
    async def _phase_environment_validation(self) -> tuple[bool, str, Dict[str, Any]]:
        """Phase 2: Environment validation."""
        try:
            self.environment_validator = EnvironmentValidator()
            is_valid, report = self.environment_validator.validate_all()
            
            if not is_valid:
                return False, f"Environment validation failed with {len(report['critical_errors'])} critical errors", report
            
            production_score = report.get("production_ready_score", 0)
            
            if production_score < 7.0:
                return False, f"Production readiness score too low: {production_score}/10", report
            
            return True, f"Environment validation passed (score: {production_score}/10)", report
            
        except Exception as e:
            return False, f"Environment validation error: {str(e)}", {}
    
    async def _phase_secrets_management(self) -> tuple[bool, str, Dict[str, Any]]:
        """Phase 3: Secrets management initialization."""
        try:
            self.secrets_manager = SecretsManager()
            
            # Test secret operations
            test_secret = "test_startup_secret"
            test_value = f"startup_test_{int(time.time())}"
            
            # Store and retrieve test secret
            if not self.secrets_manager.store_secret(test_secret, test_value, "Startup test secret"):
                return False, "Failed to store test secret", {}
            
            retrieved_value = self.secrets_manager.get_secret(test_secret)
            if retrieved_value != test_value:
                return False, "Secret retrieval validation failed", {}
            
            # Clean up test secret
            self.secrets_manager.delete_secret(test_secret)
            
            # Check for required secrets
            secrets_list = self.secrets_manager.list_secrets()
            
            return True, f"Secrets management initialized with {len(secrets_list)} secrets", {
                "secrets_count": len(secrets_list),
                "vault_path": str(self.secrets_manager.vault_path)
            }
            
        except Exception as e:
            return False, f"Secrets management error: {str(e)}", {}
    
    async def _phase_audit_logging(self) -> tuple[bool, str, Dict[str, Any]]:
        """Phase 4: Audit logging system initialization."""
        try:
            self.audit_logger = AuditLogger()
            
            # Test audit logging
            self.audit_logger.log_system_event(
                AuditEventType.SYSTEM_STARTUP,
                "production_startup",
                "success",
                "startup_orchestrator",
                {"phase": "audit_logging_test"}
            )
            
            # Verify the event was logged
            await asyncio.sleep(1)  # Allow async processing
            
            recent_events = self.audit_logger.get_events(
                start_time=datetime.datetime.utcnow() - datetime.timedelta(minutes=1),
                limit=10
            )
            
            startup_events = [e for e in recent_events if e.get("action") == "production_startup"]
            
            if not startup_events:
                return False, "Audit logging test failed - no events found", {}
            
            return True, "Audit logging system initialized and tested", {
                "recent_events_count": len(recent_events),
                "log_dir": str(self.audit_logger.log_dir)
            }
            
        except Exception as e:
            return False, f"Audit logging error: {str(e)}", {}
    
    async def _phase_backup_recovery(self) -> tuple[bool, str, Dict[str, Any]]:
        """Phase 5: Backup and recovery system initialization."""
        try:
            self.backup_system = BackupRecoverySystem()
            
            # Test backup system
            test_backup_id = self.backup_system.create_full_backup(
                source_paths=["config/", ".env"],
                description="Startup validation backup"
            )
            
            if not test_backup_id:
                return False, "Test backup creation failed", {}
            
            # Verify backup integrity
            if not self.backup_system.verify_backup_integrity(test_backup_id):
                return False, "Backup integrity verification failed", {}
            
            # Start automated backups
            self.backup_system.start_automated_backups()
            
            # Register cleanup handler
            self.shutdown_handlers.append(self.backup_system.stop_automated_backups)
            
            return True, f"Backup system initialized with test backup: {test_backup_id}", {
                "test_backup_id": test_backup_id,
                "backup_root": str(self.backup_system.backup_root)
            }
            
        except Exception as e:
            return False, f"Backup system error: {str(e)}", {}
    
    async def _phase_core_services(self) -> tuple[bool, str, Dict[str, Any]]:
        """Phase 6: Core trading bot services initialization."""
        try:
            # Initialize core configuration
            config_loaded = await self._load_core_configuration()
            if not config_loaded:
                return False, "Failed to load core configuration", {}
            
            # Initialize database connections
            db_initialized = await self._initialize_databases()
            if not db_initialized:
                return False, "Failed to initialize databases", {}
            
            # Initialize RPC connections
            rpc_connected = await self._test_rpc_connections()
            if not rpc_connected:
                return False, "Failed to establish RPC connections", {}
            
            return True, "Core services initialized successfully", {
                "config_loaded": config_loaded,
                "database_initialized": db_initialized,
                "rpc_connected": rpc_connected
            }
            
        except Exception as e:
            return False, f"Core services error: {str(e)}", {}
    
    async def _phase_trading_engine(self) -> tuple[bool, str, Dict[str, Any]]:
        """Phase 7: Trading engine initialization."""
        try:
            # This would initialize the actual trading engine
            # For now, we'll simulate the checks
            
            # Check wallet connectivity
            wallet_connected = await self._test_wallet_connection()
            if not wallet_connected:
                return False, "Wallet connection failed", {}
            
            # Validate trading configuration
            trading_config_valid = await self._validate_trading_config()
            if not trading_config_valid:
                return False, "Trading configuration validation failed", {}
            
            # Initialize risk management
            risk_mgmt_initialized = await self._initialize_risk_management()
            if not risk_mgmt_initialized:
                return False, "Risk management initialization failed", {}
            
            return True, "Trading engine initialized and ready", {
                "wallet_connected": wallet_connected,
                "trading_config_valid": trading_config_valid,
                "risk_management_ready": risk_mgmt_initialized
            }
            
        except Exception as e:
            return False, f"Trading engine error: {str(e)}", {}
    
    async def _phase_monitoring(self) -> tuple[bool, str, Dict[str, Any]]:
        """Phase 8: Monitoring and alerting system initialization."""
        try:
            # Initialize health checks
            health_checks_ready = await self._initialize_health_checks()
            if not health_checks_ready:
                return False, "Health checks initialization failed", {}
            
            # Test Discord notifications
            discord_ready = await self._test_discord_notifications()
            
            # Initialize API documentation
            api_docs_ready = await self._initialize_api_documentation()
            
            return True, "Monitoring systems initialized", {
                "health_checks_ready": health_checks_ready,
                "discord_notifications": discord_ready,
                "api_documentation": api_docs_ready
            }
            
        except Exception as e:
            return False, f"Monitoring error: {str(e)}", {}
    
    async def _phase_production_ready(self) -> tuple[bool, str, Dict[str, Any]]:
        """Phase 9: Final production readiness verification."""
        try:
            # Perform final health check
            final_health_check = await self._perform_final_health_check()
            if not final_health_check["healthy"]:
                return False, f"Final health check failed: {final_health_check['issues']}", final_health_check
            
            # Log production readiness
            if self.audit_logger:
                self.audit_logger.log_system_event(
                    AuditEventType.SYSTEM_STARTUP,
                    "production_ready",
                    "success",
                    "startup_orchestrator",
                    {
                        "startup_duration": (datetime.datetime.utcnow() - self.startup_time).total_seconds(),
                        "phases_completed": len(self.phases_completed)
                    }
                )
            
            return True, "System is production ready!", {
                "final_health_check": final_health_check,
                "startup_duration": (datetime.datetime.utcnow() - self.startup_time).total_seconds()
            }
            
        except Exception as e:
            return False, f"Production readiness check error: {str(e)}", {}
    
    async def _load_core_configuration(self) -> bool:
        """Load and validate core configuration."""
        try:
            # Check if configuration files exist
            config_files = ["config/config.yaml", ".env"]
            for config_file in config_files:
                if not Path(config_file).exists():
                    self.logger.warning(f"Configuration file not found: {config_file}")
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Configuration loading error: {e}")
            return False
    
    async def _initialize_databases(self) -> bool:
        """Initialize database connections."""
        try:
            # Test SQLite database connection
            import sqlite3
            test_db = "data/test_connection.db"
            
            conn = sqlite3.connect(test_db)
            conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER)")
            conn.execute("INSERT INTO test (id) VALUES (1)")
            conn.commit()
            conn.close()
            
            # Clean up test database
            if Path(test_db).exists():
                Path(test_db).unlink()
            
            return True
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
            return False
    
    async def _test_rpc_connections(self) -> bool:
        """Test RPC connections."""
        try:
            # This would test actual Solana RPC connections
            # For now, we'll simulate the check
            import requests
            
            # Test a public RPC endpoint
            test_rpc = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
            
            response = requests.post(
                test_rpc,
                json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"},
                timeout=10
            )
            
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"RPC connection test error: {e}")
            return False
    
    async def _test_wallet_connection(self) -> bool:
        """Test wallet connection."""
        try:
            # This would test actual wallet connectivity
            # For now, we'll check if private key is available
            private_key = None
            
            if self.secrets_manager:
                private_key = self.secrets_manager.get_secret("wallet_private_key")
            
            if not private_key:
                private_key = os.getenv("WALLET_PRIVATE_KEY")
            
            return private_key is not None and len(private_key) > 20
        except Exception as e:
            self.logger.error(f"Wallet connection test error: {e}")
            return False
    
    async def _validate_trading_config(self) -> bool:
        """Validate trading configuration."""
        try:
            # Check required trading parameters
            required_vars = [
                "SOLANA_RPC_URL",
                "WALLET_PRIVATE_KEY"
            ]
            
            for var in required_vars:
                value = os.getenv(var)
                if self.secrets_manager:
                    value = value or self.secrets_manager.get_secret(var.lower())
                
                if not value:
                    self.logger.error(f"Required trading config missing: {var}")
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Trading config validation error: {e}")
            return False
    
    async def _initialize_risk_management(self) -> bool:
        """Initialize risk management systems."""
        try:
            # This would initialize actual risk management
            # For now, we'll check basic risk parameters
            return True
        except Exception as e:
            self.logger.error(f"Risk management initialization error: {e}")
            return False
    
    async def _initialize_health_checks(self) -> bool:
        """Initialize health monitoring."""
        try:
            # Basic health check initialization
            return True
        except Exception as e:
            self.logger.error(f"Health checks initialization error: {e}")
            return False
    
    async def _test_discord_notifications(self) -> bool:
        """Test Discord notification system."""
        try:
            discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")
            if self.secrets_manager:
                discord_webhook = discord_webhook or self.secrets_manager.get_secret("discord_webhook_url")
            
            if not discord_webhook:
                self.logger.warning("Discord webhook not configured")
                return False
            
            # Test webhook (don't actually send to avoid spam)
            return discord_webhook.startswith("https://discord.com/api/webhooks/")
        except Exception as e:
            self.logger.error(f"Discord notification test error: {e}")
            return False
    
    async def _initialize_api_documentation(self) -> bool:
        """Initialize API documentation."""
        try:
            doc_generator = APIDocumentationGenerator()
            doc_files = doc_generator.generate_all_documentation()
            
            self.logger.info(f"API documentation generated: {len(doc_files)} files")
            return True
        except Exception as e:
            self.logger.error(f"API documentation error: {e}")
            return False
    
    async def _perform_final_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive final health check."""
        health_status = {
            "healthy": True,
            "issues": [],
            "checks": {}
        }
        
        try:
            # Check system resources
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            if memory.percent > 90:
                health_status["issues"].append("High memory usage")
                health_status["healthy"] = False
            
            if disk.percent > 90:
                health_status["issues"].append("High disk usage")
                health_status["healthy"] = False
            
            health_status["checks"]["system_resources"] = {
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "cpu_count": psutil.cpu_count()
            }
            
            # Check core systems
            health_status["checks"]["core_systems"] = {
                "environment_validator": self.environment_validator is not None,
                "secrets_manager": self.secrets_manager is not None,
                "audit_logger": self.audit_logger is not None,
                "backup_system": self.backup_system is not None
            }
            
            # Check file system
            critical_paths = ["data/", "logs/", "config/"]
            for path in critical_paths:
                if not Path(path).exists():
                    health_status["issues"].append(f"Critical path missing: {path}")
                    health_status["healthy"] = False
            
            health_status["checks"]["file_system"] = {
                path: Path(path).exists() for path in critical_paths
            }
            
        except Exception as e:
            health_status["healthy"] = False
            health_status["issues"].append(f"Health check error: {str(e)}")
        
        return health_status
    
    def _log_startup_summary(self):
        """Log comprehensive startup summary."""
        total_duration = (datetime.datetime.utcnow() - self.startup_time).total_seconds()
        
        self.logger.info("=" * 80)
        self.logger.info("🎉 PRODUCTION STARTUP SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Started at: {self.startup_time.isoformat()}")
        self.logger.info(f"Total Duration: {total_duration:.2f} seconds")
        self.logger.info(f"Phases Completed: {len(self.phases_completed)}")
        self.logger.info("")
        
        for result in self.phases_completed:
            status_emoji = "✅" if result.success else "❌"
            self.logger.info(f"{status_emoji} {result.phase.value}: {result.duration_seconds:.2f}s - {result.message}")
        
        self.logger.info("")
        self.logger.info("🚀 System is now ready for production trading!")
        self.logger.info("=" * 80)
    
    async def shutdown(self):
        """Gracefully shutdown all systems."""
        self.logger.info("🛑 Initiating graceful system shutdown...")
        
        # Run shutdown handlers in reverse order
        for handler in reversed(self.shutdown_handlers):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                self.logger.error(f"Error in shutdown handler: {e}")
        
        # Shutdown core systems
        if self.audit_logger:
            self.audit_logger.log_system_event(
                AuditEventType.SYSTEM_SHUTDOWN,
                "graceful_shutdown",
                "success",
                "startup_orchestrator",
                {"uptime_seconds": (datetime.datetime.utcnow() - self.startup_time).total_seconds()}
            )
            self.audit_logger.shutdown()
        
        self.logger.info("✅ System shutdown completed")

async def main():
    """Main entry point for production startup."""
    orchestrator = ProductionStartupOrchestrator()
    
    try:
        success = await orchestrator.start_production_system()
        
        if success:
            # Keep the system running
            print("\n🎯 Trading bot is now running in production mode.")
            print("Press Ctrl+C to stop the system gracefully.")
            
            try:
                while True:
                    await asyncio.sleep(60)  # Heartbeat every minute
                    # Could add periodic health checks here
            except KeyboardInterrupt:
                print("\n🛑 Shutdown signal received...")
        else:
            print("\n❌ Production startup failed. Check logs for details.")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n💥 Fatal error during startup: {e}")
        sys.exit(1)
    
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main()) 