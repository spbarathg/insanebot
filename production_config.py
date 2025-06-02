"""
Production Configuration for Solana Trading Bot

This module contains all production-ready configuration settings,
including security, monitoring, performance, and operational parameters.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_NAME", "trading_bot")
    username: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "")
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "20"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "30"))
    pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    pool_recycle: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    ssl_mode: str = os.getenv("DB_SSL_MODE", "require")

@dataclass
class RedisConfig:
    """Redis configuration settings"""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    database: int = int(os.getenv("REDIS_DB", "0"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    ssl: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
    retry_on_timeout: bool = True
    health_check_interval: int = 30

@dataclass
class SolanaConfig:
    """Solana blockchain configuration"""
    rpc_url: str = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    ws_url: str = os.getenv("SOLANA_WS_URL", "wss://api.mainnet-beta.solana.com")
    commitment: str = os.getenv("SOLANA_COMMITMENT", "confirmed")
    timeout: int = int(os.getenv("SOLANA_TIMEOUT", "30"))
    max_retries: int = int(os.getenv("SOLANA_MAX_RETRIES", "3"))
    backoff_factor: float = float(os.getenv("SOLANA_BACKOFF_FACTOR", "1.5"))
    
    # Fallback RPC endpoints for redundancy
    fallback_rpc_urls: List[str] = field(default_factory=lambda: [
        "https://solana-api.projectserum.com",
        "https://rpc.ankr.com/solana",
        "https://api.mainnet-beta.solana.com"
    ])

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    private_key: str = os.getenv("PRIVATE_KEY", "")
    encryption_key: str = os.getenv("ENCRYPTION_KEY", "")
    jwt_secret: str = os.getenv("JWT_SECRET", "")
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = int(os.getenv("JWT_EXPIRATION", "3600"))
    
    # API Security
    api_rate_limit: str = os.getenv("API_RATE_LIMIT", "100/minute")
    cors_origins: List[str] = field(default_factory=lambda: 
        os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else ["*"])
    
    # TLS/SSL
    tls_cert_path: Optional[str] = os.getenv("TLS_CERT_PATH")
    tls_key_path: Optional[str] = os.getenv("TLS_KEY_PATH")
    
    # Security headers
    enable_security_headers: bool = True
    hsts_max_age: int = 31536000  # 1 year

@dataclass
class TradingConfig:
    """Trading strategy and risk management configuration"""
    # Capital Management
    initial_capital: float = float(os.getenv("INITIAL_CAPITAL", "1.0"))
    max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.02"))
    max_daily_loss: float = float(os.getenv("MAX_DAILY_LOSS", "0.05"))
    
    # Ant Colony Parameters
    max_worker_ants: int = int(os.getenv("MAX_WORKER_ANTS", "100"))
    worker_split_threshold: float = float(os.getenv("WORKER_SPLIT_THRESHOLD", "2.0"))
    worker_merge_threshold: float = float(os.getenv("WORKER_MERGE_THRESHOLD", "0.1"))
    queen_capital_threshold: float = float(os.getenv("QUEEN_CAPITAL_THRESHOLD", "10.0"))
    
    # Trading Frequency
    min_trade_interval: int = int(os.getenv("MIN_TRADE_INTERVAL", "60"))  # seconds
    max_trades_per_hour: int = int(os.getenv("MAX_TRADES_PER_HOUR", "10"))
    
    # Risk Controls
    stop_loss_percentage: float = float(os.getenv("STOP_LOSS_PERCENTAGE", "0.02"))
    take_profit_percentage: float = float(os.getenv("TAKE_PROFIT_PERCENTAGE", "0.04"))
    
    # Emergency Controls
    emergency_stop_loss: float = float(os.getenv("EMERGENCY_STOP_LOSS", "0.20"))
    circuit_breaker_threshold: float = float(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "0.10"))

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    # Metrics
    metrics_enabled: bool = True
    metrics_retention_hours: int = int(os.getenv("METRICS_RETENTION_HOURS", "168"))  # 7 days
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "8000"))
    
    # Health Checks
    health_check_interval: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    health_check_timeout: int = int(os.getenv("HEALTH_CHECK_TIMEOUT", "10"))
    
    # Alerting
    alerting_enabled: bool = True
    alert_webhook_url: Optional[str] = os.getenv("ALERT_WEBHOOK_URL")
    slack_webhook_url: Optional[str] = os.getenv("SLACK_WEBHOOK_URL")
    email_alerts_enabled: bool = os.getenv("EMAIL_ALERTS_ENABLED", "false").lower() == "true"
    
    # Performance Monitoring
    performance_sampling_rate: float = float(os.getenv("PERFORMANCE_SAMPLING_RATE", "0.1"))
    trace_enabled: bool = os.getenv("TRACE_ENABLED", "false").lower() == "true"

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File logging
    log_to_file: bool = True
    log_file_path: str = os.getenv("LOG_FILE_PATH", "logs/trading_bot.log")
    log_file_max_size: int = int(os.getenv("LOG_FILE_MAX_SIZE", "100"))  # MB
    log_file_backup_count: int = int(os.getenv("LOG_FILE_BACKUP_COUNT", "10"))
    
    # Structured logging
    json_logging: bool = os.getenv("JSON_LOGGING", "true").lower() == "true"
    
    # Log rotation
    rotation_frequency: str = os.getenv("LOG_ROTATION_FREQUENCY", "daily")
    
    # External logging
    syslog_enabled: bool = os.getenv("SYSLOG_ENABLED", "false").lower() == "true"
    syslog_host: Optional[str] = os.getenv("SYSLOG_HOST")
    syslog_port: int = int(os.getenv("SYSLOG_PORT", "514"))

@dataclass
class APIConfig:
    """API server configuration"""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8080"))
    workers: int = int(os.getenv("API_WORKERS", "4"))
    
    # Performance
    keep_alive: int = int(os.getenv("API_KEEP_ALIVE", "2"))
    max_requests: int = int(os.getenv("API_MAX_REQUESTS", "1000"))
    max_requests_jitter: int = int(os.getenv("API_MAX_REQUESTS_JITTER", "100"))
    timeout: int = int(os.getenv("API_TIMEOUT", "30"))
    
    # Security
    cors_enabled: bool = True
    csrf_protection: bool = True
    
    # Documentation
    docs_enabled: bool = os.getenv("API_DOCS_ENABLED", "true").lower() == "true"
    openapi_url: str = "/openapi.json"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"

@dataclass
class BackupConfig:
    """Backup and disaster recovery configuration"""
    enabled: bool = os.getenv("BACKUP_ENABLED", "true").lower() == "true"
    
    # Local backups
    local_backup_path: str = os.getenv("LOCAL_BACKUP_PATH", "backups/")
    backup_frequency: str = os.getenv("BACKUP_FREQUENCY", "hourly")
    retention_days: int = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
    
    # Cloud backups
    s3_enabled: bool = os.getenv("S3_BACKUP_ENABLED", "false").lower() == "true"
    s3_bucket: Optional[str] = os.getenv("S3_BACKUP_BUCKET")
    s3_region: str = os.getenv("S3_BACKUP_REGION", "us-east-1")
    
    # Encryption
    backup_encryption_enabled: bool = True
    backup_encryption_key: Optional[str] = os.getenv("BACKUP_ENCRYPTION_KEY")

@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    # Async settings
    max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", "100"))
    task_timeout: int = int(os.getenv("TASK_TIMEOUT", "300"))
    
    # Memory management
    max_memory_usage_mb: int = int(os.getenv("MAX_MEMORY_USAGE_MB", "2048"))
    gc_threshold: int = int(os.getenv("GC_THRESHOLD", "700"))
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = int(os.getenv("CACHE_TTL", "300"))  # seconds
    cache_max_size: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    
    # Connection pooling
    max_http_connections: int = int(os.getenv("MAX_HTTP_CONNECTIONS", "100"))
    http_timeout: int = int(os.getenv("HTTP_TIMEOUT", "30"))

@dataclass
class ProductionConfig:
    """Complete production configuration"""
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "production")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    testing: bool = os.getenv("TESTING", "false").lower() == "true"
    
    # Application
    app_name: str = "Solana Trading Bot"
    version: str = "1.0.0"
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    solana: SolanaConfig = field(default_factory=SolanaConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
        self._setup_directories()
    
    def _validate_config(self):
        """Validate critical configuration parameters"""
        # Security validation
        if not self.security.private_key:
            raise ValueError("PRIVATE_KEY environment variable is required")
        
        if not self.security.encryption_key:
            raise ValueError("ENCRYPTION_KEY environment variable is required")
        
        # Database validation
        if not self.database.password and self.environment == "production":
            raise ValueError("Database password is required in production")
        
        # Trading validation
        if self.trading.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        if self.trading.risk_per_trade > 0.1:
            raise ValueError("Risk per trade should not exceed 10%")
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            Path(self.logging.log_file_path).parent,
            Path(self.backup.local_backup_path),
            Path("data/"),
            Path("cache/"),
            Path("temp/")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_database_url(self) -> str:
        """Get complete database URL"""
        return (f"postgresql://{self.database.username}:{self.database.password}"
                f"@{self.database.host}:{self.database.port}/{self.database.database}")
    
    def get_redis_url(self) -> str:
        """Get complete Redis URL"""
        auth = f":{self.redis.password}@" if self.redis.password else ""
        protocol = "rediss" if self.redis.ssl else "redis"
        return f"{protocol}://{auth}{self.redis.host}:{self.redis.port}/{self.redis.database}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for logging/debugging)"""
        config_dict = {}
        
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if hasattr(value, '__dataclass_fields__'):
                # Nested dataclass
                config_dict[field_name] = {
                    k: v for k, v in value.__dict__.items()
                    if not k.startswith('_') and 'password' not in k.lower() and 'key' not in k.lower()
                }
            else:
                config_dict[field_name] = value
        
        return config_dict
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        # Create logs directory
        log_dir = Path(self.logging.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.logging.log_file_path)
            ]
        )
        
        # Set specific logger levels
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)

# Global configuration instance
config = ProductionConfig()

# Environment-specific configurations
def get_config_for_environment(env: str) -> ProductionConfig:
    """Get configuration for specific environment"""
    if env == "development":
        config.environment = "development"
        config.debug = True
        config.logging.level = "DEBUG"
        config.api.docs_enabled = True
        config.monitoring.trace_enabled = True
    elif env == "testing":
        config.environment = "testing"
        config.testing = True
        config.database.database = "trading_bot_test"
        config.redis.database = 1
        config.logging.level = "DEBUG"
    elif env == "staging":
        config.environment = "staging"
        config.debug = False
        config.logging.level = "INFO"
        config.monitoring.alerting_enabled = False
    elif env == "production":
        config.environment = "production"
        config.debug = False
        config.logging.level = "INFO"
        config.api.docs_enabled = False
        config.monitoring.alerting_enabled = True
    
    return config

# Load environment-specific configuration
current_env = os.getenv("ENVIRONMENT", "production")
config = get_config_for_environment(current_env)

# Setup logging
config.setup_logging()

logger = logging.getLogger(__name__)
logger.info(f"Loaded {current_env} configuration")
logger.info(f"Database: {config.database.host}:{config.database.port}")
logger.info(f"Redis: {config.redis.host}:{config.redis.port}")
logger.info(f"API: {config.api.host}:{config.api.port}") 