"""
ConfigManager - Centralized configuration management for Ant Bot System

Manages all system configurations with dynamic loading, validation,
hot-reloading capabilities, and environment-specific settings.
"""

import json
import os
import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config.core_config import (
    CORE_CONFIG,
    MARKET_CONFIG,
    TRADING_CONFIG
)

# Import available configurations
try:
    from config.ant_princess_config import (
        ANT_PRINCESS_CONFIG, QUEEN_CONFIG as ANT_QUEEN_CONFIG, SYSTEM_CONSTANTS as AI_CONFIG
    )
except ImportError:
    # Fallback configurations if ant_princess_config is not available
    ANT_PRINCESS_CONFIG = {
        "market_weight": 0.6,
        "sentiment_weight": 0.3,
        "wallet_weight": 0.1,
        "buy_threshold": 0.7,
        "sell_threshold": -0.3,
        "base_position_size": 0.01,
        "max_position_size": 0.1,
        "multiplication_thresholds": {
            "performance_score": 0.8,
            "experience_threshold": 10
        }
    }
    
    ANT_QUEEN_CONFIG = {
        "initial_workers": 3,
        "min_workers": 2,
        "max_workers": 10,
        "performance_threshold": 0.6,
        "history_size": 100,
        "multiplication_threshold": 0.8
    }
    
    AI_CONFIG = {
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 100,
        "confidence_threshold": 0.7,
        "ensemble_weights": {
            "grok": 0.6,
            "local_llm": 0.4
        }
    }

logger = logging.getLogger(__name__)

@dataclass
class ConfigChangeEvent:
    """Represents a configuration change event"""
    config_path: str
    old_value: Any
    new_value: Any
    timestamp: float
    change_type: str  # 'update', 'add', 'delete'

@dataclass
class ConfigValidationRule:
    """Configuration validation rule"""
    path: str
    rule_type: str  # 'type', 'range', 'enum', 'custom'
    constraint: Any
    error_message: str
    is_required: bool = True

class ConfigFileHandler(FileSystemEventHandler):
    """Handles configuration file changes for hot-reloading"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.last_modified = {}
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = event.src_path
        current_time = time.time()
        
        # Debounce rapid file changes
        if file_path in self.last_modified:
            if current_time - self.last_modified[file_path] < 1.0:
                return
        
        self.last_modified[file_path] = current_time
        
        if file_path.endswith(('.json', '.yaml', '.yml')):
            logger.info(f"Configuration file changed: {file_path}")
            asyncio.create_task(self.config_manager._reload_config_file(file_path))

class ConfigManager:
    """Comprehensive configuration management system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config_data: Dict[str, Any] = {}
        self.config_history: List[ConfigChangeEvent] = []
        self.validation_rules: List[ConfigValidationRule] = []
        self.change_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # File watching for hot-reload
        self.observer = Observer()
        self.file_handler = ConfigFileHandler(self)
        self.watch_enabled = False
        
        # Environment and deployment settings
        self.environment = os.getenv('ANT_BOT_ENV', 'development')
        self.deployment_stage = os.getenv('DEPLOYMENT_STAGE', 'local')
        
        # Configuration cache and performance
        self.config_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_update = {}
        
        # Feature flags
        self.feature_flags: Dict[str, bool] = {}
        
        logger.info(f"ConfigManager initialized for environment: {self.environment}")
    
    def _get_default_config_path(self) -> str:
        """Get default configuration path"""
        current_dir = Path(__file__).parent
        config_dir = current_dir.parent.parent / "config"
        return str(config_dir)
    
    async def initialize(self) -> bool:
        """Initialize the configuration manager"""
        try:
            # Load base configurations
            await self._load_base_configurations()
            
            # Load environment-specific configurations
            await self._load_environment_configs()
            
            # Initialize validation rules
            await self._initialize_validation_rules()
            
            # Initialize feature flags
            await self._initialize_feature_flags()
            
            # Validate all configurations
            validation_result = await self.validate_all_configs()
            if not validation_result["is_valid"]:
                logger.error(f"Configuration validation failed: {validation_result['errors']}")
                return False
            
            # Setup file watching for hot-reload
            await self._setup_file_watching()
            
            logger.info("ConfigManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ConfigManager: {e}")
            return False
    
    async def _load_base_configurations(self):
        """Load base system configurations"""
        try:
            # Import existing configurations
            self.config_data.update({
                "core": CORE_CONFIG,
                "market": MARKET_CONFIG,
                "trading": TRADING_CONFIG,
                "ant_princess": ANT_PRINCESS_CONFIG,
                "ant_queen": ANT_QUEEN_CONFIG,
                "ai": AI_CONFIG
            })
            
            # Load additional configurations
            await self._load_compounding_configs()
            await self._load_flywheel_configs()
            await self._load_colony_configs()
            
        except Exception as e:
            logger.error(f"Error loading base configurations: {e}")
            raise
    
    async def _load_compounding_configs(self):
        """Load compounding layer configurations"""
        self.config_data["compounding"] = {
            "monetary_layer": {
                "base_compound_rate": 1.05,
                "max_compound_rate": 1.25,
                "min_compound_rate": 1.01,
                "adaptation_factor": 0.1,
                "compound_interval": 3600.0
            },
            "worker_layer": {
                "base_multiplication_rate": 5,
                "max_workers_per_split": 10,
                "min_workers_per_split": 2,
                "efficiency_decay_rate": 0.95,
                "efficiency_update_interval": 1800.0
            },
            "carwash_layer": {
                "base_compound_rates": {
                    "memory": 1.015,
                    "storage": 1.012,
                    "errors": 1.025,
                    "redundancy": 1.020,
                    "full": 1.018
                },
                "cleanup_schedules": {
                    "memory": 3600,
                    "storage": 86400,
                    "errors": 1800,
                    "redundancy": 21600,
                    "full": 259200
                }
            },
            "intelligence_layer": {
                "learning_compound_rates": {
                    "experience": 1.008,
                    "pattern": 1.012,
                    "strategic": 1.015,
                    "cross_domain": 1.020
                },
                "knowledge_domains": [
                    "trading", "market_analysis", "risk_management",
                    "pattern_recognition", "strategy_optimization", "cross_domain"
                ]
            },
            "data_layer": {
                "pattern_compound_rates": {
                    "trend": 1.010,
                    "cycle": 1.015,
                    "correlation": 1.020,
                    "anomaly": 1.025
                },
                "data_retention": {
                    "price_history": 10000,
                    "volume_history": 10000,
                    "pattern_history": 1000
                }
            }
        }
    
    async def _load_flywheel_configs(self):
        """Load flywheel system configurations"""
        self.config_data["flywheel"] = {
            "architecture_iteration": {
                "iteration_interval": 3600.0,
                "metrics_history_size": 1000,
                "bottleneck_threshold": 0.8
            },
            "performance_amplification": {
                "amplification_threshold": 1.2,
                "momentum_decay_rate": 0.95,
                "max_amplification_factor": 3.0
            },
            "feedback_loops": {
                "learning_rate": 0.1,
                "adaptation_speed": 0.05,
                "stability_threshold": 0.8
            }
        }
    
    async def _load_colony_configs(self):
        """Load ant colony configurations"""
        self.config_data["colony"] = {
            "founding_queen": {
                "initial_capital": 2.0,
                "split_threshold": 1500.0,
                "risk_tolerance": 0.05
            },
            "ant_queen": {
                "optimization_frequency": 86400,
                "performance_threshold": 0.7,
                "max_princesses": 10
            },
            "worker_ants": {
                "trades_per_coin": {"min": 5, "max": 10},
                "target_returns": {"min": 1.03, "max": 1.50},
                "position_sizing": {"base": 0.01, "max": 0.1}
            },
            "ant_drone": {
                "ai_monitoring_interval": 300,
                "trend_analysis_depth": 24,
                "decision_confidence_threshold": 0.6
            }
        }
    
    async def _load_environment_configs(self):
        """Load environment-specific configurations"""
        try:
            env_config_path = Path(self.config_path) / f"env_{self.environment}.json"
            if env_config_path.exists():
                with open(env_config_path, 'r') as f:
                    env_config = json.load(f)
                    
                # Deep merge environment configs
                self._deep_merge_config(self.config_data, env_config)
                logger.info(f"Loaded environment config for: {self.environment}")
            
        except Exception as e:
            logger.warning(f"Could not load environment config: {e}")
    
    def _deep_merge_config(self, base: Dict, override: Dict) -> Dict:
        """Deep merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_config(base[key], value)
            else:
                base[key] = value
        return base
    
    async def _initialize_validation_rules(self):
        """Initialize configuration validation rules"""
        self.validation_rules = [
            # Core configuration validation
            ConfigValidationRule(
                path="core.trading.min_liquidity",
                rule_type="range",
                constraint=(0, float('inf')),
                error_message="min_liquidity must be positive"
            ),
            ConfigValidationRule(
                path="core.trading.max_slippage",
                rule_type="range",
                constraint=(0, 1),
                error_message="max_slippage must be between 0 and 1"
            ),
            
            # Trading configuration validation
            ConfigValidationRule(
                path="trading.max_position_size",
                rule_type="range",
                constraint=(0, 1),
                error_message="max_position_size must be between 0 and 1"
            ),
            
            # Compounding configuration validation
            ConfigValidationRule(
                path="compounding.monetary_layer.base_compound_rate",
                rule_type="range",
                constraint=(1.0, 2.0),
                error_message="base_compound_rate must be between 1.0 and 2.0"
            ),
            
            # Colony configuration validation
            ConfigValidationRule(
                path="colony.founding_queen.initial_capital",
                rule_type="range",
                constraint=(0.1, 1000.0),
                error_message="initial_capital must be between 0.1 and 1000.0"
            )
        ]
    
    async def _initialize_feature_flags(self):
        """Initialize feature flags"""
        self.feature_flags = {
            "enable_hot_reload": True,
            "enable_compounding": True,
            "enable_flywheel": True,
            "enable_advanced_analytics": self.environment in ["staging", "production"],
            "enable_debug_logging": self.environment == "development",
            "enable_performance_monitoring": True,
            "enable_security_features": self.environment in ["staging", "production"]
        }
    
    async def _setup_file_watching(self):
        """Setup file watching for configuration hot-reload"""
        if not self.feature_flags.get("enable_hot_reload", False):
            return
        
        try:
            config_dir = Path(self.config_path)
            if config_dir.exists():
                self.observer.schedule(self.file_handler, str(config_dir), recursive=True)
                self.observer.start()
                self.watch_enabled = True
                logger.info("Configuration file watching enabled")
                
        except Exception as e:
            logger.warning(f"Could not setup file watching: {e}")
    
    async def _reload_config_file(self, file_path: str):
        """Reload a specific configuration file"""
        try:
            file_path_obj = Path(file_path)
            if file_path_obj.suffix == '.json':
                with open(file_path, 'r') as f:
                    new_config = json.load(f)
            elif file_path_obj.suffix in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    new_config = yaml.safe_load(f)
            else:
                return
            
            # Trigger configuration reload
            await self._trigger_config_update(file_path_obj.stem, new_config)
            
        except Exception as e:
            logger.error(f"Error reloading config file {file_path}: {e}")
    
    async def _trigger_config_update(self, config_key: str, new_config: Dict[str, Any]):
        """Trigger configuration update and notify callbacks"""
        old_config = self.config_data.get(config_key, {})
        
        # Update configuration
        self.config_data[config_key] = new_config
        
        # Record change event
        change_event = ConfigChangeEvent(
            config_path=config_key,
            old_value=old_config,
            new_value=new_config,
            timestamp=time.time(),
            change_type='update'
        )
        self.config_history.append(change_event)
        
        # Notify callbacks
        for callback in self.change_callbacks.get(config_key, []):
            try:
                await callback(change_event)
            except Exception as e:
                logger.error(f"Error in config change callback: {e}")
        
        logger.info(f"Configuration updated: {config_key}")
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by path (e.g., 'core.trading.min_liquidity')"""
        try:
            # Check cache first
            if path in self.config_cache:
                cache_time = self.last_cache_update.get(path, 0)
                if time.time() - cache_time < self.cache_ttl:
                    return self.config_cache[path]
            
            # Navigate through nested dictionaries
            keys = path.split('.')
            value = self.config_data
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            # Cache the result
            self.config_cache[path] = value
            self.last_cache_update[path] = time.time()
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting config value for path '{path}': {e}")
            return default
    
    def set(self, path: str, value: Any) -> bool:
        """Set configuration value by path"""
        try:
            keys = path.split('.')
            config = self.config_data
            
            # Navigate to parent
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            # Set the value
            old_value = config.get(keys[-1])
            config[keys[-1]] = value
            
            # Clear cache for this path
            if path in self.config_cache:
                del self.config_cache[path]
            
            # Record change
            change_event = ConfigChangeEvent(
                config_path=path,
                old_value=old_value,
                new_value=value,
                timestamp=time.time(),
                change_type='update' if old_value is not None else 'add'
            )
            self.config_history.append(change_event)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting config value for path '{path}': {e}")
            return False
    
    def register_change_callback(self, config_path: str, callback: Callable) -> bool:
        """Register a callback for configuration changes"""
        try:
            self.change_callbacks[config_path].append(callback)
            return True
        except Exception as e:
            logger.error(f"Error registering change callback: {e}")
            return False
    
    async def validate_all_configs(self) -> Dict[str, Any]:
        """Validate all configurations against rules"""
        errors = []
        warnings = []
        
        for rule in self.validation_rules:
            validation_result = await self._validate_rule(rule)
            if not validation_result["valid"]:
                if rule.is_required:
                    errors.append(validation_result["message"])
                else:
                    warnings.append(validation_result["message"])
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _validate_rule(self, rule: ConfigValidationRule) -> Dict[str, Any]:
        """Validate a single configuration rule"""
        try:
            value = self.get(rule.path)
            
            if value is None and rule.is_required:
                return {
                    "valid": False,
                    "message": f"Required configuration missing: {rule.path}"
                }
            
            if value is None:
                return {"valid": True, "message": ""}
            
            if rule.rule_type == "type":
                if not isinstance(value, rule.constraint):
                    return {
                        "valid": False,
                        "message": f"{rule.path}: {rule.error_message}"
                    }
            
            elif rule.rule_type == "range":
                constraint = rule.constraint
                if isinstance(constraint, dict):
                    min_val = constraint.get('min', float('-inf'))
                    max_val = constraint.get('max', float('inf'))
                else:
                    min_val, max_val = constraint
                if not (min_val <= value <= max_val):
                    return {
                        "valid": False,
                        "message": f"{rule.path}: {rule.error_message}"
                    }
            
            elif rule.rule_type == "enum":
                if value not in rule.constraint:
                    return {
                        "valid": False,
                        "message": f"{rule.path}: {rule.error_message}"
                    }
            
            return {"valid": True, "message": ""}
            
        except Exception as e:
            return {
                "valid": False,
                "message": f"Validation error for {rule.path}: {e}"
            }
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature flag is enabled"""
        return self.feature_flags.get(feature_name, False)
    
    def enable_feature(self, feature_name: str):
        """Enable a feature flag"""
        self.feature_flags[feature_name] = True
    
    def disable_feature(self, feature_name: str):
        """Disable a feature flag"""
        self.feature_flags[feature_name] = False
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configuration data"""
        return self.config_data.copy()
    
    def get_environment(self) -> str:
        """Get current environment"""
        return self.environment
    
    def get_deployment_stage(self) -> str:
        """Get current deployment stage"""
        return self.deployment_stage
    
    def _count_configs(self, config_dict: Dict[str, Any]) -> int:
        """Recursively count all configuration values"""
        count = 0
        for key, value in config_dict.items():
            if isinstance(value, dict):
                count += self._count_configs(value)
            else:
                count += 1
        return count
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration manager summary"""
        return {
            "environment": self.environment,
            "deployment_stage": self.deployment_stage,
            "total_configs": self._count_configs(self.config_data),
            "config_sections": list(self.config_data.keys()),
            "feature_flags": self.feature_flags.copy(),
            "validation_rules_count": len(self.validation_rules),
            "change_callbacks_count": len(self.change_callbacks),
            "config_history_length": len(self.config_history),
            "cache_entries": len(self.config_cache),
            "hot_reload_enabled": self.watch_enabled
        }
    
    async def cleanup(self):
        """Cleanup configuration manager resources"""
        try:
            if self.watch_enabled and hasattr(self, 'observer') and self.observer:
                try:
                    self.observer.stop()
                    if self.observer.is_alive():
                        self.observer.join(timeout=2.0)
                except Exception as e:
                    logger.warning(f"Error stopping file observer: {e}")
                finally:
                    self.watch_enabled = False
            
            # Clear caches
            self.config_cache.clear()
            self.last_cache_update.clear()
            
            logger.info("ConfigManager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during ConfigManager cleanup: {e}") 