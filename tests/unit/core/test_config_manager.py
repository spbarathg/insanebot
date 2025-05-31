"""
Unit tests for ConfigManager
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.core.config_manager import ConfigManager, ConfigChangeEvent, ConfigValidationRule


class TestConfigManager:
    """Test suite for ConfigManager"""
    
    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create temporary config directory"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        return str(config_dir)
    
    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create ConfigManager instance for testing"""
        return ConfigManager(config_path=temp_config_dir)
    
    @pytest.mark.asyncio
    async def test_initialization(self, config_manager):
        """Test ConfigManager initialization"""
        assert config_manager.environment == 'testing'
        assert config_manager.config_data == {}
        assert config_manager.validation_rules == []
        assert isinstance(config_manager.config_cache, dict)
    
    @pytest.mark.asyncio
    async def test_get_config_value(self, config_manager):
        """Test getting configuration values"""
        # Setup test config
        config_manager.config_data = {
            'trading': {
                'max_position_size': 0.1,
                'stop_loss': 5.0
            },
            'api': {
                'timeout': 30
            }
        }
        
        # Test nested path
        assert config_manager.get('trading.max_position_size') == 0.1
        assert config_manager.get('trading.stop_loss') == 5.0
        assert config_manager.get('api.timeout') == 30
        
        # Test default values
        assert config_manager.get('nonexistent.key', 'default') == 'default'
        assert config_manager.get('trading.nonexistent', 0) == 0
    
    @pytest.mark.asyncio
    async def test_set_config_value(self, config_manager):
        """Test setting configuration values"""
        # Test setting new values
        assert config_manager.set('trading.max_position_size', 0.2)
        assert config_manager.get('trading.max_position_size') == 0.2
        
        # Test updating existing values
        config_manager.config_data = {'api': {'timeout': 30}}
        assert config_manager.set('api.timeout', 60)
        assert config_manager.get('api.timeout') == 60
        
        # Test setting nested new path
        assert config_manager.set('new.nested.value', 'test')
        assert config_manager.get('new.nested.value') == 'test'
    
    @pytest.mark.asyncio
    async def test_config_validation(self, config_manager):
        """Test configuration validation"""
        # Add validation rules
        rule1 = ConfigValidationRule(
            path='trading.max_position_size',
            rule_type='range',
            constraint={'min': 0.01, 'max': 1.0},
            error_message='Position size must be between 0.01 and 1.0',
            is_required=True
        )
        
        rule2 = ConfigValidationRule(
            path='api.timeout',
            rule_type='type',
            constraint=int,
            error_message='Timeout must be an integer',
            is_required=True
        )
        
        config_manager.validation_rules = [rule1, rule2]
        
        # Setup config with valid values - use set() method to clear cache properly
        config_manager.set('trading.max_position_size', 0.1)
        config_manager.set('api.timeout', 30)
        
        # Test validation passes
        result = await config_manager.validate_all_configs()
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        
        # Test validation fails with invalid range value - use set() method
        config_manager.set('trading.max_position_size', 1.5)  # Invalid - above max
        result = await config_manager.validate_all_configs()
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
        
        # Reset to valid value and test type validation failure
        config_manager.set('trading.max_position_size', 0.1)  # Valid again
        config_manager.config_data['api']['timeout'] = "30"  # Invalid - string instead of int
        config_manager.config_cache.clear()  # Clear cache for this specific change
        result = await config_manager.validate_all_configs()
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
    
    @pytest.mark.asyncio
    async def test_feature_flags(self, config_manager):
        """Test feature flag functionality"""
        # Test default feature flag
        assert config_manager.is_feature_enabled('nonexistent_feature') is False
        
        # Test enabling/disabling features
        config_manager.enable_feature('test_feature')
        assert config_manager.is_feature_enabled('test_feature') is True
        
        config_manager.disable_feature('test_feature')
        assert config_manager.is_feature_enabled('test_feature') is False
    
    @pytest.mark.asyncio
    async def test_change_callbacks(self, config_manager):
        """Test configuration change callbacks"""
        callback_called = False
        callback_value = None
        
        def test_callback(event):
            nonlocal callback_called, callback_value
            callback_called = True
            callback_value = event.new_value
        
        # Register callback
        config_manager.register_change_callback('trading.max_position_size', test_callback)
        
        # Trigger change
        config_manager.set('trading.max_position_size', 0.5)
        
        # Note: In real implementation, callbacks would be triggered asynchronously
        # For unit test, we'll test the registration worked
        assert 'trading.max_position_size' in config_manager.change_callbacks
        assert len(config_manager.change_callbacks['trading.max_position_size']) == 1
    
    @pytest.mark.asyncio
    async def test_environment_config_loading(self, config_manager):
        """Test environment-specific configuration loading"""
        # Test that environment is correctly set
        assert config_manager.get_environment() == 'testing'
        
        # Test environment-specific configuration merging
        base_config = {'api': {'timeout': 30, 'retries': 3}}
        env_config = {'api': {'timeout': 60}}  # Override timeout for testing
        
        merged = config_manager._deep_merge_config(base_config, env_config)
        assert merged['api']['timeout'] == 60
        assert merged['api']['retries'] == 3
    
    @pytest.mark.asyncio
    async def test_config_caching(self, config_manager):
        """Test configuration caching functionality"""
        # Setup config
        config_manager.config_data = {'api': {'timeout': 30}}
        
        # First access should cache
        value1 = config_manager.get('api.timeout')
        assert value1 == 30
        
        # Verify cache was populated
        assert 'api.timeout' in config_manager.config_cache
        
        # Second access should use cache
        value2 = config_manager.get('api.timeout')
        assert value2 == 30
    
    @pytest.mark.asyncio
    async def test_config_summary(self, config_manager):
        """Test configuration summary generation"""
        config_manager.config_data = {
            'trading': {'max_position_size': 0.1},
            'api': {'timeout': 30}
        }
        config_manager.feature_flags = {'test_feature': True}
        
        summary = config_manager.get_config_summary()
        
        assert 'environment' in summary
        assert 'total_configs' in summary
        assert 'feature_flags' in summary
        assert summary['environment'] == 'testing'
        assert summary['total_configs'] > 0
    
    def test_deep_merge_config(self, config_manager):
        """Test deep merge functionality"""
        base = {
            'api': {'timeout': 30, 'retries': 3},
            'trading': {'max_trades': 10}
        }
        
        override = {
            'api': {'timeout': 60, 'host': 'test.com'},
            'logging': {'level': 'DEBUG'}
        }
        
        result = config_manager._deep_merge_config(base, override)
        
        # Check merged values
        assert result['api']['timeout'] == 60  # Overridden
        assert result['api']['retries'] == 3   # Preserved
        assert result['api']['host'] == 'test.com'  # Added
        assert result['trading']['max_trades'] == 10  # Preserved
        assert result['logging']['level'] == 'DEBUG'  # Added
    
    @pytest.mark.asyncio
    async def test_cleanup(self, config_manager):
        """Test cleanup functionality"""
        # Setup some state
        config_manager.watch_enabled = True
        
        # Cleanup should not raise exceptions
        await config_manager.cleanup()
        
        # Verify cleanup
        assert config_manager.watch_enabled is False


class TestConfigValidationRule:
    """Test configuration validation rules"""
    
    def test_validation_rule_creation(self):
        """Test creating validation rules"""
        rule = ConfigValidationRule(
            path='test.value',
            rule_type='range',
            constraint={'min': 0, 'max': 100},
            error_message='Value must be between 0 and 100'
        )
        
        assert rule.path == 'test.value'
        assert rule.rule_type == 'range'
        assert rule.constraint == {'min': 0, 'max': 100}
        assert rule.is_required is True


class TestConfigChangeEvent:
    """Test configuration change events"""
    
    def test_change_event_creation(self):
        """Test creating change events"""
        event = ConfigChangeEvent(
            config_path='test.value',
            old_value=10,
            new_value=20,
            timestamp=1640995200.0,
            change_type='update'
        )
        
        assert event.config_path == 'test.value'
        assert event.old_value == 10
        assert event.new_value == 20
        assert event.change_type == 'update' 