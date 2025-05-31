"""
Pytest configuration and fixtures for Enhanced Ant Bot System tests
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, Generator

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test environment setup
os.environ.update({
    'ANT_BOT_ENV': 'testing',
    'SIMULATION_MODE': 'true',
    'LOG_LEVEL': 'DEBUG',
    'INITIAL_CAPITAL': '0.01',
    'PRIVATE_KEY': 'test_private_key_placeholder',
    'WALLET_PASSWORD': 'test_password',
    'QUICKNODE_ENDPOINT_URL': 'https://test.quicknode.com',
    'HELIUS_API_KEY': 'test_helius_key',
    'JUPITER_API_KEY': 'test_jupiter_key'
})

# Mock missing config modules that might not exist
class MockConfig:
    """Mock configuration for missing config modules"""
    
    CORE_CONFIG = {
        'system': {
            'name': 'Enhanced Ant Bot',
            'version': '1.0.0',
            'environment': 'testing'
        }
    }
    
    MARKET_CONFIG = {
        'default_slippage': 0.01,
        'max_slippage': 0.05,
        'price_impact_threshold': 0.02
    }
    
    TRADING_CONFIG = {
        'min_trade_amount': 0.001,
        'max_trade_amount': 1.0,
        'default_position_size': 0.01
    }

# Mock the config imports before they're imported by other modules
sys.modules['config.core_config'] = MockConfig()
sys.modules['config.ant_princess_config'] = MockConfig()

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_wallet_manager():
    """Mock wallet manager for testing"""
    mock = AsyncMock()
    mock.get_balance.return_value = 1.0
    mock.get_address.return_value = "test_wallet_address"
    mock.sign_transaction.return_value = "test_signature"
    mock.initialize.return_value = True
    return mock

@pytest.fixture
def mock_quicknode_service():
    """Mock QuickNode service"""
    mock = AsyncMock()
    mock.get_token_price.return_value = 100.0
    mock.get_token_metadata.return_value = {
        'symbol': 'TEST',
        'name': 'Test Token',
        'decimals': 9
    }
    return mock

@pytest.fixture
def mock_helius_service():
    """Mock Helius service"""
    mock = AsyncMock()
    mock.get_wallet_activity.return_value = []
    return mock

@pytest.fixture
def mock_jupiter_service():
    """Mock Jupiter service"""
    mock = AsyncMock()
    mock.get_quote.return_value = {
        'inputMint': 'test_input',
        'outputMint': 'test_output',
        'inAmount': '1000000',
        'outAmount': '2000000'
    }
    return mock

@pytest.fixture
def mock_titan_shield():
    """Mock Titan Shield coordinator"""
    mock = AsyncMock()
    mock.analyze_threat_level.return_value = {
        'threat_level': 'LOW',
        'threat_score': 0.1,
        'recommendations': []
    }
    mock.get_defense_mode.return_value = 'NORMAL'
    mock.initialize.return_value = True
    return mock

@pytest.fixture
def mock_grok_engine():
    """Mock Grok AI engine"""
    mock = AsyncMock()
    mock.analyze_market_sentiment.return_value = {
        'sentiment': 'bullish',
        'confidence': 0.8,
        'reasoning': 'Test analysis'
    }
    mock.initialize.return_value = True
    return mock

@pytest.fixture
def mock_local_llm():
    """Mock Local LLM"""
    mock = AsyncMock()
    mock.analyze_technical_indicators.return_value = {
        'signal': 'buy',
        'strength': 0.7,
        'indicators': {'rsi': 30, 'macd': 'bullish'}
    }
    mock.initialize.return_value = True
    return mock

@pytest.fixture
def sample_token_data():
    """Sample token data for testing"""
    return {
        'address': 'So11111111111111111111111111111111111111112',
        'symbol': 'SOL',
        'name': 'Solana',
        'decimals': 9,
        'price': 100.0,
        'market_cap': 50000000000,
        'volume_24h': 1000000000,
        'price_change_24h': 5.5
    }

@pytest.fixture
def sample_trade_data():
    """Sample trade data for testing"""
    return {
        'token_address': 'So11111111111111111111111111111111111111112',
        'action': 'buy',
        'amount': 0.1,
        'price': 100.0,
        'timestamp': 1640995200,
        'gas_fee': 0.00001
    }

@pytest.fixture
def test_config():
    """Test configuration dictionary"""
    return {
        'initial_capital': 0.01,
        'max_position_size': 0.001,
        'stop_loss_percent': 5.0,
        'take_profit_percent': 20.0,
        'max_trades_per_hour': 10,
        'threat_detection': {
            'enable_rate_limiting': True,
            'enable_anomaly_detection': True,
            'rate_limit_threshold': 100
        },
        'encryption': {
            'algorithm': 'AES-256',
            'key_rotation_interval': 86400
        },
        'access_control': {
            'token_expiry': 3600,
            'max_failed_attempts': 3
        }
    }

@pytest.fixture
def mock_ai_coordinator():
    """Mock AI coordinator"""
    mock = AsyncMock()
    mock.analyze_opportunity.return_value = {
        'action': 'buy',
        'confidence': 0.8,
        'reasoning': 'Test analysis'
    }
    return mock

@pytest.fixture
def mock_portfolio_manager():
    """Mock portfolio manager"""
    mock = AsyncMock()
    mock.get_portfolio_value.return_value = 1.0
    mock.get_positions.return_value = {}
    mock.calculate_position_size.return_value = 0.1
    return mock

@pytest.fixture
def mock_portfolio_risk_manager():
    """Mock portfolio risk manager"""
    mock = AsyncMock()
    mock.assess_risk.return_value = {
        'risk_level': 'LOW',
        'risk_score': 0.2,
        'max_position_size': 0.1
    }
    mock.initialize.return_value = True
    return mock

# Test data directories
@pytest.fixture
def test_data_dir():
    """Test data directory"""
    return Path(__file__).parent / "data"

@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test files"""
    return tmp_path

# Async test helper
@pytest.fixture
def async_test_helper():
    """Helper for running async tests"""
    def run_async(coro):
        return asyncio.get_event_loop().run_until_complete(coro)
    return run_async

# Mock environment for isolated testing
@pytest.fixture
def isolated_env():
    """Isolated environment for testing"""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)

# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer for performance tests"""
    import time
    start_time = time.time()
    yield lambda: time.time() - start_time

# Database mocking
@pytest.fixture
def mock_database():
    """Mock database for testing"""
    mock = Mock()
    mock.execute.return_value = None
    mock.fetchall.return_value = []
    mock.fetchone.return_value = None
    return mock

# Configuration patches for missing modules
@pytest.fixture(autouse=True)
def patch_missing_imports():
    """Automatically patch missing imports"""
    patches = []
    
    # Patch watchdog if not available
    try:
        import watchdog
    except ImportError:
        mock_watchdog = MagicMock()
        mock_watchdog.observers.Observer = MagicMock
        mock_watchdog.events.FileSystemEventHandler = MagicMock
        patches.append(patch.dict('sys.modules', {'watchdog': mock_watchdog, 'watchdog.observers': mock_watchdog.observers, 'watchdog.events': mock_watchdog.events}))
    
    # Patch yaml if not available
    try:
        import yaml
    except ImportError:
        mock_yaml = MagicMock()
        patches.append(patch.dict('sys.modules', {'yaml': mock_yaml}))
    
    # Start all patches
    for p in patches:
        p.start()
    
    yield
    
    # Stop all patches
    for p in patches:
        p.stop()

# Mock external services that might not be available
@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external services that tests might try to import"""
    with patch.dict('sys.modules', {
        'solana': MagicMock(),
        'anchorpy': MagicMock(),
        'solders': MagicMock(),
        'base58': MagicMock(),
        'construct': MagicMock(),
        'construct_typing': MagicMock(),
    }):
        yield 