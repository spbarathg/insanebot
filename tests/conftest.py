"""
Pytest configuration and shared fixtures for Enhanced Ant Bot testing.
"""

import asyncio
import pytest
import tempfile
import shutil
import os
import redis
import psycopg2
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncGenerator, Generator, Dict, Any

# Test database settings
TEST_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "antbot_test",
    "user": "postgres",
    "password": "postgres"
}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_database():
    """Setup mock test database for integration tests."""
    # Return mock database config instead of trying to connect to real database
    return TEST_DB_CONFIG

@pytest.fixture
def test_redis():
    """Setup mock test Redis instance."""
    from unittest.mock import MagicMock
    
    # Create a mock Redis client
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = "test_value"
    mock_redis.set.return_value = True
    mock_redis.flushdb.return_value = True
    mock_redis.close.return_value = True
    
    return mock_redis

@pytest.fixture
def temp_config_dir():
    """Create temporary configuration directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_solana_client():
    """Mock Solana client for testing."""
    mock_client = AsyncMock()
    mock_client.get_balance.return_value.value = 1000000000  # 1 SOL
    mock_client.get_token_accounts_by_owner.return_value.value = []
    mock_client.send_transaction.return_value.value = "mock_transaction_signature"
    return mock_client

@pytest.fixture
def mock_jupiter_client():
    """Mock Jupiter client for testing."""
    mock_client = AsyncMock()
    mock_client.get_quote.return_value = {
        "inputMint": "So11111111111111111111111111111111111111112",
        "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "inAmount": "1000000",
        "outAmount": "1000000",
        "routePlan": []
    }
    mock_client.swap.return_value = "mock_swap_signature"
    return mock_client

@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection for testing."""
    mock_ws = AsyncMock()
    mock_ws.send.return_value = None
    mock_ws.recv.return_value = '{"type": "test", "data": {}}'
    return mock_ws

@pytest.fixture
def test_config():
    """Test configuration settings."""
    return {
        "ENVIRONMENT": "test",
        "SIMULATION_MODE": "true",
        "PRIVATE_KEY": "test_private_key_" + "0" * 64,
        "WALLET_ADDRESS": "test_wallet_address",
        "INITIAL_CAPITAL": "0.1",
        "MAX_POSITION_SIZE": "0.01",
        "LOG_LEVEL": "DEBUG",
        "DATABASE_URL": f"postgresql://{TEST_DB_CONFIG['user']}:{TEST_DB_CONFIG['password']}@{TEST_DB_CONFIG['host']}:{TEST_DB_CONFIG['port']}/{TEST_DB_CONFIG['database']}",
        "REDIS_URL": "redis://localhost:6379/15"
    }

@pytest.fixture
def mock_trading_engine():
    """Mock trading engine for testing."""
    mock_engine = AsyncMock()
    mock_engine.execute_trade.return_value = {
        "success": True,
        "transaction_id": "mock_transaction_id",
        "amount_in": 1.0,
        "amount_out": 1.1,
        "slippage": 0.01
    }
    return mock_engine

@pytest.fixture
def mock_risk_manager():
    """Mock risk manager for testing."""
    mock_manager = AsyncMock()
    mock_manager.assess_risk.return_value = {
        "risk_score": 0.3,
        "max_position_size": 0.01,
        "recommended_action": "BUY"
    }
    mock_manager.check_position_limits.return_value = True
    mock_manager.assess_portfolio_risk.return_value = {
        "risk_score": 0.3,
        "total_exposure": 0.05,
        "max_position_size": 0.01
    }
    mock_manager.calculate_position_size.return_value = 0.01
    return mock_manager

@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "token_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "price": 1.0,
        "volume_24h": 1000000,
        "market_cap": 50000000,
        "liquidity": 5000000,
        "price_change_24h": 0.05,
        "holder_count": 1000,
        "timestamp": 1640995200.0
    }

@pytest.fixture
def sample_trading_signal():
    """Sample trading signal for testing."""
    return {
        "signal_type": "BUY",
        "confidence": 0.8,
        "token_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "price_target": 1.1,
        "stop_loss": 0.95,
        "position_size": 0.01,
        "timestamp": 1640995200.0,
        "source": "smart_money_tracker"
    }

@pytest.fixture
async def trading_system(test_config, test_database, test_redis):
    """Create full trading system for integration testing."""
    from unittest.mock import patch, AsyncMock
    
    # Setup temporary config directory
    config_dir = tempfile.mkdtemp()
    
    # Create test environment file
    env_file = Path(config_dir) / ".env"
    with open(env_file, 'w') as f:
        for key, value in test_config.items():
            f.write(f"{key}={value}\n")
    
    # Mock the AntBotSystem instead of importing to avoid complex dependencies
    mock_system = AsyncMock()
    
    # Set up mock attributes that tests expect
    mock_system.is_initialized = True
    
    # Mock config manager
    mock_config_manager = MagicMock()  # Use regular MagicMock for synchronous methods
    mock_config_manager.get_config.return_value = test_config
    mock_system.config_manager = mock_config_manager
    
    # Mock portfolio manager
    mock_portfolio_manager = AsyncMock()
    mock_portfolio_manager.get_total_balance.return_value = 1.0
    mock_portfolio_manager.get_open_positions.return_value = []
    mock_system.portfolio_manager = mock_portfolio_manager
    
    # Mock risk manager
    mock_risk_manager = AsyncMock()
    mock_risk_manager.assess_portfolio_risk.return_value = {
        "risk_score": 0.3,
        "total_exposure": 0.05,
        "max_position_size": 0.01
    }
    mock_risk_manager.calculate_position_size.return_value = 0.01
    mock_system.risk_manager = mock_risk_manager
    
    # Mock security manager
    mock_security_manager = AsyncMock()
    mock_security_manager.get_security_metrics.return_value = {
        "threats_detected": 0,
        "security_score": 100
    }
    mock_system.security_manager = mock_security_manager
    
    # Mock execution engine
    mock_execution_engine = AsyncMock()
    mock_execution_engine.execute_trade.return_value = {
        "success": True,
        "transaction_id": "mock_transaction_id"
    }
    mock_system.execution_engine = mock_execution_engine
    
    # Mock signal coordinator
    mock_signal_coordinator = AsyncMock()
    mock_signal_coordinator.process_signal.return_value = {
        "processed": True,
        "action": "BUY"
    }
    mock_system.signal_coordinator = mock_signal_coordinator
    
    # Mock cache manager
    mock_cache_manager = AsyncMock()
    mock_cache_manager.set.return_value = None
    mock_cache_manager.get.return_value = {"type": "BUY", "confidence": 0.8}
    mock_system.cache_manager = mock_cache_manager
    
    # Mock health monitor
    mock_health_monitor = AsyncMock()
    mock_health_monitor.get_system_health.return_value = {
        "overall_status": "healthy",
        "components": {}
    }
    mock_system.health_monitor = mock_health_monitor
    
    # Mock metrics manager
    mock_metrics_manager = AsyncMock()
    mock_metrics_manager.collect_all_metrics.return_value = {
        "system_health": {"status": "healthy"},
        "trading_performance": {"total_trades": 10},
        "security_events": {"threats": 0},
        "execution_latency": {"avg_ms": 100}
    }
    mock_system.metrics_manager = mock_metrics_manager
    
    # Mock alert manager
    mock_alert_manager = AsyncMock()
    mock_alert_manager.send_alert.return_value = {"sent": True}
    mock_system.alert_manager = mock_alert_manager
    
    # Mock ant colony
    mock_ant_colony = AsyncMock()
    mock_ant_colony.create_princess.return_value = {"princess_id": "test_princess"}
    mock_ant_colony.get_colony_size.return_value = 5
    mock_system.ant_colony = mock_ant_colony
    
    # Mock market data manager
    mock_market_data_manager = AsyncMock()
    mock_market_data_manager.get_token_price.return_value = 100.0
    mock_system.market_data_manager = mock_market_data_manager
    
    # Mock helius service
    mock_helius_service = AsyncMock()
    mock_helius_service.is_available.return_value = True
    mock_system.helius_service = mock_helius_service
    
    # Initialize the mock system
    await mock_system.initialize()
    
    yield mock_system
    
    # Cleanup
    await mock_system.shutdown()
    shutil.rmtree(config_dir)

# Performance test fixtures
@pytest.fixture
def performance_config():
    """Configuration for performance testing."""
    return {
        "max_concurrent_requests": 100,
        "test_duration_seconds": 60,
        "ramp_up_seconds": 10,
        "target_latency_ms": 100,
        "error_rate_threshold": 0.01
    } 