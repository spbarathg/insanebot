"""
Comprehensive test suite for the Solana trading bot.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any

from src.main import SimpleTradingBot
from src.core.error_handler import ErrorHandler, CircuitBreaker
from src.core.dex import RaydiumDEX
from src.core.wallet import WalletManager

@pytest.fixture
def mock_rpc_client():
    """Mock Solana RPC client."""
    client = AsyncMock()
    client.get_balance.return_value = {"result": {"value": 1000000000}}  # 1 SOL
    return client

@pytest.fixture
def mock_dex():
    """Mock DEX interface."""
    dex = Mock(spec=RaydiumDEX)
    dex.get_token_price.return_value = {
        "price": 0.01,
        "price_change_1h": 0.05,
        "volume_change_1h": 0.03
    }
    dex.get_liquidity.return_value = 50000.0
    return dex

@pytest.fixture
def mock_wallet():
    """Mock wallet manager."""
    wallet = Mock(spec=WalletManager)
    wallet.get_balance.return_value = 1.0
    return wallet

@pytest.fixture
def trading_bot(mock_rpc_client, mock_dex, mock_wallet):
    """Create trading bot instance with mocks."""
    return SimpleTradingBot(
        wallet_address="test_wallet",
        simulation_mode=True
    )

class TestTradingBot:
    """Test suite for trading bot functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, trading_bot):
        """Test bot initialization."""
        await trading_bot.initialize()
        assert trading_bot.is_running is False
        assert len(trading_bot.known_tokens) == 0
    
    @pytest.mark.asyncio
    async def test_token_discovery(self, trading_bot, mock_rpc_client):
        """Test token discovery functionality."""
        # Mock RPC response
        mock_rpc_client.get_program_accounts.return_value = {
            "result": [
                {"pubkey": "token1"},
                {"pubkey": "token2"}
            ]
        }
        
        tokens = await trading_bot._get_tokens_to_check()
        assert len(tokens) == 2
        assert "token1" in tokens
        assert "token2" in tokens
    
    @pytest.mark.asyncio
    async def test_token_data_validation(self, trading_bot, mock_dex):
        """Test token data validation."""
        token_data = await trading_bot._get_token_data("test_token")
        assert token_data is not None
        assert "price" in token_data
        assert "liquidity" in token_data
        assert "should_buy" in token_data
        assert "should_sell" in token_data
    
    @pytest.mark.asyncio
    async def test_trade_execution(self, trading_bot, mock_dex, mock_wallet):
        """Test trade execution."""
        # Mock successful trade
        mock_wallet.send_transaction.return_value = True
        
        success = await trading_bot.execute_trade("test_token", is_buy=True)
        assert success is True
        assert "test_token" in trading_bot.active_trades
    
    @pytest.mark.asyncio
    async def test_error_handling(self, trading_bot):
        """Test error handling and circuit breaker."""
        # Simulate network error
        with patch.object(trading_bot.rpc_client, 'get_balance', side_effect=Exception("Network error")):
            with pytest.raises(Exception):
                await trading_bot._initialize_wallet()
        
        # Check error handler state
        assert trading_bot.error_handler.should_stop_trading() is False
    
    @pytest.mark.asyncio
    async def test_trading_cycle(self, trading_bot, mock_rpc_client, mock_dex):
        """Test complete trading cycle."""
        # Mock token discovery
        mock_rpc_client.get_program_accounts.return_value = {
            "result": [{"pubkey": "test_token"}]
        }
        
        # Mock successful trade
        mock_dex.create_swap_transaction.return_value = True
        
        # Run one trading cycle
        await trading_bot._process_trading_cycle()
        
        # Verify state
        assert len(trading_bot.active_trades) > 0
    
    @pytest.mark.asyncio
    async def test_cleanup(self, trading_bot):
        """Test cleanup and shutdown."""
        await trading_bot.close()
        assert trading_bot.is_running is False

class TestErrorHandler:
    """Test suite for error handling functionality."""
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=1)
        
        # Simulate failures
        for _ in range(3):
            breaker.record_failure()
        
        assert breaker.is_open is True
        assert breaker.can_execute() is False
        
        # Wait for reset
        breaker.last_failure_time = 0
        assert breaker.can_execute() is True
    
    def test_error_recording(self):
        """Test error recording and categorization."""
        handler = ErrorHandler()
        
        # Record different types of errors
        handler.add_error(Exception("Network error"), "network")
        handler.add_error(Exception("Transaction failed"), "transaction")
        
        summary = handler.get_error_summary()
        assert summary["total_errors"] == 2
        assert "network" in summary["error_types"]
        assert "transaction" in summary["error_types"]
    
    def test_error_cooldown(self):
        """Test error cooldown mechanism."""
        handler = ErrorHandler(error_cooldown=1)
        
        # Record errors
        for _ in range(5):
            handler.add_error(Exception("Test error"))
        
        # Should stop trading due to error frequency
        assert handler.should_stop_trading() is True
        
        # Clear errors
        handler.clear_errors()
        assert handler.should_stop_trading() is False

class TestDEX:
    """Test suite for DEX interaction."""
    
    @pytest.mark.asyncio
    async def test_price_caching(self, mock_rpc_client):
        """Test price caching mechanism."""
        dex = RaydiumDEX(mock_rpc_client)
        
        # First call should cache
        price1 = await dex.get_token_price("test_token")
        price2 = await dex.get_token_price("test_token")
        
        assert price1 == price2
        assert "test_token" in dex._price_cache
    
    @pytest.mark.asyncio
    async def test_liquidity_caching(self, mock_rpc_client):
        """Test liquidity caching mechanism."""
        dex = RaydiumDEX(mock_rpc_client)
        
        # First call should cache
        liq1 = await dex.get_liquidity("test_token")
        liq2 = await dex.get_liquidity("test_token")
        
        assert liq1 == liq2
        assert "test_token" in dex._liquidity_cache
    
    @pytest.mark.asyncio
    async def test_swap_transaction(self, mock_rpc_client):
        """Test swap transaction creation."""
        dex = RaydiumDEX(mock_rpc_client)
        
        # Mock successful transaction creation
        transaction = await dex.create_swap_transaction(
            "test_token",
            1.0,
            True,
            Mock()
        )
        
        assert transaction is not None

class TestWallet:
    """Test suite for wallet management."""
    
    def test_wallet_encryption(self, mock_rpc_client):
        """Test wallet encryption and decryption."""
        wallet = WalletManager(mock_rpc_client)
        
        # Test wallet loading
        wallet.load_wallet("test_wallet.enc")
        assert wallet.keypair is not None
    
    @pytest.mark.asyncio
    async def test_balance_check(self, mock_rpc_client):
        """Test balance checking functionality."""
        wallet = WalletManager(mock_rpc_client)
        wallet.load_wallet("test_wallet.enc")
        
        balance = await wallet.get_balance()
        assert balance > 0
    
    @pytest.mark.asyncio
    async def test_transaction_sending(self, mock_rpc_client):
        """Test transaction sending with retry logic."""
        wallet = WalletManager(mock_rpc_client)
        wallet.load_wallet("test_wallet.enc")
        
        # Mock successful transaction
        mock_rpc_client.send_transaction.return_value = {"result": "success"}
        
        success = await wallet.send_transaction(Mock())
        assert success is True 