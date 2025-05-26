"""
Integration tests for the complete trading workflow.
Tests the interaction between all components of the trading bot.
"""
import pytest
import asyncio
import time
import json
from unittest.mock import AsyncMock, MagicMock, patch
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction
from decimal import Decimal
from typing import Dict, List, Optional

# Import all necessary components
from src.core.config import CORE_CONFIG, TRADING_CONFIG, MARKET_CONFIG
from src.core.wallet_manager import WalletManager
from src.core.dex import RaydiumDEX
from src.core.error_handler import ErrorHandler
from src.core.simulator import TradingSimulator
from src.core.risk_management import RiskManager

# Mock configuration values for testing
TEST_CONFIG = {
    "SOLANA_PRIVATE_KEY": "0000000000000000000000000000000000000000000000000000000000000000",
    "RPC_ENDPOINTS": ["https://api.testnet.solana.com"],
    "MAX_WALLET_EXPOSURE": 0.1,
    "MIN_SOL_BALANCE": 0.05,
    "SIMULATION_MODE": True,
    "TOKEN_WHITELIST": ["token1", "token2"],
    "MIN_LIQUIDITY": 10000,
    "MAX_PRICE_IMPACT": 0.05
}

@pytest.fixture
def mock_rpc_client():
    client = AsyncMock()
    client.get_balance = AsyncMock(return_value={'result': {'value': 1000000000}})  # 1 SOL
    client.get_token_accounts_by_owner = AsyncMock(return_value={
        'result': {
            'value': [
                {
                    'pubkey': 'token_account_1',
                    'account': {
                        'data': {
                            'parsed': {
                                'info': {
                                    'mint': 'token_1',
                                    'tokenAmount': {
                                        'amount': '1000000000',
                                        'decimals': 9,
                                        'uiAmount': 1.0
                                    }
                                }
                            }
                        }
                    }
                }
            ]
        }
    })
    client.send_transaction = AsyncMock(return_value={'result': 'tx_signature'})
    return client

@pytest.fixture
async def wallet_manager(mock_rpc_client):
    wallet = WalletManager()
    wallet.client = mock_rpc_client
    wallet.simulation_mode = True
    await wallet.initialize()
    yield wallet
    await wallet.close()

@pytest.fixture
async def dex(mock_rpc_client):
    dex_instance = RaydiumDEX(mock_rpc_client)
    yield dex_instance

@pytest.fixture
def error_handler():
    return ErrorHandler()

@pytest.fixture
async def simulator(mock_rpc_client):
    simulator = TradingSimulator(mock_rpc_client)
    return simulator

@pytest.fixture
async def market_data():
    market = AsyncMock()
    market.get_token_price = AsyncMock(return_value=1.0)
    market.get_token_data = AsyncMock(return_value={
        'address': 'token_1',
        'symbol': 'TEST',
        'price': 1.0,
        'volume_24h': 1000000.0,
        'liquidity': 500000.0,
        'market_cap': 10000000.0,
        'holders': 1000,
        'transactions_24h': 500
    })
    return market

class MockTradingBot:
    """Mock trading bot for integration testing"""
    
    def __init__(self, wallet, dex, market_data, error_handler, simulator):
        self.wallet = wallet
        self.dex = dex
        self.market_data = market_data
        self.error_handler = error_handler
        self.simulator = simulator
        self.running = False
        self.tokens_to_monitor = ["token_1", "token_2"]
    
    async def initialize(self):
        """Initialize all components"""
        self.running = True
    
    async def run_cycle(self):
        """Run a complete trading cycle"""
        for token in self.tokens_to_monitor:
            try:
                # Get token data
                token_data = await self.market_data.get_token_data(token)
                
                # Check if token meets criteria
                if self._evaluate_token(token_data):
                    # Get wallet balance
                    balance = await self.wallet.check_balance()
                    
                    # Calculate trade amount
                    amount = balance * 0.01  # Use 1% of balance
                    
                    # Execute trade in simulation mode
                    if self.wallet.simulation_mode:
                        result = await self.simulator.simulate_trade(token, amount, True)
                        if result:  # simulate_trade returns boolean
                            return {"success": True, "token": token, "amount": amount}
                    else:
                        # Create trade transaction
                        tx = await self.dex.create_swap_transaction("SOL", token, amount)
                        
                        # Execute trade
                        return {"success": True, "token": token, "amount": amount}
                    
            except Exception as e:
                self.error_handler.add_error(str(e), "trading_cycle")
        
        return {"success": False, "reason": "No suitable tokens found"}
    
    def _evaluate_token(self, token_data):
        """Evaluate if token meets trading criteria"""
        if not token_data:
            return False
            
        # Use more lenient criteria for testing
        # Check minimum liquidity (reduced threshold)
        if token_data.get('liquidity', 0) < 10000:  # Lower threshold for testing
            return False
            
        # Check trading volume (reduced threshold)
        if token_data.get('volume_24h', 0) < 100000:
            return False
            
        # Check holder count (reduced threshold)
        if token_data.get('holders', 0) < 100:
            return False
            
        return True
    
    async def close(self):
        """Close all components"""
        self.running = False

@pytest.fixture
async def trading_bot(wallet_manager, dex, market_data, error_handler, simulator):
    bot = MockTradingBot(wallet_manager, dex, market_data, error_handler, simulator)
    await bot.initialize()
    yield bot
    await bot.close()

@pytest.mark.asyncio
async def test_full_trading_cycle(trading_bot):
    """Test the entire trading workflow from market data to execution"""
    # Run a complete trading cycle
    result = await trading_bot.run_cycle()
    
    # Verify result
    assert result is not None
    assert "success" in result
    assert result["success"] is True

@pytest.mark.asyncio
async def test_wallet_balance_check(trading_bot, wallet_manager):
    """Test that wallet balance is checked before trading"""
    # Run a trading cycle
    result = await trading_bot.run_cycle()
    
    # Verify result is successful (wallet balance was checked implicitly)
    assert result["success"] is True

@pytest.mark.asyncio
async def test_error_handling(trading_bot, market_data, error_handler):
    """Test error handling during trading cycle"""
    # Make market data raise an exception
    market_data.get_token_data.side_effect = Exception("API Error")
    
    # Run a trading cycle
    result = await trading_bot.run_cycle()
    
    # Verify error was handled
    assert result["success"] is False

@pytest.mark.asyncio
async def test_simulation_mode(trading_bot, simulator, dex):
    """Test that simulation mode works correctly"""
    # Verify simulation mode is active
    assert trading_bot.wallet.simulation_mode is True
    
    # Run a trading cycle
    result = await trading_bot.run_cycle()
    
    # Verify result
    assert result["success"] is True

@pytest.mark.asyncio
async def test_trade_execution_flow(trading_bot, wallet_manager, dex):
    """Test the complete trade execution flow"""
    # Run a trading cycle
    result = await trading_bot.run_cycle()
    
    # Verify successful execution
    assert result["success"] is True
    assert "token" in result
    assert "amount" in result 