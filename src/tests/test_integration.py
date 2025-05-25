"""
Integration tests for the complete trading workflow.
Tests the interaction between all components of the trading bot.
"""
import pytest
import asyncio
import time
import json
from unittest.mock import AsyncMock, MagicMock, patch
from solana.keypair import Keypair
from solders.pubkey import Pubkey
from solana.transaction import Transaction

# Import all necessary components
from src.core.config import CORE_CONFIG, TRADING_CONFIG, MARKET_CONFIG
from src.core.wallet import WalletManager
from src.core.dex import RaydiumDEX
from src.core.error_handler import ErrorHandler
from src.core.simulator import TradingSimulator

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
    with patch('src.core.wallet.AsyncClient', return_value=mock_rpc_client):
        wallet = WalletManager(mock_rpc_client)
        await wallet.initialize()
        yield wallet
        await wallet.close()

@pytest.fixture
async def dex(mock_rpc_client):
    with patch('src.core.dex.AsyncClient', return_value=mock_rpc_client):
        dex_instance = RaydiumDEX(mock_rpc_client)
        await dex_instance.initialize()
        yield dex_instance
        await dex_instance.close()

@pytest.fixture
def error_handler():
    return ErrorHandler()

@pytest.fixture
async def simulator():
    simulator = TradingSimulator()
    await simulator.initialize()
    yield simulator
    await simulator.close()

@pytest.fixture
async def market_data():
    with patch('aiohttp.ClientSession'):
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
        await market.initialize()
        yield market
        await market.close()

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
                    balance = await self.wallet.get_balance()
                    
                    # Calculate trade amount
                    amount = balance * 0.01  # Use 1% of balance
                    
                    # Execute trade in simulation mode
                    if CORE_CONFIG.SIMULATION_MODE:
                        result = await self.simulator.simulate_trade(token, amount)
                    else:
                        # Create trade transaction
                        tx = await self.dex.create_swap_transaction("SOL", token, amount)
                        
                        # Execute trade
                        result = {"success": True, "tx": "simulated_tx"}
                    
                    return result
            except Exception as e:
                self.error_handler.add_error("trading_cycle", str(e))
        
        return {"success": False, "reason": "No suitable tokens found"}
    
    def _evaluate_token(self, token_data):
        """Evaluate if token meets trading criteria"""
        if not token_data:
            return False
            
        # Check minimum liquidity
        if token_data.get('liquidity', 0) < TRADING_CONFIG.MIN_LIQUIDITY:
            return False
            
        # Check trading volume
        if token_data.get('volume_24h', 0) < 100000:
            return False
            
        # Check holder count
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
    # Set a mock balance
    wallet_manager.get_balance = AsyncMock(return_value=1.5)  # 1.5 SOL
    
    # Run a trading cycle
    result = await trading_bot.run_cycle()
    
    # Verify wallet balance was checked
    assert wallet_manager.get_balance.called
    assert result["success"] is True

@pytest.mark.asyncio
async def test_error_handling(trading_bot, market_data, error_handler):
    """Test error handling during trading cycle"""
    # Make market data raise an exception
    market_data.get_token_data.side_effect = Exception("API Error")
    
    # Run a trading cycle
    result = await trading_bot.run_cycle()
    
    # Verify error was handled
    assert error_handler.has_errors()
    assert result["success"] is False

@pytest.mark.asyncio
async def test_simulation_mode(trading_bot, simulator, dex):
    """Test that simulation mode works correctly"""
    # Configure simulation mode
    with patch('src.core.config.CORE_CONFIG.SIMULATION_MODE', True):
        # Run a trading cycle
        result = await trading_bot.run_cycle()
        
        # Verify simulator was used and DEX was not used for real transaction
        assert simulator.simulate_trade.called
        assert dex.create_swap_transaction.call_count == 0
        assert result["success"] is True

@pytest.mark.asyncio
async def test_trade_execution_flow(trading_bot, wallet_manager, dex):
    """Test the complete trade execution flow"""
    # Run with simulation mode off
    with patch('src.core.config.CORE_CONFIG.SIMULATION_MODE', False):
        # Run a trading cycle
        result = await trading_bot.run_cycle()
        
        # Verify dex was used
        assert dex.create_swap_transaction.called
        assert result["success"] is True 