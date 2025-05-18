"""
Test suite for trade execution functionality.
"""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from solana.transaction import Transaction
from solana.keypair import Keypair

# Import directly from the project structure
from src.core.config import TRADING_CONFIG, CORE_CONFIG
from src.core.trade_execution import TradeExecution

@pytest.fixture
def mock_solana_client():
    client = AsyncMock()
    client.send_transaction = AsyncMock(return_value={"signature": "test_signature"})
    client.get_balance = AsyncMock(return_value={'result': {'value': 1000000000}})  # 1 SOL
    return client

@pytest.fixture
def mock_session():
    session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "price": 1.0,
        "priceImpactPct": 0.05,
        "swapInstruction": "mock_instruction"
    })
    session.get = AsyncMock(return_value=mock_response)
    session.post = AsyncMock(return_value=mock_response)
    return session

@pytest.fixture
async def trade_execution(mock_solana_client, mock_session):
    with patch('aiohttp.ClientSession', return_value=mock_session):
        with patch('solana.rpc.async_api.AsyncClient', return_value=mock_solana_client):
            # Configure the trade execution with test settings
            trade_exec = TradeExecution()
            trade_exec.wallet = Keypair()
            trade_exec.session = mock_session
            trade_exec.solana_client = mock_solana_client
            yield trade_exec
            await trade_exec.close()

@pytest.mark.asyncio
async def test_initialization(trade_execution):
    """Test that TradeExecution initializes correctly"""
    assert trade_execution is not None
    assert trade_execution.session is not None
    assert trade_execution.solana_client is not None
    assert isinstance(trade_execution.wallet, Keypair)
    assert len(trade_execution.active_trades) == 0

@pytest.mark.asyncio
async def test_get_liquidity_info(trade_execution, mock_session):
    """Test getting liquidity information for a token"""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        'pairs': [{
            'liquidity': {'usd': 100000},
            'lockTime': 86400,
            'fdv': 1000000
        }]
    })
    mock_session.get.return_value = mock_response
    
    liquidity_info = await trade_execution._get_liquidity_info("test_token")
    
    assert liquidity_info is not None
    assert liquidity_info['amount'] == 100000
    assert liquidity_info['lock_time'] == 86400
    assert liquidity_info['market_cap'] == 1000000
    
    mock_session.get.assert_called_once()

@pytest.mark.asyncio
async def test_get_token_info(trade_execution, mock_session):
    """Test getting token information"""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        'holderCount': 1000,
        'ownerConcentration': 0.5,
        'age': 30
    })
    mock_session.get.return_value = mock_response
    
    token_info = await trade_execution._get_token_info("test_token")
    
    assert token_info is not None
    assert token_info['holder_count'] == 1000
    assert token_info['owner_concentration'] == 0.5
    assert token_info['age'] == 30

@pytest.mark.asyncio
async def test_calculate_price_impact(trade_execution, mock_session):
    """Test calculating price impact for a trade"""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={'priceImpactPct': 0.02})
    mock_session.get.return_value = mock_response
    
    price_impact = await trade_execution._calculate_price_impact("test_token", 1.0)
    
    assert price_impact == 0.02

@pytest.mark.asyncio
async def test_get_swap_quote(trade_execution, mock_session):
    """Test getting swap quote"""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={'quote': 'test_quote'})
    mock_session.get.return_value = mock_response
    
    quote = await trade_execution._get_swap_quote()
    
    assert quote == {'quote': 'test_quote'}

@pytest.mark.asyncio
async def test_create_swap_transaction(trade_execution):
    """Test creating a swap transaction"""
    with patch.object(trade_execution, '_get_swap_quote', return_value={'quote': 'test_quote'}):
        with patch.object(trade_execution, '_create_swap_instruction', return_value='test_instruction'):
            with patch('solana.transaction.Transaction') as mock_tx:
                instance = mock_tx.return_value
                
                tx = await trade_execution._create_swap_transaction()
                
                assert instance.add.called
                assert instance.sign.called
                assert tx is not None

@pytest.mark.asyncio
async def test_check_liquidity(trade_execution):
    """Test checking token liquidity"""
    with patch.object(trade_execution, '_get_liquidity_info', return_value={
        'amount': 100000,
        'lock_time': 86400,
        'market_cap': 1000000
    }):
        result = await trade_execution.check_liquidity("test_token")
        assert result is True

@pytest.mark.asyncio
async def test_check_rug_risk(trade_execution):
    """Test checking rug risk for a token"""
    with patch.object(trade_execution, '_get_token_info', return_value={
        'holder_count': 1000,
        'owner_concentration': 0.2,
        'age': 30
    }):
        result = await trade_execution.check_rug_risk("test_token")
        assert result is False  # No rug risk detected

@pytest.mark.asyncio
async def test_get_available_balance(trade_execution, mock_solana_client):
    """Test getting available balance"""
    mock_solana_client.get_balance = AsyncMock(return_value={'result': {'value': 1000000000}})
    
    balance = trade_execution.get_available_balance()
    
    assert balance == 1.0  # 1 SOL

@pytest.mark.asyncio
async def test_get_trade_pnl(trade_execution):
    """Test getting trade PnL"""
    # Mock active trade
    trade_execution.active_trades = {
        "test_token": {
            "entry_price": 1.0,
            "amount": 100
        }
    }
    
    pnl = trade_execution.get_trade_pnl("test_token", 1.5)
    
    assert pnl == 50.0  # 50% profit 