"""
Test suite for DEX interaction functionality.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from solana.transaction import Transaction
from src.core.dex import RaydiumDEX

@pytest.fixture
async def dex(mock_rpc_client):
    """Create a DEX instance with mocked RPC client."""
    return RaydiumDEX(mock_rpc_client)

@pytest.mark.asyncio
async def test_dex_initialization(dex):
    """Test DEX initialization."""
    assert dex.rpc_client is not None
    assert dex.pools_cache is not None
    assert isinstance(dex.pools_cache, dict)

@pytest.mark.asyncio
async def test_create_swap_transaction(dex):
    """Test swap transaction creation."""
    token_address = "test_token_address"
    amount = 1.0
    is_buy = True
    keypair = MagicMock()
    
    transaction = await dex.create_swap_transaction(
        token_address,
        amount,
        is_buy,
        keypair
    )
    
    assert isinstance(transaction, Transaction)
    assert len(transaction.instructions) > 0

@pytest.mark.asyncio
async def test_get_pool_info(dex):
    """Test pool information retrieval."""
    token_address = "test_token_address"
    pool_info = await dex._get_pool_info(token_address)
    
    assert pool_info is not None
    assert "address" in pool_info
    assert "token_a" in pool_info
    assert "token_b" in pool_info

@pytest.mark.asyncio
async def test_get_token_price(dex):
    """Test token price retrieval."""
    token_address = "test_token_address"
    price_data = await dex.get_token_price(token_address)
    
    assert isinstance(price_data, dict)
    assert "price" in price_data
    assert price_data["price"] > 0

@pytest.mark.asyncio
async def test_get_liquidity(dex):
    """Test liquidity information retrieval."""
    token_address = "test_token_address"
    liquidity = await dex.get_liquidity(token_address)
    
    assert isinstance(liquidity, float)
    assert liquidity >= 0

@pytest.mark.asyncio
async def test_error_handling(dex):
    """Test error handling in DEX operations."""
    # Mock RPC error
    dex.rpc_client.get_account_info.side_effect = Exception("RPC Error")
    
    # Test error handling in price retrieval
    price = await dex.get_token_price("invalid_token")
    assert price is None
    
    # Test error handling in liquidity retrieval
    liquidity = await dex.get_liquidity("invalid_token")
    assert liquidity is None

@pytest.mark.asyncio
async def test_pool_cache(dex):
    """Test pool information caching."""
    token_address = "test_token_address"
    
    # First call should fetch from RPC
    pool_info1 = await dex._get_pool_info(token_address)
    
    # Second call should use cache
    pool_info2 = await dex._get_pool_info(token_address)
    
    assert pool_info1 == pool_info2
    assert dex.rpc_client.get_account_info.call_count == 1

@pytest.mark.asyncio
async def test_swap_validation(dex):
    """Test swap transaction validation."""
    token_address = "test_token_address"
    amount = 1.0
    is_buy = True
    keypair = MagicMock()
    
    # Test with valid parameters
    transaction = await dex.create_swap_transaction(
        token_address,
        amount,
        is_buy,
        keypair
    )
    assert transaction is not None
    
    # Test with invalid amount
    with pytest.raises(ValueError):
        await dex.create_swap_transaction(
            token_address,
            -1.0,  # Invalid amount
            is_buy,
            keypair
        )
    
    # Test with invalid token
    with pytest.raises(ValueError):
        await dex.create_swap_transaction(
            "invalid_token",
            amount,
            is_buy,
            keypair
        ) 