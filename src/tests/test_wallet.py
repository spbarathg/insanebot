"""
Test suite for wallet management functionality.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from solana.keypair import Keypair
from src.core.wallet import WalletManager

@pytest.fixture
async def wallet(mock_rpc_client):
    """Create a wallet manager instance with mocked RPC client."""
    return WalletManager(mock_rpc_client)

@pytest.mark.asyncio
async def test_wallet_initialization(wallet):
    """Test wallet initialization."""
    assert wallet.rpc_client is not None
    assert wallet.keypair is not None
    assert isinstance(wallet.keypair, Keypair)

@pytest.mark.asyncio
async def test_get_balance(wallet):
    """Test balance retrieval."""
    # Configure mock return value
    wallet.rpc_client.get_balance.return_value = {"result": {"value": 10000000000}}  # 10 SOL
    
    balance = await wallet.get_balance()
    assert isinstance(balance, float)
    assert balance >= 0

@pytest.mark.asyncio
async def test_get_token_balance(wallet):
    """Test token balance retrieval."""
    token_address = "test_token_address"
    
    # Configure mock return values
    wallet.rpc_client.get_token_accounts_by_owner.return_value = {
        "result": {
            "value": [
                {"pubkey": "test_pubkey"}
            ]
        }
    }
    
    wallet.rpc_client.get_token_account_balance.return_value = {
        "result": {
            "value": {
                "uiAmount": "10.5"
            }
        }
    }
    
    balance = await wallet.get_token_balance(token_address)
    
    assert isinstance(balance, float)
    assert balance >= 0

@pytest.mark.asyncio
async def test_get_keypair(wallet):
    """Test keypair retrieval."""
    keypair = wallet.get_keypair()
    assert isinstance(keypair, Keypair)
    assert keypair.public_key is not None

@pytest.mark.asyncio
async def test_error_handling(wallet):
    """Test error handling in wallet operations."""
    # Mock RPC error
    wallet.rpc_client.get_balance.side_effect = Exception("RPC Error")
    
    # Test error handling in balance retrieval
    balance = await wallet.get_balance()
    assert balance is None
    
    # Reset mock
    wallet.rpc_client.get_balance.side_effect = None
    wallet.rpc_client.get_balance.return_value = {"result": {"value": 10000000000}}  # 10 SOL
    
    # Mock token balance error
    wallet.rpc_client.get_token_accounts_by_owner.side_effect = Exception("RPC Error")
    
    # Test error handling in token balance retrieval
    token_balance = await wallet.get_token_balance("invalid_token")
    assert token_balance is None

@pytest.mark.asyncio
async def test_balance_cache(wallet):
    """Test balance caching."""
    # Configure mock return value
    wallet.rpc_client.get_balance.return_value = {"result": {"value": 10000000000}}  # 10 SOL
    
    # First call should fetch from RPC
    balance1 = await wallet.get_balance()
    
    # Second call should use cache
    balance2 = await wallet.get_balance()
    
    assert balance1 == balance2
    assert wallet.rpc_client.get_balance.call_count == 1

@pytest.mark.asyncio
async def test_token_validation(wallet):
    """Test token validation."""
    # Configure mock return values for valid token
    wallet.rpc_client.get_token_accounts_by_owner.return_value = {
        "result": {
            "value": [
                {"pubkey": "test_pubkey"}
            ]
        }
    }
    
    wallet.rpc_client.get_token_account_balance.return_value = {
        "result": {
            "value": {
                "uiAmount": "10.5"
            }
        }
    }
    
    # Test with valid token
    balance = await wallet.get_token_balance("valid_token")
    assert balance is not None
    
    # Test with invalid token
    with pytest.raises(ValueError):
        await wallet.get_token_balance("")

@pytest.mark.asyncio
async def test_wallet_security(wallet):
    """Test wallet security features."""
    # Test keypair protection
    keypair1 = wallet.get_keypair()
    keypair2 = wallet.get_keypair()
    assert keypair1 == keypair2  # Should return same instance
    
    # Test private key protection
    assert not hasattr(wallet, '_private_key')  # Private key should not be exposed

@pytest.mark.asyncio
async def test_balance_updates(wallet):
    """Test balance update mechanism."""
    # Mock initial balance
    wallet.rpc_client.get_balance.return_value = {"result": {"value": 10000000000}}  # 10 SOL
    
    # Get initial balance
    initial_balance = await wallet.get_balance()
    assert initial_balance == 10.0
    
    # Mock updated balance
    wallet.rpc_client.get_balance.return_value = {"result": {"value": 15000000000}}  # 15 SOL
    
    # Force cache refresh
    wallet._balance_cache = None
    
    # Get updated balance
    updated_balance = await wallet.get_balance()
    assert updated_balance == 15.0 