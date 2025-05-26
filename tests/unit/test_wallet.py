"""
Test suite for wallet management functionality.
"""
import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from solders.keypair import Keypair
from src.core.wallet_manager import WalletManager

@pytest.fixture
def wallet(mock_rpc_client):
    """Create a wallet manager instance with mocked RPC client."""
    wallet_manager = WalletManager()
    wallet_manager.client = mock_rpc_client
    wallet_manager.simulation_mode = True  # Force simulation mode for testing
    return wallet_manager

@pytest.mark.asyncio
async def test_wallet_initialization(wallet):
    """Test wallet initialization."""
    assert wallet.client is not None
    assert wallet.simulation_mode is True

@pytest.mark.asyncio
async def test_check_balance(wallet):
    """Test balance retrieval."""
    # In simulation mode, should return simulation_balance
    balance = await wallet.check_balance()
    assert isinstance(balance, float)
    assert balance >= 0
    assert balance == wallet.simulation_balance

@pytest.mark.asyncio
async def test_get_keypair_simulation(wallet):
    """Test keypair retrieval in simulation mode."""
    keypair = wallet.get_keypair()
    assert keypair == "SIMULATION_KEYPAIR"

@pytest.mark.asyncio
async def test_get_public_key_simulation(wallet):
    """Test public key retrieval in simulation mode."""
    public_key = wallet.get_public_key()
    assert public_key == "SIMULATION_PUBLIC_KEY"

@pytest.mark.asyncio
async def test_transfer_sol_simulation(wallet):
    """Test SOL transfer in simulation mode."""
    initial_balance = wallet.simulation_balance
    transfer_amount = 0.01
    
    tx_id = await wallet.transfer_sol("test_address", transfer_amount)
    
    # Should return simulation transaction ID
    assert tx_id.startswith("SIM_TRANSFER_")
    # Should update simulation balance
    assert wallet.simulation_balance == initial_balance - transfer_amount

@pytest.mark.asyncio
async def test_validate_transaction_params(wallet):
    """Test transaction parameter validation."""
    # Valid parameters
    result = await wallet.validate_transaction_params(0.01)
    assert result is True
    
    # Invalid amount (negative)
    with pytest.raises(ValueError):
        await wallet.validate_transaction_params(-0.01)
    
    # Invalid amount (too large)
    with pytest.raises(ValueError):
        await wallet.validate_transaction_params(1001.0)

@pytest.mark.asyncio
async def test_insufficient_funds_error(wallet):
    """Test insufficient funds error."""
    # Try to transfer more than available
    with pytest.raises(Exception):  # Should raise InsufficientFundsError
        await wallet.transfer_sol("test_address", wallet.simulation_balance + 1.0)

@pytest.mark.asyncio
async def test_wallet_security(wallet):
    """Test wallet security features."""
    # Test that private data is not exposed
    assert not hasattr(wallet, '_private_key')
    
    # Test simulation mode protection
    assert wallet.simulation_mode is True

@pytest.mark.asyncio
async def test_transaction_status_simulation(wallet):
    """Test transaction status retrieval in simulation mode."""
    status = await wallet.get_transaction_status("SIM_12345")
    
    assert status["status"] == "confirmed"
    assert "slot" in status
    assert "confirmations" in status
    assert "block_time" in status

@pytest.mark.asyncio
async def test_wallet_close(wallet):
    """Test wallet cleanup."""
    await wallet.close()
    # Verify cleanup was called (keypair should be None after close)
    assert wallet.keypair is None 