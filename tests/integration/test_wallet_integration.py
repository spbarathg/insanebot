import pytest
import asyncio
from unittest.mock import AsyncMock
from solders.keypair import Keypair
from src.core.wallet import WalletManager

@pytest.mark.asyncio
async def test_send_transaction_with_confirmation(monkeypatch):
    # Mock RPC client
    mock_rpc = AsyncMock()
    # Simulate send_transaction returning a signature
    mock_rpc.send_transaction.return_value = {"result": "test_signature"}
    # Simulate get_signature_statuses returning confirmed
    async def mock_get_signature_statuses(signatures):
        return {"result": {"value": [{"confirmationStatus": "confirmed"}]}}
    mock_rpc.get_signature_statuses.side_effect = mock_get_signature_statuses

    wallet = WalletManager(mock_rpc)
    wallet.keypair = Keypair()  # Use dummy keypair
    # Dummy transaction object with sign method
    class DummyTx:
        def sign(self, kp):
            pass
    tx = DummyTx()
    result = await wallet.send_transaction(tx)
    assert result is True 