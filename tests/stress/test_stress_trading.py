import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch
from solders.keypair import Keypair
from src.core.wallet_manager import WalletManager

@pytest.mark.asyncio
async def test_stress_transaction_sending():
    mock_rpc = AsyncMock()
    mock_rpc.send_transaction.return_value = {"result": "test_signature"}
    async def mock_get_signature_statuses(signatures):
        return {"result": {"value": [{"confirmationStatus": "confirmed"}]}}
    mock_rpc.get_signature_statuses.side_effect = mock_get_signature_statuses

    wallet = WalletManager(mock_rpc)
    wallet.keypair = wallet.keypair or object()
    class DummyTx:
        def sign(self, kp):
            pass
    tx = DummyTx()
    n = 100
    start = time.time()
    results = await asyncio.gather(*(wallet.send_transaction(tx) for _ in range(n)))
    elapsed = time.time() - start
    assert all(results)
    print(f"Sent {n} transactions in {elapsed:.2f} seconds. TPS: {n/elapsed:.2f}")

@pytest.mark.asyncio
async def test_stress_batch_processing():
    mock_rpc = AsyncMock()
    mock_rpc.send_transaction.return_value = {"result": "test_signature"}
    async def mock_get_signature_statuses(signatures):
        return {"result": {"value": [{"confirmationStatus": "confirmed"}]}}
    mock_rpc.get_signature_statuses.side_effect = mock_get_signature_statuses

    wallet = WalletManager(mock_rpc)
    wallet.keypair = wallet.keypair or object()
    class DummyTx:
        def sign(self, kp):
            pass
    tx = DummyTx()
    batch_size = 10
    n_batches = 10
    start = time.time()
    for _ in range(n_batches):
        results = await asyncio.gather(*(wallet.send_transaction(tx) for _ in range(batch_size)))
        assert all(results)
    elapsed = time.time() - start
    print(f"Processed {n_batches * batch_size} transactions in {elapsed:.2f} seconds. TPS: {n_batches * batch_size / elapsed:.2f}") 