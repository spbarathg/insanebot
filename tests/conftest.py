"""
Common test fixtures and configuration.
"""
import os
import sys
import types
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Mock the solana modules before importing to prevent missing dependency errors
def setup_solana_mocks():
    """Setup mock solana modules for testing."""
    
    # Mock solders modules
    mock_solders = types.ModuleType('solders')
    mock_solders_commitment = types.ModuleType('solders.commitment_config')
    mock_solders_keypair = types.ModuleType('solders.keypair')
    mock_solders_pubkey = types.ModuleType('solders.pubkey')
    mock_solders_transaction = types.ModuleType('solders.transaction')
    mock_solders_system = types.ModuleType('solders.system_program')
    
    # Mock commitment levels
    class MockCommitmentLevel:
        @staticmethod
        def confirmed():
            return "confirmed"
        
        @staticmethod
        def finalized():
            return "finalized"
    
    mock_solders_commitment.CommitmentLevel = MockCommitmentLevel
    
    # Mock other solders classes
    class MockKeypair:
        def __init__(self, pubkey="mock_pubkey"):
            self.pubkey = pubkey
        
        @classmethod
        def generate(cls):
            return cls()
        
        @classmethod
        def from_secret_key_bytes(cls, bytes_data):
            return cls()
    
    class MockPubkey:
        def __init__(self, key="mock_public_key"):
            self.key = key
        
        def __str__(self):
            return self.key
        
        @classmethod
        def from_string(cls, pubkey_str):
            return cls(pubkey_str)
    
    class MockTransaction:
        def __init__(self):
            self.instructions = []
            self.recent_blockhash = "mock_blockhash"
        
        def add(self, instruction):
            self.instructions.append(instruction)
        
        def sign(self, *signers):
            self.signatures = [signer.public_key for signer in signers]
    
    class MockTransferParams:
        def __init__(self, from_pubkey, to_pubkey, lamports):
            self.from_pubkey = from_pubkey
            self.to_pubkey = to_pubkey
            self.lamports = lamports
    
    def mock_transfer(params):
        return Mock()
    
    mock_solders_keypair.Keypair = MockKeypair
    mock_solders_pubkey.Pubkey = MockPubkey
    mock_solders_transaction.Transaction = MockTransaction
    mock_solders_system.transfer = mock_transfer
    mock_solders_system.TransferParams = MockTransferParams
    
    # Register mocks
    sys.modules['solders'] = mock_solders
    sys.modules['solders.commitment_config'] = mock_solders_commitment
    sys.modules['solders.keypair'] = mock_solders_keypair
    sys.modules['solders.pubkey'] = mock_solders_pubkey
    sys.modules['solders.transaction'] = mock_solders_transaction
    sys.modules['solders.system_program'] = mock_solders_system
    
    # Mock legacy commitment module for backward compatibility
    mock_commitment = types.ModuleType('solana.rpc.commitment')
    mock_commitment.Confirmed = "confirmed"
    sys.modules['solana.rpc.commitment'] = mock_commitment

# Setup mocks before importing any modules
setup_solana_mocks()

# Mock additional solana modules for backward compatibility
mock_solana = types.ModuleType('solana')
mock_async_api = types.ModuleType('solana.rpc.async_api')
mock_keypair = types.ModuleType('solana.keypair')
mock_publickey = types.ModuleType('solana.publickey')
mock_transaction = types.ModuleType('solana.transaction')
mock_instruction = types.ModuleType('solana.instruction')
mock_account = types.ModuleType('solana.account')

# Define additional mock classes
class MockAsyncClient(AsyncMock):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.get_account_info = AsyncMock()
        self.get_balance = AsyncMock()
        self.send_transaction = AsyncMock()

class MockSolanaKeypair:
    def __init__(self, secret_key=None):
        self.secret_key = secret_key or bytes([0] * 32)
        self.public_key = MockSolanaPublicKey("mock_public_key")

    @classmethod
    def from_secret_key(cls, secret_key):
        return cls(secret_key)

class MockSolanaPublicKey:
    def __init__(self, address):
        self.address = address

    def __str__(self):
        return self.address

class MockSolanaTransaction:
    def __init__(self):
        self.instructions = []
        self.recent_blockhash = "mock_blockhash"
        
    def add(self, instruction):
        self.instructions.append(instruction)
        
    def sign(self, *signers):
        self.signatures = [signer.public_key for signer in signers]

mock_async_api.AsyncClient = MockAsyncClient
mock_keypair.Keypair = MockSolanaKeypair
mock_publickey.PublicKey = MockSolanaPublicKey
mock_transaction.Transaction = MockSolanaTransaction

# Set up the module hierarchy  
mock_solana.rpc = types.ModuleType('solana.rpc')
mock_solana.rpc.async_api = mock_async_api
mock_solana.keypair = mock_keypair
mock_solana.publickey = mock_publickey
mock_solana.transaction = mock_transaction
mock_solana.instruction = mock_instruction
mock_solana.account = mock_account

# Add the modules to sys.modules
sys.modules['solana'] = mock_solana
sys.modules['solana.rpc.async_api'] = mock_async_api
sys.modules['solana.keypair'] = mock_keypair
sys.modules['solana.publickey'] = mock_publickey
sys.modules['solana.transaction'] = mock_transaction
sys.modules['solana.instruction'] = mock_instruction
sys.modules['solana.account'] = mock_account

# Now import the project modules
from config.core_config import CORE_CONFIG, MARKET_CONFIG, TRADING_CONFIG
from src.core.wallet_manager import WalletManager
from src.core.dex import RaydiumDEX
from src.core.simulator import TradingSimulator

@pytest.fixture
def mock_rpc_client():
    """Mock RPC client for testing."""
    client = AsyncMock()
    client.get_account_info.return_value = {
        "result": {
            "value": {
                "data": "mock_data",
                "owner": "mock_owner"
            }
        }
    }
    return client

@pytest.fixture
def mock_wallet():
    """Mock wallet manager for testing."""
    wallet = AsyncMock(spec=WalletManager)
    wallet.get_balance.return_value = 10.0  # 10 SOL
    wallet.get_keypair.return_value = MockSolanaKeypair()
    return wallet

@pytest.fixture
def mock_dex():
    """Mock DEX for testing."""
    dex = AsyncMock(spec=RaydiumDEX)
    dex.create_swap_transaction.return_value = MockSolanaTransaction()
    return dex

@pytest.fixture
def mock_simulator():
    """Mock trading simulator for testing."""
    simulator = AsyncMock(spec=TradingSimulator)
    simulator.get_balance.return_value = 10.0
    simulator.simulate_trade.return_value = True
    return simulator

@pytest.fixture
def test_config():
    """Test configuration with safe values."""
    return {
        "core": CORE_CONFIG.copy(),
        "market": MARKET_CONFIG.copy(),
        "trading": TRADING_CONFIG.copy()
    }

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close() 