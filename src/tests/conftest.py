"""
Common test fixtures and configuration.
"""
import pytest
import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

# Mock the solana modules
# Create mock solana modules
mock_solana = types.ModuleType('solana')
mock_keypair = types.ModuleType('solana.keypair')
mock_publickey = types.ModuleType('solana.publickey')
mock_transaction = types.ModuleType('solana.transaction')
mock_instruction = types.ModuleType('solana.instruction')
mock_rpc = types.ModuleType('solana.rpc')
mock_async_api = types.ModuleType('solana.rpc.async_api')
mock_commitment = types.ModuleType('solana.rpc.commitment')

# Define mock classes
class MockKeypair:
    def __init__(self, secret_key=None):
        self.secret_key = secret_key or bytes([0] * 32)
        self.public_key = MockPublicKey("mock_public_key")

    @classmethod
    def from_secret_key(cls, secret_key):
        return cls(secret_key)

class MockPublicKey:
    def __init__(self, address):
        self.address = address

    def __str__(self):
        return self.address

class MockTransaction:
    def __init__(self):
        self.instructions = []
        self.recent_blockhash = "mock_blockhash"
        
    def add(self, instruction):
        self.instructions.append(instruction)
        
    def sign(self, *signers):
        self.signatures = [signer.public_key for signer in signers]

class MockInstruction:
    def __init__(self, program_id, accounts, data):
        self.program_id = program_id
        self.accounts = accounts
        self.data = data

class MockAccountMeta:
    def __init__(self, pubkey, is_signer, is_writable):
        self.pubkey = pubkey
        self.is_signer = is_signer
        self.is_writable = is_writable

# Add the mock classes to their respective modules
mock_keypair.Keypair = MockKeypair
mock_publickey.PublicKey = MockPublicKey
mock_transaction.Transaction = MockTransaction
mock_instruction.Instruction = MockInstruction
mock_instruction.AccountMeta = MockAccountMeta
mock_async_api.AsyncClient = AsyncMock
mock_commitment.Confirmed = "confirmed"

# Set up the module hierarchy
mock_solana.keypair = mock_keypair
mock_solana.publickey = mock_publickey
mock_solana.transaction = mock_transaction
mock_solana.instruction = mock_instruction
mock_solana.rpc = mock_rpc
mock_rpc.async_api = mock_async_api
mock_rpc.commitment = mock_commitment

# Add the modules to sys.modules
sys.modules['solana'] = mock_solana
sys.modules['solana.keypair'] = mock_keypair
sys.modules['solana.publickey'] = mock_publickey
sys.modules['solana.transaction'] = mock_transaction
sys.modules['solana.instruction'] = mock_instruction
sys.modules['solana.rpc'] = mock_rpc
sys.modules['solana.rpc.async_api'] = mock_async_api
sys.modules['solana.rpc.commitment'] = mock_commitment

# Now import the project modules
from src.core.config import CORE_CONFIG, MARKET_CONFIG, TRADING_CONFIG
from src.core.wallet import WalletManager
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
    wallet.get_keypair.return_value = MockKeypair()
    return wallet

@pytest.fixture
def mock_dex():
    """Mock DEX for testing."""
    dex = AsyncMock(spec=RaydiumDEX)
    dex.create_swap_transaction.return_value = MockTransaction()
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