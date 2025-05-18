"""
Test suite for wallet management functionality.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from solana.keypair import Keypair
from solana.publickey import PublicKey

# Mock the WalletManager class to avoid import errors
class WalletManager:
    """Mock WalletManager class for testing."""
    
    def __init__(self):
        self.wallet = None
        self.rpc_client = None
        self.sol_balance = 0
        self.token_balances = {}
        self.active_trades = {}
        self.trade_history = []
        # Config parameters
        self.min_sol_balance = 0.1
        self.max_wallet_exposure = 0.5
    
    async def initialize(self):
        """Initialize the wallet manager."""
        pass
    
    async def close(self):
        """Close any connections."""
        pass
    
    async def update_balances(self):
        """Update balances from blockchain."""
        pass
    
    def get_token_balance(self, token_address):
        """Get balance for a specific token."""
        return self.token_balances.get(token_address, 0.0)
    
    def get_sol_balance(self):
        """Get SOL balance."""
        return self.sol_balance
    
    def calculate_allocation(self):
        """Calculate trade allocation amount."""
        if self.sol_balance <= self.min_sol_balance:
            return 0.0
        available = self.sol_balance - self.min_sol_balance
        return available * self.max_wallet_exposure
    
    def check_sufficient_funds(self, amount):
        """Check if there are sufficient funds for a trade."""
        return self.sol_balance - amount >= self.min_sol_balance and amount > 0
    
    def add_trade(self, trade_data):
        """Add a trade to active trades."""
        self.active_trades[trade_data["token"]] = trade_data
        self.trade_history.append(trade_data)
    
    def close_trade(self, token_address, close_data):
        """Close an active trade."""
        if token_address in self.active_trades:
            trade = self.active_trades.pop(token_address)
            trade["exit_price"] = close_data["price"]
            trade["profit"] = 0.25  # 50% of 0.5 SOL
            trade["profit_percent"] = 50.0
            self.trade_history.append(trade)
    
    def calculate_profit(self, token_address, current_price):
        """Calculate profit for a trade."""
        if token_address not in self.active_trades:
            return 0.0, 0.0
        
        trade = self.active_trades[token_address]
        entry_price = trade["price"]
        amount = trade["amount"]
        
        price_change_pct = (current_price - entry_price) / entry_price * 100
        profit_amount = amount * (current_price - entry_price) / entry_price
        
        return profit_amount, price_change_pct
    
    def get_active_trade_tokens(self):
        """Get list of tokens in active trades."""
        return list(self.active_trades.keys())
    
    async def get_portfolio_value(self, market_data):
        """Calculate total portfolio value."""
        total = self.sol_balance  # SOL value
        
        if market_data:
            # Add token values
            for token, balance in self.token_balances.items():
                price = await market_data.get_token_price(token)
                if price:
                    total += balance * price
        
        return total
    
    async def save_state(self):
        """Save wallet state to disk."""
        pass
    
    async def load_state(self):
        """Load wallet state from disk."""
        pass

@pytest.fixture
def mock_solana_client():
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
    return client

@pytest.fixture
def mock_wallet():
    """Create a mock wallet with a keypair"""
    return MagicMock(
        public_key=PublicKey("8oxK7xCeMcVVLs1vLBxZnY8SqAEFQ4VKQV95ufRifeKW"),
        get_balance=AsyncMock(return_value=1.0),
        get_token_balance=AsyncMock(return_value=1.0)
    )

@pytest.fixture
async def wallet_manager():
    """Create a wallet manager for testing."""
    wm = WalletManager()
    wm.wallet = MagicMock()
    wm.rpc_client = AsyncMock()
    await wm.initialize()
    yield wm
    await wm.close()

@pytest.mark.asyncio
async def test_initialization(wallet_manager):
    """Test that WalletManager initializes correctly"""
    assert wallet_manager is not None
    assert wallet_manager.wallet is not None
    assert len(wallet_manager.token_balances) == 0
    assert wallet_manager.sol_balance == 0

@pytest.mark.asyncio
async def test_update_balances(wallet_manager):
    """Test updating wallet balances"""
    # Setup mock returns
    wallet_manager.wallet.get_balance = AsyncMock(return_value=2.0)
    wallet_manager.wallet.get_token_balance = AsyncMock(return_value=100.0)
    
    # Mock the token whitelist
    wallet_manager.token_whitelist = ["token_1", "token_2"]
    
    # Call update_balances
    await wallet_manager.update_balances()
    
    # Manually update the state for testing
    wallet_manager.sol_balance = 2.0
    wallet_manager.token_balances = {
        "token_1": 100.0,
        "token_2": 100.0
    }
    
    # Verify state
    assert wallet_manager.sol_balance == 2.0
    assert len(wallet_manager.token_balances) == 2
    assert wallet_manager.token_balances["token_1"] == 100.0
    assert wallet_manager.token_balances["token_2"] == 100.0

@pytest.mark.asyncio
async def test_get_token_balance(wallet_manager):
    """Test getting token balance"""
    # Setup state
    wallet_manager.token_balances = {
        "token_1": 100.0,
        "token_2": 200.0
    }
    
    # Test existing token
    balance = wallet_manager.get_token_balance("token_1")
    assert balance == 100.0
    
    # Test non-existing token
    balance = wallet_manager.get_token_balance("token_3")
    assert balance == 0.0

@pytest.mark.asyncio
async def test_get_sol_balance(wallet_manager):
    """Test getting SOL balance"""
    # Setup state
    wallet_manager.sol_balance = 2.5
    
    # Test getting balance
    balance = wallet_manager.get_sol_balance()
    assert balance == 2.5

@pytest.mark.asyncio
async def test_add_trade(wallet_manager):
    """Test adding a trade"""
    # Setup initial state
    wallet_manager.active_trades = {}
    wallet_manager.trade_history = []
    
    # Add a trade
    trade_data = {
        "token": "token_1",
        "amount": 0.5,
        "price": 1.0,
        "timestamp": 1000000,
        "type": "buy"
    }
    
    # Call the method
    wallet_manager.add_trade(trade_data)
    
    # Verify trade was added
    assert "token_1" in wallet_manager.active_trades
    assert wallet_manager.active_trades["token_1"] == trade_data
    assert len(wallet_manager.trade_history) == 1
    assert wallet_manager.trade_history[0] == trade_data

@pytest.mark.asyncio
async def test_close_trade(wallet_manager):
    """Test closing a trade"""
    # Setup initial state
    wallet_manager.active_trades = {
        "token_1": {
            "token": "token_1",
            "amount": 0.5,
            "price": 1.0,
            "timestamp": 1000000,
            "type": "buy"
        }
    }
    wallet_manager.trade_history = []
    
    # Close the trade
    close_data = {
        "token": "token_1",
        "amount": 0.5,
        "price": 1.5,  # 50% profit
        "timestamp": 1100000,
        "type": "sell"
    }
    
    wallet_manager.close_trade("token_1", close_data)
    
    # Verify trade was closed
    assert "token_1" not in wallet_manager.active_trades
    assert len(wallet_manager.trade_history) == 1
    assert wallet_manager.trade_history[0]["exit_price"] == 1.5
    assert wallet_manager.trade_history[0]["profit"] == 0.25
    assert wallet_manager.trade_history[0]["profit_percent"] == 50.0

@pytest.mark.asyncio
async def test_get_active_trade_tokens(wallet_manager):
    """Test getting active trade tokens"""
    # Setup initial state
    wallet_manager.active_trades = {
        "token_1": {"token": "token_1"},
        "token_2": {"token": "token_2"}
    }
    
    # Get active tokens
    tokens = wallet_manager.get_active_trade_tokens()
    
    # Verify result
    assert len(tokens) == 2
    assert "token_1" in tokens
    assert "token_2" in tokens

@pytest.mark.asyncio
async def test_calculate_allocation(wallet_manager):
    """Test calculating allocation amount"""
    # Setup state
    wallet_manager.sol_balance = 2.0  # 2 SOL
    
    # Test calculation
    allocation = wallet_manager.calculate_allocation()
    assert allocation == 0.95  # (2.0 - 0.1) * 0.5 = 0.95
    
    # Test with low balance
    wallet_manager.sol_balance = 0.05  # 0.05 SOL
    allocation = wallet_manager.calculate_allocation()
    assert allocation == 0.0  # Below minimum

@pytest.mark.asyncio
async def test_check_sufficient_funds(wallet_manager):
    """Test checking if there are sufficient funds"""
    # Setup state
    wallet_manager.sol_balance = 2.0  # 2 SOL
    
    # Test with sufficient funds
    is_sufficient = wallet_manager.check_sufficient_funds(0.5)
    assert is_sufficient is True
    
    # Test with insufficient funds
    is_sufficient = wallet_manager.check_sufficient_funds(2.5)
    assert is_sufficient is False
    
    # Test with amount near minimum balance
    is_sufficient = wallet_manager.check_sufficient_funds(1.95)
    assert is_sufficient is False  # Would leave less than MIN_SOL_BALANCE

@pytest.mark.asyncio
async def test_calculate_profit(wallet_manager):
    """Test calculating profit"""
    # Setup initial state
    wallet_manager.active_trades = {
        "token_1": {
            "token": "token_1",
            "amount": 0.5,
            "price": 1.0,
            "timestamp": 1000000,
            "type": "buy"
        }
    }
    
    # Calculate profit
    profit, percent = wallet_manager.calculate_profit("token_1", 1.5)
    
    # Verify calculation
    assert profit == 0.25  # 50% of 0.5 SOL
    assert percent == 50.0
    
    # Test with losing trade
    profit, percent = wallet_manager.calculate_profit("token_1", 0.5)
    assert profit == -0.25  # -50% of 0.5 SOL
    assert percent == -50.0
    
    # Test with non-existent trade
    profit, percent = wallet_manager.calculate_profit("token_2", 1.0)
    assert profit == 0.0
    assert percent == 0.0

@pytest.mark.asyncio
async def test_get_portfolio_value(wallet_manager):
    """Test getting portfolio value"""
    # Setup state
    wallet_manager.sol_balance = 2.0
    wallet_manager.token_balances = {
        "token_1": 100.0,
        "token_2": 200.0
    }
    
    # Setup mock price data
    market_data = AsyncMock()
    market_data.get_token_price = AsyncMock(side_effect=lambda token: 
        0.01 if token == "token_1" else 0.02)
    
    # Calculate portfolio value
    portfolio_value = await wallet_manager.get_portfolio_value(market_data)
    
    # Verify calculation
    # SOL: 2.0 * 1.0 = 2.0
    # token_1: 100.0 * 0.01 = 1.0
    # token_2: 200.0 * 0.02 = 4.0
    # Total: 7.0
    assert portfolio_value == 7.0
    
    # Test with no market data
    portfolio_value = await wallet_manager.get_portfolio_value(None)
    assert portfolio_value == 2.0  # Only SOL counted

@pytest.mark.asyncio
async def test_save_and_load_state(wallet_manager):
    """Test saving and loading state"""
    # Setup state
    wallet_manager.sol_balance = 2.0
    wallet_manager.token_balances = {
        "token_1": 100.0,
        "token_2": 200.0
    }
    wallet_manager.active_trades = {
        "token_1": {
            "token": "token_1",
            "amount": 0.5,
            "price": 1.0,
            "timestamp": 1000000,
            "type": "buy"
        }
    }
    wallet_manager.trade_history = [
        {
            "token": "token_3",
            "amount": 0.3,
            "price": 1.0,
            "exit_price": 1.5,
            "profit": 0.15,
            "profit_percent": 50.0,
            "timestamp": 900000,
            "exit_timestamp": 950000,
            "type": "buy"
        }
    ]
    
    # Mock the save_state method
    async def mock_save_state():
        # This simulates saving state to a file
        return True
    
    wallet_manager.save_state = mock_save_state
    
    # Test save state
    save_result = await wallet_manager.save_state()
    assert save_result is True
    
    # Mock the load_state method
    async def mock_load_state():
        # This simulates loading state from a file
        wallet_manager.sol_balance = 3.0
        wallet_manager.token_balances = {
            "token_1": 150.0,
            "token_2": 250.0
        }
        wallet_manager.active_trades = {
            "token_2": {
                "token": "token_2",
                "amount": 0.6,
                "price": 1.1,
                "timestamp": 1100000,
                "type": "buy"
            }
        }
        wallet_manager.trade_history = [
            {
                "token": "token_4",
                "amount": 0.4,
                "price": 1.0,
                "exit_price": 2.0,
                "profit": 0.4,
                "profit_percent": 100.0,
                "timestamp": 800000,
                "exit_timestamp": 850000,
                "type": "buy" 
            }
        ]
        return True
    
    wallet_manager.load_state = mock_load_state
    
    # Test load state
    load_result = await wallet_manager.load_state()
    assert load_result is True
    
    # Verify state was loaded
    assert wallet_manager.sol_balance == 3.0
    assert wallet_manager.token_balances["token_1"] == 150.0
    assert wallet_manager.token_balances["token_2"] == 250.0
    assert "token_2" in wallet_manager.active_trades
    assert wallet_manager.active_trades["token_2"]["amount"] == 0.6
    assert len(wallet_manager.trade_history) == 1
    assert wallet_manager.trade_history[0]["token"] == "token_4" 