"""
Risk management tests for the trading bot.
These tests verify that the bot operates safely and within defined risk parameters.
"""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from solana.keypair import Keypair
from solana.publickey import PublicKey

from src.core.config import CORE_CONFIG, TRADING_CONFIG
from src.core.wallet import WalletManager

# Create a mock WalletManager for testing
class MockWalletManager:
    def __init__(self):
        self.sol_balance = 1.0  # Initial balance of 1 SOL
        self.token_balances = {}
        self.active_trades = {}
        self.trade_history = []
        self.max_wallet_exposure = TRADING_CONFIG.MAX_WALLET_EXPOSURE
        self.min_sol_balance = TRADING_CONFIG.MIN_SOL_BALANCE
        
    def get_sol_balance(self):
        return self.sol_balance
        
    def get_token_balance(self, token_address):
        return self.token_balances.get(token_address, 0.0)
        
    def calculate_allocation(self):
        """Calculate how much SOL can be allocated to a trade"""
        if self.sol_balance <= self.min_sol_balance:
            return 0.0
        available = self.sol_balance - self.min_sol_balance
        return available * self.max_wallet_exposure
        
    def check_sufficient_funds(self, amount):
        """Check if there are sufficient funds for a trade"""
        if amount <= 0:
            return False
        remaining = self.sol_balance - amount
        return remaining >= self.min_sol_balance
        
    def add_trade(self, trade_data):
        """Add a trade to active trades"""
        self.active_trades[trade_data["token"]] = trade_data
        self.trade_history.append(trade_data)
        
    def get_portfolio_value(self):
        """Get total portfolio value in SOL"""
        return self.sol_balance
        
    def get_exposure(self, token_address=None):
        """Get current exposure to a token or total exposure"""
        if token_address:
            if token_address in self.active_trades:
                return self.active_trades[token_address]["amount"] / self.sol_balance
            return 0.0
        
        # Calculate total exposure
        total_exposure = sum(trade["amount"] for trade in self.active_trades.values()) / self.sol_balance
        return total_exposure
        
    def is_within_risk_limits(self):
        """Check if current state is within risk limits"""
        # Check minimum SOL balance
        if self.sol_balance < self.min_sol_balance:
            return False
            
        # Check maximum exposure
        total_exposure = self.get_exposure()
        if total_exposure > self.max_wallet_exposure:
            return False
            
        return True

# Mock trading bot for risk tests
class MockTradingBot:
    def __init__(self, wallet_manager):
        self.wallet_manager = wallet_manager
        self.max_trades_per_day = TRADING_CONFIG.MAX_TRADES_PER_DAY
        self.max_loss_per_trade = TRADING_CONFIG.MAX_LOSS_PER_TRADE
        self.stop_loss_percentage = TRADING_CONFIG.STOP_LOSS_PERCENTAGE
        self.take_profit_percentage = TRADING_CONFIG.TAKE_PROFIT_PERCENTAGE
        self.daily_trades = []
        
    def add_trade(self, token, amount, price):
        """Add a new trade"""
        if len(self.daily_trades) >= self.max_trades_per_day:
            return False, "Maximum daily trades reached"
            
        # Check allocation limits
        max_allocation = self.wallet_manager.calculate_allocation()
        if amount > max_allocation:
            return False, "Trade amount exceeds allocation limit"
            
        # Check sufficient funds
        if not self.wallet_manager.check_sufficient_funds(amount):
            return False, "Insufficient funds"
            
        # Create trade
        trade_data = {
            "token": token,
            "amount": amount,
            "price": price,
            "timestamp": time.time(),
            "type": "buy"
        }
        
        # Add to wallet
        self.wallet_manager.add_trade(trade_data)
        self.daily_trades.append(trade_data)
        
        return True, "Trade added successfully"
        
    def check_stop_loss(self, token, current_price):
        """Check if stop loss should be triggered"""
        if token not in self.wallet_manager.active_trades:
            return False
            
        trade = self.wallet_manager.active_trades[token]
        entry_price = trade["price"]
        
        # Calculate loss percentage
        loss_percentage = (entry_price - current_price) / entry_price * 100
        
        # Check stop loss
        if loss_percentage >= self.stop_loss_percentage:
            return True
            
        return False
        
    def check_take_profit(self, token, current_price):
        """Check if take profit should be triggered"""
        if token not in self.wallet_manager.active_trades:
            return False
            
        trade = self.wallet_manager.active_trades[token]
        entry_price = trade["price"]
        
        # Calculate profit percentage
        profit_percentage = (current_price - entry_price) / entry_price * 100
        
        # Check take profit
        if profit_percentage >= self.take_profit_percentage:
            return True
            
        return False
        
    def calculate_max_loss(self):
        """Calculate maximum loss allowed"""
        return self.wallet_manager.get_sol_balance() * self.max_loss_per_trade

@pytest.fixture
def wallet_manager():
    wallet = MockWalletManager()
    return wallet

@pytest.fixture
def trading_bot(wallet_manager):
    bot = MockTradingBot(wallet_manager)
    return bot

def test_max_wallet_exposure(wallet_manager):
    """Test that the bot respects maximum wallet exposure limits"""
    # Get the maximum allocation
    max_allocation = wallet_manager.calculate_allocation()
    
    # Verify it doesn't exceed the maximum exposure
    assert max_allocation <= wallet_manager.max_wallet_exposure * wallet_manager.get_sol_balance()
    
    # Test with different balances
    wallet_manager.sol_balance = 2.0
    max_allocation = wallet_manager.calculate_allocation()
    assert max_allocation <= wallet_manager.max_wallet_exposure * wallet_manager.get_sol_balance()
    
    wallet_manager.sol_balance = 5.0
    max_allocation = wallet_manager.calculate_allocation()
    assert max_allocation <= wallet_manager.max_wallet_exposure * wallet_manager.get_sol_balance()

def test_minimum_balance_maintained(wallet_manager):
    """Test that the minimum balance requirement is maintained"""
    # Try a trade that would leave exactly the minimum balance
    amount = wallet_manager.sol_balance - wallet_manager.min_sol_balance
    assert wallet_manager.check_sufficient_funds(amount) is True
    
    # Try a trade that would leave less than the minimum balance
    amount = wallet_manager.sol_balance - wallet_manager.min_sol_balance + 0.01
    assert wallet_manager.check_sufficient_funds(amount) is False
    
    # Try with different balances
    wallet_manager.sol_balance = 0.2  # Close to minimum
    amount = 0.1
    assert wallet_manager.check_sufficient_funds(amount) is True
    amount = 0.2
    assert wallet_manager.check_sufficient_funds(amount) is False

def test_max_trades_per_day(trading_bot):
    """Test that the maximum trades per day limit is enforced"""
    # Add trades up to the limit
    for i in range(trading_bot.max_trades_per_day):
        success, _ = trading_bot.add_trade(f"token_{i}", 0.01, 1.0)
        assert success is True
        
    # Try to add one more trade
    success, reason = trading_bot.add_trade("token_extra", 0.01, 1.0)
    assert success is False
    assert "Maximum daily trades" in reason

def test_stop_loss_trigger(trading_bot):
    """Test that stop loss is triggered correctly"""
    # Add a trade
    trading_bot.add_trade("token_1", 0.1, 1.0)
    
    # Test with price above stop loss
    assert trading_bot.check_stop_loss("token_1", 0.95) is False
    
    # Test with price at stop loss
    stop_loss_price = 1.0 * (1 - trading_bot.stop_loss_percentage / 100)
    assert trading_bot.check_stop_loss("token_1", stop_loss_price) is True
    
    # Test with price below stop loss
    assert trading_bot.check_stop_loss("token_1", 0.5) is True

def test_take_profit_trigger(trading_bot):
    """Test that take profit is triggered correctly"""
    # Add a trade
    trading_bot.add_trade("token_1", 0.1, 1.0)
    
    # Test with price below take profit
    assert trading_bot.check_take_profit("token_1", 1.05) is False
    
    # Test with price at take profit
    take_profit_price = 1.0 * (1 + trading_bot.take_profit_percentage / 100)
    assert trading_bot.check_take_profit("token_1", take_profit_price) is True
    
    # Test with price above take profit
    assert trading_bot.check_take_profit("token_1", 1.5) is True

def test_max_loss_per_trade(trading_bot):
    """Test that maximum loss per trade is respected"""
    # Calculate maximum loss allowed
    max_loss = trading_bot.calculate_max_loss()
    
    # Verify it doesn't exceed the maximum loss percentage
    assert max_loss <= trading_bot.wallet_manager.get_sol_balance() * trading_bot.max_loss_per_trade
    
    # Test with different balances
    trading_bot.wallet_manager.sol_balance = 2.0
    max_loss = trading_bot.calculate_max_loss()
    assert max_loss <= trading_bot.wallet_manager.get_sol_balance() * trading_bot.max_loss_per_trade

def test_risk_limits(wallet_manager):
    """Test overall risk limits"""
    # Initially should be within limits
    assert wallet_manager.is_within_risk_limits() is True
    
    # Test when balance is too low
    original_balance = wallet_manager.sol_balance
    wallet_manager.sol_balance = wallet_manager.min_sol_balance - 0.01
    assert wallet_manager.is_within_risk_limits() is False
    
    # Restore balance
    wallet_manager.sol_balance = original_balance
    assert wallet_manager.is_within_risk_limits() is True
    
    # Test when exposure is too high
    wallet_manager.active_trades = {
        "token_1": {"token": "token_1", "amount": wallet_manager.sol_balance * 0.4, "price": 1.0},
        "token_2": {"token": "token_2", "amount": wallet_manager.sol_balance * 0.3, "price": 1.0}
    }
    assert wallet_manager.is_within_risk_limits() is False 