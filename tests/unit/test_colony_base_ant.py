"""
Unit tests for BaseAnt class and related components.
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from src.colony.base_ant import (
    BaseAnt, AntRole, AntStatus, AntCapital, AntPerformance
)


class TestAntCapital:
    """Test the AntCapital data class."""
    
    def test_initial_state(self):
        """Test initial capital state."""
        capital = AntCapital()
        assert capital.current_balance == 0.0
        assert capital.allocated_capital == 0.0
        assert capital.available_capital == 0.0
        assert capital.total_trades == 0
        assert capital.profit_loss == 0.0
        assert capital.hedged_amount == 0.0
        assert capital.last_updated > 0
    
    def test_update_balance(self):
        """Test balance update functionality."""
        capital = AntCapital(current_balance=100.0, allocated_capital=20.0)
        capital.available_capital = 80.0  # Set initial available
        
        old_time = capital.last_updated
        time.sleep(0.01)  # Ensure time difference
        capital.update_balance(150.0)
        
        assert capital.current_balance == 150.0
        assert capital.profit_loss == 50.0
        assert capital.available_capital == 130.0  # 150 - 20 allocated
        assert capital.last_updated > old_time
    
    def test_allocate_capital_success(self):
        """Test successful capital allocation."""
        capital = AntCapital(current_balance=100.0)
        capital.available_capital = 100.0
        
        result = capital.allocate_capital(30.0)
        
        assert result is True
        assert capital.allocated_capital == 30.0
        assert capital.available_capital == 70.0
    
    def test_allocate_capital_insufficient(self):
        """Test capital allocation with insufficient funds."""
        capital = AntCapital(current_balance=100.0)
        capital.available_capital = 50.0
        
        result = capital.allocate_capital(60.0)
        
        assert result is False
        assert capital.allocated_capital == 0.0
        assert capital.available_capital == 50.0
    
    def test_release_capital(self):
        """Test capital release functionality."""
        capital = AntCapital(allocated_capital=50.0, available_capital=25.0)
        
        capital.release_capital(20.0)
        
        assert capital.allocated_capital == 30.0
        assert capital.available_capital == 45.0
    
    def test_release_capital_more_than_allocated(self):
        """Test releasing more capital than allocated."""
        capital = AntCapital(allocated_capital=30.0, available_capital=25.0)
        
        capital.release_capital(50.0)
        
        assert capital.allocated_capital == 0.0
        assert capital.available_capital == 75.0


class TestAntPerformance:
    """Test the AntPerformance data class."""
    
    def test_initial_state(self):
        """Test initial performance state."""
        performance = AntPerformance()
        assert performance.total_trades == 0
        assert performance.successful_trades == 0
        assert performance.total_profit == 0.0
        assert performance.win_rate == 0.0
        assert performance.profit_per_trade == 0.0
    
    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        performance = AntPerformance(successful_trades=7, total_trades=10)
        assert performance.win_rate == 70.0
        
        # Test zero trades
        performance_zero = AntPerformance()
        assert performance_zero.win_rate == 0.0
    
    def test_profit_per_trade_calculation(self):
        """Test profit per trade calculation."""
        performance = AntPerformance(total_profit=500.0, total_trades=10)
        assert performance.profit_per_trade == 50.0
        
        # Test zero trades
        performance_zero = AntPerformance()
        assert performance_zero.profit_per_trade == 0.0
    
    def test_update_trade_result_successful(self):
        """Test updating performance with successful trade."""
        performance = AntPerformance()
        
        performance.update_trade_result(profit=50.0, trade_time=30.0, success=True)
        
        assert performance.total_trades == 1
        assert performance.successful_trades == 1
        assert performance.total_profit == 50.0
        assert performance.best_trade == 50.0
        assert performance.average_trade_time == 30.0
        assert performance.compound_factor > 1.0
        assert performance.trades_this_cycle == 1
    
    def test_update_trade_result_failed(self):
        """Test updating performance with failed trade."""
        performance = AntPerformance()
        
        performance.update_trade_result(profit=-20.0, trade_time=15.0, success=False)
        
        assert performance.total_trades == 1
        assert performance.successful_trades == 0
        assert performance.total_profit == -20.0
        assert performance.worst_trade == -20.0
        assert performance.average_trade_time == 15.0
        assert performance.compound_factor == 1.0  # No change for failed trades
    
    def test_update_trade_result_multiple_trades(self):
        """Test updating performance with multiple trades."""
        performance = AntPerformance()
        
        # First trade
        performance.update_trade_result(profit=30.0, trade_time=20.0, success=True)
        # Second trade
        performance.update_trade_result(profit=-10.0, trade_time=40.0, success=False)
        
        assert performance.total_trades == 2
        assert performance.successful_trades == 1
        assert performance.total_profit == 20.0
        assert performance.best_trade == 30.0
        assert performance.worst_trade == -10.0
        assert performance.average_trade_time == 30.0  # (20 + 40) / 2


class MockAnt(BaseAnt):
    """Mock implementation of BaseAnt for testing."""
    
    async def initialize(self) -> bool:
        return True
    
    async def execute_cycle(self) -> dict:
        return {"status": "executed"}
    
    async def cleanup(self):
        pass


class TestBaseAnt:
    """Test the BaseAnt abstract base class."""
    
    def test_ant_initialization(self):
        """Test basic ant initialization."""
        ant = MockAnt("test-ant-001", AntRole.WORKER, parent_id="parent-001")
        
        assert ant.ant_id == "test-ant-001"
        assert ant.role == AntRole.WORKER
        assert ant.parent_id == "parent-001"
        assert ant.status == AntStatus.ACTIVE
        assert ant.created_at > 0
        assert ant.last_activity > 0
        assert isinstance(ant.capital, AntCapital)
        assert isinstance(ant.performance, AntPerformance)
        assert ant.children == []
        assert isinstance(ant.metadata, dict)
        assert isinstance(ant.config, dict)
    
    def test_get_role_config_founding_queen(self):
        """Test role configuration for Founding Queen."""
        ant = MockAnt("test-ant", AntRole.FOUNDING_QUEEN)
        config = ant.config
        
        assert config["max_children"] == 10
        assert config["split_threshold"] == 20.0
        assert config["merge_threshold"] == 1.0
        assert config["max_trades"] == float('inf')
        assert config["retirement_trades"] is None
        assert "compound_layers" in config
    
    def test_get_role_config_queen(self):
        """Test role configuration for Queen."""
        ant = MockAnt("test-ant", AntRole.QUEEN)
        config = ant.config
        
        assert config["max_children"] == 50
        assert config["split_threshold"] == 2.0
        assert config["merge_threshold"] == 0.5
        assert config["target_split_amount"] == 1500.0
    
    def test_get_role_config_worker(self):
        """Test role configuration for Worker."""
        ant = MockAnt("test-ant", AntRole.WORKER)
        config = ant.config
        
        assert config["max_children"] == 0
        assert config["split_threshold"] == 2.0
        assert config["merge_threshold"] == 0.1
        assert config["max_trades"] == 10
        assert config["retirement_trades"] == (5, 10)
        assert config["target_return_range"] == (1.03, 1.50)
    
    def test_get_role_config_drone(self):
        """Test role configuration for Drone."""
        ant = MockAnt("test-ant", AntRole.DRONE)
        config = ant.config
        
        assert config["max_children"] == 0
        assert config["split_threshold"] is None
        assert config["ai_sync_interval"] == 60
    
    def test_get_role_config_accounting(self):
        """Test role configuration for Accounting ant."""
        ant = MockAnt("test-ant", AntRole.ACCOUNTING)
        config = ant.config
        
        assert config["max_children"] == 0
        assert config["hedge_percentage"] == 0.1
    
    def test_get_role_config_princess(self):
        """Test role configuration for Princess."""
        ant = MockAnt("test-ant", AntRole.PRINCESS)
        config = ant.config
        
        assert config["accumulation_threshold"] == 10.0
    
    def test_should_split_with_threshold(self):
        """Test split decision with capital threshold."""
        ant = MockAnt("test-ant", AntRole.WORKER)
        ant.capital.available_capital = 3.0  # Above 2.0 threshold
        
        result = ant.should_split()
        assert result is True
    
    def test_should_split_below_threshold(self):
        """Test split decision below capital threshold."""
        ant = MockAnt("test-ant", AntRole.WORKER)
        ant.capital.available_capital = 1.0  # Below 2.0 threshold
        
        result = ant.should_split()
        assert result is False
    
    def test_should_split_no_threshold(self):
        """Test split decision for role with no threshold."""
        ant = MockAnt("test-ant", AntRole.DRONE)
        ant.capital.available_capital = 100.0
        
        result = ant.should_split()
        assert result is False
    
    def test_should_merge_below_threshold(self):
        """Test merge decision below threshold."""
        ant = MockAnt("test-ant", AntRole.WORKER)
        ant.capital.available_capital = 0.05  # Below 0.1 threshold
        ant.capital.current_balance = 0.05  # Set current balance to match available capital
        
        result = ant.should_merge()
        assert result is True
    
    def test_should_merge_above_threshold(self):
        """Test merge decision above threshold."""
        ant = MockAnt("test-ant", AntRole.WORKER)
        ant.capital.available_capital = 0.5  # Above 0.1 threshold
        ant.capital.current_balance = 0.5  # Set current balance to match available capital
        
        result = ant.should_merge()
        assert result is False
    
    def test_should_retire_by_trades(self):
        """Test retirement decision based on trade count."""
        ant = MockAnt("test-ant", AntRole.WORKER)
        ant.performance.total_trades = 12  # Above max of 10
        
        result = ant.should_retire()
        assert result is True
    
    def test_should_retire_not_yet(self):
        """Test retirement decision when not ready."""
        ant = MockAnt("test-ant", AntRole.WORKER)
        ant.performance.total_trades = 3  # Below max of 10
        
        result = ant.should_retire()
        assert result is False
    
    def test_should_retire_no_limit(self):
        """Test retirement for role with no trade limit."""
        ant = MockAnt("test-ant", AntRole.QUEEN)
        ant.performance.total_trades = 1000
        
        result = ant.should_retire()
        assert result is False
    
    def test_update_activity(self):
        """Test activity timestamp update."""
        ant = MockAnt("test-ant", AntRole.WORKER)
        old_activity = ant.last_activity
        
        time.sleep(0.01)
        ant.update_activity()
        
        assert ant.last_activity > old_activity
    
    def test_get_status_summary(self):
        """Test status summary generation."""
        ant = MockAnt("test-ant", AntRole.WORKER, parent_id="parent-001")
        ant.capital.current_balance = 5.0
        ant.performance.total_trades = 3
        ant.performance.successful_trades = 2
        ant.children = ["child-001", "child-002"]
        
        summary = ant.get_status_summary()
        
        assert summary["ant_id"] == "test-ant"
        assert summary["role"] == "worker"
        assert summary["status"] == "active"
        assert summary["parent_id"] == "parent-001"
        assert summary["capital"] == 5.0
        assert summary["trades"] == 3
        assert summary["win_rate"] == 2.0 / 3 * 100  # 2/3 * 100 = 66.67%
        assert summary["children_count"] == 2
        assert "uptime" in summary
        assert "last_activity" in summary


@pytest.mark.asyncio
class TestBaseAntAsync:
    """Test async methods of BaseAnt."""
    
    async def test_abstract_methods_implemented(self):
        """Test that abstract methods are properly implemented in mock."""
        ant = MockAnt("test-ant", AntRole.WORKER)
        
        init_result = await ant.initialize()
        assert init_result is True
        
        cycle_result = await ant.execute_cycle()
        assert cycle_result == {"status": "executed"}
        
        # cleanup should not raise an exception
        await ant.cleanup()


class TestAntEnums:
    """Test the enum classes."""
    
    def test_ant_role_enum(self):
        """Test AntRole enum values."""
        assert AntRole.FOUNDING_QUEEN.value == "founding_queen"
        assert AntRole.QUEEN.value == "queen"
        assert AntRole.WORKER.value == "worker"
        assert AntRole.DRONE.value == "drone"
        assert AntRole.ACCOUNTING.value == "accounting"
        assert AntRole.PRINCESS.value == "princess"
    
    def test_ant_status_enum(self):
        """Test AntStatus enum values."""
        assert AntStatus.ACTIVE.value == "active"
        assert AntStatus.SPLITTING.value == "splitting"
        assert AntStatus.MERGING.value == "merging"
        assert AntStatus.RETIRING.value == "retiring"
        assert AntStatus.DORMANT.value == "dormant"
        assert AntStatus.ERROR.value == "error"


@pytest.mark.parametrize("role,expected_max_children", [
    (AntRole.FOUNDING_QUEEN, 10),
    (AntRole.QUEEN, 50),
    (AntRole.WORKER, 0),
    (AntRole.DRONE, 0),
    (AntRole.ACCOUNTING, 0),
    (AntRole.PRINCESS, 0),
])
def test_role_max_children_config(role, expected_max_children):
    """Test that each role has correct max_children configuration."""
    ant = MockAnt("test-ant", role)
    assert ant.config["max_children"] == expected_max_children


@pytest.mark.parametrize("capital,threshold,expected", [
    (5.0, 2.0, True),
    (1.0, 2.0, False),
    (2.0, 2.0, True),
    (0.0, 2.0, False),
])
def test_should_split_various_capitals(capital, threshold, expected):
    """Test split decision with various capital amounts."""
    ant = MockAnt("test-ant", AntRole.WORKER)
    ant.config["split_threshold"] = threshold
    ant.capital.available_capital = capital
    
    assert ant.should_split() == expected 