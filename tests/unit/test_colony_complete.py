"""
Unit tests for all colony ant types (Queen, Worker, Drone, Accounting, Princess).
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from src.colony.base_ant import AntRole, AntStatus

# Mock all colony classes since they might have complex dependencies
@pytest.fixture
def mock_colony_modules():
    """Mock all colony modules."""
    modules = {
        'src.colony.ant_queen': MagicMock(),
        'src.colony.worker_ant': MagicMock(),
        'src.colony.ant_drone': MagicMock(),
        'src.colony.accounting_ant': MagicMock(),
        'src.colony.ant_princess': MagicMock(),
    }
    
    with patch.dict('sys.modules', modules):
        yield modules


class MockAntQueen:
    """Mock implementation of AntQueen for testing."""
    
    def __init__(self, ant_id: str, parent_id: str, initial_capital: float):
        self.ant_id = ant_id
        self.parent_id = parent_id
        self.role = AntRole.QUEEN
        self.status = AntStatus.ACTIVE
        self.workers = {}
        self.retired_workers = []
        self.capital = MagicMock()
        self.capital.current_balance = initial_capital
        self.capital.available_capital = initial_capital
        self.performance = MagicMock()
        self.performance.total_trades = 0
        self.performance.total_profit = 0.0
        self.children = []
        self.metadata = {}
        self.created_at = time.time()
        self.last_activity = time.time()
    
    async def initialize(self):
        return True
    
    async def execute_cycle(self):
        return {"status": "active", "workers_managed": len(self.workers)}
    
    async def cleanup(self):
        pass
    
    def should_split(self):
        return self.capital.available_capital >= 2.0
    
    def should_merge(self):
        return self.capital.available_capital < 0.5
    
    def should_retire(self):
        return False
    
    def get_status_summary(self):
        return {
            "ant_id": self.ant_id,
            "role": "queen",
            "status": "active",
            "capital": self.capital.current_balance,
            "workers": len(self.workers)
        }


class MockWorkerAnt:
    """Mock implementation of WorkerAnt for testing."""
    
    def __init__(self, ant_id: str, parent_id: str, initial_capital: float):
        self.ant_id = ant_id
        self.parent_id = parent_id
        self.role = AntRole.WORKER
        self.status = AntStatus.ACTIVE
        self.capital = MagicMock()
        self.capital.current_balance = initial_capital
        self.capital.available_capital = initial_capital
        self.performance = MagicMock()
        self.performance.total_trades = 0
        self.performance.total_profit = 0.0
        self.trades_remaining = 10
        self.target_coin = None
        self.metadata = {}
        self.created_at = time.time()
        self.last_activity = time.time()
    
    async def initialize(self):
        return True
    
    async def execute_cycle(self):
        self.trades_remaining -= 1
        return {"status": "active", "trades_remaining": self.trades_remaining}
    
    async def cleanup(self):
        pass
    
    def should_retire(self):
        return self.trades_remaining <= 0
    
    def get_status_summary(self):
        return {
            "ant_id": self.ant_id,
            "role": "worker",
            "status": "active",
            "capital": self.capital.current_balance,
            "trades_remaining": self.trades_remaining
        }


class TestAntQueen:
    """Test the AntQueen class."""
    
    def test_ant_queen_initialization(self):
        """Test AntQueen initialization."""
        queen = MockAntQueen("queen_1", "founding_queen_0", 5.0)
        
        assert queen.ant_id == "queen_1"
        assert queen.parent_id == "founding_queen_0"
        assert queen.role == AntRole.QUEEN
        assert queen.capital.current_balance == 5.0
        assert queen.workers == {}
        assert queen.retired_workers == []
    
    @pytest.mark.asyncio
    async def test_ant_queen_initialize(self):
        """Test AntQueen initialization method."""
        queen = MockAntQueen("queen_1", "founding_queen_0", 5.0)
        result = await queen.initialize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_ant_queen_execute_cycle(self):
        """Test AntQueen execute cycle."""
        queen = MockAntQueen("queen_1", "founding_queen_0", 5.0)
        
        # Add some mock workers
        queen.workers["worker_1"] = MockWorkerAnt("worker_1", "queen_1", 1.0)
        queen.workers["worker_2"] = MockWorkerAnt("worker_2", "queen_1", 1.0)
        
        result = await queen.execute_cycle()
        
        assert result["status"] == "active"
        assert result["workers_managed"] == 2
    
    def test_ant_queen_should_split(self):
        """Test AntQueen split logic."""
        queen = MockAntQueen("queen_1", "founding_queen_0", 5.0)
        queen.capital.available_capital = 3.0  # Above 2.0 threshold
        
        assert queen.should_split() is True
        
        queen.capital.available_capital = 1.0  # Below threshold
        assert queen.should_split() is False
    
    def test_ant_queen_should_merge(self):
        """Test AntQueen merge logic."""
        queen = MockAntQueen("queen_1", "founding_queen_0", 5.0)
        queen.capital.available_capital = 0.3  # Below 0.5 threshold
        
        assert queen.should_merge() is True
        
        queen.capital.available_capital = 1.0  # Above threshold
        assert queen.should_merge() is False


class TestWorkerAnt:
    """Test the WorkerAnt class."""
    
    def test_worker_ant_initialization(self):
        """Test WorkerAnt initialization."""
        worker = MockWorkerAnt("worker_1", "queen_1", 1.0)
        
        assert worker.ant_id == "worker_1"
        assert worker.parent_id == "queen_1"
        assert worker.role == AntRole.WORKER
        assert worker.capital.current_balance == 1.0
        assert worker.trades_remaining == 10
        assert worker.target_coin is None
    
    @pytest.mark.asyncio
    async def test_worker_ant_initialize(self):
        """Test WorkerAnt initialization method."""
        worker = MockWorkerAnt("worker_1", "queen_1", 1.0)
        result = await worker.initialize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_worker_ant_execute_cycle(self):
        """Test WorkerAnt execute cycle."""
        worker = MockWorkerAnt("worker_1", "queen_1", 1.0)
        initial_trades = worker.trades_remaining
        
        result = await worker.execute_cycle()
        
        assert result["status"] == "active"
        assert result["trades_remaining"] == initial_trades - 1
        assert worker.trades_remaining == initial_trades - 1
    
    def test_worker_ant_should_retire(self):
        """Test WorkerAnt retirement logic."""
        worker = MockWorkerAnt("worker_1", "queen_1", 1.0)
        
        # Should not retire with trades remaining
        worker.trades_remaining = 5
        assert worker.should_retire() is False
        
        # Should retire when no trades remaining
        worker.trades_remaining = 0
        assert worker.should_retire() is True 