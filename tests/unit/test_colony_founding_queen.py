"""
Unit tests for FoundingAntQueen class.
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from src.colony.founding_queen import FoundingAntQueen
from src.colony.base_ant import AntRole, AntStatus


class TestFoundingAntQueen:
    """Test the FoundingAntQueen class."""
    
    def test_initialization(self):
        """Test FoundingAntQueen initialization."""
        initial_capital = 25.0
        queen = FoundingAntQueen("test_founding_queen", initial_capital)
        
        assert queen.ant_id == "test_founding_queen"
        assert queen.role == AntRole.FOUNDING_QUEEN
        assert queen.capital.current_balance == initial_capital
        assert queen.queens == {}
        assert queen.retired_queens == []
        assert queen.system_metrics["total_capital"] == initial_capital
        assert queen.system_metrics["total_ants"] == 1
        assert queen.system_metrics["queens_created"] == 0
    
    def test_default_initialization(self):
        """Test FoundingAntQueen with default parameters."""
        queen = FoundingAntQueen()
        
        assert queen.ant_id == "founding_queen_0"
        assert queen.capital.current_balance == 20.0
        assert queen.system_metrics["total_capital"] == 20.0
    
    @pytest.mark.asyncio
    async def test_initialize_success_with_capital(self):
        """Test successful initialization with sufficient capital."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            # Mock queen creation
            mock_queen = AsyncMock()
            mock_queen.initialize.return_value = True
            MockAntQueen.return_value = mock_queen
            
            queen = FoundingAntQueen("test_queen", 25.0)
            result = await queen.initialize()
            
            assert result is True
            assert len(queen.queens) == 1
            assert "queen_1" in queen.queens
            assert queen.system_metrics["queens_created"] == 1
            assert len(queen.children) == 1
    
    @pytest.mark.asyncio
    async def test_initialize_insufficient_capital(self):
        """Test initialization with insufficient capital."""
        queen = FoundingAntQueen("test_queen", 1.0)  # Less than 2.0 needed
        result = await queen.initialize()
        
        assert result is True  # Still succeeds, just no initial queen
        assert len(queen.queens) == 0
        assert queen.system_metrics["queens_created"] == 0
    
    @pytest.mark.asyncio
    async def test_initialize_queen_creation_fails(self):
        """Test initialization when queen creation fails."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            # Mock queen creation failure
            mock_queen = AsyncMock()
            mock_queen.initialize.return_value = False
            MockAntQueen.return_value = mock_queen
            
            queen = FoundingAntQueen("test_queen", 25.0)
            result = await queen.initialize()
            
            assert result is True  # Initialize still succeeds
            assert len(queen.queens) == 0
            assert queen.capital.allocated_capital == 0  # Capital should be released
    
    @pytest.mark.asyncio
    async def test_execute_cycle_inactive_status(self):
        """Test execute_cycle when status is not active."""
        queen = FoundingAntQueen("test_queen", 25.0)
        queen.status = AntStatus.ERROR
        
        result = await queen.execute_cycle()
        
        assert result["status"] == "inactive"
        assert "Founding Queen status: error" in result["reason"]
    
    @pytest.mark.asyncio
    async def test_execute_cycle_active(self):
        """Test successful execute_cycle."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            # Setup mock queen
            mock_queen = AsyncMock()
            mock_queen.status = AntStatus.ACTIVE
            mock_queen.execute_cycle.return_value = {
                "status": "active",
                "queen_metrics": {
                    "queen_specific": {
                        "workers": {
                            "currently_active": 3,
                            "total_worker_profit": 15.0
                        }
                    }
                }
            }
            MockAntQueen.return_value = mock_queen
            
            queen = FoundingAntQueen("test_queen", 25.0)
            await queen.initialize()
            
            result = await queen.execute_cycle()
            
            assert result["status"] == "active"
            assert "queen_coordination" in result
            assert "queen_management" in result
            assert "system_metrics" in result
            assert "queens_status" in result
    
    @pytest.mark.asyncio
    async def test_create_initial_queen_success(self):
        """Test successful initial queen creation."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            mock_queen = AsyncMock()
            mock_queen.initialize.return_value = True
            MockAntQueen.return_value = mock_queen
            
            queen = FoundingAntQueen("test_queen", 25.0)
            result = await queen._create_initial_queen()
            
            assert result is True
            assert queen.capital.allocated_capital == 2.0
            assert "queen_1" in queen.queens
            assert "queen_1" in queen.children
            assert queen.system_metrics["queens_created"] == 1
    
    @pytest.mark.asyncio
    async def test_create_initial_queen_insufficient_capital(self):
        """Test initial queen creation with insufficient capital."""
        queen = FoundingAntQueen("test_queen", 1.0)
        result = await queen._create_initial_queen()
        
        assert result is False
        assert len(queen.queens) == 0
    
    @pytest.mark.asyncio
    async def test_create_initial_queen_initialization_fails(self):
        """Test initial queen creation when initialization fails."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            mock_queen = AsyncMock()
            mock_queen.initialize.return_value = False
            MockAntQueen.return_value = mock_queen
            
            queen = FoundingAntQueen("test_queen", 25.0)
            result = await queen._create_initial_queen()
            
            assert result is False
            assert queen.capital.allocated_capital == 0  # Capital released
            assert len(queen.queens) == 0
    
    @pytest.mark.asyncio
    async def test_coordinate_queens_empty(self):
        """Test queen coordination with no queens."""
        queen = FoundingAntQueen("test_queen", 25.0)
        result = await queen._coordinate_queens()
        
        assert result["queens_coordinated"] == 0
        assert result["total_workers"] == 0
        assert result["total_trades"] == 0
        assert result["total_profit"] == 0.0
        assert result["queen_results"] == {}
    
    @pytest.mark.asyncio
    async def test_coordinate_queens_with_active_queens(self):
        """Test queen coordination with active queens."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            # Setup mock queen
            mock_queen = AsyncMock()
            mock_queen.status = AntStatus.ACTIVE
            mock_queen.execute_cycle.return_value = {
                "status": "active",
                "queen_metrics": {
                    "queen_specific": {
                        "workers": {
                            "currently_active": 5,
                            "total_worker_profit": 25.0
                        }
                    }
                }
            }
            MockAntQueen.return_value = mock_queen
            
            queen = FoundingAntQueen("test_queen", 25.0)
            queen.queens["queen_1"] = mock_queen
            
            result = await queen._coordinate_queens()
            
            assert result["queens_coordinated"] == 1
            assert result["total_workers"] == 5
            assert result["total_profit"] == 25.0
            assert "queen_1" in result["queen_results"]
    
    @pytest.mark.asyncio
    async def test_coordinate_queens_with_splitting_queen(self):
        """Test queen coordination when a queen is splitting."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            # Setup mock queen that's splitting
            mock_queen = AsyncMock()
            mock_queen.status = AntStatus.ACTIVE
            mock_queen.execute_cycle.return_value = {
                "status": "splitting",
                "split_result": {
                    "type": "queen_split",
                    "new_queen_capital": 1.5,
                    "inheritance_data": {
                        "ai_learning_data": {"strategy": "test"},
                        "worker_strategies": {"strategy1": "data"},
                        "performance_patterns": {"pattern1": "value"}
                    }
                }
            }
            MockAntQueen.return_value = mock_queen
            
            queen = FoundingAntQueen("test_queen", 25.0)
            queen.queens["queen_1"] = mock_queen
            
            with patch.object(queen, '_handle_queen_split_request') as mock_split_handler:
                mock_split_handler.return_value = True
                result = await queen._coordinate_queens()
                
                mock_split_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_queen_split_request_success(self):
        """Test successful queen split handling."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            # Mock new queen creation
            mock_new_queen = AsyncMock()
            mock_new_queen.initialize.return_value = True
            mock_new_queen.metadata = {}
            MockAntQueen.return_value = mock_new_queen
            
            # Mock original queen
            mock_original_queen = AsyncMock()
            
            queen = FoundingAntQueen("test_queen", 25.0)
            queen.queens["queen_1"] = mock_original_queen
            
            split_data = {
                "type": "queen_split",
                "new_queen_capital": 1.5,
                "inheritance_data": {
                    "ai_learning_data": {"strategy": "test"},
                    "worker_strategies": {"strategy1": "data"},
                    "performance_patterns": {"pattern1": "value"}
                }
            }
            
            with patch.object(queen, '_retire_queen') as mock_retire:
                mock_retire.return_value = True
                result = await queen._handle_queen_split_request("queen_1", split_data)
                
                assert result is True
                assert "queen_2" in queen.queens
                assert queen.system_metrics["queens_created"] == 1
                mock_retire.assert_called_once_with("queen_1")
    
    @pytest.mark.asyncio
    async def test_handle_queen_split_request_invalid_data(self):
        """Test queen split handling with invalid data."""
        queen = FoundingAntQueen("test_queen", 25.0)
        
        # Test with None data
        result = await queen._handle_queen_split_request("queen_1", None)
        assert result is False
        
        # Test with wrong type
        split_data = {"type": "worker_split"}
        result = await queen._handle_queen_split_request("queen_1", split_data)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_apply_inheritance(self):
        """Test applying inheritance data to new queen."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            mock_queen = AsyncMock()
            mock_queen.metadata = {}
            MockAntQueen.return_value = mock_queen
            
            queen = FoundingAntQueen("test_queen", 25.0)
            
            inheritance_data = {
                "ai_learning_data": {"strategy": "test"},
                "worker_strategies": {"strategy1": "data"},
                "performance_patterns": {"pattern1": "value"}
            }
            
            await queen._apply_inheritance(mock_queen, inheritance_data)
            
            assert mock_queen.metadata["inherited_ai_data"] == {"strategy": "test"}
            assert mock_queen.metadata["inherited_strategies"] == {"strategy1": "data"}
            assert mock_queen.metadata["inherited_patterns"] == {"pattern1": "value"}
    
    @pytest.mark.asyncio
    async def test_manage_queen_lifecycle(self):
        """Test queen lifecycle management."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            mock_queen = AsyncMock()
            mock_queen.should_retire.return_value = False
            mock_queen.should_merge.return_value = False
            MockAntQueen.return_value = mock_queen
            
            queen = FoundingAntQueen("test_queen", 25.0)
            queen.queens["queen_1"] = mock_queen
            
            result = await queen._manage_queen_lifecycle()
            
            assert "queens_checked" in result
            assert "retirement_actions" in result
            assert "merge_actions" in result
    
    @pytest.mark.asyncio
    async def test_retire_queen(self):
        """Test queen retirement."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            mock_queen = AsyncMock()
            # Properly configure async mock to return a coroutine
            mock_queen.cleanup = AsyncMock(return_value=None)
            mock_queen.capital.current_balance = 1.5
            MockAntQueen.return_value = mock_queen
            
            queen = FoundingAntQueen("test_queen", 25.0)
            queen.queens["queen_1"] = mock_queen
            queen.children = ["queen_1"]
            
            result = await queen._retire_queen("queen_1")
            
            assert result is True
            assert "queen_1" not in queen.queens
            assert "queen_1" in queen.retired_queens
            assert "queen_1" not in queen.children
            assert queen.system_metrics["queens_retired"] == 1
            # Await the async mock call
            mock_queen.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retire_queen_nonexistent(self):
        """Test retiring a queen that doesn't exist."""
        queen = FoundingAntQueen("test_queen", 25.0)
        result = await queen._retire_queen("nonexistent_queen")
        
        assert result is False
    
    def test_should_create_queen_true(self):
        """Test should create queen when conditions are met."""
        queen = FoundingAntQueen("test_queen", 25.0)
        queen.capital.available_capital = 25.0  # Above 20.0 threshold
        
        result = queen._should_create_queen()
        assert result is True
    
    def test_should_create_queen_false(self):
        """Test should create queen when conditions are not met."""
        queen = FoundingAntQueen("test_queen", 25.0)
        queen.capital.available_capital = 15.0  # Below 20.0 threshold
        
        result = queen._should_create_queen()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_create_queen_success(self):
        """Test successful queen creation."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            mock_queen = AsyncMock()
            mock_queen.initialize.return_value = True
            MockAntQueen.return_value = mock_queen
            
            queen = FoundingAntQueen("test_queen", 25.0)
            queen.capital.available_capital = 25.0
            
            result = await queen._create_queen()
            
            assert result["success"] is True
            assert "queen_id" in result
            assert result["capital_allocated"] == 20.0  # Split threshold is 20.0 SOL for founding queen
            assert len(queen.queens) == 1
    
    @pytest.mark.asyncio
    async def test_create_queen_insufficient_capital(self):
        """Test queen creation with insufficient capital."""
        queen = FoundingAntQueen("test_queen", 25.0)
        queen.capital.available_capital = 1.0
        
        result = await queen._create_queen()
        
        assert result["success"] is False
        assert result["reason"] == "insufficient_capital"
    
    @pytest.mark.asyncio
    async def test_update_system_metrics(self):
        """Test system metrics update."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            mock_queen = AsyncMock()
            mock_queen.capital.current_balance = 5.0
            mock_queen.performance.total_trades = 10
            mock_queen.performance.total_profit = 2.5
            # Configure workers as AsyncMock to prevent warnings
            mock_queen.workers = AsyncMock(return_value={})
            MockAntQueen.return_value = mock_queen
            
            queen = FoundingAntQueen("test_queen", 25.0)
            queen.queens["queen_1"] = mock_queen
            
            await queen._update_system_metrics()
            
            assert queen.system_metrics["total_ants"] >= 1
            assert queen.system_metrics["total_capital"] >= 25.0
            assert queen.system_metrics["total_trades"] == 10
            assert queen.system_metrics["system_profit"] == 2.5
    
    def test_get_queens_status(self):
        """Test getting queens status."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            mock_queen = AsyncMock()
            mock_queen.status = AntStatus.ACTIVE  # Set the status attribute
            mock_queen.get_status_summary.return_value = {
                "ant_id": "queen_1",
                "status": "active",
                "capital": 5.0
            }
            mock_queen.capital.current_balance = 5.0
            mock_queen.performance.total_profit = 1.0
            mock_queen.workers = {}  # Initialize workers as empty dict
            MockAntQueen.return_value = mock_queen
            
            queen = FoundingAntQueen("test_queen", 25.0)
            queen.queens["queen_1"] = mock_queen
            
            result = queen._get_queens_status()
            
            assert result["total_queens"] == 1
            assert result["active_queens"] == 1
            assert "queen_1" in result["queen_details"]
    
    def test_get_system_status(self):
        """Test getting complete system status."""
        queen = FoundingAntQueen("test_queen", 25.0)
        
        result = queen.get_system_status()
        
        assert result["founding_queen_id"] == "test_queen"
        assert result["system_metrics"] == queen.system_metrics
        assert "founding_queen_status" in result
        assert "queens_overview" in result
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup method."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            mock_queen = AsyncMock()
            # Properly configure cleanup as AsyncMock
            mock_queen.cleanup = AsyncMock(return_value=None)
            MockAntQueen.return_value = mock_queen
            
            queen = FoundingAntQueen("test_queen", 25.0)
            queen.queens["queen_1"] = mock_queen
            
            await queen.cleanup()
            
            mock_queen.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_cycle_exception_handling(self):
        """Test execute_cycle exception handling."""
        queen = FoundingAntQueen("test_queen", 25.0)
        
        with patch.object(queen, '_coordinate_queens', side_effect=Exception("Test error")):
            result = await queen.execute_cycle()
            
            assert result["status"] == "error"
            assert "Test error" in result["error"]


@pytest.mark.asyncio
class TestFoundingAntQueenIntegration:
    """Integration tests for FoundingAntQueen."""
    
    async def test_full_lifecycle_simulation(self):
        """Test a complete lifecycle simulation."""
        with patch('src.colony.founding_queen.AntQueen') as MockAntQueen:
            # Mock queen behavior
            mock_queen = AsyncMock()
            mock_queen.status = AntStatus.ACTIVE
            mock_queen.initialize.return_value = True
            mock_queen.execute_cycle.return_value = {
                "status": "active",
                "queen_metrics": {
                    "queen_specific": {
                        "workers": {
                            "currently_active": 2,
                            "total_worker_profit": 10.0
                        }
                    }
                }
            }
            mock_queen.should_retire.return_value = False
            mock_queen.should_merge.return_value = False
            mock_queen.capital.current_balance = 3.0
            mock_queen.performance.total_trades = 5
            mock_queen.performance.total_profit = 1.0
            mock_queen.get_status_summary.return_value = {
                "ant_id": "queen_1",
                "status": "active",
                "capital": 3.0
            }
            MockAntQueen.return_value = mock_queen
            
            # Initialize and run cycles
            queen = FoundingAntQueen("test_founding_queen", 30.0)
            
            # Initialize
            init_result = await queen.initialize()
            assert init_result is True
            
            # Run several cycles
            for i in range(3):
                cycle_result = await queen.execute_cycle()
                assert cycle_result["status"] == "active"
            
            # Check final status
            system_status = queen.get_system_status()
            assert system_status["founding_queen_id"] == "test_founding_queen"
            
            # Cleanup
            await queen.cleanup() 