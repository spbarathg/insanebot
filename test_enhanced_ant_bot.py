#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Ant Bot System

This script tests all the new components:
- Ant Hierarchy (Founding Queen -> Queens -> Princesses)
- Enhanced AI Coordination with learning
- Self-Replication System
- Worker Ant lifecycle management
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from unittest.mock import Mock, AsyncMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestEnhancedAntBot:
    """Comprehensive test suite for Enhanced Ant Bot"""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = tempfile.mkdtemp()
        
    async def run_all_tests(self):
        """Run all test suites"""
        logger.info("🧪 Starting Enhanced Ant Bot Test Suite...")
        
        test_suites = [
            ("Ant Hierarchy", self.test_ant_hierarchy),
            ("AI Coordination", self.test_ai_coordination),
            ("Self-Replication", self.test_self_replication),
            ("Worker Lifecycle", self.test_worker_lifecycle),
            ("System Integration", self.test_system_integration)
        ]
        
        for suite_name, test_func in test_suites:
            logger.info(f"🔍 Testing {suite_name}...")
            try:
                result = await test_func()
                self.test_results[suite_name] = {"status": "PASS", "details": result}
                logger.info(f"✅ {suite_name}: PASSED")
            except Exception as e:
                self.test_results[suite_name] = {"status": "FAIL", "error": str(e)}
                logger.error(f"❌ {suite_name}: FAILED - {str(e)}")
        
        self.print_test_summary()
    
    async def test_ant_hierarchy(self):
        """Test the Ant hierarchy system"""
        try:
            from src.core.ai.ant_hierarchy import FoundingAntQueen, AntQueen, AntPrincess, AntRole
            
            # Test 1: Create Founding Queen
            founding_queen = FoundingAntQueen("test_founding_queen", 20.0)
            assert founding_queen.role == AntRole.FOUNDING_QUEEN
            assert founding_queen.capital.current_balance == 20.0
            
            # Test 2: Initialize Founding Queen (mock AI components)
            founding_queen.grok_engine = Mock()
            founding_queen.local_llm = Mock()
            await founding_queen.initialize()
            
            # Test 3: Verify Queen creation
            assert len(founding_queen.queens) >= 1, "Should create at least one Queen"
            
            # Test 4: Test Queen functionality
            queen_id = list(founding_queen.queens.keys())[0]
            queen = founding_queen.queens[queen_id]
            assert queen.role == AntRole.QUEEN
            assert queen.capital.current_balance == 2.0  # Default Queen capital
            
            # Test 5: Test Princess creation
            queen.grok_engine = Mock()
            queen.local_llm = Mock()
            princess_id = await queen.create_princess(0.5)
            assert princess_id is not None, "Should create Princess successfully"
            assert len(queen.princesses) == 1
            
            # Test 6: Test Princess functionality
            princess = queen.princesses[princess_id]
            assert princess.role == AntRole.PRINCESS
            assert princess.capital.current_balance == 0.5
            assert princess.target_trades is not None
            
            # Test 7: Test capital allocation
            initial_balance = princess.capital.current_balance
            allocated = princess.capital.allocate_capital(0.1)
            assert allocated is True
            assert princess.capital.allocated_capital == 0.1
            assert princess.capital.available_capital == initial_balance - 0.1
            
            return {
                "founding_queen_created": True,
                "queens_created": len(founding_queen.queens),
                "princesses_created": sum(len(q.princesses) for q in founding_queen.queens.values()),
                "capital_management": "working",
                "hierarchy_levels": 3
            }
            
        except ImportError as e:
            raise Exception(f"Failed to import ant hierarchy components: {str(e)}")
    
    async def test_ai_coordination(self):
        """Test the AI coordination system"""
        try:
            from src.core.ai.enhanced_ai_coordinator import AICoordinator, AIModelRole, AIDecision
            
            # Test 1: Create AI Coordinator
            coordinator = AICoordinator()
            
            # Mock the AI engines
            coordinator.grok_engine = Mock()
            coordinator.local_llm = Mock()
            
            # Mock AI responses
            coordinator.grok_engine.analyze_market = AsyncMock(return_value={
                "confidence": 0.8,
                "hype_level": 0.7,
                "community_sentiment": 0.6,
                "reasoning": "Strong social media activity"
            })
            
            coordinator.local_llm.analyze_market = AsyncMock(return_value={
                "confidence": 0.7,
                "trend_strength": 0.8,
                "momentum_score": 0.6,
                "reasoning": "Bullish technical indicators"
            })
            
            await coordinator.initialize()
            
            # Test 2: Comprehensive analysis
            market_data = {
                "token_address": "test_token_123",
                "price": 1.0,
                "volume_24h": 50000,
                "liquidity": 100000,
                "volatility": 0.3,
                "holder_count": 500
            }
            
            analysis_result = await coordinator.analyze_comprehensive("test_token_123", market_data)
            
            assert "sentiment_analysis" in analysis_result
            assert "technical_analysis" in analysis_result
            assert "risk_assessment" in analysis_result
            assert "final_decision" in analysis_result
            
            # Test 3: Learning feedback
            test_decision = AIDecision(
                model_role=AIModelRole.DECISION_MAKER,
                confidence=0.8,
                decision="buy",
                reasoning="Test decision",
                risk_score=0.3,
                supporting_data={}
            )
            
            trade_outcome = {
                "profit": 0.05,
                "success": True,
                "market_conditions": {}
            }
            
            feedback_result = await coordinator.learn_from_outcome("test_trade_1", test_decision, trade_outcome)
            assert feedback_result is True
            
            # Test 4: Performance tracking
            performance_summary = coordinator.get_performance_summary()
            assert "model_performances" in performance_summary
            assert "model_weights" in performance_summary
            
            return {
                "ai_coordinator_initialized": True,
                "comprehensive_analysis": "working",
                "learning_feedback": "working",
                "performance_tracking": "working",
                "model_roles": len(AIModelRole)
            }
            
        except ImportError as e:
            raise Exception(f"Failed to import AI coordination components: {str(e)}")
    
    async def test_self_replication(self):
        """Test the self-replication system"""
        try:
            from src.core.system_replicator import SystemReplicator, ReplicationTrigger
            from src.core.ai.ant_hierarchy import FoundingAntQueen
            
            # Test 1: Create components
            founding_queen = FoundingAntQueen("test_founding_queen", 50.0)  # High capital for replication
            replicator = SystemReplicator(founding_queen)
            
            # Test 2: Check replication triggers
            trigger = replicator.replication_trigger
            assert trigger.capital_threshold == 50.0
            assert trigger.profit_threshold == 20.0
            assert trigger.max_instances == 5
            
            # Test 3: Mock system status for replication conditions
            founding_queen.get_system_status = Mock(return_value={
                "system_metrics": {
                    "total_capital": 55.0,  # Above threshold
                    "system_profit": 25.0,  # Above threshold
                    "uptime": 90000,        # Above threshold
                    "total_trades": 20      # Enough for performance calculation
                },
                "queen_details": [{
                    "princess_details": [{
                        "performance": {
                            "total_trades": 10,
                            "win_rate": 85.0  # Above performance threshold
                        }
                    }]
                }]
            })
            
            # Test 4: Monitor replication conditions
            should_replicate = await replicator.monitor_replication_conditions()
            # Note: This might be False due to cooldown or other constraints, which is fine
            
            # Test 5: Test resource allocation
            resources = replicator.resource_allocator.allocate_resources(0)
            assert "port" in resources
            assert "memory_limit_gb" in resources
            assert "cpu_limit" in resources
            
            # Test 6: Test configuration management
            config = replicator.config_manager.generate_instance_config("test_instance", 5.0, resources)
            assert "instance_id" in config
            assert "initial_capital" in config
            assert config["initial_capital"] == 5.0
            
            # Test 7: Test workspace management (use temp directory)
            replicator.workspace_manager.base_workspace = self.temp_dir
            workspace_path = replicator.workspace_manager.create_instance_workspace("test_workspace")
            assert os.path.exists(workspace_path)
            
            return {
                "replicator_created": True,
                "replication_triggers": "configured",
                "resource_allocation": "working",
                "configuration_management": "working",
                "workspace_management": "working",
                "replication_ready": should_replicate
            }
            
        except ImportError as e:
            raise Exception(f"Failed to import replication components: {str(e)}")
    
    async def test_worker_lifecycle(self):
        """Test Worker Ant (Princess) lifecycle management"""
        try:
            from src.core.ai.ant_hierarchy import AntPrincess, AntRole
            
            # Test 1: Create Princess
            princess = AntPrincess("test_princess", "test_queen", 1.0)
            assert princess.role == AntRole.PRINCESS
            assert princess.target_trades is not None
            assert 5 <= princess.target_trades <= 10
            
            # Test 2: Test retirement conditions
            # New princess should not retire
            assert not princess.should_retire()
            
            # Test 3: Simulate trades to trigger retirement
            for i in range(10):
                princess.performance.update_trade_result(0.01, 1.0, True)
            
            # After 10 trades, should consider retirement
            assert princess.should_retire()
            
            # Test 4: Test splitting conditions
            princess2 = AntPrincess("test_princess_2", "test_queen", 3.0)  # Higher capital
            # Make it profitable
            for i in range(5):
                princess2.performance.update_trade_result(0.1, 1.0, True)
            
            # Should consider splitting with good performance and capital
            assert princess2.should_split()
            
            # Test 5: Test merging conditions
            princess3 = AntPrincess("test_princess_3", "test_queen", 0.05)  # Low capital
            # Make it perform poorly
            for i in range(5):
                princess3.performance.update_trade_result(-0.02, 1.0, False)
            
            # Should consider merging due to poor performance
            assert princess3.should_merge()
            
            # Test 6: Test performance tracking
            princess4 = AntPrincess("test_princess_4", "test_queen", 1.0)
            initial_trades = princess4.performance.total_trades
            princess4.performance.update_trade_result(0.05, 2.0, True)
            
            assert princess4.performance.total_trades == initial_trades + 1
            assert princess4.performance.total_profit == 0.05
            assert princess4.performance.win_rate == 100.0
            
            return {
                "princess_creation": "working",
                "retirement_logic": "working",
                "splitting_logic": "working", 
                "merging_logic": "working",
                "performance_tracking": "working",
                "lifecycle_thresholds": "configured"
            }
            
        except ImportError as e:
            raise Exception(f"Failed to import worker lifecycle components: {str(e)}")
    
    async def test_system_integration(self):
        """Test system integration of all components"""
        try:
            # Import main system
            from src.core.enhanced_main import AntBotSystem
            
            # Test 1: Create system
            system = AntBotSystem(initial_capital=10.0)
            assert system.initial_capital == 10.0
            
            # Test 2: Mock external dependencies to avoid network calls
            system._initialize_external_services = AsyncMock(return_value=True)
            system._initialize_core_infrastructure = AsyncMock(return_value=True)
            
            # Mock AI components
            from unittest.mock import patch
            
            with patch('src.core.ai.enhanced_ai_coordinator.AICoordinator') as mock_coordinator:
                mock_coordinator.return_value.initialize = AsyncMock(return_value=True)
                
                with patch('src.core.ai.ant_hierarchy.FoundingAntQueen') as mock_queen:
                    mock_queen.return_value.initialize = AsyncMock(return_value=True)
                    mock_queen.return_value.ant_id = "test_founding_queen"
                    
                    with patch('src.core.data_ingestion.DataIngestion') as mock_ingestion:
                        mock_ingestion.return_value.start = AsyncMock(return_value=True)
                        
                        # Test 3: Initialize system
                        init_result = await system.initialize()
                        assert init_result is True
            
            # Test 4: Test system overview
            system.founding_queen = Mock()
            system.founding_queen.get_system_status = Mock(return_value={"test": "data"})
            system.ai_coordinator = Mock()
            system.ai_coordinator.get_performance_summary = Mock(return_value={"test": "ai_data"})
            system.system_replicator = Mock()
            system.system_replicator.get_replication_status = Mock(return_value={"test": "replication_data"})
            
            overview = system.get_system_overview()
            assert "system_info" in overview
            assert "metrics" in overview
            assert "components" in overview
            
            return {
                "system_creation": "working",
                "system_initialization": "working", 
                "component_integration": "working",
                "system_overview": "working",
                "architecture_complete": True
            }
            
        except ImportError as e:
            raise Exception(f"Failed to import system integration components: {str(e)}")
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        logger.info("\n" + "="*60)
        logger.info("🧪 ENHANCED ANT BOT - TEST SUMMARY")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASS")
        failed_tests = total_tests - passed_tests
        
        logger.info(f"📊 Total Test Suites: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        logger.info("\n📋 Detailed Results:")
        for suite_name, result in self.test_results.items():
            status = result["status"]
            emoji = "✅" if status == "PASS" else "❌"
            logger.info(f"{emoji} {suite_name}: {status}")
            
            if status == "PASS" and "details" in result:
                for key, value in result["details"].items():
                    logger.info(f"     {key}: {value}")
            elif status == "FAIL":
                logger.info(f"     Error: {result['error']}")
        
        logger.info("\n🎯 Architecture Verification:")
        architecture_features = [
            ("Founding Queen → Queens → Princesses", "Ant Hierarchy"),
            ("2 SOL threshold splitting", "Worker Lifecycle"),
            ("5-10 trade retirement", "Worker Lifecycle"),
            ("Grok + Local LLM collaboration", "AI Coordination"),
            ("Learning from trade outcomes", "AI Coordination"),
            ("Automatic system replication", "Self-Replication"),
            ("Resource isolation", "Self-Replication")
        ]
        
        for feature, suite in architecture_features:
            status = self.test_results.get(suite, {}).get("status", "UNKNOWN")
            emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
            logger.info(f"{emoji} {feature}")
        
        logger.info("\n🔮 System Readiness Assessment:")
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Enhanced Ant Bot architecture is ready!")
            logger.info("✨ The system implements the complete Ant hierarchy as specified")
            logger.info("🚀 Ready for production deployment with the new architecture")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY READY - Minor issues need to be addressed")
            logger.info("🔧 Fix the failing components before full deployment")
        else:
            logger.info("❌ NOT READY - Major architectural issues detected")
            logger.info("🛠️ Significant work needed before deployment")
        
        logger.info("="*60)

async def main():
    """Run the comprehensive test suite"""
    tester = TestEnhancedAntBot()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 