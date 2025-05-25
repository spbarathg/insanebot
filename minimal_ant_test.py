#!/usr/bin/env python3
"""
Minimal Ant Bot Test - Tests core logic without heavy dependencies
"""

import asyncio
import logging
import sys
import os
import time
from unittest.mock import Mock, AsyncMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_core_ant_classes():
    """Test core Ant classes without external dependencies"""
    try:
        logger.info("Testing core Ant classes...")
        
        # Import only the core classes we need to test
        import importlib.util
        import types
        
        # Load ant_hierarchy module without importing its dependencies
        spec = importlib.util.spec_from_file_location(
            "ant_hierarchy", 
            "src/core/ai/ant_hierarchy.py"
        )
        ant_module = importlib.util.module_from_spec(spec)
        
        # Mock the problematic imports before loading
        sys.modules['src.core.ai.grok_engine'] = Mock()
        sys.modules['src.core.local_llm'] = Mock() 
        sys.modules['src.core.wallet_manager'] = Mock()
        sys.modules['src.core.portfolio_risk_manager'] = Mock()
        
        # Execute the module
        spec.loader.exec_module(ant_module)
        
        # Test 1: Test Ant Roles and Status Enums
        assert hasattr(ant_module, 'AntRole')
        assert hasattr(ant_module, 'AntStatus')
        
        AntRole = ant_module.AntRole
        AntStatus = ant_module.AntStatus
        
        assert AntRole.FOUNDING_QUEEN.value == "founding_queen"
        assert AntRole.QUEEN.value == "queen"
        assert AntRole.PRINCESS.value == "princess"
        logger.info("✅ Ant roles and status enums working")
        
        # Test 2: Test AntCapital class
        AntCapital = ant_module.AntCapital
        capital = AntCapital(current_balance=10.0)
        capital.update_balance(12.0)
        assert capital.current_balance == 12.0
        assert capital.profit_loss == 2.0
        
        # Test capital allocation
        success = capital.allocate_capital(5.0)
        assert success is True
        assert capital.allocated_capital == 5.0
        assert capital.available_capital == 7.0
        logger.info("✅ AntCapital class working")
        
        # Test 3: Test AntPerformance class  
        AntPerformance = ant_module.AntPerformance
        performance = AntPerformance()
        performance.update_trade_result(0.05, 2.0, True)
        
        assert performance.total_trades == 1
        assert performance.successful_trades == 1
        assert performance.total_profit == 0.05
        assert performance.win_rate == 100.0
        logger.info("✅ AntPerformance class working")
        
        # Test 4: Test BaseAnt class
        BaseAnt = ant_module.BaseAnt
        base_ant = BaseAnt("test_ant", AntRole.PRINCESS, "parent_ant")
        
        assert base_ant.ant_id == "test_ant"
        assert base_ant.role == AntRole.PRINCESS
        assert base_ant.parent_id == "parent_ant"
        assert base_ant.status == AntStatus.ACTIVE
        
        # Test configuration
        config = base_ant._get_role_config()
        assert config["max_trades"] == 10
        assert config["retirement_trades"] == (5, 10)
        logger.info("✅ BaseAnt class working")
        
        # Test 5: Test Princess specific logic
        AntPrincess = ant_module.AntPrincess
        princess = AntPrincess("test_princess", "parent_queen", 2.0)
        
        assert princess.role == AntRole.PRINCESS
        assert princess.capital.current_balance == 2.0
        assert not princess.should_retire()  # New princess shouldn't retire
        
        # Simulate trades to test retirement
        for i in range(10):
            princess.performance.update_trade_result(0.01, 1.0, True)
        
        assert princess.should_retire()  # Should retire after 10 trades
        logger.info("✅ AntPrincess lifecycle working")
        
        # Test 6: Test Queen logic
        AntQueen = ant_module.AntQueen
        queen = AntQueen("test_queen", None, 5.0)
        
        assert queen.role == AntRole.QUEEN
        assert queen.capital.current_balance == 5.0
        assert len(queen.princesses) == 0
        
        # Test Queen should_split logic
        queen.capital.update_balance(10.0)  # Add more capital
        queen.performance.update_trade_result(1.0, 1.0, True)  # Make profitable
        assert queen.should_split()
        logger.info("✅ AntQueen logic working")
        
        # Test 7: Test Founding Queen
        FoundingAntQueen = ant_module.FoundingAntQueen
        founding_queen = FoundingAntQueen("founding_queen", 20.0)
        
        assert founding_queen.role == AntRole.FOUNDING_QUEEN
        assert founding_queen.capital.current_balance == 20.0
        assert len(founding_queen.queens) == 0
        logger.info("✅ FoundingAntQueen creation working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Core classes test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_coordinator_core():
    """Test AI coordinator core logic without external dependencies"""
    try:
        logger.info("Testing AI coordinator core...")
        
        # Mock the external imports
        sys.modules['src.core.ai.grok_engine'] = Mock()
        sys.modules['src.core.local_llm'] = Mock()
        
        # Import AI coordinator classes
        from src.core.ai.enhanced_ai_coordinator import (
            AIModelRole, AIDecision, ModelPerformance, PromptStrategy
        )
        
        # Test 1: Enums
        assert AIModelRole.SENTIMENT_ANALYZER.value == "sentiment_analyzer"
        assert AIModelRole.TECHNICAL_ANALYST.value == "technical_analyst"
        logger.info("✅ AI coordinator enums working")
        
        # Test 2: AIDecision
        decision = AIDecision(
            model_role=AIModelRole.DECISION_MAKER,
            confidence=0.8,
            decision="buy",
            reasoning="Test decision",
            risk_score=0.3,
            supporting_data={"test": "data"}
        )
        assert decision.confidence == 0.8
        assert decision.decision == "buy"
        logger.info("✅ AIDecision class working")
        
        # Test 3: ModelPerformance
        performance = ModelPerformance(AIModelRole.SENTIMENT_ANALYZER)
        performance.update_performance(True, 0.05, 0.8)
        
        assert performance.total_decisions == 1
        assert performance.correct_decisions == 1
        assert performance.accuracy == 1.0
        logger.info("✅ ModelPerformance class working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AI coordinator test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_system_integration_minimal():
    """Test minimal system integration"""
    try:
        logger.info("Testing minimal system integration...")
        
        # This tests the integration without actually initializing heavy components
        from src.core.enhanced_main import AntBotSystem
        
        # Create system
        system = AntBotSystem(initial_capital=10.0)
        assert system.initial_capital == 10.0
        assert system.start_time > 0
        logger.info("✅ AntBotSystem creation working")
        
        # Test system overview with mocked components
        system.founding_queen = Mock()
        system.founding_queen.get_system_status = Mock(return_value={
            "system_metrics": {"total_capital": 10.0}
        })
        system.ai_coordinator = Mock()
        system.ai_coordinator.get_performance_summary = Mock(return_value={
            "model_performances": {}
        })
        system.system_replicator = Mock()
        system.system_replicator.get_replication_status = Mock(return_value={
            "total_instances": 0
        })
        
        overview = system.get_system_overview()
        assert "system_info" in overview
        assert "metrics" in overview
        assert "components" in overview
        logger.info("✅ System overview working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run minimal tests"""
    logger.info("🧪 Starting Minimal Ant Bot Tests...")
    
    tests = [
        ("Core Ant Classes", test_core_ant_classes),
        ("AI Coordinator Core", test_ai_coordinator_core), 
        ("System Integration", test_system_integration_minimal)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"🔍 Running {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
            logger.info(f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results[test_name] = "FAIL"
            logger.error(f"❌ {test_name}: FAILED - {str(e)}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("🧪 MINIMAL ANT BOT TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    
    logger.info(f"📊 Total Tests: {total}")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {total - passed}")
    logger.info(f"📈 Success Rate: {(passed/total)*100:.1f}%")
    
    logger.info("\n📋 Detailed Results:")
    for test_name, result in results.items():
        emoji = "✅" if result == "PASS" else "❌"
        logger.info(f"{emoji} {test_name}: {result}")
    
    logger.info("\n🎯 Core Architecture Status:")
    architecture_tests = [
        ("Ant Role Hierarchy", "Core Ant Classes"),
        ("Capital Management", "Core Ant Classes"),
        ("Worker Lifecycle", "Core Ant Classes"),
        ("AI Model Coordination", "AI Coordinator Core"),
        ("System Integration", "System Integration")
    ]
    
    for feature, test_suite in architecture_tests:
        status = results.get(test_suite, "UNKNOWN")
        emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        logger.info(f"{emoji} {feature}")
    
    if passed == total:
        logger.info("\n🎉 ALL CORE TESTS PASSED!")
        logger.info("✨ Core Ant Bot architecture is working correctly")
        logger.info("🚀 Ready to proceed with full system testing")
    elif passed >= total * 0.8:
        logger.info("\n⚠️ MOSTLY WORKING - Minor issues detected")
    else:
        logger.info("\n❌ CORE ISSUES DETECTED - Architecture needs fixes")
    
    logger.info("="*60)

if __name__ == "__main__":
    asyncio.run(main()) 