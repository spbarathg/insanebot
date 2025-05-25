#!/usr/bin/env python3
"""
Simple test to isolate issues with the Enhanced Ant Bot system
"""

import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test basic imports"""
    try:
        logger.info("Testing basic imports...")
        
        # Test 1: Basic ant hierarchy import
        logger.info("Importing ant hierarchy...")
        from src.core.ai.ant_hierarchy import FoundingAntQueen, AntQueen, AntPrincess, AntRole
        logger.info("✅ Ant hierarchy imported successfully")
        
        # Test 2: AI coordinator import
        logger.info("Importing AI coordinator...")
        from src.core.ai.enhanced_ai_coordinator import AICoordinator, AIModelRole
        logger.info("✅ AI coordinator imported successfully")
        
        # Test 3: System replicator import
        logger.info("Importing system replicator...")
        from src.core.system_replicator import SystemReplicator, ReplicationTrigger
        logger.info("✅ System replicator imported successfully")
        
        # Test 4: Enhanced main import
        logger.info("Importing enhanced main...")
        from src.core.enhanced_main import AntBotSystem
        logger.info("✅ Enhanced main imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {str(e)}")
        return False

def test_basic_creation():
    """Test basic object creation without initialization"""
    try:
        logger.info("Testing basic object creation...")
        
        from src.core.ai.ant_hierarchy import FoundingAntQueen, AntRole
        
        # Test creating a Founding Queen
        logger.info("Creating Founding Queen...")
        founding_queen = FoundingAntQueen("test_queen", 20.0)
        
        assert founding_queen.role == AntRole.FOUNDING_QUEEN
        assert founding_queen.capital.current_balance == 20.0
        logger.info("✅ Founding Queen created successfully")
        
        # Test creating AI Coordinator
        logger.info("Creating AI Coordinator...")
        from src.core.ai.enhanced_ai_coordinator import AICoordinator
        coordinator = AICoordinator()
        logger.info("✅ AI Coordinator created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Object creation failed: {str(e)}")
        return False

def main():
    """Run simple tests"""
    logger.info("🧪 Starting Simple Ant Bot Tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Creation Test", test_basic_creation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"🔍 Running {test_name}...")
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
            logger.info(f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results[test_name] = "FAIL"
            logger.error(f"❌ {test_name}: FAILED - {str(e)}")
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 Test Summary:")
    for test_name, result in results.items():
        emoji = "✅" if result == "PASS" else "❌"
        logger.info(f"{emoji} {test_name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    logger.info(f"📈 Success Rate: {passed}/{total} ({(passed/total)*100:.1f}%)")
    logger.info("="*50)

if __name__ == "__main__":
    main() 