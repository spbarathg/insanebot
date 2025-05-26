#!/usr/bin/env python3
"""
Test script specifically for the Portfolio Risk Manager
"""
import sys
import os
import warnings
sys.path.append('.')

# Suppress numpy warnings for testing
warnings.filterwarnings('ignore', category=RuntimeWarning)

import asyncio

async def test_risk_manager():
    """Test portfolio risk manager initialization and basic functionality"""
    print("🛡️ Testing Portfolio Risk Manager...")
    
    try:
        from src.core.portfolio_manager import PortfolioManager
        from src.core.portfolio_risk_manager import PortfolioRiskManager
        
        # Initialize portfolio manager first
        pm = PortfolioManager()
        success = pm.initialize(0.1)
        
        if not success:
            print("   ❌ Portfolio Manager initialization failed")
            return False
        
        print("   ✅ Portfolio Manager initialized")
        
        # Initialize risk manager
        risk_manager = PortfolioRiskManager(pm)
        init_success = await risk_manager.initialize()
        
        if not init_success:
            print("   ❌ Risk Manager initialization failed")
            return False
        
        print("   ✅ Risk Manager initialized successfully")
        
        # Test basic risk assessment
        risk_metrics = await risk_manager.assess_portfolio_risk()
        
        print(f"   ✅ Risk assessment completed")
        print(f"      - Total Value: {risk_metrics.total_value}")
        print(f"      - Total Positions: {risk_metrics.total_positions}")
        print(f"      - Portfolio VAR (1d): {risk_metrics.portfolio_var_1d}")
        print(f"      - Max Drawdown: {risk_metrics.max_drawdown}")
        print(f"      - Overall Risk Level: {risk_metrics.overall_risk_level}")
        
        # Test risk violations check
        violations = await risk_manager.check_risk_violations(risk_metrics)
        print(f"   ✅ Risk violations check: {len(violations)} violations found")
        
        # Test get risk stats
        stats = risk_manager.get_risk_stats()
        print(f"   ✅ Risk stats retrieved: {len(stats)} metrics")
        
        # Clean up
        await risk_manager.close()
        print("   ✅ Risk Manager closed properly")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import Error: {str(e)}")
        print("      This might be due to missing numpy or other dependencies")
        return False
    except Exception as e:
        print(f"   ❌ Risk Manager Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_risk_manager_with_positions():
    """Test risk manager with some test positions"""
    print("🔍 Testing Risk Manager with Test Positions...")
    
    try:
        from src.core.portfolio_manager import PortfolioManager
        from src.core.portfolio_risk_manager import PortfolioRiskManager
        
        # Initialize portfolio manager
        pm = PortfolioManager()
        pm.initialize(1.0)  # Start with 1 SOL
        
        # Add some test positions
        await pm.add_position("token1", 0.1, 10.0)  # 0.1 tokens at 10 SOL each
        await pm.add_position("token2", 0.05, 20.0)  # 0.05 tokens at 20 SOL each
        
        print("   ✅ Test positions added")
        
        # Initialize risk manager
        risk_manager = PortfolioRiskManager(pm)
        await risk_manager.initialize()
        
        print("   ✅ Risk Manager initialized with positions")
        
        # Update prices to create some PnL
        await pm.update_position("token1", 12.0)  # +20% price increase
        await pm.update_position("token2", 18.0)  # -10% price decrease
        
        print("   ✅ Position prices updated")
        
        # Assess risk
        risk_metrics = await risk_manager.assess_portfolio_risk()
        
        print(f"   ✅ Risk assessment with positions:")
        print(f"      - Total Positions: {risk_metrics.total_positions}")
        print(f"      - Total Value: {risk_metrics.total_value:.4f}")
        print(f"      - Unrealized PnL: {risk_metrics.total_unrealized_pnl:.4f}")
        print(f"      - Risk Level: {risk_metrics.overall_risk_level}")
        
        # Check violations
        violations = await risk_manager.check_risk_violations(risk_metrics)
        if violations:
            print(f"   ⚠️ Risk violations found: {violations}")
        else:
            print("   ✅ No risk violations")
        
        await risk_manager.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

async def main():
    """Run risk manager tests"""
    print("🎯 Portfolio Risk Manager Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Risk Manager", test_risk_manager),
        ("Risk Manager with Positions", test_risk_manager_with_positions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print()
        try:
            if await test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: CRASHED - {str(e)}")
    
    print()
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Risk Manager tests PASSED!")
        return True
    else:
        print("⚠️ Some tests FAILED - check output above")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 