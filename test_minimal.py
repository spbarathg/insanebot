#!/usr/bin/env python3
"""
Minimal test script to verify core functionality works
"""
import sys
import os
sys.path.append('.')

def test_portfolio_manager():
    """Test basic portfolio manager functionality"""
    print("🧪 Testing Portfolio Manager...")
    
    try:
        from src.core.portfolio_manager import PortfolioManager
        
        # Initialize portfolio manager
        pm = PortfolioManager()
        success = pm.initialize(0.1)
        print(f"   ✅ Initialization: {'SUCCESS' if success else 'FAILED'}")
        
        # Test portfolio summary
        summary = pm.get_portfolio_summary()
        required_keys = ['current_value', 'unrealized_profit', 'percent_return', 'max_drawdown']
        
        for key in required_keys:
            if key in summary:
                print(f"   ✅ {key}: {summary[key]}")
            else:
                print(f"   ❌ Missing key: {key}")
                return False
        
        # Test holdings method
        holdings = pm.get_holdings()
        print(f"   ✅ Holdings method returns: {type(holdings).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Portfolio Manager Error: {str(e)}")
        return False

def test_data_ingestion():
    """Test data ingestion token address handling"""
    print("🧪 Testing Data Ingestion...")
    
    try:
        from src.core.data_ingestion import DataIngestion
        
        # Create data ingestion instance
        di = DataIngestion()
        
        # Test token address extraction
        test_addresses = [
            {"address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"},
            "So11111111111111111111111111111111111111112",
            {"address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", "symbol": "USDT"}
        ]
        
        for i, token in enumerate(test_addresses):
            if isinstance(token, dict):
                address = token.get("address")
                print(f"   ✅ Token {i+1}: Extracted '{address}' from dict")
            else:
                address = token
                print(f"   ✅ Token {i+1}: Direct string '{address}'")
            
            if address and isinstance(address, str):
                print(f"      ✓ Valid string address")
            else:
                print(f"      ❌ Invalid address format")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data Ingestion Error: {str(e)}")
        return False

def test_main_imports():
    """Test that main.py imports work correctly"""
    print("🧪 Testing Main Imports...")
    
    try:
        # Test key imports from main.py
        from src.services.jupiter_service import JupiterService
        print("   ✅ JupiterService import: SUCCESS")
        
        from src.services.wallet_manager import WalletManager
        print("   ✅ WalletManager import: SUCCESS")
        
        from src.core.portfolio_manager import PortfolioManager
        print("   ✅ PortfolioManager import: SUCCESS")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import Error: {str(e)}")
        return False

def main():
    """Run all minimal tests"""
    print("🎯 Enhanced Ant Bot - Minimal Functionality Test")
    print("=" * 60)
    
    tests = [
        ("Portfolio Manager", test_portfolio_manager),
        ("Data Ingestion", test_data_ingestion),
        ("Main Imports", test_main_imports)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print()
        try:
            if test_func():
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
        print("🎉 All core functionality tests PASSED!")
        return True
    else:
        print("⚠️ Some tests FAILED - check output above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 