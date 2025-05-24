#!/usr/bin/env python3
"""
Simple test script to check what components can be imported.
"""

def test_imports():
    """Test individual imports to identify issues."""
    
    print("Testing basic imports...")
    
    try:
        from src.core.validation import TradingValidator
        print("✅ TradingValidator imported successfully")
    except Exception as e:
        print(f"❌ TradingValidator import failed: {e}")
    
    try:
        from src.core.wallet_manager import WalletManager
        print("✅ WalletManager imported successfully")
    except Exception as e:
        print(f"❌ WalletManager import failed: {e}")
    
    try:
        from src.core.helius_service import HeliusService
        print("✅ HeliusService imported successfully")
    except Exception as e:
        print(f"❌ HeliusService import failed: {e}")
    
    try:
        from src.core.jupiter_service import JupiterService
        print("✅ JupiterService imported successfully")
    except Exception as e:
        print(f"❌ JupiterService import failed: {e}")
    
    print("\nTesting main bot import...")
    
    try:
        from src.core.main import MemeCoinBot
        print("✅ MemeCoinBot imported successfully")
    except Exception as e:
        print(f"❌ MemeCoinBot import failed: {e}")
    
    print("\nTesting validation functionality...")
    
    try:
        from src.core.validation import AddressValidator
        result = AddressValidator.is_valid_solana_address("So11111111111111111111111111111111111111112")
        print(f"✅ Address validation works: {result}")
    except Exception as e:
        print(f"❌ Address validation failed: {e}")

if __name__ == "__main__":
    test_imports() 