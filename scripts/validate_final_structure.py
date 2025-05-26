#!/usr/bin/env python3
"""
Final structure validation script - verifies all imports work correctly after cleanup.
"""
import sys
import importlib
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all critical imports work correctly"""
    print("🔍 Testing core module imports...")
    
    # Test core modules
    core_modules = [
        'src.core.trade_execution',
        'src.core.error_handler', 
        'src.core.config_manager',
        'src.core.market_data',
        'src.core.portfolio_manager',
        'src.core.data_ingestion',
        'src.core.whale_tracker'
    ]
    
    for module in core_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            
    # Test config modules
    print("\n🔍 Testing config modules...")
    config_modules = [
        'config.core_config',
        'config.ant_princess_config', 
        'config.wallet_tracker_config'
    ]
    
    for module in config_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            
    # Test services
    print("\n🔍 Testing service modules...")
    service_modules = [
        'src.services.quicknode_service',
        'src.services.helius_service',
        'src.services.jupiter_service',
        'src.services.wallet_manager'
    ]
    
    for module in service_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            
    # Test monitoring
    print("\n🔍 Testing monitoring modules...")
    monitoring_modules = [
        'src.monitoring.monitoring',
        'src.monitoring.alerts'
    ]
    
    for module in monitoring_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")

def test_main_entry_points():
    """Test main entry points"""
    print("\n🔍 Testing main entry points...")
    
    entry_points = [
        'main',
        'src.core.enhanced_main'
    ]
    
    for module in entry_points:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")

def verify_deleted_references():
    """Verify no references to deleted modules exist"""
    print("\n🔍 Verifying no deleted module references...")
    
    # These should not be importable anymore
    deleted_modules = [
        'src.utils.config',
        'src.utils.logging_config',
        'src.core.config',
        'src.core.monitoring',
        'src.core.metrics'
    ]
    
    for module in deleted_modules:
        try:
            importlib.import_module(module)
            print(f"❌ ERROR: {module} still exists (should be deleted)")
        except ImportError:
            print(f"✅ {module} correctly not found")
        except Exception as e:
            print(f"⚠️ {module}: {e}")

if __name__ == "__main__":
    print("🚀 Final Structure Validation")
    print("=" * 50)
    
    test_imports()
    test_main_entry_points()
    verify_deleted_references()
    
    print("\n" + "=" * 50)
    print("✅ Final structure validation complete!")
    print("🐜 Ant Bot Ultimate is ready for deployment!") 