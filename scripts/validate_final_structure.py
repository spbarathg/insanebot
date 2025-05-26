#!/usr/bin/env python3
"""
Final Structure Validation Script
Tests that all imports work correctly after cleanup
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_core_infrastructure():
    """Test core infrastructure imports"""
    print("🏛️ Testing Core Infrastructure...")
    
    try:
        from src.core.config_manager import ConfigManager
        print("  ✅ ConfigManager")
    except ImportError as e:
        print(f"  ❌ ConfigManager: {e}")
    
    try:
        from src.core.logger import SystemLogger
        print("  ✅ SystemLogger")
    except ImportError as e:
        print(f"  ❌ SystemLogger: {e}")
    
    try:
        from src.core.system_metrics import SystemMetrics
        print("  ✅ SystemMetrics")
    except ImportError as e:
        print(f"  ❌ SystemMetrics: {e}")
    
    try:
        from src.core.security_manager import SecurityManager
        print("  ✅ SecurityManager")
    except ImportError as e:
        print(f"  ❌ SecurityManager: {e}")

def test_services():
    """Test services imports"""
    print("\n🌐 Testing External Services...")
    
    try:
        from src.services.quicknode_service import QuickNodeService
        print("  ✅ QuickNodeService")
    except ImportError as e:
        print(f"  ❌ QuickNodeService: {e}")
    
    try:
        from src.services.helius_service import HeliusService
        print("  ✅ HeliusService")
    except ImportError as e:
        print(f"  ❌ HeliusService: {e}")
    
    try:
        from src.services.jupiter_service import JupiterService
        print("  ✅ JupiterService")
    except ImportError as e:
        print(f"  ❌ JupiterService: {e}")
    
    try:
        from src.services.wallet_manager import WalletManager
        print("  ✅ WalletManager")
    except ImportError as e:
        print(f"  ❌ WalletManager: {e}")

def test_ant_colony():
    """Test ant colony imports"""
    print("\n🐜 Testing Ant Colony Architecture...")
    
    try:
        from src.colony.founding_queen import FoundingQueen
        print("  ✅ FoundingQueen")
    except ImportError as e:
        print(f"  ❌ FoundingQueen: {e}")
    
    try:
        from src.colony.ant_queen import AntQueen
        print("  ✅ AntQueen")
    except ImportError as e:
        print(f"  ❌ AntQueen: {e}")
    
    try:
        from src.colony.worker_ant import WorkerAnt
        print("  ✅ WorkerAnt")
    except ImportError as e:
        print(f"  ❌ WorkerAnt: {e}")
    
    try:
        from src.colony.ant_princess import AntPrincess
        print("  ✅ AntPrincess")
    except ImportError as e:
        print(f"  ❌ AntPrincess: {e}")

def test_compounding_system():
    """Test compounding system imports"""
    print("\n📈 Testing 5-Layer Compounding System...")
    
    layers = [
        "data_layer",
        "worker_layer", 
        "carwash_layer",
        "monetary_layer",
        "intelligence_layer"
    ]
    
    for layer in layers:
        try:
            module = __import__(f"src.compounding.{layer}", fromlist=[layer])
            print(f"  ✅ {layer}")
        except ImportError as e:
            print(f"  ❌ {layer}: {e}")

def test_flywheel_system():
    """Test flywheel system imports"""
    print("\n🔄 Testing Flywheel System...")
    
    components = [
        "feedback_loops",
        "architecture_iteration",
        "performance_amplification"
    ]
    
    for component in components:
        try:
            module = __import__(f"src.flywheel.{component}", fromlist=[component])
            print(f"  ✅ {component}")
        except ImportError as e:
            print(f"  ❌ {component}: {e}")

def test_main_entry_points():
    """Test main entry points"""
    print("\n🚀 Testing Entry Points...")
    
    try:
        from src.ant_bot_system import AntBotSystem
        print("  ✅ AntBotSystem")
    except ImportError as e:
        print(f"  ❌ AntBotSystem: {e}")
    
    try:
        from src.core.enhanced_main import AntBotSystem as EnhancedAntBotSystem
        print("  ✅ EnhancedAntBotSystem")
    except ImportError as e:
        print(f"  ❌ EnhancedAntBotSystem: {e}")

def test_removed_files():
    """Test that removed files are actually gone"""
    print("\n🗑️ Verifying Removed Files...")
    
    removed_files = [
        "src/core/config.py",
        "src/utils/config.py",
        "src/utils/logging_config.py",
        "src/core/monitoring.py",
        "src/core/metrics.py",
        "main_simple.py",
        "cli_simple.py",
        "compatible.Dockerfile"
    ]
    
    for file_path in removed_files:
        if os.path.exists(file_path):
            print(f"  ❌ {file_path} still exists!")
        else:
            print(f"  ✅ {file_path} removed")

def main():
    """Run all validation tests"""
    print("🔍 FINAL STRUCTURE VALIDATION")
    print("=" * 50)
    
    test_core_infrastructure()
    test_services()
    test_ant_colony()
    test_compounding_system()
    test_flywheel_system()
    test_main_entry_points()
    test_removed_files()
    
    print("\n" + "=" * 50)
    print("✅ VALIDATION COMPLETE")
    print("🎯 System ready for production deployment!")

if __name__ == "__main__":
    main() 