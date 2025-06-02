#!/usr/bin/env python3
"""
Setup Script for Trading Bot Comprehensive Monitoring

This script ensures all dependencies are installed and the system is ready.

Usage:
    python setup_monitoring.py
"""

import subprocess
import sys
from pathlib import Path
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_dependencies():
    """Check and install required packages"""
    required_packages = [
        "rich>=13.0.0",
        "psutil>=5.9.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ]
    
    print("🔧 Checking dependencies...")
    
    missing_packages = []
    
    for package in required_packages:
        package_name = package.split(">=")[0]
        try:
            __import__(package_name)
            print(f"✅ {package_name} - OK")
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            print(f"   Installing {package}...")
            if install_package(package):
                print(f"   ✅ {package} installed successfully")
            else:
                print(f"   ❌ Failed to install {package}")
                return False
    else:
        print("\n✅ All dependencies are already installed!")
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "logs/comprehensive",
        "data",
        "monitoring"
    ]
    
    print("\n📁 Creating directories...")
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {directory}")
        else:
            print(f"   ✅ Exists: {directory}")

def test_comprehensive_logging():
    """Test that comprehensive logging works"""
    print("\n🧪 Testing comprehensive logging...")
    
    try:
        # Test imports
        from monitoring.comprehensive_bot_logger import get_bot_logger, log_all_methods
        from monitoring.bot_analyzer_ai import BotAnalyzerAI
        
        print("   ✅ All imports successful")
        
        # Test logger creation
        logger = get_bot_logger("test_bot")
        print("   ✅ Logger creation successful")
        
        # Test analyzer creation
        analyzer = BotAnalyzerAI()
        print("   ✅ AI analyzer creation successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🤖 Trading Bot Comprehensive Monitoring Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor} - OK")
    
    # Install dependencies
    if not check_and_install_dependencies():
        print("\n❌ Failed to install some dependencies")
        return False
    
    # Create directories
    create_directories()
    
    # Test the system
    if not test_comprehensive_logging():
        print("\n❌ System test failed")
        return False
    
    print("\n🎉 Setup Complete!")
    print("=" * 30)
    print()
    print("✅ All dependencies installed")
    print("✅ Directories created")
    print("✅ Comprehensive logging system ready")
    print()
    print("🚀 Ready to start your bot:")
    print("   python start_bot.py")
    print()
    print("📊 Or run analysis on existing logs:")
    print("   python monitoring/bot_analyzer_ai.py --full-analysis")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 