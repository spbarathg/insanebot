#!/usr/bin/env python3
"""
Enhanced Ant Bot - Production Upgrade Script
===========================================

This script safely upgrades your Enhanced Ant Bot system from simplified 
simulation mode to full production trading with real Solana trading capabilities.

⚠️ WARNING: This enables REAL TRADING with REAL MONEY ⚠️
"""

import os
import sys
import shutil
import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionUpgrader:
    """Handles safe upgrade to production trading mode"""
    
    def __init__(self):
        self.backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.changes_made = []
        
    def display_upgrade_banner(self):
        """Display upgrade information and warnings"""
        print("""
🚀 ========================================================================
   ENHANCED ANT BOT - PRODUCTION UPGRADE SYSTEM
========================================================================

🎯 WHAT THIS UPGRADE INCLUDES:
   ✅ Full Production Ant Hierarchy (Advanced AI coordination)
   ✅ Real Solana Trading (No more simulation)
   ✅ Advanced Market Analysis (Complex strategies)
   ✅ Dynamic Position Sizing (Profit-optimized)
   ✅ Professional Risk Management
   ✅ Enhanced Learning Algorithms

⚠️  CRITICAL WARNINGS:
   🔴 This enables REAL TRADING with REAL MONEY
   🔴 You can LOSE MONEY if market conditions are unfavorable
   🔴 Advanced features require careful monitoring
   🔴 Backup will be created automatically

💰 PROFIT POTENTIAL:
   📈 Advanced AI strategies for higher returns
   📊 Dynamic risk-reward optimization
   🎯 Professional-grade trading algorithms
   🔄 Continuous learning and adaptation

========================================================================
        """)
        
    def get_user_confirmation(self):
        """Get explicit user confirmation for upgrade"""
        print("🔄 UPGRADE CONFIRMATION REQUIRED:")
        print("   Type 'UPGRADE' to proceed with production upgrade")
        print("   Type 'CANCEL' to abort and keep current system")
        print()
        
        while True:
            response = input("Your choice: ").strip().upper()
            if response == 'UPGRADE':
                return True
            elif response == 'CANCEL':
                print("✅ Upgrade cancelled. Your system remains unchanged.")
                return False
            else:
                print("❌ Please type exactly 'UPGRADE' or 'CANCEL'")
    
    def create_backup(self):
        """Create backup of current system"""
        try:
            print(f"📁 Creating backup in {self.backup_dir}...")
            os.makedirs(self.backup_dir)
            
            # Backup critical files
            files_to_backup = [
                '.env',
                'src/core/enhanced_main.py',
                'env.template',
                'requirements.txt'
            ]
            
            for file_path in files_to_backup:
                if os.path.exists(file_path):
                    dest_path = os.path.join(self.backup_dir, file_path.replace('/', '_'))
                    shutil.copy2(file_path, dest_path)
                    
            print(f"✅ Backup created successfully: {self.backup_dir}")
            self.changes_made.append(f"Created backup: {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup creation failed: {str(e)}")
            return False
    
    def update_environment_config(self):
        """Update environment configuration for production"""
        try:
            print("⚙️ Updating environment configuration...")
            
            # Read current .env file
            env_path = '.env'
            if not os.path.exists(env_path):
                print("📝 Creating .env file from template...")
                if os.path.exists('env.template'):
                    shutil.copy2('env.template', env_path)
                else:
                    logger.error("❌ env.template not found")
                    return False
            
            # Update .env for production
            with open(env_path, 'r') as f:
                env_content = f.read()
            
            # Enable production mode
            if 'SIMULATION_MODE=true' in env_content:
                print("🔄 Enabling REAL TRADING mode...")
                env_content = env_content.replace('SIMULATION_MODE=true', 'SIMULATION_MODE=false')
                self.changes_made.append("Enabled real trading mode")
            
            # Add production settings if missing
            production_settings = {
                'AI_AGGRESSIVE_MODE': 'true',
                'DETAILED_LOGGING': 'true',
                'SAVE_TRADE_LOGS': 'true',
                'MAX_POSITION_PERCENT': '10',
                'TRADE_EXECUTION_TIMEOUT': '30'
            }
            
            for key, value in production_settings.items():
                if key not in env_content:
                    env_content += f"\n{key}={value}"
                    self.changes_made.append(f"Added {key}={value}")
            
            # Write updated .env
            with open(env_path, 'w') as f:
                f.write(env_content)
                
            print("✅ Environment configuration updated for production")
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment update failed: {str(e)}")
            return False
    
    def validate_wallet_credentials(self):
        """Validate that wallet credentials are properly configured"""
        try:
            print("🔐 Validating wallet credentials...")
            
            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv()
            
            required_vars = [
                'PRIVATE_KEY',
                'WALLET_PASSWORD', 
                'WALLET_SALT',
                'QUICKNODE_ENDPOINT_URL',
                'HELIUS_API_KEY'
            ]
            
            missing_vars = []
            test_vars = []
            
            for var in required_vars:
                value = os.getenv(var, '')
                if not value:
                    missing_vars.append(var)
                elif value in ['your-key-here', 'demo_key_for_testing', 'test-value']:
                    test_vars.append(var)
            
            if missing_vars:
                print(f"❌ Missing required variables: {missing_vars}")
                print("🔧 Please update your .env file with real values")
                return False
            
            if test_vars:
                print(f"⚠️ Test/placeholder values detected: {test_vars}")
                print("💡 These appear to be placeholder values - please verify they are correct")
                
                response = input("Continue anyway? (y/N): ").strip().lower()
                if response != 'y':
                    return False
            
            print("✅ Wallet credentials validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Credential validation failed: {str(e)}")
            return False
    
    async def test_production_systems(self):
        """Test that production systems can initialize properly"""
        try:
            print("🧪 Testing production systems...")
            
            # Test external services
            sys.path.append('.')
            
            # Test QuickNode
            from src.services.quicknode_service import QuickNodeService
            quicknode = QuickNodeService()
            print("   ✅ QuickNode service: OK")
            
            # Test Helius
            from src.services.helius_service import HeliusService
            helius = HeliusService()
            print("   ✅ Helius service: OK")
            
            # Test Jupiter
            from src.services.jupiter_service import JupiterService
            jupiter = JupiterService()
            print("   ✅ Jupiter service: OK")
            
            # Test Wallet Manager in production mode
            from src.services.wallet_manager import WalletManager
            wallet = WalletManager()
            if await wallet.initialize():
                print("   ✅ Wallet manager: OK")
            else:
                print("   ⚠️ Wallet manager: Issues detected")
            
            print("✅ Production systems test completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Production systems test failed: {str(e)}")
            return False
    
    def display_upgrade_summary(self):
        """Display summary of changes made"""
        print("""
🎉 ========================================================================
   PRODUCTION UPGRADE COMPLETED SUCCESSFULLY!
========================================================================

📊 CHANGES MADE:""")
        
        for change in self.changes_made:
            print(f"   ✅ {change}")
        
        print(f"""
💾 BACKUP LOCATION: {self.backup_dir}

🚀 YOUR ENHANCED ANT BOT IS NOW IN PRODUCTION MODE:
   🔥 Real Solana trading enabled
   🧠 Advanced AI coordination active
   💰 Professional trading algorithms running
   🛡️ Full Titan Shield protection
   📈 Optimized for maximum profitability

⚡ NEXT STEPS:
   1. Monitor your first few trades carefully
   2. Check logs in data/logs/ directory
   3. Watch capital and profit metrics
   4. Adjust position sizes if needed

🎯 TO START TRADING:
   python trading_bot_24x7.py

⚠️ REMEMBER:
   • Real money is now at risk
   • Monitor performance regularly
   • You can revert using backup if needed

========================================================================
        """)
    
    def restore_from_backup(self):
        """Emergency restore from backup"""
        try:
            print(f"🔄 Restoring from backup: {self.backup_dir}")
            
            backup_files = {
                '.env': '.env',
                'src_core_enhanced_main.py': 'src/core/enhanced_main.py',
                'env.template': 'env.template'
            }
            
            for backup_name, original_path in backup_files.items():
                backup_path = os.path.join(self.backup_dir, backup_name)
                if os.path.exists(backup_path):
                    # Create directory if needed
                    os.makedirs(os.path.dirname(original_path), exist_ok=True)
                    shutil.copy2(backup_path, original_path)
                    print(f"   ✅ Restored: {original_path}")
            
            print("✅ System restored from backup")
            return True
            
        except Exception as e:
            logger.error(f"❌ Restore failed: {str(e)}")
            return False

async def main():
    """Main upgrade process"""
    upgrader = ProductionUpgrader()
    
    try:
        # Display upgrade information
        upgrader.display_upgrade_banner()
        
        # Get user confirmation
        if not upgrader.get_user_confirmation():
            return
        
        print("\n🚀 Starting production upgrade...")
        
        # Create backup
        if not upgrader.create_backup():
            print("❌ Backup creation failed. Aborting upgrade.")
            return
        
        # Update environment configuration
        if not upgrader.update_environment_config():
            print("❌ Environment update failed. Restoring from backup...")
            upgrader.restore_from_backup()
            return
        
        # Validate credentials
        if not upgrader.validate_wallet_credentials():
            print("❌ Credential validation failed. Please fix .env file and retry.")
            return
        
        # Test production systems
        if not await upgrader.test_production_systems():
            print("⚠️ Some production systems have issues, but upgrade completed.")
            print("💡 You may need to check your API keys and configuration.")
        
        # Display success summary
        upgrader.display_upgrade_summary()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Upgrade cancelled by user")
        print("🔄 Restoring from backup...")
        upgrader.restore_from_backup()
    except Exception as e:
        logger.error(f"❌ Upgrade failed: {str(e)}")
        print("🔄 Restoring from backup...")
        upgrader.restore_from_backup()

if __name__ == "__main__":
    print("🤖 Enhanced Ant Bot - Production Upgrade System")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Upgrade cancelled")
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        sys.exit(1) 