#!/usr/bin/env python3
"""
Production Setup Script for Solana Trading Bot
Helps configure real wallets, API keys, and production settings.
"""

import os
import sys
import getpass
from pathlib import Path
from solders.keypair import Keypair
from solders.pubkey import Pubkey
import secrets
import base58

def main():
    print("🚀 Solana Trading Bot - Production Setup")
    print("=" * 50)
    
    # Check if .env exists
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  .env file already exists!")
        overwrite = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("❌ Setup cancelled")
            return
    
    print("\n📋 Phase 1: Wallet Configuration")
    print("-" * 30)
    
    # Option 1: Generate new wallet
    print("Choose wallet setup option:")
    print("1. Generate new wallet (recommended for testing)")
    print("2. Import existing wallet from private key")
    print("3. Import existing wallet from seed phrase")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        # Generate new wallet
        keypair = Keypair()
        private_key = base58.b58encode(bytes(keypair)).decode('utf-8')
        public_key = str(keypair.pubkey())
        
        print(f"\n✅ New wallet generated:")
        print(f"📍 Public Address: {public_key}")
        print(f"🔐 Private Key: {private_key}")
        print("\n⚠️  IMPORTANT: Save these credentials securely!")
        print("   - Fund this address with SOL before starting the bot")
        print("   - Never share your private key with anyone")
        
    elif choice == "2":
        # Import from private key
        private_key = getpass.getpass("🔐 Enter your private key (base58 format): ").strip()
        try:
            keypair = Keypair.from_base58_string(private_key)
            public_key = str(keypair.pubkey())
            print(f"✅ Wallet imported successfully: {public_key}")
        except Exception as e:
            print(f"❌ Invalid private key: {str(e)}")
            return
            
    elif choice == "3":
        print("❌ Seed phrase import not implemented yet")
        print("   Please use option 1 or 2")
        return
    else:
        print("❌ Invalid choice")
        return
    
    print("\n📋 Phase 2: API Keys Configuration")
    print("-" * 30)
    
    # Helius API Key
    print("🔗 Helius API Key (get from https://helius.xyz/)")
    helius_key = input("Enter Helius API key (or press Enter for demo): ").strip()
    if not helius_key:
        helius_key = "demo_helius_key_for_testing"
        print("⚠️  Using demo key - get real key for production!")
    
    # Jupiter API Key
    print("\n🔗 Jupiter API Key (optional, get from https://docs.jup.ag/)")
    jupiter_key = input("Enter Jupiter API key (or press Enter for demo): ").strip()
    if not jupiter_key:
        jupiter_key = "demo_jupiter_key_for_testing"
        print("⚠️  Using demo key - get real key for better rate limits!")
    
    print("\n📋 Phase 3: Trading Configuration")
    print("-" * 30)
    
    # Simulation vs Production
    print("🎯 Trading Mode:")
    print("1. Simulation Mode (safe for testing)")
    print("2. Production Mode (real trading with real funds)")
    
    mode_choice = input("Enter choice (1-2): ").strip()
    simulation_mode = "true" if mode_choice == "1" else "false"
    
    if mode_choice == "2":
        print("\n⚠️  WARNING: Production mode will use REAL FUNDS!")
        confirmation = input("Type 'I UNDERSTAND' to confirm: ").strip()
        if confirmation != "I UNDERSTAND":
            print("❌ Production mode not confirmed - using simulation mode")
            simulation_mode = "true"
    
    # Risk parameters
    print(f"\n⚙️  Risk Management (for {'simulation' if simulation_mode == 'true' else 'PRODUCTION'}):")
    
    if simulation_mode == "true":
        max_position = "0.01"  # 0.01 SOL for simulation
        daily_loss = "0.05"    # 0.05 SOL daily loss limit
        capital = "0.1"        # 0.1 SOL simulation capital
    else:
        max_position = input("Max position size (SOL) [0.01]: ").strip() or "0.01"
        daily_loss = input("Daily loss limit (SOL) [0.05]: ").strip() or "0.05"
        capital = input("Starting capital (SOL) [0.1]: ").strip() or "0.1"
    
    print("\n📋 Phase 4: Security Configuration")
    print("-" * 30)
    
    # Generate secure encryption salt
    encryption_salt = secrets.token_hex(32)
    
    # Wallet password for encryption
    wallet_password = getpass.getpass("🔐 Set wallet encryption password: ").strip()
    if not wallet_password:
        wallet_password = "default_password_change_me"
        print("⚠️  Using default password - change this in production!")
    
    print("\n📝 Creating .env file...")
    
    # Create .env content
    env_content = f"""# Solana Trading Bot - Production Configuration
# Generated on {os.popen('date').read().strip()}
# ⚠️  KEEP THIS FILE SECURE - CONTAINS PRIVATE KEYS

# ============================================================================
# OPERATION MODE
# ============================================================================
SIMULATION_MODE={simulation_mode}

# ============================================================================
# SOLANA WALLET CONFIGURATION  
# ============================================================================
SOLANA_PRIVATE_KEY={private_key}
WALLET_PRIVATE_KEY={private_key}
WALLET_PUBLIC_KEY={public_key}
WALLET_PASSWORD={wallet_password}
WALLET_SALT={encryption_salt}

# Solana RPC Configuration
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com

# ============================================================================
# API CREDENTIALS
# ============================================================================
HELIUS_API_KEY={helius_key}
JUPITER_API_KEY={jupiter_key}
JUPITER_API_URL=https://quote-api.jup.ag/v6

# ============================================================================
# TRADING CONFIGURATION
# ============================================================================
SIMULATION_CAPITAL={capital}
MAX_POSITION_SIZE={max_position}
DAILY_LOSS_LIMIT={daily_loss}
MIN_TRADE_INTERVAL=300
MAX_TRADES_PER_DAY=50
DEFAULT_SLIPPAGE=1.5
MAX_SLIPPAGE=5.0

# ============================================================================
# RISK MANAGEMENT
# ============================================================================
MIN_LIQUIDITY=10000
MAX_PRICE_IMPACT=0.05
TARGET_PROFIT=20.0
STOP_LOSS=10.0
MIN_CONFIDENCE=0.7

# ============================================================================
# SYSTEM SETTINGS
# ============================================================================
LOG_LEVEL=INFO
LOOP_INTERVAL=30
TRADE_COOLDOWN=300
PYTHONUNBUFFERED=1

# ============================================================================
# MONITORING
# ============================================================================
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
"""

    # Write .env file
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ .env file created successfully!")
    
    print("\n📋 Phase 5: Final Steps")
    print("-" * 30)
    
    if simulation_mode == "false":
        print("🎯 PRODUCTION MODE SETUP:")
        print(f"1. 💰 Fund your wallet: {public_key}")
        print("2. 🔄 Send a small test transaction first")
        print("3. 📊 Monitor the bot closely when it starts")
        print("4. 🛑 Set up emergency stop procedures")
    else:
        print("🧪 SIMULATION MODE SETUP:")
        print("1. ✅ Configuration complete - ready for testing")
        print("2. 🚀 Start the bot: docker-compose up -d")
        print("3. 📊 Monitor logs: docker-compose logs -f trading-bot")
        print("4. 🔄 Switch to production mode later when ready")
    
    print("\n🚀 Setup Complete!")
    print("Run the bot with: docker-compose up -d")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
    except Exception as e:
        print(f"\n💥 Setup error: {str(e)}")
        sys.exit(1) 