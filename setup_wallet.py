#!/usr/bin/env python3
"""
Secure wallet setup script for Solana trading bot.
"""
import os
import base58
from solders.keypair import Keypair
from solders.pubkey import Pubkey
import json

def generate_new_wallet():
    """Generate a new Solana wallet for trading."""
    print("🔐 Generating new Solana wallet...")
    
    # Generate new keypair
    keypair = Keypair()
    
    # Get public key (wallet address)
    public_key = str(keypair.pubkey())
    
    # Get private key (keep this SECRET!)
    private_key = base58.b58encode(keypair.secret()).decode('utf-8')
    
    print(f"✅ New wallet generated!")
    print(f"📍 Public Address: {public_key}")
    print(f"🔑 Private Key: {private_key}")
    print("\n⚠️  SECURITY WARNINGS:")
    print("1. NEVER share your private key with anyone")
    print("2. Store it in a secure password manager")
    print("3. This wallet will be used for trading - fund appropriately")
    print("4. Consider using a hardware wallet for large amounts")
    
    return {
        "public_key": public_key,
        "private_key": private_key
    }

def save_wallet_config(wallet_info):
    """Save wallet configuration securely."""
    print("\n📝 Creating wallet configuration...")
    
    # Create secure .env file
    env_content = f"""# Solana Trading Bot - Production Configuration
# ⚠️  KEEP THIS FILE SECURE - CONTAINS PRIVATE KEYS

# Wallet Configuration
SOLANA_PRIVATE_KEY={wallet_info['private_key']}
WALLET_PRIVATE_KEY={wallet_info['private_key']}
WALLET_PUBLIC_KEY={wallet_info['public_key']}

# API Keys (Replace with your real keys)
JUPITER_API_KEY=your_real_jupiter_api_key_here
JUPITER_API_URL=https://quote-api.jup.ag/v6
HELIUS_API_KEY=your_real_helius_api_key_here

# RPC Configuration
RPC_URL=https://api.mainnet-beta.solana.com
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com

# Trading Configuration
SIMULATION_MODE=false
LOG_LEVEL=INFO
MAX_POSITION_SIZE=0.1
STOP_LOSS_PERCENTAGE=10
TAKE_PROFIT_PERCENTAGE=50
MIN_LIQUIDITY_USD=10000

# Risk Management
MAX_DAILY_LOSS=0.05
MAX_TRADES_PER_HOUR=10
MIN_TRADE_SIZE=0.001
MAX_TRADE_SIZE=0.1

# Security
PYTHONUNBUFFERED=1
"""
    
    with open('.env.production', 'w') as f:
        f.write(env_content)
    
    print("✅ Configuration saved to .env.production")
    print("📋 Next steps:")
    print("1. Replace API keys with your real ones")
    print("2. Fund your wallet address with SOL")
    print("3. Rename .env.production to .env when ready")
    print("4. Set SIMULATION_MODE=false for live trading")

def main():
    """Main wallet setup function."""
    print("🏦 Solana Trading Bot - Wallet Setup")
    print("=" * 50)
    
    choice = input("Choose option:\n1. Generate new wallet\n2. Use existing wallet\nEnter (1/2): ")
    
    if choice == "1":
        wallet_info = generate_new_wallet()
        save_wallet_config(wallet_info)
    elif choice == "2":
        print("📝 Enter your existing wallet details:")
        private_key = input("Private Key (base58): ")
        try:
            # Validate private key
            keypair = Keypair.from_base58_string(private_key)
            public_key = str(keypair.pubkey())
            
            wallet_info = {
                "public_key": public_key,
                "private_key": private_key
            }
            save_wallet_config(wallet_info)
        except Exception as e:
            print(f"❌ Invalid private key: {e}")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 