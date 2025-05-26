#!/usr/bin/env python3
"""
Fix Enhanced Ant Bot startup issues
This script addresses common startup problems including missing RISK_LIMITS and wallet credentials.
"""

import os
import sys
import json
import secrets
import string
from pathlib import Path

def generate_secure_password(length=16):
    """Generate a secure random password"""
    characters = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(characters) for _ in range(length))

def generate_salt():
    """Generate a 32-byte hex salt for wallet encryption"""
    return secrets.token_hex(32)

def check_config_file():
    """Check and update config.json with required risk settings"""
    config_file = Path("config.json")
    
    if not config_file.exists():
        print("❌ config.json not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check if risk_management section exists and has required fields
        if 'risk_management' not in config:
            config['risk_management'] = {}
        
        # Add missing risk management fields
        risk_defaults = {
            "daily_loss_limit": 0.2,
            "portfolio_risk_limit": 0.3,
            "max_token_exposure": 0.2,
            "max_portfolio_exposure": 0.8,
            "max_exposure": 0.7,
            "max_drawdown": 0.1,
            "max_token_risk": {
                "low": 0.1,
                "medium": 0.05,
                "high": 0.02,
                "extreme": 0.01
            }
        }
        
        updated = False
        for key, default_value in risk_defaults.items():
            if key not in config['risk_management']:
                config['risk_management'][key] = default_value
                updated = True
                print(f"✅ Added missing risk setting: {key}")
        
        # Add position limits if missing
        if 'position_limits' not in config:
            config['position_limits'] = {
                "max_position_size": 0.1,
                "min_position_size": 0.001,
                "max_positions": 10,
                "position_concentration": 0.3
            }
            updated = True
            print("✅ Added position_limits section")
        
        if updated:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            print("✅ Updated config.json with missing risk settings")
        else:
            print("✅ config.json already has all required risk settings")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating config.json: {e}")
        return False

def check_env_file():
    """Check and setup .env file with wallet credentials"""
    env_file = Path(".env")
    
    if not env_file.exists():
        # Copy from template if it exists
        template_file = Path("env.template")
        if template_file.exists():
            print("📋 Creating .env from template...")
            try:
                with open(template_file, 'r') as f:
                    template_content = f.read()
                with open(env_file, 'w') as f:
                    f.write(template_content)
                print("✅ Created .env from env.template")
            except Exception as e:
                print(f"❌ Error creating .env from template: {e}")
                return False
        else:
            print("❌ No .env file and no env.template found")
            return False
    
    # Check if wallet credentials exist
    try:
        with open(env_file, 'r') as f:
            env_content = f.read()
        
        has_password = 'WALLET_PASSWORD=' in env_content and not env_content.split('WALLET_PASSWORD=')[1].split('\n')[0].strip().endswith('_here')
        has_salt = 'WALLET_SALT=' in env_content and not env_content.split('WALLET_SALT=')[1].split('\n')[0].strip().endswith('_here')
        
        if not has_password or not has_salt:
            print("🔐 Generating missing wallet credentials...")
            password = generate_secure_password()
            salt = generate_salt()
            
            # Append or update credentials
            lines = env_content.split('\n')
            updated_lines = []
            password_updated = False
            salt_updated = False
            
            for line in lines:
                if line.startswith('WALLET_PASSWORD=') and not password_updated:
                    updated_lines.append(f'WALLET_PASSWORD={password}')
                    password_updated = True
                elif line.startswith('WALLET_SALT=') and not salt_updated:
                    updated_lines.append(f'WALLET_SALT={salt}')
                    salt_updated = True
                else:
                    updated_lines.append(line)
            
            # Add missing credentials if not found in file
            if not password_updated:
                updated_lines.extend(['', '# Wallet encryption credentials', f'WALLET_PASSWORD={password}'])
            if not salt_updated:
                updated_lines.append(f'WALLET_SALT={salt}')
            
            with open(env_file, 'w') as f:
                f.write('\n'.join(updated_lines))
            
            print("✅ Generated and added wallet credentials to .env")
            print(f"   WALLET_PASSWORD={password}")
            print(f"   WALLET_SALT={salt}")
        else:
            print("✅ Wallet credentials already configured in .env")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking/updating .env file: {e}")
        return False

def main():
    print("🔧 Enhanced Ant Bot - Startup Fix Tool")
    print("=" * 50)
    
    print("\n1. Checking configuration files...")
    config_ok = check_config_file()
    
    print("\n2. Checking environment variables...")
    env_ok = check_env_file()
    
    print("\n" + "=" * 50)
    
    if config_ok and env_ok:
        print("✅ All startup issues fixed!")
        print("\n🚀 Your bot should now start successfully.")
        print("\n💡 Next steps:")
        print("   1. Update your .env file with real API keys")
        print("   2. Add your wallet private key")
        print("   3. Run: docker-compose up enhanced-ant-bot")
    else:
        print("❌ Some issues remain. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 