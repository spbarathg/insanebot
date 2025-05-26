#!/usr/bin/env python3
"""
Generate wallet credentials for Enhanced Ant Bot
This script helps generate secure WALLET_PASSWORD and WALLET_SALT values.
"""

import os
import secrets
import string

def generate_secure_password(length=16):
    """Generate a secure random password"""
    characters = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(characters) for _ in range(length))

def generate_salt():
    """Generate a 32-byte hex salt for wallet encryption"""
    return secrets.token_hex(32)

def main():
    print("🔐 Enhanced Ant Bot - Wallet Credentials Generator")
    print("=" * 50)
    
    # Generate credentials
    password = generate_secure_password()
    salt = generate_salt()
    
    print("\n✅ Generated Secure Credentials:")
    print(f"WALLET_PASSWORD={password}")
    print(f"WALLET_SALT={salt}")
    
    print("\n📝 Instructions:")
    print("1. Copy the above values to your .env file")
    print("2. Keep these credentials secure and backed up")
    print("3. Do NOT share these credentials with anyone")
    print("4. The salt is used for encryption - losing it means losing access to encrypted wallets")
    
    # Check if .env exists and offer to append
    if os.path.exists('.env'):
        response = input("\n❓ Do you want to append these to your .env file? (y/n): ")
        if response.lower() in ['y', 'yes']:
            try:
                with open('.env', 'a') as f:
                    f.write(f"\n# Wallet encryption credentials (generated {os.path.basename(__file__)})\n")
                    f.write(f"WALLET_PASSWORD={password}\n")
                    f.write(f"WALLET_SALT={salt}\n")
                print("✅ Credentials appended to .env file")
            except Exception as e:
                print(f"❌ Error writing to .env file: {e}")
    else:
        print("\n💡 Tip: Create a .env file with these credentials based on env.template")

if __name__ == "__main__":
    main() 