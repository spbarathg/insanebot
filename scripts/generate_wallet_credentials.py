#!/usr/bin/env python3
"""
Comprehensive Wallet Management Tool for Enhanced Ant Bot
Generate credentials, validate wallets, check security, and manage configurations
"""

import os
import secrets
import string
import argparse
import json
import base58
from pathlib import Path
from datetime import datetime

def generate_secure_password(length=16):
    """Generate a secure random password"""
    characters = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(characters) for _ in range(length))

def generate_salt():
    """Generate a 32-byte hex salt for wallet encryption"""
    return secrets.token_hex(32)

def generate_private_key():
    """Generate a new Solana private key"""
    # Generate 32 random bytes for Solana private key
    private_bytes = secrets.token_bytes(32)
    return base58.b58encode(private_bytes).decode()

def validate_wallet_address(address):
    """Validate Solana wallet address format"""
    try:
        # Solana addresses are base58 encoded and 32 bytes (44 chars)
        decoded = base58.b58decode(address)
        return len(decoded) == 32 and len(address) <= 44
    except:
        return False

def check_env_security():
    """Check .env file security and configuration"""
    env_file = Path(".env")
    if not env_file.exists():
        return {"status": "missing", "issues": ["No .env file found"]}
    
    issues = []
    config = {}
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
            
        # Check for required fields
        required_fields = [
            'WALLET_PASSWORD', 'WALLET_SALT', 'HELIUS_API_KEY', 
            'QUICKNODE_ENDPOINT', 'JUPITER_API_KEY'
        ]
        
        for field in required_fields:
            if field not in content:
                issues.append(f"Missing {field}")
            elif f"{field}=your_" in content or f"{field}=placeholder" in content:
                issues.append(f"{field} not configured (placeholder value)")
            else:
                # Extract value
                lines = content.split('\n')
                for line in lines:
                    if line.startswith(f"{field}="):
                        config[field] = line.split('=', 1)[1].strip()
                        break
        
        # Check password strength
        if 'WALLET_PASSWORD' in config:
            password = config['WALLET_PASSWORD']
            if len(password) < 12:
                issues.append("WALLET_PASSWORD too short (minimum 12 characters)")
            if not any(c.isupper() for c in password):
                issues.append("WALLET_PASSWORD missing uppercase letters")
            if not any(c.islower() for c in password):
                issues.append("WALLET_PASSWORD missing lowercase letters")
            if not any(c.isdigit() for c in password):
                issues.append("WALLET_PASSWORD missing numbers")
        
        # Check salt format
        if 'WALLET_SALT' in config:
            salt = config['WALLET_SALT']
            if len(salt) != 64:  # 32 bytes hex = 64 chars
                issues.append("WALLET_SALT invalid format (should be 64 hex characters)")
        
        return {"status": "checked", "issues": issues, "config": config}
        
    except Exception as e:
        return {"status": "error", "issues": [f"Error reading .env: {str(e)}"]}

def backup_wallet_config():
    """Create a backup of wallet configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [".env", "config/config.json"]
    backed_up = []
    
    for file_path in files_to_backup:
        if Path(file_path).exists():
            backup_name = f"{Path(file_path).name}_{timestamp}.backup"
            backup_path = backup_dir / backup_name
            
            try:
                with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
                backed_up.append(str(backup_path))
            except Exception as e:
                print(f"❌ Failed to backup {file_path}: {e}")
    
    return backed_up

def show_wallet_info():
    """Show comprehensive wallet information"""
    print(f"\n💰 Wallet Information:")
    
    # Check .env configuration
    security_check = check_env_security()
    if security_check["status"] == "checked":
        config = security_check["config"]
        
        print(f"Configuration Status: {'✅ Complete' if not security_check['issues'] else '⚠️  Issues Found'}")
        
        if security_check["issues"]:
            print(f"\n⚠️  Security Issues:")
            for issue in security_check["issues"]:
                print(f"   • {issue}")
        
        # Show configured services
        print(f"\n🔗 Configured Services:")
        services = {
            'HELIUS_API_KEY': 'Helius RPC',
            'QUICKNODE_ENDPOINT': 'QuickNode',
            'JUPITER_API_KEY': 'Jupiter DEX'
        }
        
        for key, name in services.items():
            status = "✅" if key in config else "❌"
            print(f"   {status} {name}")
    
    elif security_check["status"] == "missing":
        print(f"Configuration Status: ❌ Missing .env file")
        for issue in security_check["issues"]:
            print(f"   • {issue}")
        print(f"\n💡 Tip: Run --setup to create a new wallet configuration")
    
    else:
        print(f"Configuration Status: ❌ {security_check['status']}")
        for issue in security_check["issues"]:
            print(f"   • {issue}")
        print(f"\n💡 Tip: Check .env file format and permissions")

def setup_new_wallet():
    """Interactive setup for a new wallet"""
    print(f"\n🆕 New Wallet Setup")
    print(f"This will generate a complete wallet configuration.")
    
    # Generate credentials
    password = generate_secure_password(16)
    salt = generate_salt()
    private_key = generate_private_key()
    
    print(f"\n🔐 Generated Credentials:")
    print(f"WALLET_PASSWORD={password}")
    print(f"WALLET_SALT={salt}")
    print(f"PRIVATE_KEY={private_key}")
    
    # Create .env if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        template_file = Path("env.template")
        if template_file.exists():
            print(f"\n📋 Creating .env from template...")
            with open(template_file, 'r') as f:
                template_content = f.read()
            
            # Replace placeholders
            content = template_content.replace('your_wallet_password_here', password)
            content = content.replace('your_wallet_salt_here', salt)
            content = content.replace('your_private_key_here', private_key)
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print(f"✅ .env file created with secure credentials")
        else:
            print(f"❌ No env.template found. Please create manually.")
    else:
        print(f"\n⚠️  .env file already exists. Use --update to modify.")

def update_credentials():
    """Update existing wallet credentials"""
    env_file = Path(".env")
    if not env_file.exists():
        print(f"❌ No .env file found. Use --setup for new wallet.")
        return
    
    # Backup first
    backed_up = backup_wallet_config()
    if backed_up:
        print(f"✅ Created backup: {backed_up[0]}")
    
    # Generate new credentials
    password = generate_secure_password(16)
    salt = generate_salt()
    
    # Update .env file
    with open(env_file, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    updated_lines = []
    
    for line in lines:
        if line.startswith('WALLET_PASSWORD='):
            updated_lines.append(f'WALLET_PASSWORD={password}')
        elif line.startswith('WALLET_SALT='):
            updated_lines.append(f'WALLET_SALT={salt}')
        else:
            updated_lines.append(line)
    
    with open(env_file, 'w') as f:
        f.write('\n'.join(updated_lines))
    
    print(f"\n🔄 Updated Credentials:")
    print(f"WALLET_PASSWORD={password}")
    print(f"WALLET_SALT={salt}")
    print(f"✅ Credentials updated successfully")

def validate_address_interactive():
    """Interactive address validation"""
    while True:
        address = input("\n🔍 Enter Solana address to validate (or 'exit'): ").strip()
        if address.lower() == 'exit':
            break
        
        if validate_wallet_address(address):
            print(f"✅ Valid Solana address: {address}")
        else:
            print(f"❌ Invalid Solana address format")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Wallet Management Tool")
    
    # Generation options
    parser.add_argument("--generate", action="store_true", help="Generate new credentials only")
    parser.add_argument("--length", type=int, default=16, help="Password length (default: 16)")
    parser.add_argument("--setup", action="store_true", help="Setup new wallet with full configuration")
    parser.add_argument("--update", action="store_true", help="Update existing credentials")
    
    # Information and validation
    parser.add_argument("--info", action="store_true", help="Show wallet information and security status")
    parser.add_argument("--validate", nargs='?', const='interactive', help="Validate wallet address")
    parser.add_argument("--check-security", action="store_true", help="Check security configuration")
    
    # Utility options
    parser.add_argument("--backup", action="store_true", help="Backup wallet configuration")
    parser.add_argument("--new-key", action="store_true", help="Generate new private key")
    
    # Output options
    parser.add_argument("--no-prompt", action="store_true", help="Don't prompt to save to .env file")
    parser.add_argument("--output-only", action="store_true", help="Only output credentials, no extra text")
    
    args = parser.parse_args()
    
    # Handle different operations
    if args.setup:
        setup_new_wallet()
    elif args.update:
        update_credentials()
    elif args.info:
        show_wallet_info()
    elif args.validate:
        if args.validate == 'interactive':
            validate_address_interactive()
        else:
            if validate_wallet_address(args.validate):
                print(f"✅ Valid Solana address: {args.validate}")
            else:
                print(f"❌ Invalid Solana address format")
    elif args.check_security:
        security_check = check_env_security()
        if security_check["issues"]:
            print(f"⚠️  Security Issues Found:")
            for issue in security_check["issues"]:
                print(f"   • {issue}")
        else:
            print(f"✅ Security configuration looks good")
    elif args.backup:
        backed_up = backup_wallet_config()
        if backed_up:
            print(f"✅ Backup created:")
            for backup in backed_up:
                print(f"   📁 {backup}")
        else:
            print(f"❌ No files to backup")
    elif args.new_key:
        private_key = generate_private_key()
        print(f"🔑 New Private Key: {private_key}")
        print(f"⚠️  Keep this secure and never share it!")
    elif args.generate or not any(vars(args).values()):
        # Default generation behavior
        if not args.output_only:
            print("🔐 Enhanced Ant Bot - Wallet Credentials Generator")
            print("=" * 50)
        
        # Generate credentials
        password = generate_secure_password(args.length)
        salt = generate_salt()
        
        if args.output_only:
            print(f"WALLET_PASSWORD={password}")
            print(f"WALLET_SALT={salt}")
            return
        
        print("\n✅ Generated Secure Credentials:")
        print(f"WALLET_PASSWORD={password}")
        print(f"WALLET_SALT={salt}")
        
        print("\n📝 Instructions:")
        print("1. Copy the above values to your .env file")
        print("2. Keep these credentials secure and backed up")
        print("3. Do NOT share these credentials with anyone")
        print("4. The salt is used for encryption - losing it means losing access to encrypted wallets")
        
        # Check if .env exists and offer to append
        if not args.no_prompt and os.path.exists('.env'):
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
        elif not args.no_prompt:
            print("\n💡 Tip: Create a .env file with these credentials based on env.template")

if __name__ == "__main__":
    main() 