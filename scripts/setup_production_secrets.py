#!/usr/bin/env python3
"""
Production Secrets Setup Script

This script generates secure secrets for production deployment:
- Master encryption keys
- API authentication tokens
- Database passwords
- Monitoring passwords
- SSL certificates

Usage: python scripts/setup_production_secrets.py [--output .env.secrets]
"""

import os
import sys
import secrets
import string
import base64
import hashlib
import json
from pathlib import Path
from typing import Dict, Any
import argparse

class ProductionSecretsGenerator:
    """Generate and manage production secrets securely"""
    
    def __init__(self, output_file: str = ".env.secrets"):
        self.output_file = output_file
        self.secrets_data = {}
        
    def generate_secure_password(self, length: int = 32, include_symbols: bool = True) -> str:
        """Generate a cryptographically secure password"""
        alphabet = string.ascii_letters + string.digits
        if include_symbols:
            alphabet += "!@#$%^&*"
        
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        return password
    
    def generate_encryption_key(self, length: int = 32) -> str:
        """Generate a secure encryption key"""
        key_bytes = secrets.token_bytes(length)
        return base64.b64encode(key_bytes).decode('utf-8')
    
    def generate_api_token(self, length: int = 32) -> str:
        """Generate a secure API token"""
        return secrets.token_urlsafe(length)
    
    def generate_database_credentials(self) -> Dict[str, str]:
        """Generate secure database credentials"""
        return {
            'username': f"trading_user_{secrets.token_hex(4)}",
            'password': self.generate_secure_password(24),
            'admin_password': self.generate_secure_password(24)
        }
    
    def generate_jwt_secret(self) -> str:
        """Generate JWT signing secret"""
        return base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
    
    def generate_all_secrets(self) -> Dict[str, Any]:
        """Generate all required production secrets"""
        
        # Encryption keys
        master_encryption_key = self.generate_encryption_key(32)
        wallet_encryption_password = self.generate_secure_password(24)
        
        # API tokens
        api_auth_token = self.generate_api_token(32)
        monitoring_auth_token = self.generate_api_token(32)
        prometheus_auth_token = self.generate_api_token(24)
        
        # Database credentials
        db_creds = self.generate_database_credentials()
        
        # Monitoring passwords
        grafana_admin_password = self.generate_secure_password(16, False)
        
        # JWT secrets
        jwt_secret = self.generate_jwt_secret()
        
        # Generate secure session key
        session_key = self.generate_encryption_key(32)
        
        secrets_dict = {
            # Encryption
            'MASTER_ENCRYPTION_KEY': master_encryption_key,
            'WALLET_ENCRYPTION_PASSWORD': wallet_encryption_password,
            'SESSION_SECRET_KEY': session_key,
            'JWT_SECRET': jwt_secret,
            
            # API Authentication
            'API_AUTH_TOKEN': api_auth_token,
            'MONITORING_AUTH_TOKEN': monitoring_auth_token,
            'PROMETHEUS_AUTH_TOKEN': prometheus_auth_token,
            
            # Database
            'DB_USERNAME': db_creds['username'],
            'DB_PASSWORD': db_creds['password'],
            'DB_ADMIN_PASSWORD': db_creds['admin_password'],
            
            # Monitoring
            'GRAFANA_ADMIN_PASSWORD': grafana_admin_password,
            
            # Additional security
            'CSRF_SECRET': self.generate_api_token(24),
            'COOKIE_SECRET': self.generate_encryption_key(16),
            
            # Generate deployment timestamp and build number
            'DEPLOYMENT_TIMESTAMP': str(int(time.time())),
            'BUILD_NUMBER': f"build_{secrets.token_hex(4)}",
            
            # Application version
            'APP_VERSION': '2.0.0-production',
        }
        
        self.secrets_data = secrets_dict
        return secrets_dict
    
    def save_secrets_file(self, secrets_dict: Dict[str, Any]) -> None:
        """Save secrets to environment file"""
        
        header = """# ===============================================================================
# 🔐 PRODUCTION SECRETS - GENERATED AUTOMATICALLY
# ===============================================================================
# WARNING: This file contains sensitive production secrets
# - Never commit this file to version control
# - Restrict file permissions: chmod 600
# - Use in production environment only
# - Rotate secrets regularly
# ===============================================================================

"""
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(header)
            
            # Group secrets by category
            categories = {
                'Encryption Keys': [
                    'MASTER_ENCRYPTION_KEY', 'WALLET_ENCRYPTION_PASSWORD', 
                    'SESSION_SECRET_KEY', 'JWT_SECRET'
                ],
                'API Authentication': [
                    'API_AUTH_TOKEN', 'MONITORING_AUTH_TOKEN', 'PROMETHEUS_AUTH_TOKEN'
                ],
                'Database Credentials': [
                    'DB_USERNAME', 'DB_PASSWORD', 'DB_ADMIN_PASSWORD'
                ],
                'Monitoring': [
                    'GRAFANA_ADMIN_PASSWORD'
                ],
                'Additional Security': [
                    'CSRF_SECRET', 'COOKIE_SECRET'
                ],
                'Deployment Info': [
                    'DEPLOYMENT_TIMESTAMP', 'BUILD_NUMBER', 'APP_VERSION'
                ]
            }
            
            for category, keys in categories.items():
                f.write(f"\n# {category}\n")
                for key in keys:
                    if key in secrets_dict:
                        f.write(f"{key}={secrets_dict[key]}\n")
        
        # Set secure file permissions
        os.chmod(self.output_file, 0o600)
        print(f"✅ Secrets saved to {self.output_file} with secure permissions (600)")
    
    def generate_docker_secrets(self) -> None:
        """Generate Docker secrets for compose deployment"""
        docker_secrets_dir = Path("secrets")
        docker_secrets_dir.mkdir(exist_ok=True)
        
        # Key secrets for Docker
        docker_secrets = {
            'master_key': self.secrets_data.get('MASTER_ENCRYPTION_KEY', ''),
            'api_token': self.secrets_data.get('API_AUTH_TOKEN', ''),
            'db_password': self.secrets_data.get('DB_PASSWORD', ''),
            'grafana_password': self.secrets_data.get('GRAFANA_ADMIN_PASSWORD', '')
        }
        
        for name, value in docker_secrets.items():
            secret_file = docker_secrets_dir / name
            with open(secret_file, 'w', encoding='utf-8') as f:
                f.write(value)
            os.chmod(secret_file, 0o600)
        
        print(f"✅ Docker secrets generated in {docker_secrets_dir}/")
    
    def validate_secrets_strength(self) -> bool:
        """Validate that generated secrets meet security requirements"""
        validations = []
        
        # Check master encryption key length
        master_key = self.secrets_data.get('MASTER_ENCRYPTION_KEY', '')
        if len(master_key) >= 44:  # Base64 encoded 32 bytes
            validations.append("✅ Master encryption key: Strong")
        else:
            validations.append("❌ Master encryption key: Too weak")
        
        # Check password complexity
        password = self.secrets_data.get('WALLET_ENCRYPTION_PASSWORD', '')
        if len(password) >= 24 and any(c.isupper() for c in password) and any(c.islower() for c in password) and any(c.isdigit() for c in password):
            validations.append("✅ Wallet password: Strong")
        else:
            validations.append("❌ Wallet password: Too weak")
        
        # Check API tokens
        api_token = self.secrets_data.get('API_AUTH_TOKEN', '')
        if len(api_token) >= 32:
            validations.append("✅ API token: Strong")
        else:
            validations.append("❌ API token: Too weak")
        
        print("\n🔐 Security Validation:")
        for validation in validations:
            print(f"   {validation}")
        
        return all("✅" in v for v in validations)
    
    def generate_setup_instructions(self) -> str:
        """Generate setup instructions for production deployment"""
        
        instructions = f"""
🚀 PRODUCTION SECRETS SETUP COMPLETE

📋 Next Steps:
1. Copy the generated secrets to your production environment:
   cp {self.output_file} .env

2. Set secure file permissions:
   chmod 600 .env

3. Configure your specific API keys and wallet:
   - Add your PRIVATE_KEY and WALLET_ADDRESS
   - Add your HELIUS_API_KEY and QUICKNODE_ENDPOINT
   - Add your Discord webhook and other notification URLs

4. Validate the configuration:
   python scripts/production_readiness_check.py

5. Test in simulation mode first:
   export SIMULATION_MODE=true
   python enhanced_trading_main.py

6. Deploy to production:
   ./deploy.sh production live

🔐 Security Reminders:
- Never commit .env files to version control
- Rotate secrets regularly (monthly recommended)
- Monitor access logs for unauthorized access
- Use hardware security modules (HSM) for high-value deployments

📊 Generated Secrets Summary:
- Master encryption key: 256-bit
- Wallet password: 24+ characters
- API tokens: 32+ characters  
- Database credentials: Secure random
- Monitoring passwords: Secure random

🛡️ Additional Security Recommendations:
- Enable 2FA on all external accounts
- Use VPN for remote access
- Monitor system logs regularly
- Set up automated backups
- Test disaster recovery procedures
"""
        return instructions

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate production secrets')
    parser.add_argument('--output', default='.env.secrets', 
                       help='Output file for secrets (default: .env.secrets)')
    parser.add_argument('--docker', action='store_true',
                       help='Generate Docker secrets as well')
    args = parser.parse_args()
    
    print("🔐 Generating Production Secrets...")
    
    generator = ProductionSecretsGenerator(args.output)
    
    # Generate all secrets
    secrets_dict = generator.generate_all_secrets()
    
    # Save to file
    generator.save_secrets_file(secrets_dict)
    
    # Validate strength
    if generator.validate_secrets_strength():
        print("✅ All secrets meet security requirements")
    else:
        print("⚠️ Some secrets may be weak - review and regenerate if needed")
    
    # Generate Docker secrets if requested
    if args.docker:
        generator.generate_docker_secrets()
    
    # Show setup instructions
    print(generator.generate_setup_instructions())

if __name__ == "__main__":
    import time
    main() 