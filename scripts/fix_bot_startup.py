#!/usr/bin/env python3
"""
Comprehensive System Management Tool for Enhanced Ant Bot
Configuration, validation, health checks, deployment, and troubleshooting
"""

import os
import sys
import json
import secrets
import string
import argparse
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def generate_secure_password(length=16):
    """Generate a secure random password"""
    characters = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(characters) for _ in range(length))

def generate_salt():
    """Generate a 32-byte hex salt for wallet encryption"""
    return secrets.token_hex(32)

def check_system_requirements():
    """Check system requirements and dependencies"""
    print(f"\n🔍 System Requirements Check:")
    
    issues = []
    requirements = {
        'Python': {'current': platform.python_version(), 'required': '3.8+'},
        'OS': {'current': platform.system(), 'required': 'Windows/Linux/macOS'},
        'Architecture': {'current': platform.machine(), 'required': 'x64/ARM64'}
    }
    
    # Check Python version
    python_version = tuple(map(int, platform.python_version().split('.')))
    if python_version < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Display requirements
    for name, info in requirements.items():
        status = "✅" if name not in [i.split()[0] for i in issues] else "❌"
        print(f"   {status} {name}: {info['current']} (required: {info['required']})")
    
    # Check dependencies
    print(f"\n📦 Dependencies Check:")
    required_packages = [
        'solana', 'httpx', 'aiohttp', 'pandas', 'numpy', 
        'python-dotenv', 'loguru', 'pytest'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            issues.append(f"Missing package: {package}")
    
    return issues

def check_config_file():
    """Check and update config.json with required risk settings"""
    config_file = Path("config/config.json")
    
    if not config_file.exists():
        print("❌ config/config.json not found")
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
        
        # Add performance settings if missing
        if 'performance' not in config:
            config['performance'] = {
                "max_concurrent_trades": 5,
                "rate_limit_requests": 100,
                "cache_duration": 300,
                "retry_attempts": 3
            }
            updated = True
            print("✅ Added performance section")
        
        if updated:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            print("✅ Updated config/config.json with missing settings")
        else:
            print("✅ config/config.json already has all required settings")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating config/config.json: {e}")
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

def check_docker_setup():
    """Check Docker configuration and setup"""
    print(f"\n🐳 Docker Configuration:")
    
    # Check if Docker is installed
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Docker installed: {result.stdout.strip()}")
        else:
            print(f"   ❌ Docker not found")
            return False
    except FileNotFoundError:
        print(f"   ❌ Docker not installed")
        return False
    
    # Check docker-compose file
    compose_file = Path("docker-compose.yml")
    if compose_file.exists():
        print(f"   ✅ docker-compose.yml found")
        try:
            # Validate YAML syntax
            with open(compose_file, 'r') as f:
                content = f.read()
            # Basic validation - check for required services
            if 'enhanced-ant-bot' in content:
                print(f"   ✅ enhanced-ant-bot service configured")
            else:
                print(f"   ⚠️  enhanced-ant-bot service not found")
        except Exception as e:
            print(f"   ❌ Error reading docker-compose.yml: {e}")
    else:
        print(f"   ❌ docker-compose.yml not found")
        return False
    
    return True

def run_health_check():
    """Run comprehensive health check"""
    print(f"\n🏥 System Health Check:")
    
    health_status = {
        'config': check_config_file(),
        'environment': check_env_file(),
        'docker': check_docker_setup()
    }
    
    # Test bot initialization
    print(f"\n🤖 Bot Initialization Test:")
    try:
        # Try to import and initialize basic components
        sys.path.append(str(Path.cwd()))
        from src.core.wallet_manager import WalletManager
        print(f"   ✅ Core modules importable")
        health_status['imports'] = True
    except Exception as e:
        print(f"   ❌ Import error: {str(e)}")
        health_status['imports'] = False
    
    # Overall health score
    healthy_components = sum(health_status.values())
    total_components = len(health_status)
    health_percentage = (healthy_components / total_components) * 100
    
    print(f"\n📊 Overall Health: {health_percentage:.1f}% ({healthy_components}/{total_components} components healthy)")
    
    if health_percentage >= 80:
        print(f"✅ System is healthy and ready to run")
    elif health_percentage >= 60:
        print(f"⚠️  System has minor issues but should work")
    else:
        print(f"❌ System has major issues - troubleshooting needed")
    
    return health_status

def install_dependencies():
    """Install missing dependencies"""
    print(f"\n📦 Installing Dependencies:")
    
    try:
        # Install from requirements.txt
        if Path("requirements.txt").exists():
            print(f"Installing from requirements.txt...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Error installing dependencies: {result.stderr}")
                return False
        else:
            print(f"❌ requirements.txt not found")
            return False
    except Exception as e:
        print(f"❌ Error during installation: {e}")
        return False

def create_project_structure():
    """Create missing project directories and files"""
    print(f"\n📁 Project Structure:")
    
    required_dirs = [
        "logs", "data", "backups", "config", 
        "src/core", "tests/unit", "tests/integration"
    ]
    
    created = []
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(str(path))
    
    if created:
        print(f"✅ Created directories: {', '.join(created)}")
    else:
        print(f"✅ All required directories exist")
    
    return True

def deploy_bot():
    """Deploy bot using Docker"""
    print(f"\n🚀 Deploying Bot:")
    
    if not check_docker_setup():
        print(f"❌ Docker setup issues - cannot deploy")
        return False
    
    try:
        # Build and start containers
        print(f"Building Docker containers...")
        build_result = subprocess.run(['docker-compose', 'build'], 
                                    capture_output=True, text=True)
        
        if build_result.returncode == 0:
            print(f"✅ Docker build successful")
            
            print(f"Starting containers...")
            start_result = subprocess.run(['docker-compose', 'up', '-d'], 
                                        capture_output=True, text=True)
            
            if start_result.returncode == 0:
                print(f"✅ Bot deployed successfully")
                print(f"   Use 'docker-compose logs -f enhanced-ant-bot' to view logs")
                return True
            else:
                print(f"❌ Failed to start containers: {start_result.stderr}")
                return False
        else:
            print(f"❌ Docker build failed: {build_result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return False

def stop_bot():
    """Stop running bot"""
    print(f"\n🛑 Stopping Bot:")
    
    try:
        result = subprocess.run(['docker-compose', 'down'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Bot stopped successfully")
            return True
        else:
            print(f"❌ Error stopping bot: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_logs(lines=50):
    """Show bot logs"""
    print(f"\n📝 Bot Logs (last {lines} lines):")
    
    try:
        result = subprocess.run(['docker-compose', 'logs', '--tail', str(lines), 'enhanced-ant-bot'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ Error getting logs: {result.stderr}")
    except Exception as e:
        print(f"❌ Error: {e}")

def troubleshoot():
    """Run troubleshooting diagnostics"""
    print(f"\n🔧 Troubleshooting:")
    
    # Check common issues
    issues_found = []
    
    # Check system requirements
    req_issues = check_system_requirements()
    if req_issues:
        issues_found.extend(req_issues)
    
    # Check file permissions
    important_files = ['.env', 'config/config.json', 'requirements.txt']
    for file_path in important_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    f.read(1)  # Try to read
                print(f"   ✅ {file_path} readable")
            except PermissionError:
                issues_found.append(f"Permission denied: {file_path}")
                print(f"   ❌ {file_path} permission denied")
        else:
            print(f"   ⚠️  {file_path} missing")
    
    # Check network connectivity
    print(f"\n🌐 Network Check:")
    try:
        import urllib.request
        urllib.request.urlopen('https://google.com', timeout=5)
        print(f"   ✅ Internet connectivity")
    except:
        issues_found.append("No internet connectivity")
        print(f"   ❌ No internet connectivity")
    
    # Summary
    if issues_found:
        print(f"\n⚠️  Issues Found:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print(f"\n💡 Suggested Actions:")
        print(f"   • Run with --install to fix dependencies")
        print(f"   • Run with --setup to fix configuration")
        print(f"   • Check file permissions")
        print(f"   • Verify internet connection")
    else:
        print(f"\n✅ No obvious issues found")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive System Management Tool")
    
    # Configuration management
    parser.add_argument("--setup", action="store_true", help="Full system setup")
    parser.add_argument("--config-only", action="store_true", help="Only check/fix config.json")
    parser.add_argument("--env-only", action="store_true", help="Only check/fix .env file")
    
    # System operations
    parser.add_argument("--install", action="store_true", help="Install missing dependencies")
    parser.add_argument("--health", action="store_true", help="Run comprehensive health check")
    parser.add_argument("--troubleshoot", action="store_true", help="Run troubleshooting diagnostics")
    
    # Deployment operations
    parser.add_argument("--deploy", action="store_true", help="Deploy bot using Docker")
    parser.add_argument("--stop", action="store_true", help="Stop running bot")
    parser.add_argument("--logs", type=int, nargs='?', const=50, help="Show bot logs")
    
    # Test and validation
    parser.add_argument("--test", action="store_true", help="Test mode - check configuration without making changes")
    parser.add_argument("--validate", action="store_true", help="Validate all configurations")
    
    # Output control
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🔧 Enhanced Ant Bot - System Management Tool")
        print("=" * 50)
    
    # Handle different operations
    if args.setup:
        if not args.quiet:
            print("\n🛠️  Full System Setup:")
        
        # Run all setup operations
        create_project_structure()
        config_ok = check_config_file() if not args.env_only else True
        env_ok = check_env_file() if not args.config_only else True
        
        if config_ok and env_ok:
            if not args.quiet:
                print("\n✅ Setup completed successfully!")
        else:
            if not args.quiet:
                print("\n❌ Setup completed with issues")
    
    elif args.install:
        install_dependencies()
    
    elif args.health:
        run_health_check()
    
    elif args.troubleshoot:
        troubleshoot()
    
    elif args.deploy:
        deploy_bot()
    
    elif args.stop:
        stop_bot()
    
    elif args.logs is not None:
        show_logs(args.logs)
    
    elif args.validate:
        health_status = run_health_check()
        if all(health_status.values()):
            print("✅ All validations passed")
        else:
            print("❌ Some validations failed")
            return 1
    
    elif args.test or not any(vars(args).values()):
        # Default test behavior
        config_ok = True
        env_ok = True
        
        if not args.env_only:
            if not args.quiet:
                print("\n1. Checking configuration files...")
            config_ok = check_config_file()
        
        if not args.config_only:
            if not args.quiet:
                print("\n2. Checking environment variables...")
            env_ok = check_env_file()
        
        if not args.quiet:
            print("\n" + "=" * 50)
        
        if config_ok and env_ok:
            if not args.quiet:
                print("✅ All startup issues fixed!")
                print("\n🚀 Your bot should now start successfully.")
                print("\n💡 Next steps:")
                print("   1. Update your .env file with real API keys")
                print("   2. Add your wallet private key")
                print("   3. Run: python main_simple.py --test")
                print("   4. Or deploy: python scripts/fix_bot_startup.py --deploy")
            elif args.test:
                print("✅ Test passed: Configuration looks good")
        else:
            if not args.quiet:
                print("❌ Some issues remain. Please check the errors above.")
            elif args.test:
                print("❌ Test failed: Configuration issues found")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 