#!/usr/bin/env python3
"""
🚀 Docker Deployment Validation Script

This script validates that your Solana Trading Bot is ready for Docker deployment.
It checks all files, configurations, and dependencies required for production.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    print(f"{Colors.PURPLE}{Colors.BOLD}")
    print("=" * 70)
    print("🚀 DOCKER DEPLOYMENT VALIDATION")
    print("=" * 70)
    print(f"{Colors.END}")

def print_section(title: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}📋 {title}{Colors.END}")
    print("-" * (len(title) + 4))

def print_check(item: str, status: bool, details: str = ""):
    icon = "✅" if status else "❌"
    color = Colors.GREEN if status else Colors.RED
    print(f"{color}{icon} {item}{Colors.END}")
    if details:
        print(f"   {details}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message: str):
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def check_docker_files() -> Dict[str, bool]:
    """Check if all required Docker files exist"""
    print_section("Docker Configuration Files")
    
    required_files = {
        "Dockerfile": "Main container configuration",
        "docker-compose.prod.yml": "Production stack configuration",
        "env.production": "Production environment template",
        ".dockerignore": "Docker build optimization"
    }
    
    results = {}
    for file, description in required_files.items():
        exists = Path(file).exists()
        results[file] = exists
        print_check(f"{file} - {description}", exists)
        if not exists and file == ".dockerignore":
            print_info("Optional file - Docker will include all files")
    
    return results

def check_monitoring_files() -> Dict[str, bool]:
    """Check monitoring configuration files"""
    print_section("Monitoring Configuration")
    
    monitoring_files = {
        "monitoring/prometheus.yml": "Metrics collection config",
        "monitoring/alert_rules.yml": "Alert configuration",
        "monitoring/grafana/": "Dashboard configuration (directory)"
    }
    
    results = {}
    for file, description in monitoring_files.items():
        if file.endswith("/"):
            exists = Path(file).is_dir()
        else:
            exists = Path(file).exists()
        results[file] = exists
        print_check(f"{file} - {description}", exists)
    
    return results

def check_application_files() -> Dict[str, bool]:
    """Check core application files"""
    print_section("Core Application Files")
    
    core_files = {
        "enhanced_trading_main.py": "Main application entry point",
        "requirements.txt": "Python dependencies",
        "src/": "Source code directory",
        "logs/": "Logging directory (auto-created)"
    }
    
    results = {}
    for file, description in core_files.items():
        if file.endswith("/"):
            exists = Path(file).is_dir()
        else:
            exists = Path(file).exists()
        results[file] = exists
        print_check(f"{file} - {description}", exists)
    
    return results

def check_environment_config() -> Dict[str, bool]:
    """Check environment configuration"""
    print_section("Environment Configuration")
    
    # Check if .env exists
    env_exists = Path(".env").exists()
    env_prod_exists = Path("env.production").exists()
    
    print_check(".env file", env_exists, 
                "Required for Docker Compose" if not env_exists else "Ready for deployment")
    
    print_check("env.production template", env_prod_exists,
                "Template for creating .env file")
    
    if not env_exists and env_prod_exists:
        print_warning("Create .env file by copying env.production and filling in your values")
        print_info("Command: cp env.production .env")
    
    results = {
        ".env": env_exists,
        "env.production": env_prod_exists
    }
    
    return results

def check_docker_availability() -> Dict[str, bool]:
    """Check if Docker and Docker Compose are available"""
    print_section("Docker Availability")
    
    results = {}
    
    # Check Docker
    try:
        result = subprocess.run(["docker", "--version"], 
                              capture_output=True, text=True, timeout=10)
        docker_available = result.returncode == 0
        if docker_available:
            version = result.stdout.strip()
            print_check("Docker Engine", True, f"Found: {version}")
        else:
            print_check("Docker Engine", False, "Docker not found or not running")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        docker_available = False
        print_check("Docker Engine", False, "Docker not installed or not accessible")
    
    results["docker"] = docker_available
    
    # Check Docker Compose
    try:
        result = subprocess.run(["docker", "compose", "version"], 
                              capture_output=True, text=True, timeout=10)
        compose_available = result.returncode == 0
        if compose_available:
            version = result.stdout.strip().split('\n')[0]
            print_check("Docker Compose", True, f"Found: {version}")
        else:
            print_check("Docker Compose", False, "Docker Compose not found")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        compose_available = False
        print_check("Docker Compose", False, "Docker Compose not available")
    
    results["compose"] = compose_available
    
    return results

def check_resource_requirements() -> Dict[str, bool]:
    """Check system resource requirements"""
    print_section("System Resources")
    
    results = {}
    
    try:
        import psutil
        
        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_adequate = memory_gb >= 8  # Minimum 8GB recommended
        
        print_check(f"Available RAM: {memory_gb:.1f} GB", memory_adequate,
                    "Minimum 8GB recommended for production deployment")
        results["memory"] = memory_adequate
        
        # Check CPU
        cpu_cores = psutil.cpu_count()
        cpu_adequate = cpu_cores >= 2  # Minimum 2 cores
        
        print_check(f"CPU Cores: {cpu_cores}", cpu_adequate,
                    "Minimum 2 cores recommended")
        results["cpu"] = cpu_adequate
        
        # Check disk space
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        disk_adequate = disk_free_gb >= 20  # Minimum 20GB free
        
        print_check(f"Free Disk Space: {disk_free_gb:.1f} GB", disk_adequate,
                    "Minimum 20GB free space recommended")
        results["disk"] = disk_adequate
        
    except ImportError:
        print_warning("psutil not available - cannot check system resources")
        results = {"memory": True, "cpu": True, "disk": True}  # Assume adequate
    
    return results

def validate_dockerfile_syntax() -> bool:
    """Validate Dockerfile syntax"""
    print_section("Dockerfile Validation")
    
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        print_check("Dockerfile syntax", False, "Dockerfile not found")
        return False
    
    try:
        # Read and basic validation
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for essential instructions
        essential_instructions = ["FROM", "COPY", "CMD"]
        missing_instructions = []
        
        for instruction in essential_instructions:
            if instruction not in content:
                missing_instructions.append(instruction)
        
        if missing_instructions:
            print_check("Dockerfile syntax", False, 
                       f"Missing instructions: {', '.join(missing_instructions)}")
            return False
        else:
            print_check("Dockerfile syntax", True, "All essential instructions present")
            return True
            
    except Exception as e:
        print_check("Dockerfile syntax", False, f"Error reading Dockerfile: {e}")
        return False

def generate_deployment_summary(all_results: Dict[str, Dict[str, bool]]) -> None:
    """Generate deployment readiness summary"""
    print_section("Deployment Readiness Summary")
    
    total_checks = 0
    passed_checks = 0
    critical_failures = []
    
    for category, results in all_results.items():
        for check, status in results.items():
            total_checks += 1
            if status:
                passed_checks += 1
            else:
                # Mark critical failures
                if check in ["Dockerfile", "docker-compose.prod.yml", "enhanced_trading_main.py", 
                           "docker", "compose"]:
                    critical_failures.append(f"{category}: {check}")
    
    success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    print(f"\n{Colors.BOLD}Overall Readiness: {success_rate:.1f}% ({passed_checks}/{total_checks} checks passed){Colors.END}")
    
    if success_rate >= 90:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 READY FOR DEPLOYMENT!{Colors.END}")
        print(f"{Colors.GREEN}Your trading bot is fully prepared for Docker deployment.{Colors.END}")
    elif success_rate >= 75:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  MOSTLY READY - Minor Issues{Colors.END}")
        print(f"{Colors.YELLOW}Address the remaining issues for optimal deployment.{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ NOT READY - Critical Issues{Colors.END}")
        print(f"{Colors.RED}Resolve critical issues before deployment.{Colors.END}")
    
    if critical_failures:
        print(f"\n{Colors.RED}{Colors.BOLD}Critical Issues:{Colors.END}")
        for failure in critical_failures:
            print(f"{Colors.RED}  • {failure}{Colors.END}")

def provide_next_steps() -> None:
    """Provide next steps for deployment"""
    print_section("Next Steps")
    
    print("1. Ensure all checks pass (90%+ success rate)")
    print("2. Create .env file: cp env.production .env")
    print("3. Fill in your API keys and credentials in .env")
    print("4. Deploy with: docker compose -f docker-compose.prod.yml up -d")
    print("5. Monitor logs: docker logs -f solana-trading-bot")
    print("6. Access Grafana dashboard: http://localhost:3000")
    
    print(f"\n{Colors.CYAN}For detailed instructions, see: DOCKER_DEPLOYMENT_GUIDE.md{Colors.END}")

def main():
    """Main validation function"""
    print_header()
    
    # Run all validation checks
    all_results = {
        "Docker Files": check_docker_files(),
        "Monitoring": check_monitoring_files(),
        "Application": check_application_files(),
        "Environment": check_environment_config(),
        "Docker Tools": check_docker_availability(),
        "System Resources": check_resource_requirements()
    }
    
    # Additional validations
    dockerfile_valid = validate_dockerfile_syntax()
    all_results["Dockerfile Validation"] = {"syntax": dockerfile_valid}
    
    # Generate summary
    generate_deployment_summary(all_results)
    
    # Provide next steps
    provide_next_steps()
    
    print(f"\n{Colors.PURPLE}{Colors.BOLD}Validation Complete!{Colors.END}")

if __name__ == "__main__":
    main() 