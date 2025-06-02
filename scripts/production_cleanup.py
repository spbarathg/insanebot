#!/usr/bin/env python3
"""
🧹 Production Cleanup Script

Automatically removes development files and prepares the codebase for production deployment.
This script will:
1. Remove all development/testing files
2. Clean up build artifacts
3. Prepare the minimal production structure
4. Validate essential files remain

Run this before deploying to your server to minimize size and security exposure.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Set
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Files and directories to REMOVE for production
REMOVE_FILES = {
    # Documentation files (keep only essential)
    "PRODUCTION_READINESS_REPORT.md",
    "UPDATED_PRODUCTION_READINESS_REPORT.md", 
    "RECOMMENDATIONS_COMPLETED_SUMMARY.md",
    "COST_EFFICIENT_DEPLOYMENT_GUIDE.md",
    
    # Test coverage and reports
    ".coverage",
    "pytest.ini",
    
    # Development files  
    "integrate_monitoring_with_existing_bot.py",
    "main_trading_bot_with_monitoring.py",
    "setup_monitoring.py",
    "create_env.py",
    
    # Security scan reports (development artifacts)
    "security_scan_current.json",
    "bandit_report_new.json", 
    "bandit_report.json",
    
    # Template files (use actual env file)
    "env.template",
    "cost_efficient_max_performance.env",
    
    # Git files
    ".gitignore",
    
    # Development docker
    "docker-compose.yml"
}

REMOVE_DIRECTORIES = {
    # Test directories
    "tests",
    "htmlcov", 
    ".pytest_cache",
    
    # Git directory (clean deployment)
    ".git",
    
    # CI/CD files
    ".github"
}

REMOVE_SCRIPT_FILES = {
    "scripts/production_readiness_check.py",
    "scripts/fix_random_security.py", 
    "scripts/security_hardening.py"
}

# Essential files that MUST remain
ESSENTIAL_FILES = {
    # Core application
    "enhanced_trading_main.py",
    "start_bot.py", 
    "production_config.py",
    "requirements.txt",
    
    # Production infrastructure
    "Dockerfile",
    "docker-compose.prod.yml",
    "launch_production.sh",
    "deploy.sh",
    
    # Configuration
    "env.production",
    "production_setup_report.json"
}

ESSENTIAL_DIRECTORIES = {
    "src",
    "config", 
    "monitoring",
    "scripts",  # Keep essential scripts only
    "nginx",
    "logs",
    "data", 
    "backups"
}

def safe_remove_file(file_path: Path) -> bool:
    """Safely remove a file"""
    try:
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            logger.info(f"✅ Removed file: {file_path}")
            return True
        elif file_path.exists():
            logger.warning(f"⚠️  Skipped (not a file): {file_path}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to remove {file_path}: {e}")
        return False

def safe_remove_directory(dir_path: Path) -> bool:
    """Safely remove a directory"""
    try:
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)
            logger.info(f"✅ Removed directory: {dir_path}")
            return True
        elif dir_path.exists():
            logger.warning(f"⚠️  Skipped (not a directory): {dir_path}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to remove {dir_path}: {e}")
        return False

def cleanup_development_files() -> int:
    """Remove development files and return count of files removed"""
    removed_count = 0
    project_root = Path.cwd()
    
    logger.info("🧹 Starting cleanup of development files...")
    
    # Remove individual files
    for filename in REMOVE_FILES:
        file_path = project_root / filename
        if safe_remove_file(file_path):
            removed_count += 1
    
    # Remove directories
    for dirname in REMOVE_DIRECTORIES:
        dir_path = project_root / dirname
        if safe_remove_directory(dir_path):
            removed_count += 1
    
    # Remove specific script files
    for script_file in REMOVE_SCRIPT_FILES:
        script_path = project_root / script_file
        if safe_remove_file(script_path):
            removed_count += 1
    
    return removed_count

def cleanup_python_cache() -> int:
    """Remove Python cache files"""
    removed_count = 0
    project_root = Path.cwd()
    
    logger.info("🧹 Cleaning Python cache files...")
    
    # Remove __pycache__ directories
    for pycache in project_root.rglob("__pycache__"):
        if safe_remove_directory(pycache):
            removed_count += 1
    
    # Remove .pyc files
    for pyc_file in project_root.rglob("*.pyc"):
        if safe_remove_file(pyc_file):
            removed_count += 1
    
    # Remove .pyo files
    for pyo_file in project_root.rglob("*.pyo"):
        if safe_remove_file(pyo_file):
            removed_count += 1
    
    return removed_count

def validate_essential_files() -> bool:
    """Validate that essential files still exist"""
    project_root = Path.cwd()
    missing_files = []
    missing_dirs = []
    
    logger.info("✅ Validating essential files...")
    
    # Check essential files
    for filename in ESSENTIAL_FILES:
        file_path = project_root / filename
        if not file_path.exists():
            missing_files.append(filename)
    
    # Check essential directories
    for dirname in ESSENTIAL_DIRECTORIES:
        dir_path = project_root / dirname
        if not dir_path.exists():
            missing_dirs.append(dirname)
    
    if missing_files:
        logger.error(f"❌ Missing essential files: {missing_files}")
    
    if missing_dirs:
        logger.error(f"❌ Missing essential directories: {missing_dirs}")
    
    return len(missing_files) == 0 and len(missing_dirs) == 0

def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes"""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception as e:
        logger.warning(f"Could not calculate size for {path}: {e}")
    return total

def format_size(size_bytes: int) -> str:
    """Format size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def show_cleanup_summary(removed_count: int, cache_removed: int):
    """Show summary of cleanup operation"""
    project_root = Path.cwd()
    final_size = get_directory_size(project_root)
    
    logger.info("📊 Cleanup Summary:")
    logger.info(f"   • Files/directories removed: {removed_count}")
    logger.info(f"   • Cache files removed: {cache_removed}")
    logger.info(f"   • Final project size: {format_size(final_size)}")
    
    # Show remaining structure
    logger.info("\n📁 Production structure ready:")
    logger.info("   ├── enhanced_trading_main.py (MAIN ENTRY POINT)")
    logger.info("   ├── production_config.py")
    logger.info("   ├── requirements.txt")
    logger.info("   ├── env.production (copy to .env)")
    logger.info("   ├── src/ (core trading logic)")
    logger.info("   ├── config/ (configuration)")
    logger.info("   ├── monitoring/ (production monitoring)")
    logger.info("   ├── scripts/ (essential scripts only)")
    logger.info("   └── docker-compose.prod.yml")

def create_production_env():
    """Create .env from env.production if it doesn't exist"""
    project_root = Path.cwd()
    env_file = project_root / ".env"
    env_production = project_root / "env.production"
    
    if not env_file.exists() and env_production.exists():
        try:
            shutil.copy(env_production, env_file)
            logger.info("✅ Created .env from env.production")
            logger.warning("⚠️  IMPORTANT: Edit .env file with your real API keys!")
        except Exception as e:
            logger.error(f"❌ Failed to create .env: {e}")

def main():
    """Main cleanup function"""
    logger.info("🚀 Production Deployment Cleanup Starting...")
    logger.info("=" * 60)
    
    # Get initial size
    project_root = Path.cwd()
    initial_size = get_directory_size(project_root)
    logger.info(f"📊 Initial project size: {format_size(initial_size)}")
    
    # Perform cleanup
    removed_count = cleanup_development_files()
    cache_removed = cleanup_python_cache()
    
    # Create production env if needed
    create_production_env()
    
    # Validate essential files remain
    if not validate_essential_files():
        logger.error("❌ Essential files missing! Cleanup may have failed.")
        sys.exit(1)
    
    # Show summary
    show_cleanup_summary(removed_count, cache_removed)
    
    # Calculate space saved
    final_size = get_directory_size(project_root)
    space_saved = initial_size - final_size
    if space_saved > 0:
        logger.info(f"💾 Space saved: {format_size(space_saved)}")
    
    logger.info("\n🎯 Next Steps:")
    logger.info("1. Edit .env file with your real API keys")
    logger.info("2. Run: python production_config.py --validate")
    logger.info("3. Deploy: python enhanced_trading_main.py")
    logger.info("\n✅ Production cleanup complete!")

if __name__ == "__main__":
    main() 