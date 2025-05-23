#!/usr/bin/env python3
"""
Pre-commit check script to validate codebase before GitHub commit.
"""
import os
import sys
import py_compile
import json
import subprocess
from pathlib import Path

def check_syntax():
    """Check Python syntax for all .py files."""
    print("🔍 Checking Python syntax...")
    errors = []
    
    for root, dirs, files in os.walk('.'):
        # Skip cache and git directories
        dirs[:] = [d for d in dirs if not d.startswith(('.git', '__pycache__', '.pytest_cache'))]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    py_compile.compile(filepath, doraise=True)
                    print(f"  ✓ {filepath}")
                except py_compile.PyCompileError as e:
                    print(f"  ✗ {filepath}: {e}")
                    errors.append(f"Syntax error in {filepath}: {e}")
    
    return errors

def check_sensitive_info():
    """Check for potentially sensitive information."""
    print("🔐 Checking for sensitive information...")
    warnings = []
    
    # Check for common sensitive patterns
    sensitive_patterns = [
        'private_key',
        'secret_key',
        'api_key',
        'password',
        'token'
    ]
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if not d.startswith(('.git', '__pycache__', 'logs', 'data'))]
        
        for file in files:
            if file.endswith(('.py', '.json', '.yml', '.yaml', '.env')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    for pattern in sensitive_patterns:
                        if pattern in content and 'example' not in file and 'template' not in file:
                            # Check if it's just a variable name or actual sensitive data
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if pattern in line and '=' in line:
                                    value = line.split('=')[1].strip().strip('"\'')
                                    if len(value) > 10 and not value.startswith(('your-', 'example', 'test', '0000')):
                                        warnings.append(f"Potential sensitive data in {filepath}:{i+1}")
                                        
                except Exception:
                    continue
    
    return warnings

def check_required_files():
    """Check for required project files."""
    print("📋 Checking required files...")
    required_files = [
        'README.md',
        'requirements.txt',
        '.gitignore',
        'env.example'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"  ✓ {file}")
    
    for file in missing_files:
        print(f"  ✗ Missing: {file}")
    
    return missing_files

def check_json_files():
    """Check JSON files for valid syntax."""
    print("📄 Checking JSON files...")
    errors = []
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if not d.startswith(('.git', '__pycache__'))]
        
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        json.load(f)
                    print(f"  ✓ {filepath}")
                except json.JSONDecodeError as e:
                    print(f"  ✗ {filepath}: {e}")
                    errors.append(f"JSON error in {filepath}: {e}")
                except Exception as e:
                    print(f"  ✗ {filepath}: {e}")
                    errors.append(f"Error reading {filepath}: {e}")
    
    return errors

def check_imports():
    """Check for missing imports in key files."""
    print("📦 Checking imports in key files...")
    warnings = []
    
    key_files = ['main.py', 'src/main.py', 'run_bot.py']
    
    for filepath in key_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for common import patterns
                if 'import asyncio' in content and 'asyncio.run' in content:
                    print(f"  ✓ {filepath}: asyncio usage looks good")
                elif 'asyncio' in content:
                    warnings.append(f"Check asyncio usage in {filepath}")
                    
            except Exception:
                continue
    
    return warnings

def main():
    """Main validation function."""
    print("🚀 Running pre-commit checks for Solana Trading Bot...")
    print("=" * 50)
    
    all_errors = []
    all_warnings = []
    
    # Run all checks
    syntax_errors = check_syntax()
    sensitive_warnings = check_sensitive_info()
    missing_files = check_required_files()
    json_errors = check_json_files()
    import_warnings = check_imports()
    
    all_errors.extend(syntax_errors)
    all_errors.extend([f"Missing required file: {f}" for f in missing_files])
    all_errors.extend(json_errors)
    
    all_warnings.extend(sensitive_warnings)
    all_warnings.extend(import_warnings)
    
    print("\n" + "=" * 50)
    
    if all_errors:
        print("❌ ERRORS FOUND:")
        for error in all_errors:
            print(f"  • {error}")
        print(f"\nTotal errors: {len(all_errors)}")
        return 1
    
    if all_warnings:
        print("⚠️  WARNINGS:")
        for warning in all_warnings:
            print(f"  • {warning}")
        print(f"\nTotal warnings: {len(all_warnings)}")
    
    print("✅ Pre-commit checks completed!")
    if not all_errors:
        print("✅ Codebase is ready for commit!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 