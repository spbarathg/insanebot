#!/usr/bin/env python3
"""
Fix random module usage in security contexts by replacing with secrets module
"""

import os
import re
from pathlib import Path

def fix_random_security():
    """Fix random module usage in security contexts"""
    
    # Files that need random module replacement
    files_to_fix = [
        'src/compounding/carwash_layer.py',
        'src/core/dex.py', 
        'src/core/execution_engine.py',
        'src/core/local_llm.py',
        'src/services/advanced_onchain_analytics.py',
        'src/services/jupiter_service.py',
        'src/services/memecoin_exit_engine.py',
        'src/services/websocket_manager.py',
        'src/core/ai/ant_hierarchy.py',
        'src/core/ai/grok_engine.py',
        'src/core/ml_engine/sentiment_analyzer.py',
        'src/core/trading/transaction_warfare_system.py'
    ]

    for file_path in files_to_fix:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Only replace if it's in security context (has security-related keywords)
                if any(keyword in content.lower() for keyword in ['password', 'token', 'key', 'salt', 'nonce', 'secret']):
                    # Add secrets import and helper functions if not present
                    if 'import secrets' not in content and 'import random' in content:
                        content = content.replace('import random', '''import random
import secrets

def secure_choice(seq):
    """Secure random choice from sequence"""
    return seq[secrets.randbelow(len(seq))]''')
                    
                    # Replace random calls with secure alternatives in security contexts
                    content = re.sub(r'random\.randint\((\d+),\s*(\d+)\)', r'secrets.randbelow(\2 - \1 + 1) + \1', content)
                    content = re.sub(r'random\.choice\(([^)]+)\)', r'secure_choice(\1)', content)
                    content = re.sub(r'random\.random\(\)', 'secrets.randbelow(10000) / 10000.0', content)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f'Fixed: {file_path}')
            except Exception as e:
                print(f'Error fixing {file_path}: {e}')

    print('Random module security fixes completed')

if __name__ == "__main__":
    fix_random_security() 