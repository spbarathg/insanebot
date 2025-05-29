#!/usr/bin/env python3
"""
Enhanced Ant Bot - Easy CLI Startup
Simple launcher for the CLI control center
"""

import os
import sys

def main():
    """Launch the CLI control center"""
    print("🚀 Starting Enhanced Ant Bot CLI...")
    
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    try:
        # Import and run the CLI
        from src.cli import main as cli_main
        import asyncio
        
        asyncio.run(cli_main())
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed all dependencies with 'pip install -r requirements.txt'")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main() 