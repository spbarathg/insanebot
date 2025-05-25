#!/usr/bin/env python3
"""
Run Enhanced Ant Bot - Clean Production Version
QuickNode Primary + Helius Backup + Jupiter DEX
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import the cleaned Enhanced Ant Bot
from enhanced_main_entry import main

if __name__ == "__main__":
    print("🎯 Enhanced Ant Bot - Production Ready")
    print("="*50)
    print("🚀 QuickNode Primary + Helius Backup")
    print("🏰 Ant Hierarchy + AI Collaboration")
    print("🔄 Self-Replication Enabled")
    print("="*50)
    
    # Run the Enhanced Ant Bot
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Enhanced Ant Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Enhanced Ant Bot error: {str(e)}")
        sys.exit(1) 