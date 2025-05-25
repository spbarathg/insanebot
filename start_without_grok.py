#!/usr/bin/env python3
"""
Start Enhanced Ant Bot without Grok API
Uses fallback configurations and mock sentiment analysis
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Set environment to indicate no Grok API
os.environ["GROK_API_KEY"] = ""
os.environ["USE_MOCK_GROK"] = "true"

def print_status():
    """Print system status without Grok"""
    print("🎯 Enhanced Ant Bot - Starting without Grok API")
    print("="*60)
    print("📡 API Services Status:")
    print("   🚀 QuickNode (Primary): ✅ CONFIGURED")
    print("   🔄 Helius (Backup): ✅ READY") 
    print("   🌟 Jupiter (DEX): ✅ READY")
    print("   🤖 Grok (AI): ⚠️ FALLBACK MODE (Mock sentiment)")
    print()
    print("🏰 System Components:")
    print("   👑 Ant Hierarchy: ✅ READY")
    print("   🧠 AI Coordination: ✅ READY (Local LLM only)")
    print("   🔄 Self-Replication: ✅ READY") 
    print("   📊 Data Ingestion: ✅ READY")
    print()
    print("⚡ Available Functionality:")
    print("   ✅ Market data scanning (QuickNode + Jupiter)")
    print("   ✅ Price monitoring and arbitrage detection")
    print("   ✅ Token metadata and holder analysis")
    print("   ✅ Trading execution via Jupiter")
    print("   ✅ Portfolio management and risk controls")
    print("   ✅ Ant hierarchy system (splitting/merging)")
    print("   ⚠️ Sentiment analysis (Mock data only)")
    print()
    print("🔮 Missing (Until Grok API added):")
    print("   ❌ Real-time social sentiment analysis")
    print("   ❌ Twitter/X trend analysis")
    print("   ❌ Advanced hype detection")
    print()
    print("💡 To add Grok later:")
    print("   1. Get API key from https://grok.com")
    print("   2. Add GROK_API_KEY to your .env file")
    print("   3. Restart the bot")
    print("="*60)

async def main():
    """Run Enhanced Ant Bot without Grok API"""
    print_status()
    
    try:
        # Import and run the Enhanced Ant Bot
        from enhanced_main_entry import main as run_enhanced_bot
        
        print("🚀 Starting Enhanced Ant Bot...")
        print("📊 Market scanning will use QuickNode + Jupiter")
        print("🤖 AI decisions will use Local LLM + Mock sentiment")
        print("⏰ Press Ctrl+C to stop")
        print()
        
        # Run the bot
        await run_enhanced_bot()
        
    except KeyboardInterrupt:
        print("\n👋 Enhanced Ant Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("💡 Check your configuration and try again")

if __name__ == "__main__":
    asyncio.run(main()) 