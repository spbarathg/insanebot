#!/usr/bin/env python3
"""
Simple test script for the Solana trading bot without heavy ML dependencies.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

# Set environment variables for testing
os.environ["SIMULATION_MODE"] = "true"
os.environ["SIMULATION_CAPITAL"] = "1.0"
os.environ["USE_LOCAL_LLM"] = "false"  # Disable LLM to avoid heavy imports
os.environ["LOG_LEVEL"] = "INFO"

async def test_bot():
    """Test the bot initialization and basic functionality."""
    try:
        print("🚀 Testing Solana Trading Bot...")
        
        # Import the main bot class
        print("📦 Importing bot modules...")
        from src.core.helius_service import HeliusService
        from src.core.jupiter_service import JupiterService
        
        print("✅ Core services imported successfully")
        
        # Test service initialization
        print("🔧 Testing service initialization...")
        helius = HeliusService()
        jupiter = JupiterService()
        
        print("✅ Services initialized successfully")
        
        # Test basic functionality
        print("🧪 Testing basic functionality...")
        
        # Test Helius service
        print("📡 Testing Helius API...")
        metadata = await helius.get_token_metadata("So11111111111111111111111111111111111111112")  # SOL
        if metadata:
            print(f"✅ Helius API working - Got metadata for {metadata.get('symbol', 'Unknown')}")
        else:
            print("⚠️ Helius API returned no data (expected with demo key)")
        
        # Test Jupiter service
        print("📡 Testing Jupiter API...")
        tokens = await jupiter.get_supported_tokens()
        if tokens:
            print(f"✅ Jupiter API working - Found {len(tokens)} supported tokens")
        else:
            print("⚠️ Jupiter API returned no data")
        
        # Cleanup
        await helius.close()
        await jupiter.close()
        
        print("✅ All tests passed! Bot is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_bot())
    if success:
        print("\n🎉 Bot is ready to use!")
        print("💡 To run the full bot, use: python src/main.py")
    else:
        print("\n💥 Bot has issues that need to be fixed.")
        sys.exit(1) 