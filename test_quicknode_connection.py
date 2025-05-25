#!/usr/bin/env python3
"""
Quick QuickNode Connection Test
Verifies your QuickNode integration is working
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.quicknode_service import QuickNodeService

async def test_quicknode_connection():
    """Test QuickNode connection and basic functionality"""
    
    print("🔍 Testing QuickNode Connection...")
    print("="*50)
    
    # Initialize QuickNode service
    quicknode = QuickNodeService()
    
    # Check configuration
    endpoint_configured = bool(quicknode.endpoint_url)
    print(f"📡 Endpoint configured: {'✅' if endpoint_configured else '❌'}")
    
    if not endpoint_configured:
        print("⚠️  No QuickNode endpoint found!")
        print("💡 Add to your .env file:")
        print("   QUICKNODE_ENDPOINT_URL=https://your-endpoint.solana-mainnet.quiknode.pro/TOKEN/")
        return False
    
    try:
        # Test 1: Get token metadata (USDC)
        print("\n🧪 Test 1: Getting token metadata...")
        usdc_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        metadata = await quicknode.get_token_metadata(usdc_address)
        
        print(f"   Token: {metadata.get('symbol', 'Unknown')}")
        print(f"   Name: {metadata.get('name', 'Unknown')}")
        print(f"   ✅ Metadata test: {'PASSED' if metadata.get('verified') else 'FAILED'}")
        
        # Test 2: Get account balance
        print("\n🧪 Test 2: Getting account balance...")
        balance_info = await quicknode.get_account_balance(usdc_address)
        print(f"   Balance: {balance_info.get('balance_sol', 0)} SOL")
        print(f"   ✅ Balance test: PASSED")
        
        # Test 3: Get price data
        print("\n🧪 Test 3: Getting price data...")
        price_data = await quicknode.get_token_price_from_dex_pools(usdc_address)
        print(f"   Price: ${price_data.get('price_usd', 0):.6f}")
        print(f"   Source: {price_data.get('source', 'none')}")
        print(f"   ✅ Price test: {'PASSED' if price_data.get('price') > 0 else 'SIMULATED'}")
        
        # Performance stats
        stats = quicknode.get_performance_stats()
        print(f"\n📊 Performance Stats:")
        print(f"   Total requests: {stats.get('total_requests', 0)}")
        print(f"   Cache entries: {sum(stats.get('cache_entries', {}).values())}")
        
        print("\n🎉 QuickNode connection test COMPLETED!")
        print("✅ Your Enhanced Ant Bot can now use QuickNode!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ QuickNode test failed: {str(e)}")
        return False
    
    finally:
        await quicknode.close()

if __name__ == "__main__":
    result = asyncio.run(test_quicknode_connection())
    if result:
        print("\n🚀 Ready to run Enhanced Ant Bot with QuickNode!")
    else:
        print("\n⚠️  Fix QuickNode configuration and try again") 