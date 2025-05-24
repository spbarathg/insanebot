#!/usr/bin/env python3
"""
Quick test script to verify environment variables are loaded correctly
"""
import os
from dotenv import load_dotenv

def test_env_vars():
    print("🔍 Testing Environment Variables Loading...")
    
    # Load .env file
    load_dotenv()
    
    # Check API keys
    helius_key = os.getenv("HELIUS_API_KEY", "NOT_FOUND")
    jupiter_key = os.getenv("JUPITER_API_KEY", "NOT_FOUND")
    private_key = os.getenv("PRIVATE_KEY", "NOT_FOUND")
    
    print(f"\n📊 Environment Variables Status:")
    print(f"HELIUS_API_KEY: {'✅ SET' if helius_key != 'NOT_FOUND' and helius_key != 'demo_key_for_testing' else '❌ NOT SET'}")
    print(f"JUPITER_API_KEY: {'✅ SET' if jupiter_key != 'NOT_FOUND' and jupiter_key != 'demo_key_for_testing' else '❌ NOT SET'}")
    print(f"PRIVATE_KEY: {'✅ SET' if private_key != 'NOT_FOUND' and private_key != 'demo_private_key_for_testing' else '❌ NOT SET'}")
    
    if helius_key != "NOT_FOUND":
        print(f"\nHELIUS_API_KEY starts with: {helius_key[:8]}...")
    if jupiter_key != "NOT_FOUND":
        print(f"JUPITER_API_KEY starts with: {jupiter_key[:8]}...")
    
    # Test API connections
    print(f"\n🧪 Testing API Connections...")
    
    # Test Helius
    if helius_key and helius_key != "demo_key_for_testing":
        try:
            import aiohttp
            import asyncio
            
            async def test_helius():
                async with aiohttp.ClientSession() as session:
                    url = f"https://api.helius.xyz/v0/tokens/metadata"
                    headers = {"Authorization": f"Bearer {helius_key}"}
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            print("✅ Helius API: Connected successfully")
                        elif response.status == 401:
                            print("❌ Helius API: Authentication failed - check API key")
                        else:
                            print(f"⚠️ Helius API: Status {response.status}")
            
            asyncio.run(test_helius())
        except Exception as e:
            print(f"❌ Helius API test failed: {str(e)}")
    else:
        print("⚠️ Helius API: No valid key found")
    
    # Test Jupiter
    try:
        import aiohttp
        import asyncio
        
        async def test_jupiter():
            async with aiohttp.ClientSession() as session:
                url = "https://quote-api.jup.ag/v6/tokens"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        print("✅ Jupiter API: Connected successfully")
                    else:
                        print(f"❌ Jupiter API: Status {response.status}")
        
        asyncio.run(test_jupiter())
    except Exception as e:
        print(f"❌ Jupiter API test failed: {str(e)}")

if __name__ == "__main__":
    test_env_vars() 