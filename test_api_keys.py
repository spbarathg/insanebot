#!/usr/bin/env python3
"""
Test script to verify API keys and connections
"""
import asyncio
import aiohttp
import logging
import json
from dotenv import load_dotenv
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api_test")

# Load environment variables
load_dotenv()

async def test_helius_api():
    """Test Helius API connection"""
    api_key = os.getenv("HELIUS_API_KEY")
    if not api_key:
        logger.error("❌ HELIUS_API_KEY not found in .env file")
        return False
    
    url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={
                "jsonrpc": "2.0",
                "id": "helius-test",
                "method": "getLatestBlockhash",
                "params": []
            }) as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data:
                        logger.info(f"✅ Helius API connection successful")
                        return True
                    else:
                        logger.error(f"❌ Helius API error: {data}")
                        return False
                else:
                    logger.error(f"❌ Helius API HTTP error: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"❌ Helius API connection error: {str(e)}")
        return False

async def test_solana_rpc():
    """Test Solana RPC connection with wallet address"""
    # Use a simple RPC call to test connection
    url = "https://api.mainnet-beta.solana.com"
    
    try:
        # Get recent block hash as a simple test
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={
                "jsonrpc": "2.0",
                "id": "solana-test",
                "method": "getLatestBlockhash",
                "params": []
            }) as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data:
                        logger.info(f"✅ Solana RPC connection successful")
                        return True
                    else:
                        logger.error(f"❌ Solana RPC error: {data}")
                        return False
                else:
                    logger.error(f"❌ Solana RPC HTTP error: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"❌ Solana RPC connection error: {str(e)}")
        return False

async def test_jupiter_api():
    """Test Jupiter API connection"""
    try:
        # Just get the available tokens to verify API is accessible
        url = "https://token.jup.ag/all"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list) and len(data) > 0:
                        logger.info(f"✅ Jupiter API connection successful - Found {len(data)} tokens")
                        return True
                    else:
                        logger.error(f"❌ Jupiter API unexpected response format")
                        return False
                else:
                    logger.error(f"❌ Jupiter API HTTP error: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"❌ Jupiter API connection error: {str(e)}")
        return False

async def main():
    """Run all tests"""
    logger.info("Starting API key tests...")
    
    helius_success = await test_helius_api()
    solana_success = await test_solana_rpc()
    jupiter_success = await test_jupiter_api()
    
    logger.info("\n" + "="*50)
    logger.info("API TEST RESULTS")
    logger.info("="*50)
    logger.info(f"Helius API: {'✅ PASS' if helius_success else '❌ FAIL'}")
    logger.info(f"Solana RPC: {'✅ PASS' if solana_success else '❌ FAIL'}")
    logger.info(f"Jupiter API: {'✅ PASS' if jupiter_success else '❌ FAIL'}")
    logger.info("="*50)
    
    if helius_success and solana_success and jupiter_success:
        logger.info("✅ All API tests passed! Your configuration is ready for trading.")
    else:
        logger.info("❌ Some API tests failed. Please check the error messages above.")

if __name__ == "__main__":
    asyncio.run(main()) 