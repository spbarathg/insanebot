#!/usr/bin/env python3
"""
Network connectivity test for trading bot APIs
"""

import asyncio
import aiohttp
import time
import socket
from typing import List, Dict

async def test_dns_resolution(hostname: str) -> bool:
    """Test DNS resolution for a hostname"""
    try:
        socket.gethostbyname(hostname)
        print(f"✅ DNS resolution for {hostname}: SUCCESS")
        return True
    except socket.gaierror as e:
        print(f"❌ DNS resolution for {hostname}: FAILED - {str(e)}")
        return False

async def test_http_endpoint(url: str, timeout: int = 5) -> Dict:
    """Test HTTP endpoint connectivity"""
    try:
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response_time = time.time() - start_time
                
                result = {
                    'url': url,
                    'status': response.status,
                    'response_time': response_time,
                    'success': response.status == 200,
                    'content_type': response.headers.get('content-type', 'unknown')
                }
                
                if result['success']:
                    print(f"✅ {url}: {response.status} ({response_time:.2f}s)")
                else:
                    print(f"❌ {url}: {response.status} ({response_time:.2f}s)")
                
                return result
                
    except asyncio.TimeoutError:
        print(f"⏰ {url}: TIMEOUT (>{timeout}s)")
        return {'url': url, 'status': 'timeout', 'success': False}
    except Exception as e:
        print(f"💥 {url}: ERROR - {str(e)}")
        return {'url': url, 'status': 'error', 'success': False, 'error': str(e)}

async def main():
    """Test all API endpoints"""
    print("🌐 Testing Network Connectivity for Trading Bot")
    print("=" * 60)
    
    # DNS Tests
    print("\n🔍 DNS Resolution Tests:")
    hostnames = [
        "api.mainnet-beta.solana.com",
        "quote-api.jup.ag", 
        "price.jup.ag",
        "api.jup.ag",
        "api.helius.xyz",
        "mainnet.helius-rpc.com",
        "api.orca.so"
    ]
    
    dns_results = []
    for hostname in hostnames:
        result = await test_dns_resolution(hostname)
        dns_results.append(result)
    
    # HTTP Endpoint Tests
    print("\n🌐 HTTP Endpoint Tests:")
    endpoints = [
        "https://api.mainnet-beta.solana.com",
        "https://quote-api.jup.ag/v6/tokens",
        "https://price.jup.ag/v4/price?ids=So11111111111111111111111111111111111111112",
        "https://api.helius.xyz/v0/tokens/metadata",
        "https://api.orca.so/v1/whirlpool/list"
    ]
    
    http_results = []
    for endpoint in endpoints:
        result = await test_http_endpoint(endpoint)
        http_results.append(result)
    
    # Results Summary
    print("\n📊 Test Results Summary:")
    print("=" * 60)
    
    dns_success = sum(1 for r in dns_results if r)
    print(f"DNS Resolution: {dns_success}/{len(dns_results)} successful")
    
    http_success = sum(1 for r in http_results if r['success'])
    print(f"HTTP Endpoints: {http_success}/{len(http_results)} successful")
    
    if dns_success < len(dns_results):
        print("\n⚠️  DNS Issues Detected:")
        print("This suggests network/DNS configuration problems.")
        print("Possible solutions:")
        print("1. Check your internet connection")
        print("2. Try different DNS servers (8.8.8.8, 1.1.1.1)")
        print("3. Check firewall settings")
        print("4. Restart network services")
    
    if http_success < len(http_results):
        print("\n⚠️  HTTP Connectivity Issues:")
        failed_endpoints = [r for r in http_results if not r['success']]
        for endpoint in failed_endpoints:
            print(f"Failed: {endpoint['url']} - {endpoint.get('error', 'Unknown error')}")
    
    if dns_success == len(dns_results) and http_success == len(http_results):
        print("\n✅ All network tests passed! Your connectivity is good.")
    
    print("\n🔧 Next Steps:")
    if dns_success < len(dns_results) or http_success < len(http_results):
        print("1. Fix network connectivity issues")
        print("2. Start Docker Desktop")
        print("3. Run: docker-compose up -d")
    else:
        print("1. Start Docker Desktop")
        print("2. Run: docker-compose up -d")
        print("3. Monitor logs: docker-compose logs -f trading-bot")

if __name__ == "__main__":
    asyncio.run(main()) 