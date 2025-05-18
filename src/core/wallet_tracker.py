from typing import Dict, List, Set, Any
import asyncio
import time
from datetime import datetime, timedelta
from collections import defaultdict
import aiohttp
import json
from loguru import logger

class WalletTracker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.wallets: Dict[str, Dict] = {}  # Active wallets being tracked
        self.wallet_groups: Dict[str, Set[str]] = defaultdict(set)  # Group wallets by activity level
        self.last_update: Dict[str, float] = {}  # Last update time for each wallet
        self.activity_cache: Dict[str, Dict] = {}  # Cache for wallet activities
        self.session = None
        self.update_interval = 60  # Base update interval in seconds
        self.cache_ttl = 300  # Cache TTL in seconds
        self.batch_size = 20  # Number of wallets to process in parallel
        self.active_wallets: Set[str] = set()  # Currently active wallets
        
    async def initialize(self):
        """Initialize the wallet tracker"""
        self.session = aiohttp.ClientSession()
        await self._load_wallets()
        await self._categorize_wallets()
        
    async def _load_wallets(self):
        """Load wallet addresses from configuration"""
        try:
            with open(self.config['wallet_file'], 'r') as f:
                wallet_data = json.load(f)
                for wallet in wallet_data:
                    self.wallets[wallet['address']] = {
                        'address': wallet['address'],
                        'category': wallet.get('category', 'default'),
                        'priority': wallet.get('priority', 1),
                        'last_activity': 0,
                        'activity_score': 0
                    }
            logger.info(f"Loaded {len(self.wallets)} wallets")
        except Exception as e:
            logger.error(f"Error loading wallets: {e}")
            
    async def _categorize_wallets(self):
        """Categorize wallets based on activity level"""
        for address, data in self.wallets.items():
            if data['activity_score'] > 0.8:
                self.wallet_groups['high'].add(address)
            elif data['activity_score'] > 0.5:
                self.wallet_groups['medium'].add(address)
            else:
                self.wallet_groups['low'].add(address)
                
    async def track_wallets(self):
        """Main tracking loop"""
        while True:
            try:
                # Update high priority wallets more frequently
                await self._update_wallet_group('high', interval=30)
                
                # Update medium priority wallets
                await self._update_wallet_group('medium', interval=60)
                
                # Update low priority wallets less frequently
                await self._update_wallet_group('low', interval=300)
                
                # Clean up cache
                await self._cleanup_cache()
                
                # Sleep for a short interval
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
                await asyncio.sleep(10)
                
    async def _update_wallet_group(self, group: str, interval: int):
        """Update a group of wallets"""
        current_time = time.time()
        wallets_to_update = [
            addr for addr in self.wallet_groups[group]
            if current_time - self.last_update.get(addr, 0) >= interval
        ]
        
        if not wallets_to_update:
            return
            
        # Process wallets in batches
        for i in range(0, len(wallets_to_update), self.batch_size):
            batch = wallets_to_update[i:i + self.batch_size]
            await self._process_wallet_batch(batch)
            
    async def _process_wallet_batch(self, addresses: List[str]):
        """Process a batch of wallet addresses"""
        try:
            # Create tasks for parallel processing
            tasks = [
                self._fetch_wallet_data(addr)
                for addr in addresses
            ]
            
            # Execute tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for addr, result in zip(addresses, results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching data for {addr}: {result}")
                    continue
                    
                if result:
                    await self._update_wallet_activity(addr, result)
                    
        except Exception as e:
            logger.error(f"Error processing wallet batch: {e}")
            
    async def _fetch_wallet_data(self, address: str) -> Dict:
        """Fetch wallet data from Solana RPC"""
        try:
            # Check cache first
            if address in self.activity_cache:
                cache_data = self.activity_cache[address]
                if time.time() - cache_data['timestamp'] < self.cache_ttl:
                    return cache_data['data']
                    
            # Fetch new data
            async with self.session.get(
                f"{self.config['solana_rpc_url']}/account/{address}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache the result
                    self.activity_cache[address] = {
                        'data': data,
                        'timestamp': time.time()
                    }
                    
                    return data
                    
            return None
            
        except Exception as e:
            logger.error(f"Error fetching wallet data: {e}")
            return None
            
    async def _update_wallet_activity(self, address: str, data: Dict):
        """Update wallet activity information"""
        try:
            # Calculate activity score
            activity_score = self._calculate_activity_score(data)
            
            # Update wallet data
            self.wallets[address].update({
                'last_activity': time.time(),
                'activity_score': activity_score,
                'last_data': data
            })
            
            # Update last update time
            self.last_update[address] = time.time()
            
            # Check if wallet should change category
            await self._update_wallet_category(address, activity_score)
            
        except Exception as e:
            logger.error(f"Error updating wallet activity: {e}")
            
    def _calculate_activity_score(self, data: Dict) -> float:
        """Calculate activity score based on wallet data"""
        try:
            score = 0.0
            
            # Check for recent transactions
            if 'recentTransactions' in data:
                tx_count = len(data['recentTransactions'])
                score += min(tx_count / 10, 0.3)  # Up to 0.3 points for transaction frequency
                
                # Check for large transactions
                large_tx = any(tx.get('amount', 0) > 100 for tx in data['recentTransactions'])
                if large_tx:
                    score += 0.2
                    
            # Check balance changes
            if 'balanceChanges' in data:
                changes = data['balanceChanges']
                if changes:
                    avg_change = sum(abs(c) for c in changes) / len(changes)
                    score += min(avg_change / 1000, 0.2)  # Up to 0.2 points for balance volatility
                    
            # Check token holdings
            if 'tokenHoldings' in data:
                holdings = data['tokenHoldings']
                if holdings:
                    # Score based on number of different tokens
                    score += min(len(holdings) / 20, 0.2)  # Up to 0.2 points for token diversity
                    
                    # Score based on total value of holdings
                    total_value = sum(h.get('value', 0) for h in holdings)
                    score += min(total_value / 10000, 0.2)  # Up to 0.2 points for portfolio value
                    
            # Check program interactions
            if 'programInteractions' in data:
                interactions = data['programInteractions']
                if interactions:
                    # Score based on interaction with known programs
                    known_programs = {
                        'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA',  # Token program
                        '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8',  # Raydium
                        '9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin'   # Serum
                    }
                    program_score = sum(1 for p in interactions if p in known_programs) / len(known_programs)
                    score += program_score * 0.1  # Up to 0.1 points for program interactions
                    
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating activity score: {e}")
            return 0.0
            
    async def _update_wallet_category(self, address: str, activity_score: float):
        """Update wallet category based on activity score"""
        try:
            # Remove from all groups
            for group in self.wallet_groups.values():
                group.discard(address)
                
            # Add to appropriate group
            if activity_score > 0.8:
                self.wallet_groups['high'].add(address)
            elif activity_score > 0.5:
                self.wallet_groups['medium'].add(address)
            else:
                self.wallet_groups['low'].add(address)
                
        except Exception as e:
            logger.error(f"Error updating wallet category: {e}")
            
    async def _cleanup_cache(self):
        """Clean up expired cache entries"""
        try:
            current_time = time.time()
            expired_keys = [
                addr for addr, data in self.activity_cache.items()
                if current_time - data['timestamp'] > self.cache_ttl
            ]
            
            for key in expired_keys:
                del self.activity_cache[key]
                
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            
    async def get_active_wallets(self) -> List[Dict]:
        """Get currently active wallets"""
        try:
            current_time = time.time()
            active_wallets = []
            
            for address, data in self.wallets.items():
                if current_time - data['last_activity'] < 3600:  # Active in last hour
                    active_wallets.append({
                        'address': address,
                        'category': data['category'],
                        'activity_score': data['activity_score'],
                        'last_activity': data['last_activity']
                    })
                    
            return active_wallets
            
        except Exception as e:
            logger.error(f"Error getting active wallets: {e}")
            return []
            
    async def get_wallet_activity(self, address: str) -> Dict:
        """Get activity data for a specific wallet"""
        try:
            if address in self.wallets:
                return {
                    'address': address,
                    'data': self.wallets[address],
                    'cached_data': self.activity_cache.get(address)
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting wallet activity: {e}")
            return None
            
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close() 