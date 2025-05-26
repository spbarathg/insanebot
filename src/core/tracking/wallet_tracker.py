from typing import Dict, List, Optional
import asyncio
import aiohttp
from datetime import datetime, timedelta
import logging
from config.wallet_tracker_config import WALLET_TRACKER_CONFIG as WALLET_CONFIG

logger = logging.getLogger(__name__)

class WalletTracker:
    def __init__(self):
        self.config = WALLET_CONFIG
        self._wallets = {}
        self._activity_scores = {}
        self._last_update = {}
        
    async def track_wallets(self, wallet_addresses: List[str]):
        """Track multiple wallet addresses."""
        tasks = []
        for address in wallet_addresses:
            if self._should_update(address):
                tasks.append(self._update_wallet(address))
                
        if tasks:
            await asyncio.gather(*tasks)
            
    def _should_update(self, address: str) -> bool:
        """Determine if a wallet should be updated based on its activity level."""
        if address not in self._last_update:
            return True
            
        activity_level = self._get_activity_level(address)
        update_interval = self.config["update_intervals"][activity_level]
        
        cache_age = datetime.now() - self._last_update[address]
        return cache_age > timedelta(seconds=update_interval)
        
    def _get_activity_level(self, address: str) -> str:
        """Get the activity level of a wallet."""
        score = self._activity_scores.get(address, 0)
        
        if score >= self.config["activity_thresholds"]["high"]:
            return "high"
        elif score >= self.config["activity_thresholds"]["medium"]:
            return "medium"
        return "low"
        
    async def _update_wallet(self, address: str):
        """Update wallet data."""
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch wallet data
                wallet_data = await self._fetch_wallet_data(session, address)
                
                # Update activity score
                self._update_activity_score(address, wallet_data)
                
                # Update last update time
                self._last_update[address] = datetime.now()
                
        except Exception as e:
            logger.error(f"Error updating wallet {address}: {str(e)}")
            
    async def _fetch_wallet_data(self, session: aiohttp.ClientSession, address: str) -> Dict:
        """Fetch wallet data from the blockchain."""
        try:
            # Fetch recent transactions
            async with session.get(f"https://api.mainnet-beta.solana.com", json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [address, {"limit": 10}]
            }) as response:
                if response.status == 200:
                    data = await response.json()
                    transactions = data.get("result", [])
                    
                    # Fetch token balances
                    async with session.get(f"https://api.mainnet-beta.solana.com", json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "getTokenAccountsByOwner",
                        "params": [address, {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"}]
                    }) as balance_response:
                        if balance_response.status == 200:
                            balance_data = await balance_response.json()
                            balances = balance_data.get("result", {}).get("value", [])
                            
                            return {
                                "transactions": transactions,
                                "balances": balances,
                                "timestamp": datetime.now().isoformat()
                            }
                return {"transactions": [], "balances": [], "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error fetching wallet data: {str(e)}")
            return {"transactions": [], "balances": [], "timestamp": datetime.now().isoformat()}
        
    def _update_activity_score(self, address: str, wallet_data: Dict):
        """Update the activity score for a wallet."""
        try:
            # Calculate activity score based on:
            # 1. Number of recent transactions
            # 2. Total value of token balances
            # 3. Transaction frequency
            
            transactions = wallet_data.get("transactions", [])
            balances = wallet_data.get("balances", [])
            
            # Transaction score (0-1)
            tx_score = min(len(transactions) / 10, 1.0)
            
            # Balance score (0-1)
            total_balance = sum(float(b.get("account", {}).get("data", {}).get("parsed", {}).get("info", {}).get("tokenAmount", {}).get("uiAmount", 0)) for b in balances)
            balance_score = min(total_balance / 1000, 1.0)  # Normalize to 1000 tokens
            
            # Combine scores with weights
            self._activity_scores[address] = (tx_score * 0.6) + (balance_score * 0.4)
            
        except Exception as e:
            logger.error(f"Error updating activity score: {str(e)}")
            self._activity_scores[address] = 0.0 