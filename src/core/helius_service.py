"""
Helius API service for Solana token data and transactions (simulated).
"""
import logging
from typing import Dict, List, Optional
import time
import random
import math
from datetime import datetime

logger = logging.getLogger(__name__)

class HeliusService:
    """
    Simplified Helius service for simulating Solana token data.
    """
    
    def __init__(self):
        """Initialize the Helius service."""
        self.api_key = "simulated_api_key"
        self.api_url = "https://api.helius.xyz/v0"
        # Storage for price history to create continuous price movements
        self._price_history = {}
        # Real token data for popular tokens
        self.real_tokens = {
            # SOL
            "So11111111111111111111111111111111111111112": {
                "name": "Wrapped SOL",
                "symbol": "SOL",
                "price_usd": 144.75,
                "market_cap": 65000000000,
                "volatility": 0.04,  # 4% daily volatility
                "holders": 312000,
                "trend": 0.3  # Slight upward trend
            },
            # USDC
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": {
                "name": "USD Coin",
                "symbol": "USDC",
                "price_usd": 1.0,
                "market_cap": 35000000000,
                "volatility": 0.001,  # Very stable
                "holders": 560000,
                "trend": 0.0
            },
            # BONK
            "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": {
                "name": "Bonk",
                "symbol": "BONK",
                "price_usd": 0.00002823,
                "market_cap": 1800000000,
                "volatility": 0.12,  # High volatility
                "holders": 150000,
                "trend": 0.1
            },
            # JTO
            "JTO9c5fHf2xHjdJwEiXBXJ4DFXm7nDY7ix6Esw4qGAiA": {
                "name": "Jito",
                "symbol": "JTO",
                "price_usd": 4.23,
                "market_cap": 487000000,
                "volatility": 0.09,
                "holders": 42000,
                "trend": -0.15
            },
            # PYTH
            "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3": {
                "name": "Pyth Network",
                "symbol": "PYTH",
                "price_usd": 0.58,
                "market_cap": 1150000000,
                "volatility": 0.06,
                "holders": 88000,
                "trend": 0.05
            },
        }
        # Initialize price history with starting prices
        for token_address, data in self.real_tokens.items():
            self._price_history[token_address] = {
                "last_price": data["price_usd"],
                "last_update": time.time(),
                "data_points": [(time.time(), data["price_usd"])]
            }
        
    async def initialize(self) -> bool:
        """Initialize the Helius service."""
        try:
            logger.info("Helius service initialized successfully (simulation mode)")
            logger.info(f"Loaded data for {len(self.real_tokens)} real tokens")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Helius service: {str(e)}")
            return False
            
    async def close(self) -> None:
        """Close the Helius service."""
        logger.info("Helius service closed")
            
    async def get_token_price(self, token_address: str) -> Optional[Dict]:
        """Get token price from Helius (simulated with realistic price movements)."""
        try:
            price = 0
            price_sol = 0
            
            # Use realistic data for known tokens with price movement simulation
            if token_address in self.real_tokens:
                token_data = self.real_tokens[token_address]
                price_history = self._price_history[token_address]
                
                # Calculate time since last update
                now = time.time()
                time_diff = now - price_history["last_update"]
                
                # Only update price if some time has passed (avoid repeated calls giving different prices)
                if time_diff > 5:  # Update price every 5 seconds of simulation time
                    # Use random walk with drift for price movement
                    volatility = token_data["volatility"] * math.sqrt(time_diff / 86400)  # Scale volatility by time
                    trend_factor = token_data["trend"] * (time_diff / 86400)  # Scale trend by time
                    
                    # Generate random component with normal distribution
                    random_component = random.normalvariate(0, 1) * volatility
                    
                    # Calculate new price with trend and random component
                    price = price_history["last_price"] * (1 + trend_factor + random_component)
                    
                    # Update price history
                    price_history["last_price"] = price
                    price_history["last_update"] = now
                    price_history["data_points"].append((now, price))
                    
                    # Keep only last 100 data points
                    if len(price_history["data_points"]) > 100:
                        price_history["data_points"] = price_history["data_points"][-100:]
                else:
                    # Use last price if not enough time has passed
                    price = price_history["last_price"]
                
                # Calculate SOL price
                sol_price = self._price_history["So11111111111111111111111111111111111111112"]["last_price"]
                price_sol = price / sol_price if sol_price > 0 else 0
            else:
                # For unknown tokens, generate random memecoin prices
                # Use consistent prices for the same token address
                random.seed(token_address)
                base_price = random.uniform(0.00000001, 0.01)
                
                # If we have price history, use that with some random movement
                if token_address in self._price_history:
                    price_history = self._price_history[token_address]
                    
                    # Calculate time since last update
                    now = time.time()
                    time_diff = now - price_history["last_update"]
                    
                    # Only update price if some time has passed
                    if time_diff > 5:
                        # High volatility for unknown tokens
                        volatility = 0.15 * math.sqrt(time_diff / 86400)
                        
                        # Random trend based on token address but changes over time
                        hash_value = sum(ord(c) for c in token_address)
                        day_of_year = datetime.now().timetuple().tm_yday
                        trend_seed = (hash_value + day_of_year) % 100
                        trend_factor = (trend_seed / 100 - 0.5) * 0.1 * (time_diff / 86400)
                        
                        # Generate random component
                        random_component = random.normalvariate(0, 1) * volatility
                        
                        # Calculate new price
                        price = price_history["last_price"] * (1 + trend_factor + random_component)
                        
                        # Update price history
                        price_history["last_price"] = price
                        price_history["last_update"] = now
                        price_history["data_points"].append((now, price))
                        
                        # Keep only last 100 data points
                        if len(price_history["data_points"]) > 100:
                            price_history["data_points"] = price_history["data_points"][-100:]
                    else:
                        # Use last price if not enough time has passed
                        price = price_history["last_price"]
                else:
                    # Initialize price history for new token
                    price = base_price
                    self._price_history[token_address] = {
                        "last_price": price,
                        "last_update": time.time(),
                        "data_points": [(time.time(), price)]
                    }
                
                # Calculate SOL price
                sol_price = self._price_history["So11111111111111111111111111111111111111112"]["last_price"]
                price_sol = price / sol_price if sol_price > 0 else 0
            
            return {
                "price": price,
                "pricePerSol": price_sol,
                "last_updated": int(time.time()),
                "price_history": self._price_history[token_address]["data_points"][-10:]  # Last 10 data points
            }
        except Exception as e:
            logger.error(f"Error getting token price: {str(e)}")
            return None
            
    async def get_token_metadata(self, token_address: str) -> Optional[Dict]:
        """Get token metadata from Helius (simulated with realistic data)."""
        try:
            # Use realistic data for known tokens
            if token_address in self.real_tokens:
                token_data = self.real_tokens[token_address]
                price = self._price_history[token_address]["last_price"]
                
                # Calculate volume based on market cap and price volatility
                daily_volume = token_data["market_cap"] * (0.05 + token_data["volatility"] * 2)
                
                return {
                    "name": token_data["name"],
                    "symbol": token_data["symbol"],
                    "address": token_address,
                    "decimals": 9 if token_data["symbol"] != "USDC" else 6,
                    "price_usd": price,
                    "market_cap": token_data["market_cap"],
                    "volumeUsd24h": daily_volume,
                    "holders": token_data["holders"],
                    "lastUpdatedAt": int(time.time())
                }
            else:
                # For unknown tokens, generate plausible memecoin data
                # Use consistent data for the same token address
                random.seed(token_address)
                
                # Generate random name for memecoins
                prefixes = ["Moon", "Doge", "Shib", "Pepe", "Ape", "Baby", "Safe", "Elon", "Based", "Chad", 
                           "Floki", "Degen", "Frog", "Magic", "Pixel", "Cyber", "Meta", "Space", "Pump"]
                suffixes = ["Inu", "Moon", "Rocket", "Elon", "Coin", "Cash", "Swap", "Finance", "Token", "AI", 
                           "DAO", "Verse", "Doge", "Chain", "Labs", "Games", "Protocol", "Network"]
                token_name = f"{random.choice(prefixes)}{random.choice(suffixes)}"
                token_symbol = "".join([c for c in token_name if c.isupper()])
                if not token_symbol:
                    token_symbol = token_name[:4].upper()
                
                # Get price from price history
                price = self._price_history[token_address]["last_price"] if token_address in self._price_history else 0.0001
                
                # Generate plausible market cap (smaller for random tokens)
                market_cap = price * random.randint(10000000, 1000000000)
                
                # Realistic volume (typically 1-20% of market cap for small tokens)
                volume_percentage = random.uniform(0.01, 0.2)
                volume = market_cap * volume_percentage
                
                # Holders (typically correlated with market cap)
                holders_base = int(math.sqrt(market_cap) / 10)
                holders_base = max(holders_base, 100)  # Ensure minimum base
                min_holders = max(100, holders_base // 2)
                max_holders = max(holders_base * 2, min_holders + 1)  # Ensure max > min
                holders = random.randint(min_holders, max_holders)
                
                return {
                    "name": token_name,
                    "symbol": token_symbol,
                    "address": token_address,
                    "decimals": 9,
                    "price_usd": price,
                    "market_cap": market_cap,
                    "volumeUsd24h": volume,
                    "holders": holders,
                    "lastUpdatedAt": int(time.time())
                }
        except Exception as e:
            logger.error(f"Error getting token metadata: {str(e)}")
            return None
            
    async def get_token_balances(self, wallet_address: str) -> Optional[Dict]:
        """Get token balances for a wallet (simulated with realistic holdings)."""
        try:
            # Generate consistent balances for the same wallet
            random.seed(wallet_address)
            
            # Base SOL balance (between 0.1 and 50 SOL)
            sol_balance = random.uniform(0.1, 50)
            sol_balance_lamports = int(sol_balance * 1e9)
            
            # Generate list of tokens
            tokens = []
            
            # Add SOL
            tokens.append({
                "mint": "So11111111111111111111111111111111111111112",
                "amount": sol_balance_lamports,
                "decimals": 9,
                "uiAmount": sol_balance
            })
            
            # Add USDC (correlated with SOL balance)
            usdc_balance = sol_balance * random.uniform(10, 1000)
            tokens.append({
                "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "amount": int(usdc_balance * 1e6),
                "decimals": 6,
                "uiAmount": usdc_balance
            })
            
            # Add some popular tokens
            popular_tokens = list(self.real_tokens.keys())[2:]  # Skip SOL and USDC
            for token in random.sample(popular_tokens, min(2, len(popular_tokens))):
                if random.random() < 0.7:  # 70% chance to have each token
                    token_data = self.real_tokens[token]
                    # Token amount inversely related to price
                    amount = random.uniform(10, 10000) / token_data["price_usd"]
                    tokens.append({
                        "mint": token,
                        "amount": int(amount * 1e9),
                        "decimals": 9,
                        "uiAmount": amount
                    })
            
            # Add some random memecoins
            for _ in range(random.randint(1, 5)):
                # Generate random token address
                random_token = f"RAND{random.randint(10000, 99999)}111111111111111111111111111"
                # Get a random amount (higher for low-value tokens)
                token_price = 0.0001 * random.random()
                amount = random.uniform(100, 1000000) * (0.0001 / max(token_price, 0.0000001))
                tokens.append({
                    "mint": random_token,
                    "amount": int(amount * 1e9),
                    "decimals": 9,
                    "uiAmount": amount
                })
            
            return {
                "tokens": tokens,
                "nativeBalance": sol_balance_lamports
            }
        except Exception as e:
            logger.error(f"Error getting token balances: {str(e)}")
            return None
            
    async def get_token_holders(self, token_address: str, limit: int = 100) -> Optional[List[Dict]]:
        """Get token holders data (simulated with realistic distribution)."""
        try:
            holders = []
            
            # Total supply simulation
            if token_address in self.real_tokens:
                token_data = self.real_tokens[token_address]
                price = token_data["price_usd"]
                market_cap = token_data["market_cap"]
                total_supply = market_cap / price if price > 0 else 1000000000000
                total_holders = token_data["holders"]
            else:
                # For unknown tokens
                random.seed(token_address)
                price = self._price_history[token_address]["last_price"] if token_address in self._price_history else 0.0001
                market_cap = price * random.randint(10000000, 1000000000)
                total_supply = market_cap / price if price > 0 else 1000000000000
                total_holders = random.randint(100, 50000)
            
            # Number of holders to generate (min of limit and total holders)
            num_holders = min(limit, total_holders)
            
            # Generate whale addresses (top holders)
            num_whales = max(1, int(num_holders * 0.05))  # Top 5% are whales
            
            # Whale distribution follows power law
            whale_supply_percentage = 0.6  # Whales hold 60% of supply
            whale_supply = total_supply * whale_supply_percentage
            
            for i in range(num_whales):
                # Power law distribution for whales
                percentage = (whale_supply_percentage / num_whales) * (1 + random.paretovariate(1.5) / 5)
                amount = total_supply * percentage
                
                holders.append({
                    "address": f"WHALE{i}111111111111111111111111111111111",
                    "amount": int(amount),
                    "percentage": percentage
                })
            
            # Regular holders
            regular_supply = total_supply * (1 - whale_supply_percentage)
            for i in range(num_whales, num_holders):
                # Exponential distribution for regular holders
                percentage = (1 - whale_supply_percentage) * (random.expovariate(10) / 5) / (num_holders - num_whales)
                amount = total_supply * percentage
                
                holders.append({
                    "address": f"HOLDER{i}111111111111111111111111111111111",
                    "amount": int(amount),
                    "percentage": percentage
                })
            
            # Sort by amount
            holders.sort(key=lambda x: x["amount"], reverse=True)
            
            # Normalize percentages
            total_percentage = sum(h["percentage"] for h in holders)
            for holder in holders:
                holder["percentage"] = holder["percentage"] / total_percentage
            
            return holders
        except Exception as e:
            logger.error(f"Error getting token holders: {str(e)}")
            return None
            
    async def get_token_liquidity(self, token_address: str) -> Optional[Dict]:
        """Get token liquidity data (simulated with realistic values)."""
        try:
            # Get token metadata for market cap and volume
            metadata = await self.get_token_metadata(token_address)
            
            if not metadata:
                return None
            
            market_cap = metadata.get("market_cap", 0)
            volume = metadata.get("volumeUsd24h", 0)
            
            # Liquidity is typically 2-20% of market cap depending on the token
            if token_address in self.real_tokens:
                # Known tokens have higher liquidity as a percentage of market cap
                liquidity_percentage = random.uniform(0.05, 0.2)
            else:
                # Unknown tokens have lower liquidity
                liquidity_percentage = random.uniform(0.01, 0.1)
            
            # Calculate liquidity
            liquidity = market_cap * liquidity_percentage
            
            # Add some randomness to liquidity (±10%)
            liquidity = liquidity * random.uniform(0.9, 1.1)
            
            # Calculate liquidity/volume ratio (health indicator)
            lv_ratio = liquidity / volume if volume > 0 else 0
            
            return {
                "liquidity": liquidity,
                "liquidity_sol": liquidity / self._price_history["So11111111111111111111111111111111111111112"]["last_price"],
                "liquidity_volume_ratio": lv_ratio,
                "timestamp": int(time.time())
            }
        except Exception as e:
            logger.error(f"Error getting token liquidity: {str(e)}")
            return None 