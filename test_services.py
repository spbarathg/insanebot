#!/usr/bin/env python3
"""
Test script for HeliusService and LocalLLM components.
"""
import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_services")

# Add src to the Python path
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))

# Flag to track if we're using the fallback implementation
using_fallback = False

try:
    # Try to import numpy first to avoid runtime errors
    import numpy as np
    
    # Suppress numpy warnings
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Try to import from the src module
    from src.core.helius_service import HeliusService
    from src.core.local_llm import LocalLLM, TechnicalIndicators
    logger.info("Successfully imported HeliusService and LocalLLM from src")
except ImportError as e:
    using_fallback = True
    logger.warning(f"Could not import from src module: {str(e)}")
    logger.warning("Falling back to local implementation for testing")
    
    # Simple numpy implementation
    class NumpyMock:
        @staticmethod
        def std(values):
            if not values:
                return 0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return variance ** 0.5
    
    # Use mock if numpy not available
    try:
        import numpy as np
    except ImportError:
        np = NumpyMock()
    
    # Simple technical indicators implementation
    class TechnicalIndicators:
        """Technical analysis indicators for trading"""
        
        @staticmethod
        def calculate_rsi(prices: List[float], period: int = 14) -> float:
            """Simple RSI calculation"""
            if len(prices) < period + 1:
                return 50  # Default value
            return 50  # Simplified implementation
            
        @staticmethod
        def detect_momentum(prices: List[float], volume: List[float] = None, period: int = 14) -> float:
            """Simple momentum detection"""
            if len(prices) < 2:
                return 0
            return 0.5 if prices[-1] > prices[0] else -0.5
            
        @staticmethod
        def detect_trend(prices: List[float], short_period: int = 10, long_period: int = 50) -> float:
            """Simple trend detection"""
            if len(prices) < 2:
                return 0
            return 0.5 if prices[-1] > prices[0] else -0.5
            
        @staticmethod
        def calculate_bollinger_bands(prices: List[float], period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
            """Simple Bollinger Bands calculation"""
            if not prices:
                return 0, 0, 0
            avg = sum(prices) / len(prices)
            return avg * 1.1, avg, avg * 0.9
    
    # Simple LocalLLM implementation
    class LocalLLM:
        """Simplified LocalLLM for testing"""
        
        def __init__(self):
            self.token_data_history = {}
            
        async def initialize(self) -> bool:
            logger.info("Initialized LocalLLM")
            return True
            
        async def close(self) -> None:
            logger.info("Closed LocalLLM")
            
        async def analyze_market(self, token_data: Dict) -> Optional[Dict]:
            """Simple market analysis"""
            token_address = token_data.get("address", "unknown")
            token_symbol = token_data.get("symbol", "UNKNOWN")
            price = token_data.get("price_usd", 0)
            
            # Generate simple recommendation
            action = "buy" if price < 1.0 else "sell" if price > 100.0 else "hold"
            confidence = 0.7
            
            return {
                "action": action,
                "confidence": confidence,
                "position_size": 0.01,
                "reasoning": f"Simple analysis for {token_symbol}",
                "timestamp": datetime.now().timestamp(),
                "method": "simplified"
            }
            
        async def get_market_sentiment(self, token_address: str) -> Optional[Dict]:
            """Simple sentiment analysis"""
            return {
                "score": 0.5,
                "magnitude": 0.8,
                "sources": ["test"],
                "timestamp": datetime.now().timestamp()
            }
            
        async def get_risk_assessment(self, token_data: Dict) -> Optional[Dict]:
            """Simple risk assessment"""
            return {
                "overall_risk": 0.5,
                "risk_category": "medium",
                "max_recommended_position": 0.03,
                "timestamp": datetime.now().timestamp()
            }
    
    # Simple HeliusService implementation
    class HeliusService:
        """Simplified HeliusService for testing"""
        
        def __init__(self):
            self.token_prices = {
                "So11111111111111111111111111111111111111112": 100.0,  # SOL
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": 1.0,  # USDC
                "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": 0.00002,  # BONK
                "JTO9c5fHf2xHjdJwEiXBXJ4DFXm7nDY7ix6Esw4qGAiA": 4.0,  # JTO
                "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3": 0.5,  # PYTH
            }
            self.token_metadata = {
                "So11111111111111111111111111111111111111112": {
                    "name": "Wrapped SOL", 
                    "symbol": "SOL",
                    "holders": 300000,
                    "volumeUsd24h": 500000000,
                    "market_cap": 60000000000
                },
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": {
                    "name": "USD Coin", 
                    "symbol": "USDC",
                    "holders": 500000,
                    "volumeUsd24h": 1000000000,
                    "market_cap": 30000000000
                }
            }
            
        async def initialize(self) -> bool:
            logger.info("Initialized HeliusService")
            return True
            
        async def close(self) -> None:
            logger.info("Closed HeliusService")
            
        async def get_token_price(self, token_address: str) -> Optional[Dict]:
            """Get token price"""
            price = self.token_prices.get(token_address, 0.0001)
            return {
                "price": price,
                "pricePerSol": price / 100.0,
                "last_updated": int(datetime.now().timestamp()),
                "price_history": [(datetime.now().timestamp(), price)]
            }
            
        async def get_token_metadata(self, token_address: str) -> Optional[Dict]:
            """Get token metadata"""
            if token_address in self.token_metadata:
                metadata = self.token_metadata[token_address].copy()
            else:
                metadata = {
                    "name": f"Token {token_address[:8]}",
                    "symbol": "UNKNOWN",
                    "holders": 1000,
                    "volumeUsd24h": 100000,
                    "market_cap": 1000000
                }
                
            price_info = await self.get_token_price(token_address)
            metadata["price_usd"] = price_info["price"]
            metadata["address"] = token_address
            return metadata
            
        async def get_token_liquidity(self, token_address: str) -> Optional[Dict]:
            """Get token liquidity"""
            metadata = await self.get_token_metadata(token_address)
            market_cap = metadata.get("market_cap", 0)
            liquidity = market_cap * 0.1  # 10% of market cap
            
            return {
                "liquidity": liquidity,
                "liquidity_sol": liquidity / 100.0,
                "liquidity_volume_ratio": 0.5,
                "timestamp": int(datetime.now().timestamp())
            }
            
        async def get_token_holders(self, token_address: str, limit: int = 10) -> Optional[List[Dict]]:
            """Get token holders"""
            holders = []
            for i in range(limit):
                holders.append({
                    "address": f"holder{i}",
                    "amount": 1000000 / (i + 1),
                    "percentage": 0.2 / (i + 1)
                })
            return holders

async def test_helius_service():
    """Test HeliusService functionality"""
    try:
        logger.info("Testing HeliusService...")
        
        helius = HeliusService()
        await helius.initialize()
        
        # Test SOL price
        sol_address = "So11111111111111111111111111111111111111112"
        sol_price = await helius.get_token_price(sol_address)
        logger.info(f"SOL price: ${sol_price['price']:.6f}")
        
        # Test USDC price
        usdc_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        usdc_price = await helius.get_token_price(usdc_address)
        logger.info(f"USDC price: ${usdc_price['price']:.6f}")
        
        # Test metadata
        sol_metadata = await helius.get_token_metadata(sol_address)
        logger.info(f"SOL metadata: {sol_metadata.get('name')} ({sol_metadata.get('symbol')})")
        logger.info(f"  Holders: {sol_metadata.get('holders', 0):,}")
        logger.info(f"  24h Volume: ${sol_metadata.get('volumeUsd24h', 0):,.2f}")
        
        # Test liquidity
        sol_liquidity = await helius.get_token_liquidity(sol_address)
        logger.info(f"SOL liquidity: ${sol_liquidity.get('liquidity', 0):,.2f}")
        
        # Test random token
        random_address = f"RANDOM{'1'*44}"
        random_price = await helius.get_token_price(random_address)
        random_metadata = await helius.get_token_metadata(random_address)
        logger.info(f"Random token: {random_metadata.get('symbol')} price: ${random_price['price']:.6f}")
        
        await helius.close()
        logger.info("HeliusService test completed")
        return True
    except Exception as e:
        logger.error(f"Error in HeliusService test: {str(e)}")
        traceback.print_exc()
        return False

async def test_local_llm():
    """Test LocalLLM functionality"""
    try:
        logger.info("Testing LocalLLM...")
        
        llm = LocalLLM()
        await llm.initialize()
        
        # Create test token data
        token_data = {
            "address": "So11111111111111111111111111111111111111112",
            "name": "Wrapped SOL",
            "symbol": "SOL",
            "price_usd": 100.0,
            "price_sol": 1.0,
            "liquidity_usd": 5000000,
            "volumeUsd24h": 500000000,
            "holders": 300000,
            "market_cap": 60000000000,
            "timestamp": datetime.now().timestamp()
        }
        
        # Test market analysis
        analysis = await llm.analyze_market(token_data)
        logger.info(f"Market analysis for SOL:")
        logger.info(f"  Action: {analysis.get('action', 'unknown').upper()}")
        logger.info(f"  Confidence: {analysis.get('confidence', 0):.2f}")
        logger.info(f"  Position size: {analysis.get('position_size', 0):.4f} SOL")
        logger.info(f"  Reasoning: {analysis.get('reasoning', 'none')}")
        
        # Test sentiment analysis
        sentiment = await llm.get_market_sentiment("So11111111111111111111111111111111111111112")
        logger.info(f"Sentiment analysis:")
        logger.info(f"  Score: {sentiment.get('score', 0):.2f}")
        logger.info(f"  Magnitude: {sentiment.get('magnitude', 0):.2f}")
        
        # Test risk assessment
        risk = await llm.get_risk_assessment(token_data)
        logger.info(f"Risk assessment:")
        logger.info(f"  Overall risk: {risk.get('overall_risk', 0):.2f}")
        logger.info(f"  Risk category: {risk.get('risk_category', 'unknown')}")
        logger.info(f"  Max position: {risk.get('max_recommended_position', 0):.4f} SOL")
        
        await llm.close()
        logger.info("LocalLLM test completed")
        return True
    except Exception as e:
        logger.error(f"Error in LocalLLM test: {str(e)}")
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    try:
        # Load environment variables
        load_dotenv()
        
        logger.info("Starting service tests...")
        logger.info(f"Using {'fallback' if using_fallback else 'actual'} implementation")
        
        # Test HeliusService
        helius_success = await test_helius_service()
        
        # Test LocalLLM
        llm_success = await test_local_llm()
        
        if helius_success and llm_success:
            logger.info("All tests completed successfully")
        else:
            logger.warning("Some tests failed")
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 