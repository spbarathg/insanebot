"""
Local LLM interface for trading bot.
"""
import logging
import json
import time
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import os

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Technical analysis indicators for trading"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """
        Calculate Relative Strength Index
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        if len(prices) < period + 1:
            return 50  # Default value if not enough data
            
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Split gains and losses
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        # Calculate average gain and loss
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100  # Prevent division by zero
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[float, float, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        MACD Line = Fast EMA - Slow EMA
        Signal Line = EMA of MACD Line
        Histogram = MACD Line - Signal Line
        """
        if len(prices) < slow_period + signal_period:
            return 0, 0, 0  # Default values if not enough data
            
        # Calculate EMAs
        fast_ema = TechnicalIndicators.calculate_ema(prices, fast_period)
        slow_ema = TechnicalIndicators.calculate_ema(prices, slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD line)
        # For simplicity, we'll use a simple moving average
        signal_line = sum(prices[-signal_period:]) / signal_period
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0  # Default to last price if not enough data
            
        # Simple moving average for initial EMA
        sma = sum(prices[-period:]) / period
        
        # Smoothing factor
        multiplier = 2 / (period + 1)
        
        # Calculate EMA
        ema = sma
        for price in prices[-period:]:
            ema = (price - ema) * multiplier + ema
            
        return ema
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands
        
        Middle Band = SMA
        Upper Band = SMA + (num_std * std_dev)
        Lower Band = SMA - (num_std * std_dev)
        """
        if len(prices) < period:
            price = prices[-1] if prices else 0
            return price, price, price  # Default values if not enough data
            
        # Calculate middle band (SMA)
        middle_band = sum(prices[-period:]) / period
        
        # Calculate standard deviation
        std_dev = math.sqrt(sum((price - middle_band) ** 2 for price in prices[-period:]) / period)
        
        # Calculate upper and lower bands
        upper_band = middle_band + (num_std * std_dev)
        lower_band = middle_band - (num_std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def detect_momentum(prices: List[float], volume: List[float] = None, period: int = 14) -> float:
        """
        Detect momentum using multiple indicators
        Returns a score between -1 and 1
        """
        if len(prices) < period:
            return 0  # Neutral if not enough data
            
        # Calculate price change percentage
        price_change = (prices[-1] / prices[-period] - 1) * 100
        
        # Calculate RSI
        rsi = TechnicalIndicators.calculate_rsi(prices, period)
        
        # Calculate MACD
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(prices)
        
        # Combine indicators to create momentum score
        # Normalize values to -1 to 1 range
        rsi_score = (rsi - 50) / 50  # -1 to 1
        macd_score = 1 if histogram > 0 else -1
        price_change_score = min(max(price_change / 10, -1), 1)  # Cap at -1 to 1
        
        # Weight the different indicators
        momentum_score = (rsi_score * 0.4) + (macd_score * 0.3) + (price_change_score * 0.3)
        
        return momentum_score
    
    @staticmethod
    def detect_trend(prices: List[float], short_period: int = 10, long_period: int = 50) -> float:
        """
        Detect trend direction and strength
        Returns a score between -1 (strong downtrend) and 1 (strong uptrend)
        """
        if len(prices) < long_period:
            return 0  # Neutral if not enough data
            
        # Calculate short and long term moving averages
        short_ma = sum(prices[-short_period:]) / short_period
        long_ma = sum(prices[-long_period:]) / long_period
        
        # Calculate MA crossover
        ma_diff = short_ma - long_ma
        ma_ratio = ma_diff / long_ma
        
        # Calculate price direction over multiple timeframes
        short_direction = prices[-1] - prices[-short_period]
        medium_direction = prices[-1] - prices[-long_period//2] if len(prices) >= long_period//2 else 0
        long_direction = prices[-1] - prices[-long_period] if len(prices) >= long_period else 0
        
        # Normalize and combine
        short_score = min(max(short_direction / (prices[-short_period] * 0.1), -1), 1)
        medium_score = min(max(medium_direction / (prices[-long_period//2] * 0.2), -1), 1) if len(prices) >= long_period//2 else 0
        long_score = min(max(long_direction / (prices[-long_period] * 0.3), -1), 1) if len(prices) >= long_period else 0
        ma_score = min(max(ma_ratio * 10, -1), 1)
        
        # Weight the different timeframes
        trend_score = (short_score * 0.2) + (medium_score * 0.3) + (long_score * 0.3) + (ma_score * 0.2)
        
        return trend_score

class LocalLLM:
    """
    Local LLM interface for trading decision-making.
    This uses technical analysis and trading algorithms.
    """
    
    def __init__(self):
        """Initialize the Local LLM service."""
        self.ready = False
        self.model_loaded = False
        self.model = None
        self.training_data = []
        self.technical_indicators = TechnicalIndicators()
        self.trade_history = []
        self.token_data_history = {}
        self.risk_appetite = 0.7  # 0 to 1, higher means more risk-tolerant
        
    async def initialize(self) -> bool:
        """Initialize the Local LLM service."""
        try:
            logger.info("Initializing Local LLM service")
            self.ready = True
            logger.info("Local LLM service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Local LLM service: {str(e)}")
            return False
            
    async def close(self) -> None:
        """Close the Local LLM service."""
        logger.info("Local LLM service closed")
    
    def _update_token_history(self, token_address: str, token_data: Dict) -> None:
        """Update token history for technical analysis"""
        if token_address not in self.token_data_history:
            self.token_data_history[token_address] = []
        
        # Add new data point
        self.token_data_history[token_address].append({
            "timestamp": token_data.get("timestamp", time.time()),
            "price": token_data.get("price_usd", 0),
            "volume": token_data.get("volumeUsd24h", 0),
            "liquidity": token_data.get("liquidity_usd", 0)
        })
        
        # Keep only last 100 data points
        if len(self.token_data_history[token_address]) > 100:
            self.token_data_history[token_address] = self.token_data_history[token_address][-100:]
    
    def _get_token_price_history(self, token_address: str) -> List[float]:
        """Get price history for technical analysis"""
        if token_address not in self.token_data_history:
            return []
        
        return [point["price"] for point in self.token_data_history[token_address]]
    
    def _get_token_volume_history(self, token_address: str) -> List[float]:
        """Get volume history for technical analysis"""
        if token_address not in self.token_data_history:
            return []
        
        return [point["volume"] for point in self.token_data_history[token_address]]
    
    def _analyze_token_fundamentals(self, token_data: Dict) -> Dict:
        """Analyze token fundamentals and market data"""
        # Extract key metrics
        liquidity = token_data.get("liquidity_usd", 0)
        market_cap = token_data.get("market_cap", 0)
        volume_24h = token_data.get("volumeUsd24h", 0)
        holders = token_data.get("holders", 0)
        
        # Calculate key ratios
        volume_to_mcap = volume_24h / market_cap if market_cap > 0 else 0
        liquidity_to_mcap = liquidity / market_cap if market_cap > 0 else 0
        
        # Score different aspects (0 to 1)
        liquidity_score = min(liquidity / 1000000, 1) * 0.5 + min(liquidity_to_mcap * 10, 1) * 0.5
        volume_score = min(volume_24h / 1000000, 1) * 0.5 + min(volume_to_mcap * 5, 1) * 0.5
        holders_score = min(holders / 10000, 1)
        
        # Combine scores
        fundamental_score = (liquidity_score * 0.4) + (volume_score * 0.4) + (holders_score * 0.2)
        
        # Risk assessment
        risk_level = 1 - fundamental_score
        
        return {
            "fundamental_score": fundamental_score,
            "liquidity_score": liquidity_score,
            "volume_score": volume_score,
            "holders_score": holders_score,
            "risk_level": risk_level
        }
    
    def _determine_position_size(self, token_data: Dict, risk_level: float, confidence: float) -> float:
        """Determine position size based on risk and confidence"""
        # Base position size (0.01 to 0.1 SOL)
        base_position = 0.01
        
        # Scale based on token value (invest less in smaller cap tokens)
        market_cap = token_data.get("market_cap", 0)
        market_cap_factor = min(math.log10(max(market_cap, 10000)) / 9, 1)
        
        # Scale based on confidence and risk
        confidence_factor = confidence
        risk_factor = 1 - risk_level  # Higher risk = smaller position
        
        # Calculate position size
        position_size = base_position * (1 + 5 * self.risk_appetite * confidence_factor * risk_factor * market_cap_factor)
        
        # Cap position size
        return min(max(position_size, 0.005), 0.2)
    
    async def analyze_market(self, token_data: Dict) -> Optional[Dict]:
        """
        Analyze market data and provide trading recommendations.
        
        Uses technical analysis and market indicators.
        """
        try:
            # Update token history
            token_address = token_data.get("address", "unknown")
            self._update_token_history(token_address, token_data)
            
            # Get price history
            price_history = self._get_token_price_history(token_address)
            
            # Analyze token fundamentals first (always available)
            fundamentals = self._analyze_token_fundamentals(token_data)
            fundamental_score = fundamentals["fundamental_score"]
            risk_level = fundamentals["risk_level"]
            
            # Use fundamental analysis if not enough history (reduced from 5 to 2)
            if len(price_history) < 2:
                # Make decisions based on fundamentals and current market conditions
                liquidity = token_data.get("liquidity_usd", 0)
                volume = token_data.get("volumeUsd24h", 0)
                price = token_data.get("price_usd", 0)
                
                # Generate signal based on fundamental strength
                if fundamental_score > 0.6 and liquidity > 100000 and volume > 50000:
                    action = "buy"
                    confidence = 0.6 + (fundamental_score * 0.3)  # 0.6 to 0.9 range
                elif fundamental_score < 0.3 or liquidity < 50000:
                    action = "sell" if token_address not in ["So11111111111111111111111111111111111111112", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"] else "hold"
                    confidence = 0.5 + ((1 - fundamental_score) * 0.2)  # 0.5 to 0.7 range
                else:
                    action = "hold"
                    confidence = 0.45 + (fundamental_score * 0.1)  # 0.45 to 0.55 range
                
                # Add simulation mode variance for more interesting trading signals
                if os.getenv("SIMULATION_MODE", "True").lower() == "true":
                    # Create more dynamic signals in simulation mode
                    random.seed(int(time.time() / 600) + hash(token_address))  # Change every 10 minutes
                    simulation_factor = random.uniform(0.8, 1.2)
                    
                    # Occasionally generate strong buy/sell signals for major tokens
                    if token_address in ["So11111111111111111111111111111111111111112", "DezXAZ8zDXzK82sYdDbGNQYJuUFzJPCL7yRNmEHYYAjK"]:  # SOL, BONK
                        signal_roll = random.random()
                        if signal_roll < 0.15:  # 15% chance for strong signal
                            action = "buy" if signal_roll < 0.08 else "sell"
                            confidence = random.uniform(0.72, 0.85)
                        elif signal_roll < 0.3:  # Another 15% for moderate signal
                            confidence = min(confidence * simulation_factor, 0.69)
                    
                    # Add some variance to confidence for other tokens
                    confidence = min(max(confidence * simulation_factor, 0.35), 0.85)
                
                position_size = self._determine_position_size(token_data, risk_level, confidence)
                reasoning = f"Fundamental analysis: {fundamentals['liquidity_score']:.2f} liquidity, {fundamentals['volume_score']:.2f} volume, {fundamentals['holders_score']:.2f} holders (no price history yet)"
                
                return {
                    "action": action,
                    "confidence": confidence,
                    "position_size": position_size,
                    "reasoning": reasoning,
                    "timestamp": time.time(),
                    "method": "fundamental-only"
                }
            
            # Technical analysis
            momentum_score = TechnicalIndicators.detect_momentum(price_history)
            trend_score = TechnicalIndicators.detect_trend(price_history)
            
            # RSI analysis
            rsi = TechnicalIndicators.calculate_rsi(price_history)
            rsi_signal = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
            
            # Bollinger Bands
            upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(price_history)
            price = price_history[-1]
            bb_signal = "oversold" if price < lower else "overbought" if price > upper else "neutral"
            
            # Combined technical signal (-1 to 1)
            technical_score = (momentum_score * 0.4) + (trend_score * 0.6)
            
            # Combined overall score (-1 to 1, negative = sell, positive = buy)
            overall_score = (technical_score * 0.7) + ((fundamental_score * 2 - 1) * 0.3)
            
            # Determine action
            if overall_score > 0.3:
                action = "buy"
                confidence = min(abs(overall_score) + 0.3, 1)
            elif overall_score < -0.3:
                action = "sell"
                confidence = min(abs(overall_score) + 0.3, 1)
            else:
                action = "hold"
                # Make hold signals more dynamic based on market conditions
                price_change = 0
                if len(price_history) >= 2:
                    price_change = (price_history[-1] - price_history[-2]) / price_history[-2]
                
                # Adjust confidence based on trend strength and fundamentals
                trend_strength = abs(technical_score)
                fundamental_strength = abs(fundamental_score - 0.5) * 2  # Convert 0-1 to 0-1 where 0.5 becomes 0
                
                # Dynamic confidence: 0.4 to 0.65 range
                base_confidence = 0.4
                confidence_adjustment = (trend_strength * 0.1) + (fundamental_strength * 0.1) + (abs(price_change) * 50 * 0.05)
                confidence = min(base_confidence + confidence_adjustment, 0.65)
            
            # Determine position size
            position_size = self._determine_position_size(token_data, risk_level, confidence)
            
            # Generate reasoning
            technical_direction = "bullish" if technical_score > 0.2 else "bearish" if technical_score < -0.2 else "neutral"
            fundamental_quality = "strong" if fundamental_score > 0.7 else "weak" if fundamental_score < 0.3 else "average"
            
            reasoning = (
                f"Technical: {technical_direction} (momentum: {momentum_score:.2f}, trend: {trend_score:.2f}, RSI: {rsi:.1f}), "
                f"Fundamentals: {fundamental_quality} (score: {fundamental_score:.2f}, risk: {risk_level:.2f})"
            )
            
            return {
                "action": action,
                "confidence": confidence,
                "position_size": position_size,
                "reasoning": reasoning,
                "timestamp": time.time(),
                "method": "technical-fundamental",
                "metrics": {
                    "technical_score": technical_score,
                    "fundamental_score": fundamental_score,
                    "overall_score": overall_score,
                    "rsi": rsi,
                    "trend": trend_score,
                    "momentum": momentum_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market: {str(e)}")
            # Fallback to simple analysis
            return {
                "action": "hold",
                "confidence": 0.5,
                "position_size": 0.01,
                "reasoning": f"Analysis error: {str(e)}",
                "timestamp": time.time(),
                "method": "fallback"
            }
            
    async def get_market_sentiment(self, token_address: str) -> Optional[Dict]:
        """
        Get market sentiment for a token.
        
        Simulates social media and news sentiment analysis.
        """
        try:
            # In a real implementation, we would analyze social media, news, etc.
            # Here we generate simulated sentiment based on token address
            random.seed(token_address + str(int(time.time() / 3600)))  # Change hourly
            
            # Generate base sentiment (-1 to 1)
            base_sentiment = random.uniform(-0.7, 0.7)
            
            # Generate magnitude (0 to 1)
            magnitude = random.uniform(0.3, 1.0)
            
            # More popular tokens (like SOL) have more sources
            num_sources = 5
            if token_address == "So11111111111111111111111111111111111111112":  # SOL
                num_sources = 20
                # Bias sentiment positively for SOL
                base_sentiment = (base_sentiment + 0.5) / 1.5
            elif token_address == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v":  # USDC
                num_sources = 15
                # Neutral sentiment for stablecoins
                base_sentiment = base_sentiment * 0.3
            
            # Generate sources
            sources = []
            source_types = ["twitter", "telegram", "discord", "reddit", "news"]
            for i in range(num_sources):
                source_sentiment = base_sentiment + random.uniform(-0.3, 0.3)
                source_sentiment = max(min(source_sentiment, 1.0), -1.0)  # Clamp to -1 to 1
                
                sources.append({
                    "type": random.choice(source_types),
                    "sentiment": source_sentiment,
                    "timestamp": time.time() - random.uniform(0, 86400)  # Last 24 hours
                })
            
            # Calculate overall sentiment
            overall_sentiment = sum(s["sentiment"] for s in sources) / len(sources)
            
            return {
                "score": overall_sentiment,
                "magnitude": magnitude,
                "sources": [s["type"] for s in sources],
                "positive_sources": len([s for s in sources if s["sentiment"] > 0.2]),
                "negative_sources": len([s for s in sources if s["sentiment"] < -0.2]),
                "neutral_sources": len([s for s in sources if -0.2 <= s["sentiment"] <= 0.2]),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error getting market sentiment: {str(e)}")
            return None
            
    async def get_risk_assessment(self, token_data: Dict) -> Optional[Dict]:
        """
        Assess the risk level of a token.
        """
        try:
            # Extract key metrics
            token_address = token_data.get("address", "unknown")
            market_cap = token_data.get("market_cap", 0)
            liquidity = token_data.get("liquidity_usd", 0)
            volume = token_data.get("volumeUsd24h", 0)
            holders = token_data.get("holders", 0)
            price = token_data.get("price_usd", 0)
            
            # Calculate risk factors (0 to 1, higher is riskier)
            
            # Market cap risk (lower market cap = higher risk)
            mcap_risk = 1.0 - min(market_cap / 1e9, 1.0)
            
            # Liquidity risk (lower liquidity = higher risk)
            liquidity_risk = 1.0 - min(liquidity / 1e6, 1.0)
            
            # Liquidity/Market Cap ratio risk
            liq_mcap_ratio = liquidity / market_cap if market_cap > 0 else 0
            liq_mcap_risk = 1.0 - min(liq_mcap_ratio * 10, 1.0)
            
            # Volume risk (lower volume = higher risk)
            volume_risk = 1.0 - min(volume / 1e6, 1.0)
            
            # Holder concentration risk
            holder_risk = 1.0 - min(holders / 10000, 1.0)
            
            # Price history volatility risk
            price_history = self._get_token_price_history(token_address)
            if len(price_history) > 5:
                # Calculate price volatility
                returns = [price_history[i] / price_history[i-1] - 1 for i in range(1, len(price_history))]
                volatility = np.std(returns) if len(returns) > 0 else 0
                volatility_risk = min(volatility * 10, 1.0)
            else:
                volatility_risk = 0.5  # Default if not enough history
            
            # Token age risk (not implemented in simulation, default to medium)
            age_risk = 0.5
            
            # Overall risk score (weighted average)
            overall_risk = (
                mcap_risk * 0.2 +
                liquidity_risk * 0.2 +
                liq_mcap_risk * 0.15 +
                volume_risk * 0.15 +
                holder_risk * 0.1 +
                volatility_risk * 0.1 +
                age_risk * 0.1
            )
            
            # Risk categorization
            risk_category = "extreme" if overall_risk > 0.8 else "high" if overall_risk > 0.6 else "medium" if overall_risk > 0.4 else "low"
            
            return {
                "overall_risk": overall_risk,
                "risk_category": risk_category,
                "risk_factors": {
                    "market_cap_risk": mcap_risk,
                    "liquidity_risk": liquidity_risk,
                    "liquidity_mcap_ratio_risk": liq_mcap_risk,
                    "volume_risk": volume_risk,
                    "holder_risk": holder_risk,
                    "volatility_risk": volatility_risk,
                    "age_risk": age_risk
                },
                "max_recommended_position": 0.05 * (1 - overall_risk) + 0.01,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            return None 