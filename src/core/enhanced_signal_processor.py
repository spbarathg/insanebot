"""
Enhanced Signal Processing Engine with Advanced Technical Analysis

Implements sophisticated signal aggregation with:
- Rebalanced signal weights for better profitability
- Technical analysis integration (volume profile, RSI, MACD)
- Confidence intervals and signal quality scoring
- Dynamic weight adjustment based on market conditions
- Multi-timeframe analysis for better entries/exits
"""

import asyncio
import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

# Mock talib with simple numpy implementations for Windows compatibility
try:
    import talib
except ImportError:
    # Create a simple mock talib using pure numpy to avoid pandas_ta compatibility issues
    import numpy as np
    
    class MockTalib:
        @staticmethod
        def RSI(prices, timeperiod=14):
            """Simple RSI implementation"""
            if len(prices) < timeperiod + 1:
                return np.full(len(prices), 50.0)
            
            prices = np.array(prices)
            deltas = np.diff(prices)
            gain = np.where(deltas > 0, deltas, 0)
            loss = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.convolve(gain, np.ones(timeperiod)/timeperiod, mode='valid')
            avg_loss = np.convolve(loss, np.ones(timeperiod)/timeperiod, mode='valid')
            
            rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            
            # Pad the beginning with default values
            result = np.full(len(prices), 50.0)
            result[timeperiod:] = rsi
            return result
        
        @staticmethod
        def MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9):
            """Simple MACD implementation"""
            if len(prices) < slowperiod:
                return np.zeros(len(prices)), np.zeros(len(prices)), np.zeros(len(prices))
            
            prices = np.array(prices)
            
            # Simple exponential moving averages
            fast_ema = np.zeros(len(prices))
            slow_ema = np.zeros(len(prices))
            
            # Initialize first values
            fast_ema[0] = prices[0]
            slow_ema[0] = prices[0]
            
            # Calculate EMAs
            fast_alpha = 2.0 / (fastperiod + 1)
            slow_alpha = 2.0 / (slowperiod + 1)
            
            for i in range(1, len(prices)):
                fast_ema[i] = fast_alpha * prices[i] + (1 - fast_alpha) * fast_ema[i-1]
                slow_ema[i] = slow_alpha * prices[i] + (1 - slow_alpha) * slow_ema[i-1]
            
            # MACD line
            macd = fast_ema - slow_ema
            
            # Signal line (EMA of MACD)
            signal = np.zeros(len(macd))
            signal[0] = macd[0]
            signal_alpha = 2.0 / (signalperiod + 1)
            
            for i in range(1, len(macd)):
                signal[i] = signal_alpha * macd[i] + (1 - signal_alpha) * signal[i-1]
            
            # Histogram
            histogram = macd - signal
            
            return macd, signal, histogram
        
        @staticmethod
        def BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
            """Simple Bollinger Bands implementation"""
            if len(prices) < timeperiod:
                price_avg = np.mean(prices)
                return (np.full(len(prices), price_avg * 1.02), 
                       np.array(prices), 
                       np.full(len(prices), price_avg * 0.98))
            
            prices = np.array(prices)
            
            # Calculate moving average and standard deviation
            middle = np.convolve(prices, np.ones(timeperiod)/timeperiod, mode='same')
            
            # Calculate rolling standard deviation
            std = np.zeros(len(prices))
            for i in range(timeperiod-1, len(prices)):
                window = prices[i-timeperiod+1:i+1]
                std[i] = np.std(window)
            
            # Bollinger bands
            upper = middle + (nbdevup * std)
            lower = middle - (nbdevdn * std)
            
            return upper, middle, lower
    
    talib = MockTalib()

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market conditions affecting signal weights"""
    BULL_MARKET = "bull"
    BEAR_MARKET = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"

class SignalQuality(Enum):
    """Signal quality levels"""
    EXCELLENT = "excellent"  # >90% confidence
    GOOD = "good"           # 70-90% confidence
    AVERAGE = "average"     # 50-70% confidence
    POOR = "poor"          # 30-50% confidence
    REJECT = "reject"      # <30% confidence

@dataclass
class TechnicalIndicators:
    """Technical analysis indicators"""
    rsi: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    bb_position: float = 0.0  # Bollinger band position (0-1)
    volume_ratio: float = 0.0  # Current vs average volume
    price_momentum: float = 0.0
    support_resistance_score: float = 0.0
    trend_strength: float = 0.0
    volatility_percentile: float = 0.0

@dataclass
class CompositeSignal:
    """Enhanced composite signal with confidence and technical analysis"""
    token_address: str
    signal_type: str  # BUY, SELL, HOLD
    composite_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    signal_quality: SignalQuality
    technical_indicators: TechnicalIndicators
    signal_components: Dict[str, float]
    reasoning: str
    risk_factors: List[str]
    entry_price: float
    target_prices: List[float]  # Multiple profit targets
    stop_loss: float
    position_size_pct: float
    urgency: str
    expires_at: float
    timestamp: float = field(default_factory=time.time)

class TechnicalAnalyzer:
    """Advanced technical analysis engine"""
    
    def __init__(self):
        self.price_history = defaultdict(deque)
        self.volume_history = defaultdict(deque)
        self.indicator_cache = {}
        self.cache_duration = 60  # 1 minute cache
        
    def update_market_data(self, token_address: str, price: float, volume: float, timestamp: float = None):
        """Update price and volume data for technical analysis"""
        if timestamp is None:
            timestamp = time.time()
            
        self.price_history[token_address].append({'price': price, 'timestamp': timestamp})
        self.volume_history[token_address].append({'volume': volume, 'timestamp': timestamp})
        
        # Keep last 200 data points (enough for technical indicators)
        if len(self.price_history[token_address]) > 200:
            self.price_history[token_address].popleft()
            self.volume_history[token_address].popleft()
    
    async def calculate_technical_indicators(self, token_address: str) -> TechnicalIndicators:
        """Calculate comprehensive technical indicators"""
        try:
            cache_key = f"{token_address}_{int(time.time() // self.cache_duration)}"
            if cache_key in self.indicator_cache:
                return self.indicator_cache[cache_key]
            
            prices = [p['price'] for p in list(self.price_history[token_address])]
            volumes = [v['volume'] for v in list(self.volume_history[token_address])]
            
            if len(prices) < 50:  # Need minimum data for reliable indicators
                return TechnicalIndicators()
            
            prices_array = np.array(prices, dtype=float)
            volumes_array = np.array(volumes, dtype=float)
            
            # Calculate RSI
            rsi = talib.RSI(prices_array, timeperiod=14)[-1] if len(prices_array) >= 14 else 50.0
            
            # Calculate MACD
            macd, macd_signal, macd_hist = talib.MACD(prices_array)
            macd_signal_value = macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else 0.0
            macd_histogram = macd_hist[-1] if len(macd_hist) > 0 and not np.isnan(macd_hist[-1]) else 0.0
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(prices_array)
            current_price = prices_array[-1]
            bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] != bb_lower[-1] else 0.5
            
            # Volume analysis
            avg_volume = np.mean(volumes_array[-20:]) if len(volumes_array) >= 20 else volumes_array[-1]
            current_volume = volumes_array[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price momentum (rate of change)
            if len(prices_array) >= 10:
                price_momentum = (prices_array[-1] / prices_array[-10] - 1) * 100
            else:
                price_momentum = 0.0
            
            # Support/Resistance analysis
            support_resistance_score = await self._calculate_support_resistance(prices_array)
            
            # Trend strength using ADX
            trend_strength = self._calculate_trend_strength(prices_array)
            
            # Volatility percentile
            volatility_percentile = self._calculate_volatility_percentile(prices_array)
            
            indicators = TechnicalIndicators(
                rsi=float(rsi),
                macd_signal=float(macd_signal_value),
                macd_histogram=float(macd_histogram),
                bb_position=float(bb_position),
                volume_ratio=float(volume_ratio),
                price_momentum=float(price_momentum),
                support_resistance_score=float(support_resistance_score),
                trend_strength=float(trend_strength),
                volatility_percentile=float(volatility_percentile)
            )
            
            self.indicator_cache[cache_key] = indicators
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return TechnicalIndicators()
    
    async def _calculate_support_resistance(self, prices: np.ndarray) -> float:
        """Calculate support/resistance score"""
        try:
            if len(prices) < 20:
                return 0.5
            
            current_price = prices[-1]
            recent_highs = []
            recent_lows = []
            
            # Find local highs and lows
            for i in range(5, len(prices) - 5):
                if all(prices[i] >= prices[j] for j in range(i-5, i+6) if j != i):
                    recent_highs.append(prices[i])
                if all(prices[i] <= prices[j] for j in range(i-5, i+6) if j != i):
                    recent_lows.append(prices[i])
            
            # Calculate proximity to support/resistance
            support_score = 0.5
            if recent_lows:
                nearest_support = max([low for low in recent_lows if low <= current_price], default=current_price * 0.9)
                support_distance = (current_price - nearest_support) / current_price
                support_score = min(1.0, support_distance * 10)  # Closer to support = lower score
            
            resistance_score = 0.5
            if recent_highs:
                nearest_resistance = min([high for high in recent_highs if high >= current_price], default=current_price * 1.1)
                resistance_distance = (nearest_resistance - current_price) / current_price
                resistance_score = min(1.0, resistance_distance * 10)  # Closer to resistance = higher score
            
            return (support_score + resistance_score) / 2
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return 0.5
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using simplified ADX logic"""
        try:
            if len(prices) < 20:
                return 0.5
            
            # Calculate directional movement
            highs = prices
            lows = prices  # Simplified - using same array
            
            plus_dm = np.maximum(highs[1:] - highs[:-1], 0)
            minus_dm = np.maximum(lows[:-1] - lows[1:], 0)
            
            # True range
            tr = np.maximum(highs[1:] - lows[1:], 
                           np.maximum(np.abs(highs[1:] - prices[:-1]),
                                    np.abs(lows[1:] - prices[:-1])))
            
            # Smooth the values
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            plus_di = np.mean(plus_dm[-14:]) / atr if atr > 0 else 0
            minus_di = np.mean(minus_dm[-14:]) / atr if atr > 0 else 0
            
            # ADX calculation
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
            return min(1.0, dx)
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.5
    
    def _calculate_volatility_percentile(self, prices: np.ndarray) -> float:
        """Calculate current volatility percentile"""
        try:
            if len(prices) < 30:
                return 0.5
            
            # Calculate returns
            returns = np.diff(np.log(prices))
            
            # Current volatility (last 10 periods)
            current_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
            
            # Historical volatility distribution (rolling 10-period windows)
            historical_vols = []
            for i in range(10, len(returns)):
                vol = np.std(returns[i-10:i])
                historical_vols.append(vol)
            
            if not historical_vols:
                return 0.5
            
            # Calculate percentile
            percentile = (sum(1 for vol in historical_vols if vol <= current_vol) / len(historical_vols))
            return percentile
            
        except Exception as e:
            logger.error(f"Error calculating volatility percentile: {e}")
            return 0.5

class EnhancedSignalProcessor:
    """
    Advanced signal processing engine with improved profitability focus
    
    Key improvements:
    - Rebalanced signal weights favoring technical analysis
    - Dynamic confidence scoring
    - Multi-target profit taking
    - Enhanced risk assessment
    """
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Improved signal weights based on audit recommendations
        self.base_signal_weights = {
            'technical_analysis': 0.30,    # NEW: Highest weight for technical signals
            'pump_fun': 0.25,             # Reduced from 35%
            'smart_money': 0.25,          # Keep existing
            'social_sentiment': 0.15,      # Reduced from 25%
            'ai_analysis': 0.05           # Minimal weight, focus on risk
        }
        
        # Dynamic weights based on market conditions
        self.market_regime = MarketRegime.SIDEWAYS
        self.current_weights = self.base_signal_weights.copy()
        
        # Signal quality thresholds
        self.quality_thresholds = {
            SignalQuality.EXCELLENT: 0.9,
            SignalQuality.GOOD: 0.7,
            SignalQuality.AVERAGE: 0.5,
            SignalQuality.POOR: 0.3
        }
        
        # Enhanced risk management
        self.max_position_size = 0.02  # 2% max position size
        self.min_confidence = 0.7      # Minimum 70% confidence
        self.max_slippage = 0.05       # Reduced to 5% from 15%
        
        logger.info("🧠 Enhanced Signal Processor initialized with improved weights")
    
    async def process_composite_signal(
        self,
        token_address: str,
        pump_fun_signal=None,
        smart_money_signal=None,
        social_sentiment_signal=None,
        ai_analysis_signal=None,
        market_data: Dict = None
    ) -> CompositeSignal:
        """Process all signals into enhanced composite decision"""
        try:
            # Update technical analysis data
            if market_data:
                price = market_data.get('price', 0)
                volume = market_data.get('volume', 0)
                if price > 0 and volume > 0:
                    self.technical_analyzer.update_market_data(token_address, price, volume)
            
            # Calculate technical indicators
            technical_indicators = await self.technical_analyzer.calculate_technical_indicators(token_address)
            
            # Generate technical analysis signal
            technical_signal = await self._generate_technical_signal(technical_indicators, market_data)
            
            # Adjust weights based on market regime
            await self._adjust_weights_for_market_regime()
            
            # Calculate weighted composite score
            signal_components = {}
            total_weight = 0
            weighted_score = 0
            
            # Technical Analysis (NEW)
            if technical_signal:
                weight = self.current_weights['technical_analysis']
                score = technical_signal.get('score', 0)
                signal_components['technical_analysis'] = score
                weighted_score += score * weight
                total_weight += weight
            
            # Pump.fun signal (reduced weight)
            if pump_fun_signal:
                weight = self.current_weights['pump_fun']
                score = self._normalize_signal_score(pump_fun_signal.confidence)
                signal_components['pump_fun'] = score
                weighted_score += score * weight
                total_weight += weight
            
            # Smart money signal
            if smart_money_signal:
                weight = self.current_weights['smart_money']
                score = self._normalize_signal_score(smart_money_signal.confidence)
                signal_components['smart_money'] = score
                weighted_score += score * weight
                total_weight += weight
            
            # Social sentiment (reduced weight)
            if social_sentiment_signal:
                weight = self.current_weights['social_sentiment']
                score = self._normalize_signal_score(getattr(social_sentiment_signal, 'confidence', 0.5))
                signal_components['social_sentiment'] = score
                weighted_score += score * weight
                total_weight += weight
            
            # AI analysis (minimal weight)
            if ai_analysis_signal:
                weight = self.current_weights['ai_analysis']
                score = self._normalize_signal_score(getattr(ai_analysis_signal, 'confidence', 0.5))
                signal_components['ai_analysis'] = score
                weighted_score += score * weight
                total_weight += weight
            
            # Calculate final composite score
            composite_score = weighted_score / total_weight if total_weight > 0 else 0
            
            # Calculate confidence based on signal agreement
            confidence = await self._calculate_signal_confidence(signal_components, technical_indicators)
            
            # Determine signal quality
            signal_quality = self._determine_signal_quality(confidence, technical_indicators)
            
            # Generate trading decision
            signal_type, reasoning, risk_factors = await self._generate_trading_decision(
                composite_score, confidence, technical_indicators, signal_components
            )
            
            # Calculate position sizing and targets
            position_size_pct = self._calculate_position_size(confidence, technical_indicators)
            entry_price = market_data.get('price', 0) if market_data else 0
            target_prices = self._calculate_profit_targets(entry_price, technical_indicators)
            stop_loss = self._calculate_stop_loss(entry_price, technical_indicators)
            
            return CompositeSignal(
                token_address=token_address,
                signal_type=signal_type,
                composite_score=composite_score,
                confidence=confidence,
                signal_quality=signal_quality,
                technical_indicators=technical_indicators,
                signal_components=signal_components,
                reasoning=reasoning,
                risk_factors=risk_factors,
                entry_price=entry_price,
                target_prices=target_prices,
                stop_loss=stop_loss,
                position_size_pct=position_size_pct,
                urgency=self._determine_urgency(confidence, technical_indicators),
                expires_at=time.time() + 300  # 5 minute expiry
            )
            
        except Exception as e:
            logger.error(f"Error processing composite signal: {e}")
            return CompositeSignal(
                token_address=token_address,
                signal_type="HOLD",
                composite_score=0.0,
                confidence=0.0,
                signal_quality=SignalQuality.REJECT,
                technical_indicators=TechnicalIndicators(),
                signal_components={},
                reasoning=f"Signal processing error: {str(e)}",
                risk_factors=["Processing error"],
                entry_price=0,
                target_prices=[],
                stop_loss=0,
                position_size_pct=0,
                urgency="low",
                expires_at=time.time()
            )
    
    async def _generate_technical_signal(self, indicators: TechnicalIndicators, market_data: Dict) -> Dict:
        """Generate signal from technical indicators"""
        try:
            score = 0.5  # Neutral starting point
            signals = []
            
            # RSI signals
            if indicators.rsi < 30:  # Oversold
                score += 0.2
                signals.append("RSI oversold")
            elif indicators.rsi > 70:  # Overbought
                score -= 0.2
                signals.append("RSI overbought")
            
            # MACD signals
            if indicators.macd_histogram > 0:
                score += 0.1
                signals.append("MACD bullish")
            else:
                score -= 0.1
                signals.append("MACD bearish")
            
            # Volume confirmation
            if indicators.volume_ratio > 2.0:  # High volume
                score += 0.15
                signals.append("High volume confirmation")
            elif indicators.volume_ratio < 0.5:  # Low volume
                score -= 0.1
                signals.append("Low volume warning")
            
            # Bollinger Bands
            if indicators.bb_position < 0.2:  # Near lower band
                score += 0.1
                signals.append("Near BB support")
            elif indicators.bb_position > 0.8:  # Near upper band
                score -= 0.1
                signals.append("Near BB resistance")
            
            # Trend strength
            if indicators.trend_strength > 0.7:
                score += 0.1
                signals.append("Strong trend")
            
            # Normalize score to 0-1 range
            score = max(0, min(1, score))
            
            return {
                'score': score,
                'signals': signals,
                'recommendation': 'BUY' if score > 0.6 else 'SELL' if score < 0.4 else 'HOLD'
            }
            
        except Exception as e:
            logger.error(f"Error generating technical signal: {e}")
            return {'score': 0.5, 'signals': [], 'recommendation': 'HOLD'}
    
    def _normalize_signal_score(self, confidence: float) -> float:
        """Normalize different signal confidence scales to 0-1"""
        return max(0, min(1, confidence))
    
    async def _calculate_signal_confidence(self, signal_components: Dict, indicators: TechnicalIndicators) -> float:
        """Calculate overall confidence based on signal agreement"""
        try:
            if not signal_components:
                return 0.0
            
            # Base confidence from signal agreement
            scores = list(signal_components.values())
            mean_score = np.mean(scores)
            score_std = np.std(scores)
            
            # High agreement (low std) increases confidence
            agreement_factor = max(0, 1 - score_std * 2)
            
            # Technical confirmation factor
            tech_factor = 1.0
            if indicators.volume_ratio > 1.5:  # Volume confirmation
                tech_factor += 0.1
            if indicators.trend_strength > 0.6:  # Strong trend
                tech_factor += 0.1
            if 0.3 < indicators.bb_position < 0.7:  # Not at extremes
                tech_factor += 0.05
            
            # Final confidence calculation
            confidence = mean_score * agreement_factor * min(1.2, tech_factor)
            return max(0, min(1, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def _determine_signal_quality(self, confidence: float, indicators: TechnicalIndicators) -> SignalQuality:
        """Determine signal quality based on confidence and technical factors"""
        # Adjust confidence based on technical factors
        adjusted_confidence = confidence
        
        # Penalize extreme volatility
        if indicators.volatility_percentile > 0.9:
            adjusted_confidence *= 0.8
        
        # Reward volume confirmation
        if indicators.volume_ratio > 1.5:
            adjusted_confidence *= 1.1
        
        # Determine quality level
        if adjusted_confidence >= 0.9:
            return SignalQuality.EXCELLENT
        elif adjusted_confidence >= 0.7:
            return SignalQuality.GOOD
        elif adjusted_confidence >= 0.5:
            return SignalQuality.AVERAGE
        elif adjusted_confidence >= 0.3:
            return SignalQuality.POOR
        else:
            return SignalQuality.REJECT
    
    async def _generate_trading_decision(
        self, 
        composite_score: float, 
        confidence: float, 
        indicators: TechnicalIndicators,
        signal_components: Dict
    ) -> Tuple[str, str, List[str]]:
        """Generate final trading decision with reasoning"""
        try:
            risk_factors = []
            
            # Check minimum confidence threshold
            if confidence < self.min_confidence:
                return "HOLD", f"Low confidence: {confidence:.2f} < {self.min_confidence}", ["Low confidence"]
            
            # Analyze risk factors
            if indicators.volatility_percentile > 0.9:
                risk_factors.append("Extremely high volatility")
            
            if indicators.volume_ratio < 0.5:
                risk_factors.append("Low trading volume")
            
            if indicators.rsi > 80:
                risk_factors.append("Severely overbought")
            elif indicators.rsi < 20:
                risk_factors.append("Severely oversold")
            
            # Generate decision
            if composite_score > 0.6 and confidence > 0.7:
                signal_type = "BUY"
                reasoning = f"Strong buy signal: score={composite_score:.2f}, confidence={confidence:.2f}"
                reasoning += f", components: {signal_components}"
            elif composite_score < 0.4 or confidence < 0.5:
                signal_type = "SELL"
                reasoning = f"Sell signal: score={composite_score:.2f}, confidence={confidence:.2f}"
            else:
                signal_type = "HOLD"
                reasoning = f"Neutral signal: score={composite_score:.2f}, confidence={confidence:.2f}"
            
            return signal_type, reasoning, risk_factors
            
        except Exception as e:
            logger.error(f"Error generating trading decision: {e}")
            return "HOLD", f"Decision error: {str(e)}", ["Processing error"]
    
    def _calculate_position_size(self, confidence: float, indicators: TechnicalIndicators) -> float:
        """Calculate position size based on confidence and risk"""
        try:
            base_size = self.max_position_size  # 2% base
            
            # Adjust for confidence
            confidence_factor = confidence  # Linear scaling with confidence
            
            # Adjust for volatility
            volatility_factor = 1.0
            if indicators.volatility_percentile > 0.8:
                volatility_factor = 0.5  # Reduce size in high volatility
            elif indicators.volatility_percentile < 0.3:
                volatility_factor = 1.2  # Increase size in low volatility
            
            # Adjust for volume
            volume_factor = 1.0
            if indicators.volume_ratio > 2.0:
                volume_factor = 1.1  # Increase size with volume confirmation
            elif indicators.volume_ratio < 0.7:
                volume_factor = 0.8  # Reduce size with low volume
            
            position_size = base_size * confidence_factor * volatility_factor * volume_factor
            return max(0.005, min(self.max_position_size, position_size))  # 0.5% min, 2% max
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Default 1%
    
    def _calculate_profit_targets(self, entry_price: float, indicators: TechnicalIndicators) -> List[float]:
        """Calculate multiple profit targets for scaling out"""
        try:
            if entry_price <= 0:
                return []
            
            # Base targets: 25%, 50%, 100% profit
            base_targets = [1.25, 1.50, 2.00]
            
            # Adjust targets based on volatility
            if indicators.volatility_percentile > 0.8:
                # High volatility - more aggressive targets
                targets = [1.30, 1.75, 3.00]
            elif indicators.volatility_percentile < 0.3:
                # Low volatility - conservative targets
                targets = [1.15, 1.30, 1.50]
            else:
                targets = base_targets
            
            return [entry_price * target for target in targets]
            
        except Exception as e:
            logger.error(f"Error calculating profit targets: {e}")
            return []
    
    def _calculate_stop_loss(self, entry_price: float, indicators: TechnicalIndicators) -> float:
        """Calculate dynamic stop loss based on technical factors"""
        try:
            if entry_price <= 0:
                return 0
            
            # Base stop loss: 10%
            base_stop = 0.10
            
            # Adjust for volatility
            if indicators.volatility_percentile > 0.8:
                stop_pct = 0.15  # Wider stop in high volatility
            elif indicators.volatility_percentile < 0.3:
                stop_pct = 0.08  # Tighter stop in low volatility
            else:
                stop_pct = base_stop
            
            # Adjust for support/resistance
            if indicators.support_resistance_score > 0.7:
                stop_pct *= 0.8  # Tighter stop near support
            
            return entry_price * (1 - stop_pct)
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return entry_price * 0.9 if entry_price > 0 else 0
    
    def _determine_urgency(self, confidence: float, indicators: TechnicalIndicators) -> str:
        """Determine trade urgency"""
        try:
            if confidence > 0.9 and indicators.volume_ratio > 2.0:
                return "critical"
            elif confidence > 0.8:
                return "high"
            elif confidence > 0.6:
                return "medium"
            else:
                return "low"
        except:
            return "low"
    
    async def _adjust_weights_for_market_regime(self):
        """Dynamically adjust signal weights based on market conditions"""
        try:
            # This would analyze overall market conditions
            # For now, using base weights
            self.current_weights = self.base_signal_weights.copy()
            
            # Example adjustments based on market regime
            if self.market_regime == MarketRegime.HIGH_VOLATILITY:
                # Increase technical analysis weight in volatile markets
                self.current_weights['technical_analysis'] = 0.35
                self.current_weights['social_sentiment'] = 0.10
            elif self.market_regime == MarketRegime.BULL_MARKET:
                # Increase pump.fun weight in bull markets
                self.current_weights['pump_fun'] = 0.30
                self.current_weights['technical_analysis'] = 0.25
            
        except Exception as e:
            logger.error(f"Error adjusting weights: {e}")
    
    def get_signal_weights(self) -> Dict[str, float]:
        """Get current signal weights"""
        return self.current_weights.copy()
    
    def update_market_regime(self, regime: MarketRegime):
        """Update market regime for dynamic weight adjustment"""
        self.market_regime = regime
        logger.info(f"Market regime updated to: {regime.value}") 